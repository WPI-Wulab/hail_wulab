/*
This file contains main and supportive functions for computing Z-scores from a matrix of genotypes, covariates,
and a binary response
Reference: Zhang, Hong, and Zheyang Wu. "The generalized Fisher's combination and accurate p‐value
           calculation under dependence." Biometrics 79.2 (2023): 1159-1172.
Creators: Kylie Hoar
Last update (latest update first):
  KHoar 2025-01-16: sample format for future edits
*/

package is.hail.methods.gfisher

import breeze.linalg.{DenseMatrix => BDM, DenseVector => BDV, _}
import breeze.numerics.{sqrt, signum}

import is.hail.methods.gfisher.OptimalWeights.{getGHG_Binary, getGHG_Binary2, getGHG_Continuous}
import net.sourceforge.jdistlib.Normal

object FuncCalcuZScores {

  /*
  Helper functions
  */

  /**
    * Calculate the Z score by the marginal t statistics for Y vs. each column of G and X
    * @param g Extracted column (BDV) of genotype matrix
    * @param X Input feature matrix (rows: observations, cols: features) with first column of ones for intercept
    * @param Y Target vector (binary labels: 0 or 1).
    */
  def contZScore (g: BDV[Double], X: BDM[Double], Y: BDV[Double]): Double = {
    // Combine column of g with X
    val XwithG = BDM.horzcat(X, g.toDenseMatrix.t)
    val (beta, se, _) = stdErrLinearRegression(XwithG, Y)
    // Z-score for g
    val tStatistic = beta(-1) / se(-1)
    tStatistic
  }

  def saddleProb(q: Double, mu: BDV[Double], g: BDV[Double]): Double = {
    // Compute variance (inner product of g * g)
    val sigmaSq = g dot g

    // Ensure sigmaSq is positive to avoid division errors
    if (sigmaSq <= 0) {
        throw new IllegalArgumentException("Variance must be positive")
    }

    // Compute adjusted mean
    val muAdj = mu dot g

    // Compute standardized test statistic
    val z = (q - muAdj) / math.sqrt(sigmaSq)

    // Compute p-value using jdistlib Normal CDF
    val pValue = 2.0 * (1.0 - Normal.cumulative(Math.abs(z), 0.0, 1.0, false, false))

    pValue
  }


  /*
  Main function
  */

  /**
   * Get the standardized marginal score statistics
   * @param G A matrix of genotype (# of individuals x # of SNPs)
   * @param X A matrix of covariates, default is 1
   * @param Y A single column of response variable; it has to be 0/1 for binary trait
   * @param trait_lm indicator of "binary" (logistic regression) or "continuous" (linear regression).
   * @param use_lm_t whether to use the lm() function to get the t statistics as the Z-scores for continuous trait
   * @return Zscores A vector of Z scores of the marginal score statistics for each SNP.
   * @return scores A vector of the marginal score statistics for each SNP
   * @return M_Z the correlation matrix of Z scores when effects are fixed (including H0) given X and G fixed.
   * @return M_s the covariance matrix of the score statistics when effects are fixed (including H0) given X and G fixed.
   * @return s0 the estimated dispersion parameter. Binary trait: s0=1. Continuous trait: residual SD under the null model.
   */
  def getZMargScore(
    G: BDM[Double],
    X: BDM[Double],
    Y: BDV[Double],
    binary: Boolean = false,
    use_lm_t: Boolean = false
  ): Map[String, Any] = {

    if (binary) {
      val (ghg, _, resids) = getGHG_Binary2(G, X, Y)

      val score = G.t * resids
      val Zscore = score /:/ sqrt(diag(ghg))

      return Map(
        "Zscores" -> Zscore,
        "scores" -> score,
        "M_Z" -> cov2cor(ghg),
        "M_s" -> ghg,
        // s0 = 1 for binary trait
        "s0" -> 1.0
      )
    }

    else {
      val (ghg, s0, resids) = getGHG_Continuous(G, X, Y)
      // Compute residuals
      val score = G.t * resids / s0

      val Zscore: BDV[Double] = if (use_lm_t) {
        val XWithIntercept = BDM.horzcat(BDM.ones[Double](X.rows, 1), X)
        BDV((0 until G.cols).map { j => contZScore(G(::, j), XWithIntercept, Y) }.toArray)
      } else {
        // Z scores based on the score statistics
        score /:/ sqrt(diag(ghg))
      }

      return Map(
        "Zscores" -> Zscore,
        "scores" -> score,
        "M_Z" -> cov2cor(ghg),
        "M_s" -> ghg,
        "s0" -> s0
      )
  }
}

  def getZ_marg_score_binary_SPA (
    G: BDM[Double],
    X: BDM[Double],
    Y: BDV[Double]
  ): Map[String, Any] = {
    // Logistic regression model
    val X1 = BDM.horzcat(BDM.ones[Double](X.rows, 1), X)
    val Y0 = log_reg_predict(X, Y)

    // Calculate XVX inverse
    val sqrtY0 = sqrt(Y0 *:* (1.0 - Y0))
    val XV = ((X1(::, *) * sqrtY0).t).t
    val XVX = X1.t * XV
    val XVX_inv = inv(XVX)

    // Compute XXVX_inv and Gscale
    val XXVX_inv = X1 * XVX_inv
    val Gscale = G - (XXVX_inv.t * (XV * G.t)).t

    // Compute score
    val score = Gscale.t * Y

    // Compute p-values (using a placeholder function)
    val pval_spa = BDV((0 until score.size).map { x =>
      saddleProb(score(x), Y0, Gscale(::, x))
    }.toArray)

    // Compute Z-scores
    val Zscores_spa = BDV(pval_spa.map(p => Normal.quantile(1.0 - p / 2.0, 0.0, 1.0, false, false)).toArray) *:* signum(score)

    // Compute GHG and M
    val Xtilde = ((X(::, *) * sqrtY0).t).t
    val Hhalf = Xtilde * cholesky(inv(Xtilde.t * Xtilde)).t
    val Gtilde = ((G(::, *) * sqrtY0).t).t
    val GHhalf = Gtilde.t * Hhalf
    val GHG = (Gtilde.t * Gtilde) - (GHhalf * GHhalf.t)
    val M = cov2cor(GHG)

    // Dispersion parameter
    val s0 = 1.0

    Map(
      "Zscores" -> Zscores_spa,
      "M" -> M,
      "GHG" -> GHG,
      "s0" -> s0
    )
  }


  /**
   * Get the standardized marginal score statistics
   * @param G A matrix of genotype (# of individuals x # of SNPs)
   * @return Zscores A vector of Z scores of the marginal score statistics for each SNP.
   * @return scores A vector of the marginal score statistics for each SNP
   * @return M_Z the correlation matrix of Z scores when effects are fixed (including H0) given X and G fixed.
   * @return M_s the covariance matrix of the score statistics when effects are fixed (including H0) given X and G fixed.
   * @return s0 the estimated dispersion parameter. Binary trait: s0=1. Continuous trait: residual SD under the null model.
   */
  def getZMargScoreBinary(
    G: BDM[Double],
    HHalf: BDM[Double],
    y0: BDV[Double],
    resids: BDV[Double],
  ): Map[String, Any] = {

    val GHG = getGHG_Binary(G, HHalf, y0)

    val score = G.t * resids
    val Zscore = score /:/ sqrt(diag(GHG))

    return Map(
      "Zscores" -> Zscore,
      "scores" -> score,
      "M_Z" -> cov2cor(GHG),
      "M_s" -> GHG,
      // s0 = 1 for binary trait
      "s0" -> 1.0
    )

  }


  /**
   * Get the standardized marginal score statistics
   * @param G A matrix of genotype (# of individuals x # of SNPs)
   * @return Zscores A vector of Z scores of the marginal score statistics for each SNP.
   * @return scores A vector of the marginal score statistics for each SNP
   * @return M_Z the correlation matrix of Z scores when effects are fixed (including H0) given X and G fixed.
   * @return M_s the covariance matrix of the score statistics when effects are fixed (including H0) given X and G fixed.
   * @return s0 the estimated dispersion parameter. Binary trait: s0=1. Continuous trait: residual SD under the null model.
   */
  def getZMargScoreContinuous(
    G: BDM[Double],
    HHalf: BDM[Double],
    s0: Double,
    resids: BDV[Double],
    use_t: Boolean = false
  ): Map[String, Any] = {

    val GHG = getGHG_Continuous(G, HHalf)

    val score = G.t * resids / s0
    // TEMPORARY SOLUTION
    val Zscore: BDV[Double] = score /:/ sqrt(diag(GHG))
    // UNDO THAT LATER AND USE CODE BELOW
    // val Zscore: BDV[Double] = if (use_t) {
    //   val XWithIntercept = BDM.horzcat(BDM.ones[Double](X.rows, 1), X)
    //   BDV((0 until G.cols).map { j => contZScore(G(::, j), XWithIntercept, Y) }.toArray)
    // } else {
    //   // Z scores based on the score statistics
    //   score /:/ sqrt(diag(GHG))
    // }


    return Map(
      "Zscores" -> Zscore,
      "scores" -> score,
      "M_Z" -> cov2cor(GHG),
      "M_s" -> GHG,
      // s0 = 1 for binary trait
      "s0" -> 1.0
    )

  }
}
