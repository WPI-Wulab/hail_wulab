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
import breeze.numerics.sqrt
// import breeze.stats._
import is.hail.methods.gfisher.OptimalWeights.{getGHG_Binary, getGHG_Continuous}
object FuncCalcuZScores {

  // For this function, I created my own logistic regression function because Scala's existing logistic regression models uses the Newton-Raphson method (Fisher scoring), which is different from the simple gradient descent necessary in this function
  // Newton-Raphson method converges more quickly to an accurate solution, while gradient descent will take longer to converge and will stop at a point not as optimal as Newton-Raphson
  // Gradient descent is also more optimal for larger datasets, because calculating the Hessian matrix (2nd derivative of the loss) is computationally expensive/unavailable

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
      val (ghg, _, resids) = getGHG_Binary(G, X, Y)

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
      val XWithIntercept = BDM.horzcat(BDM.ones[Double](X.rows, 1), X)
      val (ghg, s0, resids) = getGHG_Continuous(G, X, Y)
      // Compute residuals
      val score = G.t * resids / s0

      val Zscore: BDV[Double] = if (use_lm_t) {
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
}
