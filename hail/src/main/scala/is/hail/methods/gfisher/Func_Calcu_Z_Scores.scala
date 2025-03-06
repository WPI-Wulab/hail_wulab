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
import breeze.numerics._
import breeze.stats._
import breeze.optimize.{DiffFunction, minimize}
import is.hail.stats.LogisticRegressionModel

object FuncCalcuZScores {

  // For this function, I created my own logistic regression function because Scala's existing logistic regression models uses the Newton-Raphson method (Fisher scoring), which is different from the simple gradient descent necessary in this function
  // Newton-Raphson method converges more quickly to an accurate solution, while gradient descent will take longer to converge and will stop at a point not as optimal as Newton-Raphson
  // Gradient descent is also more optimal for larger datasets, because calculating the Hessian matrix (2nd derivative of the loss) is computationally expensive/unavailable

  /*
  Helper functions
  */

  /**
   * Train the logistic regression model using gradient descent and predict probabilities for the given feature matrix.
   * @param X Input feature matrix (rows: observations, cols: features).
   * @param y Target vector (binary labels: 0 or 1).
   * @return 
   */
  def log_reg(X: BDM[Double], y: BDV[Double]): BDV[Double] = {
    // Add intercept term by appending a column of ones to X
    val XWithIntercept = BDM.horzcat(BDM.ones[Double](X.rows, 1), X)

    // Define the negative log-likelihood function and its gradient
    val logisticLoss = new DiffFunction[BDV[Double]] {
      def calculate(beta: BDV[Double]): (Double, BDV[Double]) = {
        val preds = sigmoid(XWithIntercept * beta) // Predicted probabilities
        val logLikelihood = (y dot breeze.numerics.log(preds)) + 
          ((1.0 - y) dot breeze.numerics.log(1.0 - preds))
        val gradient = XWithIntercept.t * (preds - y) // Gradient of the loss function
        (-logLikelihood, gradient) // Return negative log-likelihood and gradient
      }
    }

    // Initialize coefficients (including intercept)
    val initialCoefficients = BDV.zeros[Double](XWithIntercept.cols)

    // Minimize the negative log-likelihood to find the best coefficients
    // val optimizer = new LBFGS[BDV[Double]](maxIter = 1000, tolerance = 1e-10)
    val coefficients = minimize(logisticLoss, initialCoefficients)

    // Predict probabilities for the given feature matrix
    sigmoid(XWithIntercept * coefficients)
  }

  /**
   * Calculate the Z score by the marginal t statistics for Y vs. each column of G and X
   * @param g Extracted column (BDV) of genotype matrix
   * @param X Input feature matrix (rows: observations, cols: features).
   * @param Y Target vector (binary labels: 0 or 1).
   */
  def contZScore (g: BDV[Double], X: BDM[Double], Y: BDV[Double]): Double = {
    // Combine column of g with X
    val XwithG = BDM.horzcat(X, g.toDenseMatrix.t)
    // Fit linear model to find initialCoefficients
    val beta = inv(XwithG.t * XwithG) * XwithG.t * Y 
    // Compute predictions
    val Y_hat = XwithG * beta
    // Compute residuals
    val res = Y - Y_hat
    val variance = (res.t * res) / (XwithG.rows - XwithG.cols)
    val stdError = sqrt(diag(inv(XwithG.t * XwithG)) * variance)
    // Z-score for g
    val tStatistic = beta(-1) / stdError(-1)
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
      trait_lm: String = "binary",
      use_lm_t: Boolean = false
  ): Map[String, Any] = {

    val scores = trait_lm match {

      case "binary" =>
        // Logistic regression to compute fitted values
        val Y0 = log_reg(X, Y)

        val sqrtY0 = sqrt(Y0 *:* (1.0 - Y0))
        val Xtilde = ((X(::, *) * sqrtY0).t).t
        val Hhalf = Xtilde * (cholesky(inv(Xtilde.t * Xtilde + BDM.eye[Double](X.cols) * 1e-6)))
        val Gtilde = ((G(::, *) * sqrtY0).t).t
        val GHhalf = Gtilde.t * Hhalf
        val GHG = Gtilde.t * Gtilde - GHhalf * GHhalf.t

        val score = G.t * (Y - Y0)
        val Zscore = score /:/ sqrt(diag(GHG))

        Map(
          "Zscores" -> Zscore,
          "scores" -> score,
          "M_Z" -> cov2cor(GHG),
          "M_s" -> GHG,
          // s0 = 1 for binary trait
          "s0" -> 1.0
        )

      case "continuous" =>
        val Hhalf = X * (cholesky(inv(X.t * X)))
        val GHalf = G.t * Hhalf
        val GHG = G.t * G - GHalf * GHalf.t

        // Fitting a Gaussian Model
        val XWithIntercept = BDM.horzcat(BDM.ones[Double](X.rows, 1), X)
        // Compute beta coefficients
        val beta = inv(XWithIntercept.t * XWithIntercept) * (XWithIntercept.t * Y)
        // Compute predictions
        val Y_hat = XWithIntercept * beta
        // Compute residuals
        val res = Y - Y_hat
        // estimate of the sd of error based on the null model
        val s0 = stddev(res)
        val score = G.t * res / s0

        val Zscore: BDV[Double] = if (use_lm_t) {
            BDV((0 until G.cols).map { j => contZScore(G(::, j), XWithIntercept, Y) }.toArray)
        } else {
            // Z scores based on the score statistics
            score /:/ sqrt(diag(GHG))
        }
        
        Map(
          "Zscores" -> Zscore,
          "scores" -> score,
          "M_Z" -> cov2cor(GHG),
          "M_s" -> GHG,
          "s0" -> s0
        )
        
      case _ =>
        throw new IllegalArgumentException(s"Unknown trait type: $trait_lm")
    }

    Map(
      "Zscores" -> scores("Zscores"),
      "scores" -> scores("scores"),
      "M_Z" -> scores("M_Z"),
      "M_s" -> scores("M_s"),
      "s0" -> scores("s0")
    )
  }

  def runTests(): Unit = {
    println("Running inline tests...")

  val G = BDM(
    (1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 0.0),
    (0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0),
    (1.0, 1.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0),
    (0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0),
    (1.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0),
    (0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 1.0, 1.0),
    (1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0),
    (0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0),
    (1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0),
    (0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0),
    (1.0, 1.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0),
    (0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 1.0),
    (1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0),
    (0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0),
    (1.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0),
    (0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0),
    (1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 0.0, 1.0),
    (0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0),
    (1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0),
    (0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0),
    (1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0),
    (0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0),
    (1.0, 1.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0),
    (0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 1.0, 0.0, 1.0, 0.0),
    (1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0),
    (0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 1.0),
    (1.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0),
    (0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0),
    (1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0),
    (0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0)
  )

    val X = BDM(
        (3.879049, 5.792453),
        (4.539645, 4.679680),
        (8.117417, 8.848960),
        (5.141017, 7.326775),
        (5.258575, 7.272450),
        (8.430130, 8.592345),
        (5.921832, 7.068752),
        (2.469878, 4.111115),
        (3.626294, 4.201222),
        (4.108676, 4.293396),
        (7.448164, 5.334668),
        (5.719628, 5.443979),
        (5.801543, 3.369979),
        (5.221365, 9.948595),
        (3.888318, 7.360083),
        (8.573826, 5.040696),
        (5.995701, 5.192081),
        (1.066766, 2.600072),
        (6.402712, 7.761286),
        (4.054417, 4.860470),
        (2.864353, 4.938813),
        (4.564050, 5.224932),
        (2.947991, 4.388255),
        (3.542218, 7.508313),
        (3.749921, 4.423419),
        (1.626613, 6.846248),
        (6.675574, 3.240281),
        (5.306746, 6.822601),
        (2.723726, 4.609572),
        (7.507630, 7.185698)
    )

    val Y = BDV(0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0)

    println(s"X dimensions: ${X.rows} x ${X.cols}")
    println(s"Y length: ${Y.length}")
    println(s"G dimensions: ${G.rows} x ${G.cols}")

    val result = FuncCalcuZScores.getZMargScore(G, X, Y)

    println("getZMargScore result:")
    println(result)
  }
}

// Run tests when the file is executed
//object Main extends App {
//  is.hail.methods.gfisher.FuncCalcuZScores.runTests()
//}