package is.hail.methods

import is.hail.stats.eigSymD
import is.hail.utils.fatal

import breeze.linalg.{DenseMatrix => BDM, DenseVector => BDV, _}
import breeze.numerics.{abs, sqrt, sigmoid}
import breeze.optimize.{DiffFunction, minimize}
import net.sourceforge.jdistlib.{ChiSquare, Normal}


package object gfisher {

  def mean(x: BDV[Double]): Double = sum(x) / x.size

  /**
    * Uses method described in the _linear_skat function in statgen.py to directly compute the predicted values of the best fit model y = Xb
    *
    * @param x
    * @param y
    * @param addIntercept whether to add a column of ones to x.
    * @param method method to solve the equation. "direct" bypasses the coefficient and goes directly to the best-predicted value using the reduced QR decomposition. "qr" uses the method that Skat.scala uses to calculate beta which uses breeze's solve after using QR.
    * "breeze" simply does `X \ y`. "naive" uses the closed form equation for OLS, `inv(X.t * X) * X.t * y`
    *
    */
  def lin_reg_predict(x: BDM[Double], y: BDV[Double], addIntercept: Boolean = true, method:String = "direct"): BDV[Double] = {
    val X = if (addIntercept) {
      BDM.horzcat(BDM.ones[Double](x.rows, 1), x)
    } else {
      x
    }
    if (method == "direct") {
      val QR = qr.reduced(X)
      val q = QR.q
      return q * q.t * y
    }
    val beta = lin_solve(X, y, method, false)
    return X * beta
  }

  /**
    * wrapper for solving linear equation Xb = y for b. returns the solution. Like `np.solve(X, y)` with numpy, or `X \ y` in matlab, or `solve(X, y)` in R
    *
    * @param X matrix
    * @param y vector
    * @param method how to compute it either qr, breeze, or naive.
    * `qr` uses the method that Skat.scala uses, which still uses breeze's solve.
    * `breeze` simply does `X \ y`.
    * `naive` uses the closed form equation for OLS, `inv(X.t * X) * X.t * y`
    */
  def lin_solve(x: BDM[Double], y: BDV[Double], method: String = "qr", addIntercept: Boolean = false): BDV[Double] = {
    val X = if (addIntercept) {
      BDM.horzcat(BDM.ones[Double](x.rows, 1), x)
    } else {
      x
    }
    method match {
      case "qr" =>
        val QR = qr.reduced(x)
        val Qt = QR.q.t
        val R = QR.r
        R \ (Qt * y)
      case "breeze" => X \ y
      case "naive" => inv(X.t * X) * X.t * y
      case _ => throw new IllegalArgumentException(s"unknown method: $method")
    }
  }

  /**
    * fit coefficients for a logistic regression model
    *
    * @param X feature matrix
    * @param y response variable
    * @param addIntercept
    * @return the fitted coefficients
    */
  def log_reg_fit(X: BDM[Double], y: BDV[Double], addIntercept: Boolean=false): BDV[Double] = {
    // Add intercept term by appending a column of ones to X
    val XWithIntercept = if (addIntercept) {
      BDM.horzcat(BDM.ones[Double](X.rows, 1), X)
    } else X

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
    return coefficients
  }

  /**
    * Train the logistic regression model using gradient descent and predict probabilities for the given feature matrix.
    * @param X Input feature matrix (rows: observations, cols: features).
    * @param y Target vector (binary labels: 0 or 1).
    * @return predicted probabilities from fitted model
    */
  def log_reg_predict(X: BDM[Double], y: BDV[Double]): BDV[Double] = {
    val XWithIntercept = BDM.horzcat(BDM.ones[Double](X.rows, 1), X)
    val coefficients = log_reg_fit(X, y, false)
    // Predict probabilities for the given feature matrix
    return sigmoid(XWithIntercept * coefficients)
  }

  /**
    * GFisher transformation function 'g' for two sided p-values
    *
    * @param x
    * @param df
    */
  def gGFisher2(x: Double, df: Int=2): Double = {
    return ChiSquare.quantile(
      math.log(2.0) +
      Normal.cumulative(
        math.abs(x),
        0.0,
        1.0,
        false, //lower tail
        true //log_p
      ),
      df,
      false, //lower_tail
      true //log_p
    )
  }

  /**
    * Compute hermite polynomil for a scalar input
    *
    * @param x input
    * @param degree degree of polynomial. should be 1, 2, 3, 4, 6, or 8
    */
  def hermite_scalar(x: Double, degree: Int): Double = {
    return degree match {
      case 1 => x
      case 2 => x*x - 1.0
      case 3 => math.pow(x, 3.0) - 3.0   * x
      case 4 => math.pow(x, 4.0) - 6.0   * math.pow(x, 2.0) + 3.0
      case 5 => math.pow(x, 5.0) - 10.0  * math.pow(x, 3.0) + 15.0  * x
      case 6 => math.pow(x, 6.0) - 15.0  * math.pow(x, 4.0) + 45.0  * math.pow(x, 2.0) - 15.0
      case 7 => math.pow(x, 7.0) - 21.0  * math.pow(x, 5.0) + 105.0 * math.pow(x, 3.0) - 105.0 * x
      case _ => math.pow(x, 8.0) - 28.0  * math.pow(x, 6.0) + 210.0 * math.pow(x, 4.0) - 420.0 * math.pow(x, 2.0) + 105.0
    }
  }


  /**
    * Code to convert a covariance matrix to a correlation matrix. Should mimic R's function.
    *
    * @param X
    */
  def cov2cor(X: BDM[Double]): BDM[Double] = {
    X / (sqrt(diag(X)) * sqrt(diag(X)).t)
  }

/**
  * Computes the infinity norm of a matrix, ie, the maximum of the row sums of the absolute values.
  *
  * @param X matrix
  *
  */
  def normI(X: BDM[Double]): Double = {
    val aX = abs(X)
    return max(sum(aX(*, ::)))
  }

  /**
    * Code that copies R's code for getting the nearest positive definite matrix to an approximate one
    *
    * @TODO benchmark efficiency of var vs val and reassignment with :=
    * @param M symmetric square
    */
  def nearPD(M: BDM[Double]): BDM[Double] = {
    val n: Int = M.cols
    if (M.rows != n) fatal("Matrx M must be a square matrix")

    //tolerances
    val eigTol: Double = 1e-06
    val convTol: Double = 1e-07
    val posdTol: Double = 1e-08

    var X: BDM[Double] = M // n x n
    var DS: BDM[Double] = BDM.zeros[Double](n, n)
    var R: BDM[Double] = BDM.zeros[Double](n, n)
    var iter: Int = 0
    var converg: Boolean = false

    var conv: Double = Double.PositiveInfinity
    while (iter < 100 && !converg) {
      val Y: BDM[Double] = X
      R = Y - DS
      val eigRes = eigSymD(R)
      val (eigenvalues: BDV[Double], eigenvectors: BDM[Double]) = (eigRes.eigenvalues, eigRes.eigenvectors)

      // val adjustedEigenvalues = eigenvalues.map(ev => if (ev > eigTol * eigenvalues(0)) ev else 1e-10)
      val p: BitVector = eigenvalues >:> (eigTol * eigenvalues(0))
      if (! any(p) ) {
        fatal("Matrix is negative definite")
      }
      val Q: BDM[Double] = eigenvectors(::, p).toDenseMatrix
      val d: BDV[Double] = eigenvalues(p).toDenseVector
      // X = (Q *:* tile(d, 1, n).t) * Q.t // equivalent to below
      X = (Q(*,::) *:* d) * Q.t // equivalent to above
      DS = X - R

      diag(X) := 1.0
      conv = normI(Y - X) / normI(X)
      iter += 1
      converg = conv <= convTol
    }
    val eigRes = eigSymD(X)
    val (eigenvalues: BDV[Double], eigenvectors: BDM[Double]) = (eigRes.eigenvalues, eigRes.eigenvectors)
    val Eps = posdTol * abs(eigenvalues(0))
    if (eigenvalues(-1) < Eps) {
      eigenvalues(eigenvalues <:< Eps) := Eps
      val D = diag(X)
      X = eigenvectors * (eigenvalues * eigenvectors.t)
      D(D <:< Eps) := Eps
      D := sqrt(D / diag(X))
      X = (X * diag(D)).t *:* tile(D, 1, D.size).t
    }
    diag(X) := 1.0
    return X
  }


  def tabulateSymmetric(n: Int)(f: (Int, Int) => Double): BDM[Double] = {
    val a = new Array[Double](n*n)
    for (i <- 0 until n) {
      for (j <- i until n) {
        val v = f(i, j)
        a(i*n+j) = v
        a(j*n+i) = v
      }
    }
    new BDM(n, n, a)
  }


}
