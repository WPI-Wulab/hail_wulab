package is.hail.methods

import is.hail.stats.eigSymD
import is.hail.utils.fatal

import breeze.linalg.{DenseMatrix => BDM, DenseVector => BDV, _}
import breeze.numerics.{abs, sqrt, sigmoid}
import breeze.optimize.{DiffFunction, minimize}
import net.sourceforge.jdistlib.{ChiSquare, Normal}
import is.hail.types.physical.PStruct

package object gfisher {

  def all[T](vals: Iterable[T], condition: (T) => Boolean): Boolean = {
    for (x <- vals) {
      if (!condition(x)) {
        return false
      }
    }
    return true
  }

  def all(bools: Iterable[Boolean]): Boolean = {
    for (bool <- bools) {
      if (!bool)
        return false
    }
    return true
  }

  def cor(x: BDV[Double], y: BDV[Double]): Double = {
    val n = x.length
    val mx = sum(x) / n.toDouble
    val my = sum(y) / n.toDouble
    val xmx = x - mx
    val ymy = y - my
    return (xmx dot ymy) / sqrt(xmx dot xmx) / sqrt(ymy dot ymy)
  }

  def rowCorBad(X: BDM[Double]): BDM[Double] = {
    val rows = X.rows
    val rowsD = rows.toDouble
    // val cols = X.cols
    val res = BDM.eye[Double](rows)
    val rowMeans = new Array[Double](rows)

    for (i <- 0 until rows) {
      rowMeans(i) = sum(X(i,::)) / rowsD
      //maybe: do x - xbar before
    }
    for (i <- 0 until (rows - 1)) {
      // maybe get Xi - Xbar here
      val xmx = X(i,::) - rowMeans(i)
      for (j <- (i + 1) until rows) {
        val ymy = X(j,::) - rowMeans(j)
        res(i, j) = sum(xmx *:* ymy) / sqrt(sum(xmx *:* xmx)) / sqrt(sum(ymy *:* ymy))
        res(j,i) = res(i,j)
      }
    }
    return res
  }

  def colCorrelation(x: BDM[Double]): BDM[Double] = {
    val rows = x.rows
    val cols = x.cols
    val colsD = cols.toDouble
    val X = new BDM(rows, cols, x.data.clone())
    val res = BDM.eye[Double](cols)
    val sumSqrs = new Array[Double](cols)
    for (i <- 0 until cols) {
      var sum = 0.0
      for (j <- 0 until rows)
        sum += X(j,i)
      val mean = sum / rows.toDouble
      var sumSqr = 0.0
      for (j <- 0 until rows) {
        X(j,i) = X(j,i) - mean
        sumSqr += math.pow(X(j,i), 2.0)
      }
      sumSqrs(i) = math.sqrt(sumSqr)
    }
    for (i <- 0 until cols) {
      for (j <- (i+1) until cols) {
        var numerator = 0.0
        for (k <- 0 until rows)
          numerator += X(k, i) * X(k, j)
        val denom = sumSqrs(i) * sumSqrs(j)
        res(i,j) = numerator / denom
        res(j,i) = res(i,j)
      }
    }
    return res
  }

  /**
    * Calculate Pearson's correlation coefficient between the rows of a matrix
    *
    * This method creates a copy of the matrix to avoid modifying it.
    *
    * @param x
    */
  def rowCorrelation(x: BDM[Double]): BDM[Double] = {
    val rows = x.rows
    val cols = x.cols
    val colsD = cols.toDouble
    val X = x.copy
    val res = BDM.eye[Double](rows)
    val sumSqrs = new Array[Double](rows)
    for (i <- 0 until rows) {
      var sum = 0.0
      for (j <- 0 until cols)
        sum += X(i,j)
      val mean = sum / colsD
      var sumSqr = 0.0
      for (j <- 0 until cols) {
        X(i,j) = X(i,j) - mean
        sumSqr += math.pow(X(i,j), 2.0)
      }
      sumSqrs(i) = math.sqrt(sumSqr)
    }
    for (i <- 0 until rows) {
      for (j <- (i+1) until rows) {
        var numerator = 0.0
        for (k <- 0 until cols)
          numerator += X(i, k) * X(j, k)
        val denom = sumSqrs(i) * sumSqrs(j)
        res(i,j) = numerator / denom
        res(j,i) = res(i,j)
      }
    }
    return res
  }

  def rowCorrelationSlow(X: BDM[Double]): BDM[Double] = {
    val rows = X.rows
    val cols = X.cols
    val rowMeans = sum(X(*,::)) / cols.toDouble
    val XMX = X(::, *) - rowMeans
    val res = BDM.eye[Double](rows)
    for (i <- 0 until (rows - 1)) {
      for (j <- (i+1) until (rows)) {
        res(i, j) = (XMX(i,::) dot XMX(j,::)) / sqrt(XMX(i,::) dot XMX(i,::)) / sqrt(XMX(j,::) dot XMX(j,::))
        res(j, i) = res(i, j)
      }
    }
    return res
  }

  def getFieldIds(fullRowType: PStruct, fieldNames: String*): Array[Int] = {
    val idxs = new Array[Int](fieldNames.length)
    for (i <- 0 until fieldNames.length)
      idxs(i) = fullRowType.field(fieldNames(i)).index
    return idxs
  }


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
  def lin_reg_predict(x: BDM[Double], y: BDV[Double], method: String = "direct", addIntercept: Boolean = true): BDV[Double] = {
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

  def stdErrLinearRegression3(X: BDM[Double], y: BDV[Double]): BDV[Double] = {
    val QR = qr.reduced(X)
    val yhat = QR.q * QR.q.t * y
    val residuals: BDV[Double] = y - yhat
    val sigma2: Double = (residuals dot residuals) / (X.rows - X.cols)
    val se: BDV[Double] = sqrt(diag(inv(QR.r.t * QR.r)) * sigma2)
    return se
  }




  /**
    * fit coefficients for a logistic regression model
    *
    * Created by Kylie Hoar
    *
    * Modified by Peter Howell, 06 Feb 2025
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
