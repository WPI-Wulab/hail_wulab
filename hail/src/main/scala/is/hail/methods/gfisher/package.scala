package is.hail.methods

import is.hail.stats.eigSymD
import is.hail.utils.fatal

import breeze.linalg.{DenseMatrix => BDM, DenseVector => BDV, _}
import breeze.numerics.{abs, sqrt}

import net.sourceforge.jdistlib.{ChiSquare, Normal}


package object gfisher {

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
