package is.hail.stats

import breeze.linalg.{DenseMatrix => BDM, DenseVector => BDV, _}
import breeze.integrate.{simpson => simpsonBreeze}
import breeze.numerics.sqrt
import net.sourceforge.jdistlib.{ChiSquare, Normal}


object GFisherCoefs {

  /**
    * Compute hermite polynomil for a scalar input
    *
    * @param x input
    * @param degree degree of polynomial
    */
  def hermite_scalar(x: Double, degree: Int): Double = {
    if (degree == 1) return x
    val x2 = x * x
    if (degree == 2) return x2 - 1.0
    if (degree == 3) return (x2 * x) - (3.0 * x)

    val x4 = x2 * x2

    if (degree == 4) return x4 - (6.0 * x2) + 3.0

    val x6 = x4 * x2

    if (degree == 6) return x6 - (15.0 * x4) + (45.0 * x2) - 15.0
    assert(degree == 8)
    return x4* x4 - (28.0 * x6) + (210.0 * x4) - (420.0 * x2) + 105.0
  }

  /**
    * function to integrate if p-values are two-sided
    *
    * @param x variable of integration
    * @param df degrees of freedom
    * @param deg hermite polynomial degree
    */
  def f_to_integrate_two(x:Double, df:Int, deg: Int): Double = {
    ChiSquare.quantile(ChiSquare.cumulative(x*x, 1, true, false), df, true, false) * Normal.density(x, 0, 1.0, false) * hermite_scalar(x, deg)
  }

  /**
    * function to integrate if p-values are one-sided
    *
    * @param x variable of integration
    * @param df degrees of freedom
    * @param deg hermite polynomial degree
    */
  def f_to_integrate_one(x:Double, df:Int, deg: Int): Double = {
    ChiSquare.quantile(Normal.cumulative(x, 0, 1), df, true, false) * Normal.density(x, 0, 1.0, false) * hermite_scalar(x, deg)
  }

  /**
    * Compute necessary coefficients for estimating the covariance of the test statistics
    *
    * @param df vector of degrees of freedom
    * @param one_sided whether the p-values are one_sided
    * @param n_integ number of abscissae to use
    * @return
    */
  def getGFisherCoefs(
    df: BDV[Int],
    one_sided:Boolean = true,
    n_integ: Int = 100,
  ): (BDV[Double], BDV[Double],BDV[Double],BDV[Double]) = {

    // val x = linspace(-8, 8, n_integ)
    val coeff1 = BDV.zeros[Double](df.length)
    val coeff2 = BDV.zeros[Double](df.length)
    val coeff3 = BDV.zeros[Double](df.length)
    val coeff4 = BDV.zeros[Double](df.length)

    val absc =  if (n_integ % 2 == 0)  n_integ + 1 else  n_integ

    val (degs, f_to_integrate) = if (one_sided) {
      (BDV[Int](1,2,3,4), f_to_integrate_one(_, _, _))
    } else {
      (BDV[Int](2,4,6,8), f_to_integrate_two(_, _, _))
    }

    for (i <- 0 until df.length) {
      coeff1(i) = simpsonBreeze((x) => {f_to_integrate(x, df(i), degs(0))}, -8, 8, absc)
      coeff2(i) = simpsonBreeze((x) => {f_to_integrate(x, df(i), degs(1))}, -8, 8, absc)
      coeff3(i) = simpsonBreeze((x) => {f_to_integrate(x, df(i), degs(2))}, -8, 8, absc)
      coeff4(i) = simpsonBreeze((x) => {f_to_integrate(x, df(i), degs(3))}, -8, 8, absc)
    }
    return (coeff1, coeff2, coeff3, coeff4)
  }

}

object GFisherGM {

  /**
    * Code to convert a covariance matrix to a correlation matrix. Should mimic R's function.
    *
    * @param X
    */
  def cov2cor(X: BDM[Double]): BDM[Double] = {
    X / (sqrt(diag(X)) * sqrt(diag(X)).t)
  }

  /**
    * Calculate covariance matrix for w_iT_i,i=1,...,n in Theorem 1 / Corollary 1
    *
    * @param df n-dimensional vector of degrees of freedom
    * @param w n-dimensional vector of weights
    * @param M n by n matrix correlation matrix
    * @param one_sided whether the p-values are one-sided or not
    * @param n_integ number of points to integrate along. passed to GFisherCoefs.getGFisherCoeffs
    * @param method method of numerical integration. passed to GFisherCoefs.getGFisherCoeffs
    */
  def getGfisherGM(
    df: BDV[Int],
    w: BDV[Double],
    M: BDM[Double],
    one_sided: Boolean,
    n_integ: Int,
  ): BDM[Double] = {
    val (c1, c2, c3, c4) = GFisherCoefs.getGFisherCoefs(df, one_sided, n_integ)

    val GM = if (one_sided) {
      M *:* (c1 * c1.t) + ((M ^:^ 2.0)/ 2.0) *:* (c2 * c2.t) + ((M ^:^ 3.0)/ 6.0) *:* (c3 * c3.t) + ((M ^:^ 4.0)/ 24.0) *:* (c4 * c4.t)
    } else {
      ((M ^:^ 2.0)/ 2.0) *:* (c1 * c1.t) + ((M ^:^ 4.0)/ 24.0) *:* (c2 * c2.t) + ((M ^:^ 6.0)/ 720.0) *:* (c3 * c3.t) + ((M ^:^ 8.0) / 40320.0) *:* (c4 * c4.t)
    }
    return diag(sqrt(2*df)) * cov2cor(GM) * diag(sqrt(2*df))
  }

}
