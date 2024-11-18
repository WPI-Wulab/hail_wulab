package is.hail.methods.gfisher

import breeze.linalg.{DenseMatrix => BDM, DenseVector => BDV, _}
import net.sourceforge.jdistlib.{ChiSquare, Normal}

import is.hail.GaussKronrod


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
    return x4 * x4 - (28.0 * x6) + (210.0 * x4) - (420.0 * x2) + 105.0
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
    one_sided:Boolean = false,
  ): (BDV[Double], BDV[Double],BDV[Double],BDV[Double]) = {
    // val x = linspace(-8, 8, n_integ)
    val coeff1 = BDV.zeros[Double](df.length)
    val coeff2 = BDV.zeros[Double](df.length)
    val coeff3 = BDV.zeros[Double](df.length)
    val coeff4 = BDV.zeros[Double](df.length)

    val (degs, f_to_integrate) = if (one_sided) {
      (BDV[Int](1,2,3,4), f_to_integrate_one(_, _, _))
    } else {
      (BDV[Int](2,4,6,8), f_to_integrate_two(_, _, _))
    }
    val integrator = new GaussKronrod(1e-8, 100);
    for (i <- 0 until df.length) {
      coeff1(i) = integrator.integrate((x) => {f_to_integrate(x, df(i), degs(0))}, -8, 8).estimate
      coeff2(i) = integrator.integrate((x) => {f_to_integrate(x, df(i), degs(1))}, -8, 8).estimate
      coeff3(i) = integrator.integrate((x) => {f_to_integrate(x, df(i), degs(2))}, -8, 8).estimate
      coeff4(i) = integrator.integrate((x) => {f_to_integrate(x, df(i), degs(3))}, -8, 8).estimate
    }
    return (coeff1, coeff2, coeff3, coeff4)
  }

}
