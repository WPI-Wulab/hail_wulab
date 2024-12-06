package is.hail.methods.gfisher

import breeze.linalg.{DenseVector => BDV}
import net.sourceforge.jdistlib.{ChiSquare, Normal}

import is.hail.GaussKronrod


object GFisherCoefs {

  /**
    * Compute hermite polynomil for a scalar input
    *
    * @param x input
    * @param degree degree of polynomial. should be 1, 2, 3, 4, 6, or 8
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
    * Perform the integration to compute coefficients necessary for the GFisherGM Matrix for one-sided p-values
    *
    * @param df vector of degrees of freedom
    * @return 4 coefficient vectors, each the same length as `df`
    */
  def getGFisherCoefs1(df: BDV[Int]): (BDV[Double], BDV[Double], BDV[Double], BDV[Double]) = {
    val degs = Array(1,2,3,4)
    val dfUnique = df.data.distinct
    val integrator = new GaussKronrod(1e-8, 100)
    // a lot going here. If any of the degrees of freedom are duplicates, the coefficients will be the same
    // so we only need to calculate a coefficient once for each distinct degree of freedom. This just does that.
    // we get the distinct values and then calculate the coefficients for each of them.
    // Then we create a Map (which is just like a python dictionary), to map the distinct coefficients back to the original vector
    val c1 = dfUnique.map((d) => d -> integrator.integrate(f_to_integrate_one(_, d, degs(0)), -8, 8).estimate).toMap
    val c2 = dfUnique.map((d) => d -> integrator.integrate(f_to_integrate_one(_, d, degs(1)), -8, 8).estimate).toMap
    val c3 = dfUnique.map((d) => d -> integrator.integrate(f_to_integrate_one(_, d, degs(2)), -8, 8).estimate).toMap
    val c4 = dfUnique.map((d) => d -> integrator.integrate(f_to_integrate_one(_, d, degs(3)), -8, 8).estimate).toMap
    return (df.map(c1), df.map(c2), df.map(c3), df.map(c4))
  }

  /**
    * Perform the integration to compute coefficients necessary for the GFisherGM Matrix for two-sided p-values
    *
    * @param df vector of degrees of freedom
    * @return 4 coefficient vectors, each the same length as `df`
    */
  def getGFisherCoefs2(df: BDV[Int]): (BDV[Double], BDV[Double], BDV[Double], BDV[Double]) = {
    val degs = Array(2,4,6,8)
    val dfUnique = df.data.distinct
    val integrator = new GaussKronrod(1e-8, 100)
    // a lot going here. If any of the degrees of freedom are duplicates, the coefficients will be the same
    // so we only need to calculate a coefficient once for each distinct degree of freedom. This just does that.
    // we get the distinct values and then calculate the coefficients for each of them.
    // Then we create a Map (which is just like a python dictionary), to map the distinct coefficients back to the original vector
    val c1 = dfUnique.map((d) => d -> integrator.integrate(f_to_integrate_two(_, d, degs(0)), -8, 8).estimate).toMap
    val c2 = dfUnique.map((d) => d -> integrator.integrate(f_to_integrate_two(_, d, degs(1)), -8, 8).estimate).toMap
    val c3 = dfUnique.map((d) => d -> integrator.integrate(f_to_integrate_two(_, d, degs(2)), -8, 8).estimate).toMap
    val c4 = dfUnique.map((d) => d -> integrator.integrate(f_to_integrate_two(_, d, degs(3)), -8, 8).estimate).toMap
    return (df.map(c1), df.map(c2), df.map(c3), df.map(c4))
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
    if (one_sided) return getGFisherCoefs1(df)
    return getGFisherCoefs2(df)
  }

}
