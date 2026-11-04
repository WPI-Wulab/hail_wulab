/**
  * File that contains the functions to compute necessary coefficients for estimating the covariance of the test statistics
  * Handles both one and two-sided inputs
  * Reference: Zhang, Hong, and Zheyang Wu. "The generalized Fisher's combination and accurate p‐value
           calculation under dependence." Biometrics 79.2 (2023): 1159-1172.
  * @author Peter Howell
  * Last update (latest update first):
  *   KHoar 2025-05-07: Added header to file
  */

package is.hail.methods.gfisher

import breeze.linalg.{DenseVector => BDV}
import net.sourceforge.jdistlib.{ChiSquare, Normal}

import is.hail.GaussKronrod


object GFisherCoefs {



  /**
    * function to integrate if p-values are two-sided
    *
    * @param x variable of integration
    * @param df degrees of freedom
    * @param deg hermite polynomial degree
    */
  def f_to_integrate_two(x:Double, df:Double, deg: Int): Double = {
    ChiSquare.quantile(ChiSquare.cumulative(x*x, 1, true, false), df, true, false) * Normal.density(x, 0, 1.0, false) * hermite_scalar(x, deg)
  }

  /**
    * function to integrate if p-values are one-sided
    *
    * @param x variable of integration
    * @param df degrees of freedom
    * @param deg hermite polynomial degree
    */
  def f_to_integrate_one(x:Double, df:Double, deg: Int): Double = {
    ChiSquare.quantile(Normal.cumulative(x, 0, 1), df, true, false) * Normal.density(x, 0, 1.0, false) * hermite_scalar(x, deg)
  }

  /**
    * Perform the integration to compute coefficients necessary for the GFisherGM Matrix for one-sided p-values
    *
    * @param df vector of degrees of freedom
    * @return 4 coefficient vectors, each the same length as `df`
    */
  def getGFisherCoefs1(df: BDV[Double]): (BDV[Double], BDV[Double], BDV[Double], BDV[Double]) = {
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
  def getGFisherCoefs2(df: BDV[Double]): (BDV[Double], BDV[Double], BDV[Double], BDV[Double]) = {
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
    df: BDV[Double],
    one_sided:Boolean = false,
  ): (BDV[Double], BDV[Double],BDV[Double],BDV[Double]) = {
    if (one_sided) return getGFisherCoefs1(df)
    return getGFisherCoefs2(df)
  }

}
