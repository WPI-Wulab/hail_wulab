package is.hail.methods.gfisher

import is.hail.GaussKronrod

import breeze.linalg.{DenseVector => BDV, DenseMatrix => BDM, _}

object GFisherWeights {


  /**
    * Compute hermite polynomial of shifted value at certain degree
    *
    * @param x value
    * @param mu value to shift by
    * @param deg degree/order of hermite polynomial
    */
  def hermite_shifted(x: Double, mu: Double, deg: Int): Double = {
    return hermite_scalar(x-mu, deg)
  }


  /**
    * Calculate E[g(X)^p] where X~N(mu,sigma)
    *
    * @param g function
    * @param mu mean
    * @param p power
    * @param sigma std
    */
  def E_gX_p(g: Double => Double, mu: Double, p: Double, sigma:Double=1.0): Double = {
    val integrator = new GaussKronrod(1e-9, 100)
    return integrator.integrate((x) => math.pow(g(x), p) * Normal.density((x-mu)/sigma, 0, 1.0, false), mu-8*sigma, mu+8*sigma).estimate / sigma
  }


}
