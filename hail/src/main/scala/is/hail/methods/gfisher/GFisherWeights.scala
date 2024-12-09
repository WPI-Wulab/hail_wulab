package is.hail.methods.gfisher

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



}
