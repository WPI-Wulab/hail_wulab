package is.hail.methods.gfisher

import is.hail.GaussKronrod

import breeze.linalg.{DenseVector => BDV, DenseMatrix => BDM, _}

import net.sourceforge.jdistlib.Normal

import org.apache.commons.math3.util.CombinatoricsUtils.factorialDouble

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

  def getWts(Sigma: BDM[Double], r: BDV[Double], forcePosW: Boolean=true, normalize:Boolean = true): BDV[Double] = {
    val w = inv(Sigma) * r
    if (forcePosW) w(w <:< 0.0) := 0.0
    if (normalize) w := w /:/ sum(w)
    return w
  }

  /**
    * Calculate Covariance matrix whose elements are Cov[g(Xi),g(Yj)] where X = Z + MU1 ~ MVN(MU1, M), Y = Z + MU2 ~ MVN(MU2, M)
    *
    * @param g transformation function
    * @param mu1 mean vector of length n
    * @param mu2 another mean vector of length n
    * @param M a positive-definite correlation matrix
    * @param ORD a sequence of orders of Hermite polynomials
    */
  def covM_gXgY(g: Double => Double, mu1: BDV[Double], mu2: BDV[Double], M: BDM[Double], ORD: Seq[Int]=Seq(1,2,3,4,5,6,7,8)): BDM[Double] = {
    val integrator = new GaussKronrod(1e-9, 100)
    val coef1 = BDM.tabulate(ORD.size, mu1.size){(i, j) =>
      integrator.integrate(
        (x) => g(x) * Normal.density(x-mu1(j), 0.0, 1.0, false) * hermite_shifted(x, mu1(j), ORD(i)),
        mu1(j)-8.0,
        mu1(j)+8.0
      ).estimate
    }
    val coef2 = BDM.tabulate(ORD.size, mu2.size){(i, j) =>
      integrator.integrate(
        (x) => g(x) * Normal.density(x-mu2(j), 0.0, 1.0, false) * hermite_shifted(x, mu2(j), ORD(i)),
        mu2(j)-8.0,
        mu2(j)+8.0
      ).estimate
    }
    val M_out = BDM.zeros[Double](mu1.size, mu1.size)
    for (ord <- ORD) {
      // outer product of the rows of the coefficient matrices
      // must subtract 1 from ord because scala is 0-indexed
      M_out := M_out + (coef1(ord-1,::).t * coef2(ord-1,::)) *:* (M ^:^ (ord.toDouble) ) /:/ factorialDouble(ord)
    }

    return M_out
  }
  }

}
