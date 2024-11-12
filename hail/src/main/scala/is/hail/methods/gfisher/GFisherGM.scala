package is.hail.methods.gfisher

import breeze.linalg.{DenseMatrix => BDM, DenseVector => BDV, _}
import breeze.numerics.sqrt
// import net.sourceforge.jdistlib.{ChiSquare, Normal}

object GFisherGM {

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
  def getGFisherGM(
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
