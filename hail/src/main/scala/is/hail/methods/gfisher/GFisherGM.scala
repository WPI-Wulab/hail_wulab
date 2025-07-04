/**
  * File that contains the functions to compute covariance matrix for w_iT_i,i=1,...,n.
  * Reference: Zhang, Hong, and Zheyang Wu. "The generalized Fisher's combination and accurate p‐value
           calculation under dependence." Biometrics 79.2 (2023): 1159-1172.
  * @author Peter Howell
  * Last update (latest update first):
  *   KHoar 2025-05-07: Added header to file
  */

package is.hail.methods.gfisher

import breeze.linalg.{DenseMatrix => BDM, DenseVector => BDV, _}
import breeze.numerics.sqrt

object GFisherGM {

  /**
    * Calculate covariance matrix for w_iT_i,i=1,...,n in Theorem 1 / Corollary 1
    *
    * @param df n-dimensional vector of degrees of freedom
    * @param w n-dimensional vector of weights
    * @param M n by n matrix correlation matrix
    * @param one_sided whether the p-values are one-sided or not
    */
  def getGFisherGM(
    df: BDV[Double],
    w: BDV[Double],
    M: BDM[Double],
    one_sided: Boolean = false,
  ): BDM[Double] = {
    val (c1, c2, c3, c4) = GFisherCoefs.getGFisherCoefs(df, one_sided)

    val GM = if (one_sided) {
      M *:* (c1 * c1.t) + ((M ^:^ 2.0)/ 2.0) *:* (c2 * c2.t) + ((M ^:^ 3.0)/ 6.0) *:* (c3 * c3.t) + ((M ^:^ 4.0)/ 24.0) *:* (c4 * c4.t)
    } else {
      ((M ^:^ 2.0)/ 2.0) *:* (c1 * c1.t) + ((M ^:^ 4.0)/ 24.0) *:* (c2 * c2.t) + ((M ^:^ 6.0)/ 720.0) *:* (c3 * c3.t) + ((M ^:^ 8.0) / 40320.0) *:* (c4 * c4.t)
    }
    return diag(sqrt(2.0*df) *:* w) * cov2cor(GM) * diag(sqrt(2.0*df) *:* w)
  }

}
