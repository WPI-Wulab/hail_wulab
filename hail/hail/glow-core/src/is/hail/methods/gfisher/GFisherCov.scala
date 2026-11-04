/*
This file contains the function for computing a GFisher covariance matrix
Reference: Zhang, Hong, and Zheyang Wu. "The generalized Fisher's combination and accurate p‐value
           calculation under dependence." Biometrics 79.2 (2023): 1159-1172.
Creators: Kylie Hoar
Last update (latest update first):
  Claude Code (Opus 5), prompted by VPanwar, 2026-09-02: Fixed three defects found by the
    group-A golden-fixture suite (packages/glow-core in the GLOWh repo). (1) The M-power terms
    and the final weighted reduction used Breeze `*`, which is MATRIX multiplication, where the
    R reference uses element-wise multiplication; both now use `*:*`. This made the returned
    covariance wrong by factors of 87-115. (2) The variance correction indexed `v1(0)`/`v2(0)`
    as scalars, where R divides by the whole vector element-wise; these agreed only when every
    degree of freedom was equal. (3) getGFisherCoefs was called without `one_sided`, so the
    one-sided branch built its cross-terms from two-sided coefficients.
    Mirrors: getGFishercov(), GFisher/R/GFisher-helpers.R:206.
  PHowell 2025-02-26: Allow double degrees of freedom
  KHoar 2024-11-30: sample format for future edits
*/

package is.hail.methods.gfisher

import breeze.linalg.{DenseMatrix => BDM, DenseVector => BDV, diag, sum}
import breeze.numerics.sqrt

object GFisherCov {

  /**
    * Compute the the correlation between two GFisher statistics.
    *
    * @param D1 a vector of degrees of freedom for a GFisher statistic.
    * @param D2 a vector of degrees of freedom for another GFisher statistic.
    * @param W1 a vector of weights for a GFisher statistic.
    * @param W2 a vector of weights of freedom for another GFisher statistic.
    * @param M correlation matrix of the input Zscores from which the input p-values were obtained.
    * @param varCorrect default = TRUE to make sure the exact variance was used.
    * @param one_sided true = one-sided input p-values, false = two-sided input p-values.
    * @return covariance of T(l) and T(r) in Corollary 2
    */
  def getGFisherCov(D1: BDV[Double], D2: BDV[Double], W1: BDV[Double], W2: BDV[Double], M: BDM[Double], varCorrect: Boolean = true, one_sided: Boolean = false): Double = {
        // The coefficients depend on the sidedness of the input p-values, so the
        // flag must be threaded through: the two-sided branch below pairs them
        // with M^2, M^4, M^6, M^8 and the one-sided branch with M, M^2, M^3, M^4.
        val (coeff1_res1, coeff2_res1, coeff3_res1, coeff4_res1) = GFisherCoefs.getGFisherCoefs(D1, one_sided)
        val (coeff1_res2, coeff2_res2, coeff3_res2, coeff4_res2) = GFisherCoefs.getGFisherCoefs(D2, one_sided)
        val GM_cross = if (!one_sided) {
          // Each term is an ELEMENT-WISE product of a power of M with the outer
          // product of the two coefficient vectors (Corollary 2). Breeze's `*` on
          // two matrices is matrix multiplication, so `*:*` is required here; the
          // inner `coeff * coeff.t` is a genuine outer product and stays `*`.
          val term1 = (M.mapValues(math.pow(_, 2.0)) / 2.0) *:* (coeff1_res1 * coeff1_res2.t)
          val term2 = (M.mapValues(math.pow(_, 4.0)) / 24.0) *:* (coeff2_res1 * coeff2_res2.t)
          val term3 = (M.mapValues(math.pow(_, 6.0)) / 720.0) *:* (coeff3_res1 * coeff3_res2.t)
          val term4 = (M.mapValues(math.pow(_, 8.0)) / 40320.0) *:* (coeff4_res1 * coeff4_res2.t)
          val result = term1 + term2 + term3 + term4
          if (varCorrect) {
            val v1 = (coeff1_res1.mapValues(math.pow(_, 2.0)) / 2.0) +
                 (coeff2_res1.mapValues(math.pow(_, 2.0)) / 24.0) +
                 (coeff3_res1.mapValues(math.pow(_, 2.0)) / 720.0) +
                 (coeff4_res1.mapValues(math.pow(_, 2.0)) / 40320.0)
            val v2 = (coeff1_res2.mapValues(math.pow(_, 2.0)) / 2.0) +
                 (coeff2_res2.mapValues(math.pow(_, 2.0)) / 24.0) +
                 (coeff3_res2.mapValues(math.pow(_, 2.0)) / 720.0) +
                 (coeff4_res2.mapValues(math.pow(_, 2.0)) / 40320.0)
            // sqrt(2 * D / v) ELEMENT-WISE against the whole v vector. Indexing
            // v(0) as a scalar is correct only when every df is equal.
            val diagV1 = diag(sqrt((2.0 * D1) /:/ v1))
            val diagV2 = diag(sqrt((2.0 * D2) /:/ v2))
            diagV1 * result * diagV2
          } else {
            result
          }
        } else {
          val term1 = M *:* (coeff1_res1 * coeff1_res2.t)
          val term2 = (M.mapValues(math.pow(_, 2.0)) / 2.0) *:* (coeff2_res1 * coeff2_res2.t)
          val term3 = (M.mapValues(math.pow(_, 3.0)) / 6.0) *:* (coeff3_res1 * coeff3_res2.t)
          val term4 = (M.mapValues(math.pow(_, 4.0)) / 24.0) *:* (coeff4_res1 * coeff4_res2.t)
          val result = term1 + term2 + term3 + term4
          if (varCorrect){
            val v1 = (coeff1_res1.mapValues(math.pow(_, 2.0))) +
                 (coeff2_res1.mapValues(math.pow(_, 2.0)) / 2.0) +
                 (coeff3_res1.mapValues(math.pow(_, 2.0)) / 6.0) +
                 (coeff4_res1.mapValues(math.pow(_, 2.0)) / 24.0)
            val v2 = (coeff1_res2.mapValues(math.pow(_, 2.0))) +
                 (coeff2_res2.mapValues(math.pow(_, 2.0)) / 2.0) +
                 (coeff3_res2.mapValues(math.pow(_, 2.0)) / 6.0) +
                 (coeff4_res2.mapValues(math.pow(_, 2.0)) / 24.0)
            // sqrt(2 * D / v) ELEMENT-WISE against the whole v vector. Indexing
            // v(0) as a scalar is correct only when every df is equal.
            val diagV1 = diag(sqrt((2.0 * D1) /:/ v1))
            val diagV2 = diag(sqrt((2.0 * D2) /:/ v2))
            diagV1 * result * diagV2
          } else {
            result
          }
        }

    // Element-wise product with the outer product of the weight vectors, then
    // summed: sum(GM_cross * (W1 %*% t(W2))) in the R reference.
    sum(GM_cross *:* (W1 * W2.t))

    }
}
