package is.hail.methods.gfisher

import breeze.linalg.{DenseMatrix => BDM, DenseVector => BDV, diag, sum}

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
    * @param pType "two" = two-sided, "one" = one-sided input p-values.
    */
  def getGFisherCov(D1: BDV[Double], D2: BDV[Double], W1: BDV[Double], W2: BDV[Double], M: BDM[Double], varCorrect: Boolean = true, pType: String = "two"): Double = {

        // Get coefficients from helper function
        val (coeff1_res1, coeff2_res1, coeff3_res1, coeff4_res1) = GFisherCoefs.getGFisherCoefs(D1.mapValues(_.toInt))
        val (coeff1_res2, coeff2_res2, coeff3_res2, coeff4_res2) = GFisherCoefs.getGFisherCoefs(D2.mapValues(_.toInt))

        // Use these variables in the following calculations
        val GM_cross = if (pType == "two") {
          val term1 = (M.mapValues(math.pow(_, 2.0)) / 2.0 * (coeff1_res1 * coeff1_res2.t))
          val term2 = (M.mapValues(math.pow(_, 4.0)) / 24.0 * (coeff2_res1 * coeff2_res2.t))
          val term3 = (M.mapValues(math.pow(_, 6.0)) / 720.0 * (coeff3_res1 * coeff3_res2.t))
          val term4 = (M.mapValues(math.pow(_, 8.0)) / 40320.0 * (coeff4_res1 * coeff4_res2.t))

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

            val diagV1 = diag(((D1.mapValues(_ / v1(0) * 2.0)).map(math.sqrt)).toDenseVector)
            val diagV2 = diag(((D2.mapValues(_ / v2(0) * 2.0)).map(math.sqrt)).toDenseVector)

            diagV1 * result * diagV2
          } else {
            result
          }

        } else {
          val term1 = (M * (coeff1_res1 * coeff1_res2.t))
          val term2 = (M.mapValues(math.pow(_, 2.0)) / 2.0 * (coeff2_res1 * coeff2_res2.t))
          val term3 = (M.mapValues(math.pow(_, 3.0)) / 6.0 * (coeff3_res1 * coeff3_res2.t))
          val term4 = (M.mapValues(math.pow(_, 4.0)) / 24.0 * (coeff4_res1 * coeff4_res2.t))

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

            val diagV1 = diag(((D1.mapValues(_ / v1(0) * 2.0)).map(math.sqrt)).toDenseVector)
            val diagV2 = diag(((D2.mapValues(_ / v2(0) * 2.0)).map(math.sqrt)).toDenseVector)

            diagV1 * result * diagV2
          } else {
            result
          }
        }

    sum(GM_cross * (W1 * W2.t))

    }
}