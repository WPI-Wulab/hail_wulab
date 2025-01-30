/*
This file contains main and supportive functions for computing a GFisher correlation matrix
Reference: Zhang, Hong, and Zheyang Wu. "The generalized Fisher's combination and accurate p‐value 
           calculation under dependence." Biometrics 79.2 (2023): 1159-1172.
Creators: Kylie Hoar
Last update (latest update first): 
  KHoar 2024-11-30: sample format for future edits
*/

package is.hail.methods.gfisher

import breeze.linalg.{DenseMatrix => BDM}

object GFisherCor {

  /**
    * Compute correlation matrix
    *
    * @param DD an mxn matrix of degrees of freedom, where m is the number of GFisher statistics, n is the number of p-values to be combined by each GFisher.
    * @param W an mxn matrix of weights, where m is the number of GFisher statistics, n is the number of p-values to be combined by each GFisher.
    * @param M correlation matrix of the input Zscores from which the input p-values were obtained.
    * @param varCorrect passed to getGFishercov(). default = TRUE to make sure the exact variance was used.
    * @param one_sided true = one-sided input p-values, false = two-sided input p-values.
    * @return a correlation matrix between T(1), T(2),..., T(m) as calculated in Corollary 2.
    */
  def getGFisherCor(DD: BDM[Int], W: BDM[Double], M: BDM[Double], varCorrect: Boolean = true, one_sided: Boolean = false): BDM[Double] = {
    val m = DD.rows
    val COV = BDM.fill[Double](m, m)(Double.NaN)
    for (i <- 0 until m) {
      for (j <- i until m) {
        COV(i, j) = GFisherCov.getGFisherCov(DD(i, ::).t, DD(j, ::).t, W(i, ::).t, W(j, ::).t, M, varCorrect, one_sided)
      }
    }
    for (i <- 1 until m; j <- 0 until i) {
      COV(i, j) = COV(j, i)
    }
    cov2cor(COV)

    }
}
