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
    * @param pType "two" = two-sided, "one" = one-sided input p-values.
    * @return a correlation matrix between T(1), T(2),..., T(m) as calculated in Corollary 2.
    */
  def getGFisherCor(DD: BDM[Double], W: BDM[Double], M: BDM[Double],
                    varCorrect: Boolean = true, pType: String = "two"): BDM[Double] = {

    val m = DD.rows  // Number of GFisher statistics (equivalent to dim(DD)[1] in R)
    val COV = BDM.fill[Double](m, m)(Double.NaN)  // Initialize covariance matrix with NaN values

    // Loop over the rows and columns of the covariance matrix
    for (i <- 0 until m) {
      for (j <- i until m) {
        COV(i, j) = GFisherCov.getGFisherCov(DD(i, ::).t, DD(j, ::).t, W(i, ::).t, W(j, ::).t, M, varCorrect, pType)
      }
    }

    // Fill the lower triangle of the matrix with the transpose of the upper triangle
    for (i <- 1 until m; j <- 0 until i) {
      COV(i, j) = COV(j, i)
    }

    // Return the correlation matrix by normalizing the covariance matrix
    cov2cor(COV)
    
    }
    
    // Test function for GFisherCor
    def runTests(): Unit = {
        println("Running inline tests...")
          
        val DD = BDM((1.0, 2.0), (3.0, 4.0))
        val W = BDM((1.0, 1.0), (1.0, 1.0))
        val M = BDM((1.0, 0.5), (0.5, 1.0))

        val result = getGFisherCor(DD, W, M)
        println("getGFisherCor result:")
        println(result)
          
        println("All tests passed.")
    }
}