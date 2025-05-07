/**
  * This file contains the function to calculate eigenvalues needed for quadratic approximation.
  * Reference: Zhang, Hong, and Zheyang Wu. "The generalized Fisher's combination and accurate p‐value
           calculation under dependence." Biometrics 79.2 (2023): 1159-1172.
  * @author Peter Howell
  * Last update (latest update first):
  *   KHoar 2025-05-07: Added header to file
  */

package is.hail.methods.gfisher

import is.hail.stats.eigSymD
import breeze.linalg.{DenseMatrix => BDM, DenseVector => BDV, _}
import breeze.numerics.{abs, sqrt, signum}

// import net.sourceforge.jdistlib.{ChiSquare, Normal}


object GFisherLambda {


  /**
    * Calculate the eigenvalues needed for quadratic approximation, which only works for two-sided p-values
    *
    * @param df n-dimensional vector of degrees of freedom
    * @param w n-dimensional vector of weights
    * @param M n by n correlation matrix
    * @param GM n by n correlation matrix between w_1 T_1, ..., w_n T_n, which is the output of getGFisherGM
    */
  def getGFisherLambda(
    df: BDV[Double],
    w: BDV[Double],
    M: BDM[Double],
    GM: BDM[Double]
  ): BDV[Double] = {
    val n: Int = df.length
    val DM: BDM[Double] = min(tile(df, 1, n),  tile(df, 1, n).t)
    val Mtilde: BDM[Double] = sqrt(abs(GM) /:/ DM /:/ 2.0) *:* signum(M)
    Mtilde(Mtilde >:> 1.0) := 0.999

    if (any(eigSymD.justEigenvalues(Mtilde) <:< 1e-10)) {
      Mtilde := nearPD(Mtilde)
    }

    val MChol: BDM[Double] = cholesky(Mtilde).t // breeze cholesky returns the lower triangle. need to transpose it.
    val WMChol: BDM[Double] = MChol * diag(sqrt(w))
    var lam = reverse(eigSymD.justEigenvalues(WMChol.t * WMChol)) // eigSymD returns the eigen values in the reverse order of R
    if (max(df) > 1) {
      for (i <- 2 to max(df).toInt) {
        val Ai = BDM.eye[Double](n)
        val aDiag = diag(Ai)
        aDiag(df <:< i.toDouble) := 0.0 // this also changes Ai
        lam = BDV.vertcat(lam, reverse(eigSymD.justEigenvalues(WMChol * Ai * WMChol.t)))
      }
    }
    return lam(lam >:> 1e-10).toDenseVector
  }

}
