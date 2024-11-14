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
    df: BDV[Int],
    w: BDV[Double],
    M: BDM[Double],
    GM: BDM[Double]
  ): BDV[Double] = {
    val n: Int = df.length
    val DM: BDM[Int] = min(tile(df, 1, n),  tile(df, 1, n).t)
    val Mtilde: BDM[Double] = sqrt(abs(GM) /:/ convert(DM, Double) /:/ 2.0) *:* signum(M)
    Mtilde(Mtilde >:> 1.0) := 0.999

    if (any(eigSymD.justEigenvalues(Mtilde) <:< 1e-10)) {
      Mtilde := nearPD(Mtilde)
    }
    val WMChol: BDM[Double] = cholesky(M) * diag(sqrt(w))
    var lam: BDV[Double] = eigSymD.justEigenvalues(WMChol.t * WMChol)
    if (max(df) > 1) {
      for (i <- 2 to max(df)) {
        val Ai = BDM.eye[Double](n)
        val aDiag = diag(Ai)

        // this accomplishes `diag(Ai)(df <:< i) := 0.0`
        aDiag(df <:< i) := 0.0

        lam = BDV.vertcat(lam, eigSymD.justEigenvalues(WMChol * Ai * WMChol.t))
      }
    }


    return lam(lam >:> 1e-10).toDenseVector
  }

}
