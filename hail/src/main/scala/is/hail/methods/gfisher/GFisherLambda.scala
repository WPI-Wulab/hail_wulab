/**
  * This file contains the function to calculate eigenvalues needed for quadratic approximation.
  * Reference: Zhang, Hong, and Zheyang Wu. "The generalized Fisher's combination and accurate p‐value
           calculation under dependence." Biometrics 79.2 (2023): 1159-1172.
  * @author Peter Howell
  * Last update (latest update first):
  *   Claude Code (Opus 5.5), prompted by VPanwar, 2026-11-03: the core no longer imports Hail:
  *     eigSymD is the method package's copy and fatal come from GlowErrors (stage 1 of GLOWh's
  *     0project/glow-hail-refactor-plan.md). No change to what is computed.
  *   Claude Code (Opus 5.5), prompted by VPanwar, 2026-10-28: M_tilde is now built from the
  *     covariance of the UNWEIGHTED T_i, as formula (13) specifies, and the weights enter once,
  *     through diag(sqrt(w)). The previous code built it from GM, which already carries the
  *     weights (GM_ii = 2 d_i w_i^2), so M_tilde had diagonal w_i instead of 1 and the weights were
  *     applied twice; the nearPD repair then fired on matrices that should never have been
  *     indefinite. Upstream fix: GFisher 0.3.2. The construction is split out as getGFisherMtilde,
  *     mirroring GFisher 0.3.2's getGFisherMtilde: divide GM by w_i w_j, identity row and column
  *     for a zero weight, unit diagonal, cap at +/-0.999 (previously only the upper side), and
  *     nearPD only if the smallest eigenvalue is below 1e-10. Input checks added for dimensions
  *     and negative weights. See GLOWh's 0project/ai-reports/2026-10-25-port-plan.md section 3a.2.
  *     Mirrors: GFisher 0.3.2 getGFisherMtilde and getGFisherlam (R path).
  *   KHoar 2025-05-07: Added header to file
  */

package is.hail.methods.gfisher

import is.hail.methods.gfisher.GlowErrors.fatal
import breeze.linalg.{DenseMatrix => BDM, DenseVector => BDV, _}
import breeze.numerics.{abs, sqrt, signum}

// import net.sourceforge.jdistlib.{ChiSquare, Normal}


object GFisherLambda {

  /**
    * The correlation matrix M_tilde of formula (13): the correlation of the UNWEIGHTED T_i,
    * expressed as the correlation of underlying normals so that a Cholesky factor of it,
    * scaled by sqrt(w), gives the quadratic form whose eigenvalues approximate the null.
    *
    * GM is the covariance of the WEIGHTED w_i T_i (getGFisherGM), so
    *   cov(T_i, T_j) = GM_ij / (w_i w_j),
    * and for chi-square-type T_i the normal-scale correlation is
    *   M_tilde_ij = sign(M_ij) * sqrt( cov(T_i, T_j) / (2 min(d_i, d_j)) ).
    * For d = 1 this recovers M exactly (cov(T_i, T_j) = 2 M_ij^2).
    *
    * Edge cases, all as in GFisher 0.3.2:
    *   - a zero weight gets an identity row and column: the variant drops out of the quadratic
    *     form, and dividing by w_i w_j = 0 is never attempted;
    *   - the diagonal is set to 1 (it is 1 analytically; this removes rounding);
    *   - off-diagonal entries outside [-1, 1] are capped at +/-0.999;
    *   - if the smallest eigenvalue is below 1e-10, project to the nearest correlation matrix
    *     (nearPD keeps the unit diagonal, which is what keeps sum(lam) = sum(w * d)).
    *
    * @param df n-dimensional vector of degrees of freedom
    * @param w  n-dimensional vector of non-negative weights (any scale; M_tilde does not depend on it)
    * @param M  n by n correlation matrix of the input Z-scores
    * @param GM n by n covariance matrix of w_1 T_1, ..., w_n T_n, the output of getGFisherGM
    */
  def getGFisherMtilde(
    df: BDV[Double],
    w: BDV[Double],
    M: BDM[Double],
    GM: BDM[Double]
  ): BDM[Double] = {
    val n: Int = M.rows
    if (M.cols != n || df.length != n || w.length != n || GM.rows != n || GM.cols != n)
      fatal(s"df, w, M and GM must have matching dimensions: length(df) = ${df.length}, " +
        s"length(w) = ${w.length}, dim(M) = ${M.rows} x ${M.cols}, dim(GM) = ${GM.rows} x ${GM.cols}")
    if (any(w <:< 0.0))
      fatal(s"All weights must be non-negative; min(w) = ${min(w)}")

    val Mtilde: BDM[Double] = BDM.eye[Double](n)
    for (i <- 0 until n; j <- 0 until n if i != j && w(i) > 0.0 && w(j) > 0.0) {
      val dm = math.min(df(i), df(j))                     // min(d_i, d_j)
      val covT = math.abs(GM(i, j)) / (w(i) * w(j))       // |cov(T_i, T_j)|, weights removed
      val v = signum(M(i, j)) * math.sqrt(covT / dm / 2.0)
      // Cap at +/-0.999: rounding in GM can push a near-1 correlation just past 1.
      Mtilde(i, j) = if (v > 1.0) 0.999 else if (v < -1.0) -0.999 else v
    }

    // Same repair threshold as the R reference.
    if (any(eigSymD.justEigenvalues(Mtilde) <:< 1e-10)) nearPD(Mtilde) else Mtilde
  }

  /**
    * Calculate the eigenvalues needed for quadratic approximation, which only works for two-sided p-values
    *
    * @param df n-dimensional vector of degrees of freedom
    * @param w n-dimensional vector of weights
    * @param M n by n correlation matrix
    * @param GM n by n covariance matrix of w_1 T_1, ..., w_n T_n, which is the output of getGFisherGM
    */
  def getGFisherLambda(
    df: BDV[Double],
    w: BDV[Double],
    M: BDM[Double],
    GM: BDM[Double]
  ): BDV[Double] = {
    val n: Int = df.length
    val Mtilde: BDM[Double] = getGFisherMtilde(df, w, M, GM)

    // The weights enter ONCE here: the quadratic form is sum_k w_k * (chi-square pieces) with
    // the pieces correlated through Mtilde, so its eigenvalues are those of
    // diag(sqrt(w)) Mtilde diag(sqrt(w)) = (U diag(sqrt(w)))' (U diag(sqrt(w))), U = chol(Mtilde).
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
