/**
  * This file contains functions for calculating the optimal weights w_i's for stat S = sum_i^n w_iT_i, where T_i=g(Z_i), 
  * where Z_i's are the marginal Zscore statistics obtained under regression 
  * models with random effects:  $\beta_i = b_i C_i$, where $C_i \sim Bernoulli(\pi_i)$
  * indicates the SNV is causal or not, with the likelihood of causality $\pi_i$, 
  * and allelic effect is $b_i$.
  * Reference: Zhang, Hong, and Zheyang Wu. "The generalized Fisher's combination and accurate p‐value
           calculation under dependence." Biometrics 79.2 (2023): 1159-1172.
  * @author Peter Howell
  * Last update (latest update first):
  *   Claude Code (Opus 5.5), prompted by VPanwar, 2026-11-03: the core no longer imports Hail:
  *     eigSymD is the method package's copy and fatal/warn come from GlowErrors (stage 1 of GLOWh's
  *     0project/glow-hail-refactor-plan.md). No change to what is computed.
  *   Claude Code (Opus 5.5), prompted by VPanwar, 2026-11-02: ported GLOWr 0.3.4's quadrature and
  *     Hermite expansion (ZWuLab/GLOWr tag v0.3.4, R/optimal_weights.R and
  *     R/optimal_weights_quadrature.R). Every one-dimensional integral against the normal density
  *     (E_gX_p, the Hermite coefficients, the variances) now uses one composite 10-point
  *     Gauss-Legendre rule on [-12, 12], panels of width 0.1, cut where g's argument is zero (the
  *     kink of the Fisher transform), replacing adaptive Gauss-Kronrod, which GLOWr's
  *     integrate() counterpart showed could be far off on the stretched Fisher transform. The
  *     Mehler expansion runs to order 64 (was 8) in the normalized Hermite functions by their
  *     recurrence, with no factorials, in both the _N and the _sparse schemes. New
  *     optimalWeightMoments (GLOWr's .optimal_weight_moments) computes all four schemes from one
  *     coefficient table, each distinct (mu, s) pair integrated once; new helpers normalNodes,
  *     hermiteMoments, hermiteTable, mehlerSum. The _N null covariance keeps the truncated
  *     series on its diagonal; the sparse covariance's diagonal is the exact mixture variance by
  *     the law of total variance with centered variances (varT_mix likewise; it was
  *     E[g^2] - (E g)^2). The `fisher` flag (null mean replaced by 2.0) is removed from
  *     optimalWeightsM, getRTilde, E_T_mix and getR: the null mean is integrated, as in GLOWr.
  *     covM_Integrand and covM_IntegrandScaled are removed (no remaining callers). getWts and
  *     optimalWeightsM_Burden return equal weights 1/n, with a warning, when all weights are
  *     effectively zero (mean |w| below machine epsilon), as GLOWr's get_wts and
  *     Optimal_Weights_M do; they returned NaN (0/0). See GLOWh's
  *     0project/ai-reports/2026-10-28-fork-port-progress.md.
  *     Mirrors: GLOWr 0.3.4 Optimal_Weights_M, .optimal_weight_moments, CovM_gXgY, CovMT_mix,
  *     E_gX_p, E_T_mix, Var_T_mix, get_r_tilde, get_r, and the helpers of
  *     optimal_weights_quadrature.R.
  *   Claude Code (Opus 5.5), prompted by VPanwar, 2026-10-28: covM_gXgY standardized (GLOWh port
  *     plan change (c)). optimalWeightsM passed the covariance M + MVM to covM_gXgY for APE_N,
  *     but the Mehler expansion is in powers of a CORRELATION: order k was scaled by
  *     (M + MVM)^k / k!. That is exact for a quadratic g, yet on real windows with
  *     diag(M + MVM) up to 881 it amplified quadrature residue until SKAT APE_N was off by 12-87
  *     percent (GLOWh PGFisherProbe's real-data checks). covM_gXgY now takes sd1/sd2 (default
  *     1, so every existing caller is unchanged and bit-identical), evaluates g(mu + s*(x - mu))
  *     against a standard density and Hermite basis on the fixed window mu +/- 8 (new
  *     covM_IntegrandScaled), and rejects a non-PSD M, unequal lengths, non-positive scales and
  *     a non-unit diagonal. optimalWeightsM passes cov2cor(M + MVM) with sd = sqrt(diag(M + MVM))
  *     for APE; BE keeps plain M with no scales. Upstream fix: GLOWr 0.3.3. See GLOWh's
  *     0project/ai-reports/2026-10-25-port-plan.md section 3a.4.
  *     Mirrors: GLOWr 0.3.3 CovM_gXgY and Optimal_Weights_M.
  *   Claude Code (Opus 5.5), prompted by VPanwar, 2026-10-28: optimalWeightsM passed the VARIANCE
  *     diag(M + MVM) to getRTilde, whose argument is a standard deviation (E_gX_p integrates
  *     against N(mu, sd)). It now passes sqrt(diag(M + MVM)). This changes r_tilde and so both
  *     normal-approximation schemes, BE_N and APE_N; BE_N is now complete, APE_N also needs the
  *     covM_gXgY standardization (GLOWh port plan change (c)). Upstream fix: GLOWr 0.3.3. See
  *     GLOWh's 0project/ai-reports/2026-10-25-port-plan.md section 3a.3.
  *     Mirrors: GLOWr 0.3.3 Optimal_Weights_M (SD_tilde, get_r_tilde).
  *   Claude Code (Opus 5), prompted by VPanwar, 2026-09-09: optimalWeightsG now builds M with
  *     cov2corZeroSafe. It is currently dead code (zero call sites), but the unguarded cov2cor
  *     would put NaN into M, and optimalWeightsM propagates that to EVERY weight through
  *     inv(I + v*M) -- R's Optimal_Weights_M rejects such an M outright, where the Scala has no
  *     input check.
  *   Claude Code (Opus 5), prompted by VPanwar, 2026-09-09: Fixed defect C1, found by the group-C
  *     Z-score suite (packages/glow-core in the GLOWh repo). getH_Continuous and getH_Binary fit
  *     the null model on an intercept-augmented design matrix but built HHalf from the
  *     un-augmented X, so the projection removed a different subspace than the residuals were
  *     taken from and every GHG -- and every weight derived from it -- was formed against a null
  *     model that had not been fitted. Both now build one design matrix up front via nullDesign,
  *     adding the intercept only if X does not already carry one, and use it for the fit and for
  *     HHalf. Mirrors: GLOWr:::fit_null_model (GLOWr/R/getZ_marg_score.R).
  *   KHoar 2025-05-07: Added header to file
  */

package is.hail.methods.gfisher

import is.hail.methods.gfisher.GlowErrors.{fatal, warn}

import breeze.linalg.{DenseVector => BDV, DenseMatrix => BDM, _}
import breeze.numerics.{abs, sqrt}
import breeze.stats.{stddev, mean}
import net.sourceforge.jdistlib.Normal

object OptimalWeights {

  def optimalWeightsG(
    g: Double => Double,
    G: BDM[Double],
    b: BDV[Double],
    pi: BDV[Double],
    X: BDM[Double],
    y: BDV[Double],
    burden: Boolean,
    binary: Boolean = false,
    forcePositiveWeights: Boolean=true
  ): BDM[Double] = {
    val (bStar, gHG) = if (binary) {
      val (ghg, _, _ ) = getGHG_Binary2(G, X, y)
      (sqrt(diag(ghg)) *:* b, ghg)
    } else {
      val (hH, s0, _) = getH_Continuous(X, y)
      val GHalf = G.t * hH
      val GHG = G.t * G - GHalf * GHalf.t
      ((sqrt(diag(GHG)) *:* b) /:/ s0, GHG)
    }
    // Guarded: a monomorphic variant gives a zero GHG diagonal entry, and an
    // unguarded cov2cor would put NaN into M, which optimalWeightsM propagates
    // to EVERY weight through inv(I + v*M). R's equivalent path takes M from
    // getZ_marg_score, which guards, and Optimal_Weights_M additionally rejects
    // an M containing NA outright.
    val M = cov2corZeroSafe(gHG)
    return optimalWeightsM(g, bStar, pi, M, burden, forcePositiveWeights)
  }

  /**
    * Calculate the optimal weights based on Bstar (scaled effect size), PI (likelihood of causality), and M (correlation matrix of marginal Z-scores)
    *
    * Mirrors GLOWr 0.3.4's Optimal_Weights_M: for a non-Burden g the four weight vectors come
    * from one table of moments (optimalWeightMoments, GLOWr's .optimal_weight_moments) and
    * GLOWr's get_wts (getWts).
    *
    * @param g transformation of the Z-scores
    * @param bStar scaled effect sizes
    * @param pi likelihoods of causality, in [0, 1]
    * @param M correlation matrix of the marginal Z-scores
    * @param burden true for the Burden/Liptak test (g(x) = x), which has closed-form weights
    * @param forcePositiveWeights if true, negative weights are set to 0
    * @return a 4 x n matrix (2 x n for Burden), rows BE_N, APE_N, BE_sparse, APE_sparse
    */
  def optimalWeightsM(
    g: (Double) => Double,
    bStar: BDV[Double],
    pi: BDV[Double],
    M: BDM[Double],
    burden: Boolean,
    forcePositiveWeights: Boolean=true
  ): BDM[Double] = {

    // I couldn't find a way to check if two functions are identical in scala, so I just added another argument to do the liptak/burden test
    if (burden) {
      return optimalWeightsM_Burden(bStar, pi, M, forcePositiveWeights)
    }

    // The normal approximation (BE_N, APE_N) and the sparse mixture (BE_sparse, APE_sparse), all
    // from one coefficient table at the default 64 Hermite orders.
    val mom = optimalWeightMoments(g, bStar, pi, M)
    val wts_BE = getWts(mom.sigmaBEN, mom.rN, forcePositiveWeights)
    val wts_APE = getWts(mom.sigmaAPEN, mom.rN, forcePositiveWeights)
    val wts_BE_sparse = getWts(mom.sigmaBESparse, mom.rSparse, forcePositiveWeights)
    val wts_APE_sparse = getWts(mom.sigmaAPESparse, mom.rSparse, forcePositiveWeights)

    return BDM(wts_BE, wts_APE, wts_BE_sparse, wts_APE_sparse)
  }

  /**
    * The moments behind the non-Burden optimal weights, from one table of Hermite coefficients
    * over the null pair (0, 1), the normal-approximation pairs (MMU_i, s_i) and the sparse pairs
    * (Bstar_i, 1). Mirrors GLOWr 0.3.4's .optimal_weight_moments.
    *
    * Normal approximation: under H1 the Z-scores are N(MMU, M + MVM), MMU = M (Bstar * PI),
    * handled as MMU_i + s_i U_i with s_i = sqrt((M + MVM)_ii) and U ~ N(0, R), R the correlation
    * matrix of M + MVM (Mehler's expansion needs a correlation).
    *   rN          E[g(Z~_i)] - E[g(Z_0)]
    *   sigmaBEN    the null covariance: the truncated series in M (its diagonal is the truncated
    *               sum, not the exact variance; GLOWr found that better for the weights)
    *   sigmaAPEN   the series in R with the stretched coefficients b_k(MMU_i, s_i)
    * Sparse mixture: Z_i = Z_0i + Bstar_i C_i, C_i ~ Bernoulli(PI_i).
    *   rSparse         PI_i (E_1(i) - E_0)
    *   sigmaAPESparse  off-diagonal from the four causal combinations; diagonal the mixture's own
    *                   variance by the law of total variance, with centered variances
    *   sigmaBESparse   the null series in M, with the exact null variance on its diagonal
    *
    * @param ORD the Hermite orders summed (default 1 to 64)
    */
  def optimalWeightMoments(
    g: (Double) => Double,
    bStar: BDV[Double],
    pi: BDV[Double],
    M: BDM[Double],
    ORD: Seq[Int] = 1 to 64
  ): OptimalWeightMoments = {
    val n = bStar.length
    val ord = checkOrd(ORD)
    val K = ord.max

    val mu: BDV[Double] = bStar *:* pi                              // mean of theta
    val v: BDV[Double] = (bStar ^:^ 2.0) *:* pi *:* (1.0 - pi)      // variance of theta
    val mmu: BDV[Double] = M * mu                                   // H1 mean of the Z-scores
    val sigmaTilde: BDM[Double] = M + (v *:* M(::, *)).t * M        // M + M diag(v) M
    val sdTilde: BDV[Double] = sqrt(diag(sigmaTilde))
    val rTilde: BDM[Double] = cov2cor(sigmaTilde)
    diag(rTilde) := 1.0 // as stats::cov2cor does; this package's cov2cor leaves 1 +/- rounding

    // One table: column 0 the null pair, then the normal pairs, then the sparse pairs.
    val tabMu = BDV.vertcat(BDV(0.0), mmu, bStar)
    val tabS = BDV.vertcat(BDV(1.0), sdTilde, BDV.ones[Double](n))
    val tab = hermiteTable(g, tabMu, tabS, K)
    val b0: BDV[Double] = tab.b(::, 0).copy
    val bN: BDM[Double] = tab.b(::, 1 to n).copy
    val bS: BDM[Double] = tab.b(::, (n + 1) to (2 * n)).copy
    val B0: BDM[Double] = BDM.tabulate(K + 1, n)((k, _) => b0(k))

    // Normal approximation
    val rN: BDV[Double] = bN(0, ::).t - b0(0)
    val sigmaNull = mehlerSum(B0, B0, M, ord)
    val sigmaAPEN = mehlerSum(bN, bN, rTilde, ord)

    // Sparse mixture: off-diagonals over the causal combinations, P(C_i = a, C_j = b)
    val q: BDV[Double] = 1.0 - pi
    val cov11 = mehlerSum(bS, bS, M, ord)
    val cov10 = mehlerSum(bS, B0, M, ord)
    val sigmaAPESparse: BDM[Double] = ((pi * pi.t) *:* cov11) + ((pi * q.t) *:* cov10) +
      ((q * pi.t) *:* cov10.t) + ((q * q.t) *:* sigmaNull)
    val e1: BDV[Double] = bS(0, ::).t.copy
    val e0 = b0(0)
    val var0 = tab.variance(0)
    for (i <- 0 until n) {
      val d = e1(i) - e0
      sigmaAPESparse(i, i) = q(i) * var0 + pi(i) * tab.variance(1 + n + i) + pi(i) * q(i) * d * d
    }
    val sigmaBESparse = sigmaNull.copy
    diag(sigmaBESparse) := var0
    val rSparse: BDV[Double] = pi *:* (e1 - e0)

    OptimalWeightMoments(rN, sigmaNull, sigmaAPEN, rSparse, sigmaBESparse, sigmaAPESparse)
  }

  def optimalWeightsM_Burden(
    bStar: BDV[Double],
    pi: BDV[Double],
    M: BDM[Double],
    forcePositiveWeights: Boolean=true
  ): BDM[Double] = {
    val n = bStar.size
    val mu: BDV[Double] = bStar *:* pi  // mean of theta (random effects)
    val v: BDV[Double] = (bStar ^:^ 2.0) *:* pi *:* (1.0 - pi) // variance of theta (random effects)

    val wts_BE = mu
    val wts_APE = inv(BDM.eye[Double](n) + (v *:* M(::, *))) * mu
    if (forcePositiveWeights) {
      wts_APE(wts_APE <:< 0.0) := 0.0
      wts_BE(wts_BE <:< 0.0) := 0.0
    }
    // As GLOWr's Optimal_Weights_M: all-zero weights give equal weights 1/n with a warning.
    for ((w, name) <- Seq((wts_BE, "BE"), (wts_APE, "APE"))) {
      val meanAbs = mean(abs(w))
      if (meanAbs < math.ulp(1.0)) {
        warn(s"All $name weights are effectively zero. Returning equal weights.")
        w := 1.0 / w.length
      } else w := w / meanAbs
    }
    return BDM(wts_BE, wts_APE)
  }

  def getGHG_Binary(G: BDM[Double], HHalf: BDM[Double], y0: BDV[Double]): BDM[Double] = {
    val GTilde = sqrt(y0 *:* (1.0 - y0)) *:* G(::, *)
    val GHalf = GTilde.t * HHalf
    return GTilde.t * GTilde - GHalf * GHalf.t
  }

  def getGHG_Binary2(G: BDM[Double], X: BDM[Double], y: BDV[Double]): (BDM[Double], BDV[Double], BDV[Double]) = {
    val (hH, y0, resids) = getH_Binary(X, y)
    val GTilde = sqrt(y0 *:* (1.0 - y0)) *:* G(::, *)
    val GHalf = GTilde.t * hH
    return (GTilde.t * GTilde - GHalf * GHalf.t, y0, resids)
  }

  def getGHG_Continuous(G: BDM[Double], HHalf: BDM[Double]): BDM[Double] = {
    val GHalf = G.t * HHalf
    return G.t * G - GHalf * GHalf.t
  }

  def getGHG_Continuous(G: BDM[Double], X: BDM[Double], y: BDV[Double]): (BDM[Double], Double, BDV[Double]) = {
    val (hH, s0, resids) = getH_Continuous(X, y)
    val GHalf = G.t * hH
    return (G.t * G - GHalf * GHalf.t, s0, resids)
  }

  /** The null-model design matrix, mirroring R's fit_null_model
    * (GLOWr/R/getZ_marg_score.R): the intercept column is added ONCE, up front,
    * and only if X does not already carry one, so that the regression fit and
    * the projection H_half are built from the SAME matrix.
    *
    * Getting this wrong is not cosmetic: fitting on [1 X] while forming H_half
    * from X alone projects out a different subspace than the residuals were
    * taken from, so GHG -- and every weight derived from it -- is computed
    * against a null model that was never fitted.
    */
  def nullDesign(X: BDM[Double]): BDM[Double] = {
    val hasIntercept = X.cols > 0 && (0 until X.rows).forall(i => X(i, 0) == 1.0)
    if (hasIntercept) X else BDM.horzcat(BDM.ones[Double](X.rows, 1), X)
  }

  def getH_Continuous(X: BDM[Double], y: BDV[Double]): (BDM[Double], Double, BDV[Double]) = {
    // compute solution to Xa * beta = y, manually calculate residuals
    val Xa = nullDesign(X)

    val yPred = lin_reg_predict(Xa, y, method="direct", addIntercept=false)
    val resids = y - yPred

    val sd = stddev(resids)
    val HHalf = Xa * (cholesky(inv(Xa.t * Xa)))
    return(HHalf, sd, resids)
  }

  def getH_Binary(X: BDM[Double], y: BDV[Double]): (BDM[Double], BDV[Double], BDV[Double]) = {
    val Xa = nullDesign(X)
    val y0 = log_reg_predict(Xa, y, addIntercept=false)
    val resids = y - y0

    val XTilde = sqrt(y0 *:* (1.0-y0)) *:* Xa(::,*)
    val HHalf = XTilde * cholesky(inv(XTilde.t * XTilde))

    return (HHalf, y0, resids)
  }

  /**
    * Calculate the vector of E_1(T_i)-E_0(T_i) for i=1,...,n
    * where T_i=g(Z_i), Z_i ~ N(mu_i, sd_i^2) under H1 and Z_0 ~ N(0, 1) under H0.
    * Mirrors GLOWr 0.3.4's get_r_tilde: the null mean E_gX_p(g, 0, 1, 1) is integrated, never
    * replaced by a constant (the former `fisher` shortcut, 2.0, is gone).
    *
    * @param g transfrmation function of Z
    * @param mmu vector of means mu_i, i=1,...,n.
    * @param sd vector of standard deviations of Z_i, i=1,...,n.
    */
  def getRTilde(g: (Double) => Double, mmu: BDV[Double], sd: BDV[Double]): BDV[Double] = {
    val n = mmu.size
    val e0 = E_gX_p(g, 0.0, 1.0, 1.0)
    return BDV.tabulate(n){(i) => E_gX_p(g, mmu(i), 1.0, sd(i)) - e0}
  }

  /**
    * Calculate E[g(X)^p] where X~N(mu,sigma^2), on the composite Gauss-Legendre rule of
    * normalNodes, in the standardized variable u = (x - mu) / sigma, cut where g's argument
    * mu + sigma u is 0. Mirrors GLOWr 0.3.4's E_gX_p.
    *
    * @param g function
    * @param mu mean
    * @param p power
    * @param sigma std
    */
  def E_gX_p(g: Double => Double, mu: Double, p: Double, sigma:Double=1.0): Double = {
    if (!(sigma > 0.0) || sigma.isInfinite)
      fatal(s"sigma must be a single finite positive number; got $sigma")
    val (u, w) = normalNodes(-mu / sigma)
    var value = 0.0
    for (i <- u.indices) {
      val x = mu + sigma * u(i)
      val gx = g(x)
      if (gx.isNaN || gx.isInfinite)
        fatal(s"The transformation g returned a non-finite value at x = $x (mu = $mu, sigma = $sigma).")
      value += math.pow(gx, p) * Normal.density(u(i), 0.0, 1.0, false) * w(i)
    }
    if (value.isNaN || value.isInfinite)
      fatal(s"E[g(X)^$p] is not finite (mu = $mu, sigma = $sigma): the transformation grows too fast " +
        "for the fixed quadrature domain.")
    return value
  }

  /**
    * Calculate the optimal weights of T = sum_i^n w_iT_i under Gaussian mixture model
    *
    * @param Sigma covariance matrix of T_1,...,T_n.
    * @param r vector of E_1(T_i) - E_0(T_i), i=1,...,n.
    * @param forcePosW if true, negative weights are set to 0.
    * @param normalize if true, weights are normalized by dividing by mean
    */
  def getWts(Sigma: BDM[Double], r: BDV[Double], forcePosW: Boolean=true, normalize:Boolean = true): BDV[Double] = {
    val w = inv(Sigma) * r
    if (forcePosW) w(w <:< 0.0) := 0.0
    if (normalize) {
      // As GLOWr's get_wts: weights that are all effectively zero (e.g. every PI = 0, so
      // r = 0) cannot be normalized; return equal weights 1/n with a warning, not 0/0 = NaN.
      val meanAbs = mean(abs(w))
      if (meanAbs < math.ulp(1.0)) {
        warn("All weights are effectively zero in optimal weight calculation. Returning equal weights.")
        w := 1.0 / w.length
      } else w := w /:/ meanAbs
    }
    return w
  }

  /**
    * Calculate the covariance matrix with elements Cov[g(X_i), g(Y_j)], where
    *   X_i = mu1_i + sd1_i U_i,  Y_j = mu2_j + sd2_j U_j,  U ~ MVN(0, M),  M a CORRELATION matrix,
    * by Mehler's expansion in the normalized Hermite functions:
    *   sum over k in ORD of b_k(mu1_i, sd1_i) b_k(mu2_j, sd2_j) M_ij^k,
    * where b_k(mu, s) = E[g(mu + s U) he_k(U)] (hermiteTable). No factorials: he_k = He_k / sqrt(k!).
    * Mirrors GLOWr 0.3.4's CovM_gXgY.
    *
    * M must be a correlation matrix: the expansion is in powers of a correlation. A covariance
    * passed in its place is rejected -- pass cov2cor(Sigma) with sd = sqrt(diag(Sigma)).
    *
    * Guards, as GLOWr's CovM_gXgY: M positive semi-definite (smallest eigenvalue >= -1e-10;
    * R's check_positive_definite also WARNS below +1e-10, which is not ported); mu1, mu2, sd1 and
    * sd2 of equal length; sd1, sd2 > 0; diag(M) within 1e-8 of 1; ORD positive integers. Not
    * ported: R's identity-matrix fast path, a speed-up only -- for M = I the general sum gives the
    * same values, since the off-diagonal powers are 0.
    *
    * @param g transformation function
    * @param mu1 mean vector of length n
    * @param mu2 another mean vector of length n
    * @param M an n x n correlation matrix
    * @param ORD the orders of the expansion summed (default 1 to 64)
    * @param sd1 standard deviations of X; null (R's NULL) means all 1
    * @param sd2 standard deviations of Y; null means all 1
    */
  def covM_gXgY(
    g: Double => Double,
    mu1: BDV[Double],
    mu2: BDV[Double],
    M: BDM[Double],
    ORD: Seq[Int] = 1 to 64,
    sd1: BDV[Double] = null,
    sd2: BDV[Double] = null
  ): BDM[Double] = {
    checkPSD(M)
    val n = mu1.size
    val s1: BDV[Double] = if (sd1 == null) BDV.ones[Double](n) else sd1
    val s2: BDV[Double] = if (sd2 == null) BDV.ones[Double](n) else sd2
    if (mu2.size != n || s1.size != n || s2.size != n)
      fatal(s"mu1, mu2, sd1 and sd2 must have the same length: ${n}, ${mu2.size}, ${s1.size}, ${s2.size}")
    // !(v > 0) rather than v <= 0, so a NaN scale (sqrt of a negative variance) is caught too.
    if ((s1.toArray ++ s2.toArray).exists(v => !(v > 0.0)))
      fatal(s"sd1 and sd2 must be positive; min(sd1) = ${min(s1)}, min(sd2) = ${min(s2)}")
    if (any(abs(diag(M) - 1.0) >:> 1e-8))
      fatal("M must be a correlation matrix (unit diagonal); pass cov2cor(Sigma) with sd1 and sd2. " +
        s"Largest |diag(M) - 1| = ${max(abs(diag(M) - 1.0))}")
    val ord = checkOrd(ORD)

    // Coefficients of every distinct (mean, scale) pair among the X and Y components.
    val tab = hermiteTable(g, BDV.vertcat(mu1, mu2), BDV.vertcat(s1, s2), ord.max)
    val b1 = tab.b(::, 0 until n).copy
    val b2 = tab.b(::, n until 2 * n).copy
    return mehlerSum(b1, b2, M, ord)
  }

  /**
    * Calculate the covariance matrix Cov[T_i, T_j], where T_i=g(Z_i), T_j=g(Z_j),
    * Z_i = Z_0i + mu_i*C_i, C_i ~ Bern(pi_i), Z_j = Z_0j + mu_j*C_j, C_j ~ Bern(pi_j),
    * Cov[Z_0i, Z_0j] = M_ij. Mirrors GLOWr 0.3.4's CovMT_mix (without its cache of the null
    * covariance, a speed-up only).
    *
    * Off-diagonal: the four causal combinations, weighted by P(C_i = a, C_j = b). Diagonal: the
    * mixture's own variance by the law of total variance, (1 - pi) Var_0 + pi Var_1 +
    * pi (1 - pi) (E_1 - E_0)^2, with centered variances (the sum over combinations would treat
    * C_i as independent of itself).
    *
    * @param g transformation function
    * @param mu mean vector
    * @param pi probability vector
    * @param M a positive-semidefinite correlation matrix
    * @param ORD the orders of the expansion summed (default 1 to 64)
    */
  def covMT_mix(
    g: (Double) => Double,
    mu: BDV[Double],
    pi: BDV[Double],
    M: BDM[Double],
    ORD: Seq[Int] = 1 to 64
  ): BDM[Double] = {
    checkPSD(M)
    val n = mu.size
    val ord = checkOrd(ORD)
    val K = ord.max
    // One table: the causal pairs (mu_i, 1) and the null pair (0, 1)
    val tab = hermiteTable(g, BDV.vertcat(mu, BDV(0.0)), BDV.ones[Double](n + 1), K)
    val b1 = tab.b(::, 0 until n).copy
    val b0: BDV[Double] = tab.b(::, n).copy
    val B0: BDM[Double] = BDM.tabulate(K + 1, n)((k, _) => b0(k))

    val q: BDV[Double] = 1.0 - pi
    val covM = mehlerSum(b1, b1, M, ord)
    val covM1 = mehlerSum(b1, B0, M, ord)
    val covM3 = mehlerSum(B0, B0, M, ord)
    val M_out = ((pi * pi.t) *:* covM) + ((pi * q.t) *:* covM1) + ((q * pi.t) *:* covM1.t) + ((q * q.t) *:* covM3)
    val e0 = b0(0)
    val var0 = tab.variance(n)
    for (i <- 0 until n) {
      val d = b1(0, i) - e0
      M_out(i, i) = q(i) * var0 + pi(i) * tab.variance(i) + pi(i) * q(i) * d * d
    }
    return M_out
  }

  /**
    * Calculate Var[T_i], where T_i=g(Z_i), Z_i = Z_0i + mu_i*C_i, C_i ~ Bern(pi_i)
    *
    * @param g
    * @param mu
    * @param pi
    */
  def varT_mix(g: (Double) => Double, mu: BDV[Double], pi: BDV[Double]): BDV[Double] = {
    return BDV.tabulate(mu.length){i => varT_mix(g, mu(i), pi(i))}
  }

  /**
    * Calculate Var[T_i], where T_i=g(Z_i), Z_i = Z_0i + mu_i*C_i, C_i ~ Bern(pi_i), by the law of
    * total variance over C_i: pi Var_1 + (1 - pi) Var_0 + pi (1 - pi) (E_1 - E_0)^2, where E_a and
    * Var_a are the mean and the centered variance of g(a mu + Z_0), so a large mean does not
    * cancel against a large second moment. Mirrors GLOWr 0.3.4's Var_T_mix.
    *
    * @param g
    * @param mu
    * @param pi
    */
  def varT_mix(g: (Double) => Double, mu: Double, pi: Double): Double = {
    val m1 = hermiteMoments(g, mu, 1.0, 0)
    val m0 = hermiteMoments(g, 0.0, 1.0, 0)
    val d = m1.b(0) - m0.b(0)
    return pi * m1.variance + (1.0 - pi) * m0.variance + pi * (1.0 - pi) * d * d
  }

  /**
    * Calculate E[T_i], where T_i=g(Z_i), Z_i = Z_0i + mu_i*C_i, C_i ~ Bern(pi_i):
    * pi E[g(Z_0 + mu)] + (1 - pi) E[g(Z_0)]. Mirrors GLOWr 0.3.4's E_T_mix.
    *
    * @param g
    * @param mu
    * @param pi
    */
  def E_T_mix(g: (Double) => Double, mu: Double, pi: Double): Double = {
    return pi * E_gX_p(g, mu, 1) + (1.0 - pi) * E_gX_p(g, 0.0, 1)
  }

  def getSigma(g: (Double) => Double, mu: BDV[Double], pi: BDV[Double], M: BDM[Double], h1: Boolean = false): BDM[Double] = {
    if (h1)
      return covMT_mix(g, mu, pi, M)

    val nullMu = BDV.zeros[Double](mu.size)
    return covMT_mix(g, nullMu, nullMu, M)
  }

  def getR(g: (Double) => Double, mu: BDV[Double], pi: BDV[Double]): BDV[Double] = {
    val n = mu.size
    val e0 = E_T_mix(g, 0.0, 0.0)
    return BDV.tabulate(n){(i) => E_T_mix(g, mu(i), pi(i)) -  e0}
  }

  // -------------------------------------------------------------------------------------------
  // Quadrature and Hermite expansion. Mirrors GLOWr 0.3.4's R/optimal_weights_quadrature.R:
  // every one-dimensional integral against the standard normal density uses one composite
  // 10-point Gauss-Legendre rule on [-12, 12] with panels of width 0.1, cut where the argument
  // of g is zero (the kink of the two-sided transforms); the Hermite functions come from their
  // three-term recurrence; each distinct (mean, scale) pair is integrated once per call.
  // -------------------------------------------------------------------------------------------

  /** Gauss-Legendre nodes and weights on [-1, 1] by Golub-Welsch: the nodes are the eigenvalues
    * of the symmetric tridiagonal Jacobi matrix of the Legendre recurrence (off-diagonal
    * i / sqrt(4 i^2 - 1)), the weights twice the squared first components of its eigenvectors.
    * Mirrors GLOWr's .gl_rule. */
  private def glRule(m: Int): (Array[Double], Array[Double]) = {
    val J = BDM.zeros[Double](m, m)
    for (i <- 1 until m) {
      val b = i / math.sqrt(4.0 * i * i - 1.0)
      J(i - 1, i) = b
      J(i, i - 1) = b
    }
    val es = eigSym(J)
    val order = (0 until m).sortBy(k => es.eigenvalues(k))
    (order.map(k => es.eigenvalues(k)).toArray,
     order.map(k => 2.0 * es.eigenvectors(0, k) * es.eigenvectors(0, k)).toArray)
  }

  /** The 10-point rule, computed once. */
  private lazy val GL10: (Array[Double], Array[Double]) = glRule(10)

  /**
    * Nodes and weights of the composite rule for integrals against the standard normal density
    * (the density is not included; the caller multiplies by it). [-lim, lim] is cut into panels
    * of width at most `width`, with a cut at `cut` when it lies inside, so a kink of the
    * integrand at `cut` falls on a panel boundary; each panel carries the 10-point rule. The
    * normal mass outside [-12, 12] is 3.6e-33. Mirrors GLOWr's .normal_nodes.
    *
    * @param cut the point to cut at; NaN or infinite for no cut
    */
  def normalNodes(cut: Double, lim: Double = 12.0, width: Double = 0.1): (Array[Double], Array[Double]) = {
    val (rx, rw) = GL10
    val m = rx.length
    def panel(a: Double, b: Double): (Array[Double], Array[Double]) = {
      val k = math.max(1, math.ceil((b - a) / width).toInt)
      val by = (b - a) / k
      val edges = Array.tabulate(k + 1)(i => if (i == k) b else a + i * by)  // R's seq(a, b, length.out = k + 1)
      val x = new Array[Double](k * m)
      val w = new Array[Double](k * m)
      for (p <- 0 until k) {
        val h = (edges(p + 1) - edges(p)) / 2.0
        val c0 = (edges(p + 1) + edges(p)) / 2.0
        for (j <- 0 until m) {
          x(p * m + j) = rx(j) * h + c0
          w(p * m + j) = rw(j) * h
        }
      }
      (x, w)
    }
    if (!cut.isNaN && !cut.isInfinite && cut > -lim && cut < lim) {
      val (lx, lw) = panel(-lim, cut)
      val (hx, hw) = panel(cut, lim)
      (lx ++ hx, lw ++ hw)
    } else panel(-lim, lim)
  }

  /** Hermite moments of g(mu + s U), U ~ N(0, 1), for one (mu, s): b holds b_k = E[g he_k(U)],
    * k = 0..K (so b(0) = E[g]), with he_k = He_k / sqrt(k!); variance is E[(g - E g)^2]. */
  case class HermiteMoments(b: Array[Double], eg2: Double, variance: Double)

  /**
    * The Hermite moments of g(mu + s U) from one evaluation of g on normalNodes, cut at
    * u0 = -mu / s where the argument of g is zero. The normalized Hermite functions come from the
    * forward-stable recurrence he_{k+1}(u) = (u he_k(u) - sqrt(k) he_{k-1}(u)) / sqrt(k + 1),
    * he_0 = 1, he_1 = u. Mirrors GLOWr's .hermite_moments (with .hermite_functions).
    *
    * Validated upstream for the Fisher and polynomial transforms; a g with another kink, a cusp,
    * a discontinuity or very rapid growth needs its own convergence check.
    */
  def hermiteMoments(g: Double => Double, mu: Double, s: Double, K: Int): HermiteMoments = {
    if (mu.isNaN || mu.isInfinite || s.isNaN || s.isInfinite || s <= 0.0)
      fatal(s"The mean must be finite and the scale finite and positive (got mu = $mu, s = $s).")
    val (u, w) = normalNodes(-mu / s)
    val nNodes = u.length
    val gx = new Array[Double](nNodes)
    for (i <- 0 until nNodes) {
      gx(i) = g(mu + s * u(i))
      if (gx(i).isNaN || gx(i).isInfinite)
        fatal(s"The transformation g returned a non-finite value at x = ${mu + s * u(i)} (mu = $mu, s = $s).")
    }
    val wphi = Array.tabulate(nNodes)(i => Normal.density(u(i), 0.0, 1.0, false) * w(i))
    val gw = Array.tabulate(nNodes)(i => gx(i) * wphi(i))
    def dot(h: Array[Double]): Double = { var acc = 0.0; var i = 0; while (i < nNodes) { acc += h(i) * gw(i); i += 1 }; acc }

    // b_k = sum_i he_k(u_i) g_i wphi_i, carrying he_{k-1} and he_k through the recurrence
    val b = new Array[Double](K + 1)
    var prev = Array.fill(nNodes)(1.0)   // he_0
    b(0) = dot(prev)
    if (K >= 1) {
      var cur = u.clone()                 // he_1
      b(1) = dot(cur)
      for (k <- 1 until K) {
        val sk = math.sqrt(k.toDouble)
        val sk1 = math.sqrt(k + 1.0)
        val next = Array.tabulate(nNodes)(i => (u(i) * cur(i) - sk * prev(i)) / sk1)
        prev = cur
        cur = next
        b(k + 1) = dot(cur)
      }
    }
    var eg2 = 0.0
    var variance = 0.0
    for (i <- 0 until nNodes) {
      eg2 += gx(i) * gx(i) * wphi(i)
      variance += (gx(i) - b(0)) * (gx(i) - b(0)) * wphi(i)
    }
    if (b.exists(x => x.isNaN || x.isInfinite) || eg2.isNaN || eg2.isInfinite || variance.isNaN || variance.isInfinite)
      fatal(s"A moment of g(mu + s U) is not finite (mu = $mu, s = $s): the transformation grows too fast " +
        "for the fixed quadrature domain.")
    HermiteMoments(b, eg2, variance)
  }

  /** The Hermite moments for vectors of means and scales: b is (K + 1) x n, row k holding order
    * k; variance is indexed like mu. */
  case class HermiteTable(b: BDM[Double], variance: BDV[Double])

  /** hermiteMoments for every (mu_i, s_i), each distinct pair integrated once (pairs matched
    * exactly, as GLOWr's .hermite_table matches them at 17 significant digits). */
  def hermiteTable(g: Double => Double, mu: BDV[Double], s: BDV[Double], K: Int): HermiteTable = {
    val n = mu.length
    if (s.length != n) fatal(s"mu and s must have the same length: $n, ${s.length}")
    val seen = scala.collection.mutable.HashMap.empty[(Double, Double), HermiteMoments]
    val b = BDM.zeros[Double](K + 1, n)
    val variance = BDV.zeros[Double](n)
    for (i <- 0 until n) {
      val m = seen.getOrElseUpdate((mu(i), s(i)), hermiteMoments(g, mu(i), s(i), K))
      for (k <- 0 to K) b(k, i) = m.b(k)
      variance(i) = m.variance
    }
    HermiteTable(b, variance)
  }

  /** Mehler's sum S_ij = sum over k in ORD of C(k, i) D(k, j) M_ij^k, the powers of M accumulated
    * elementwise. With C = D coefficient tables and M a correlation matrix this is the truncated
    * covariance of the transformed variables. Mirrors GLOWr's .mehler_sum. */
  def mehlerSum(C: BDM[Double], D: BDM[Double], M: BDM[Double], ORD: Seq[Int]): BDM[Double] = {
    val n = C.cols
    val out = BDM.zeros[Double](n, n)
    var Mk = BDM.ones[Double](n, n)
    val ord = ORD.toSet
    for (k <- 1 to ORD.max) {
      Mk = Mk *:* M
      if (ord.contains(k)) out += (C(k, ::).t * D(k, ::)) *:* Mk
    }
    out
  }

  /** ORD as sorted distinct positive integers; mirrors GLOWr's .check_ord. */
  private def checkOrd(ORD: Seq[Int]): Seq[Int] = {
    if (ORD.isEmpty || ORD.exists(_ < 1))
      fatal("ORD must be a vector of positive integers (the orders of the Hermite expansion)")
    ORD.distinct.sorted
  }

  /** Stop if M is not positive semi-definite (smallest eigenvalue below -1e-10), as GLOWr's
    * check_positive_definite stops; its warning below +1e-10 is not ported. */
  private def checkPSD(M: BDM[Double]): Unit = {
    val minEig = min(eigSymD.justEigenvalues(M))
    if (minEig < -1e-10)
      fatal(f"Correlation matrix M must be positive semi-definite. Minimum eigenvalue: $minEig%.2e")
  }
}

/** The moments behind the non-Burden optimal weights (OptimalWeights.optimalWeightMoments). */
case class OptimalWeightMoments(
  rN: BDV[Double],
  sigmaBEN: BDM[Double],
  sigmaAPEN: BDM[Double],
  rSparse: BDV[Double],
  sigmaBESparse: BDM[Double],
  sigmaAPESparse: BDM[Double]
)
