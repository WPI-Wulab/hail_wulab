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

import is.hail.GaussKronrod
import is.hail.stats.eigSymD
import is.hail.utils.fatal

import breeze.linalg.{DenseVector => BDV, DenseMatrix => BDM, _}
import breeze.numerics.{abs, sqrt}
import breeze.stats.{stddev, mean}
import net.sourceforge.jdistlib.Normal

import org.apache.commons.math3.util.CombinatoricsUtils.factorialDouble

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
    * @param g
    * @param bStar
    * @param pi
    * @param M
    * @param burden
    * @param forcePositiveWeights
    */
  def optimalWeightsM(
    g: (Double) => Double,
    bStar: BDV[Double],
    pi: BDV[Double],
    M: BDM[Double],
    burden: Boolean,
    fisher: Boolean = false,
    forcePositiveWeights: Boolean=true
  ): BDM[Double] = {

    // I couldn't find a way to check if two functions are identical in scala, so I just added another argument to do the liptak/burden test
    if (burden) {
      return optimalWeightsM_Burden(bStar, pi, M, forcePositiveWeights)
    }

    val n = bStar.length
    val mu: BDV[Double] = bStar *:* pi  // mean of theta (random effects)
    val mmu: BDV[Double] = M * mu // mean of marginal Z-scores when effects are random
    val v: BDV[Double] = (bStar ^:^ 2.0) *:* pi *:* (1.0 - pi) // variance of theta (random effects)
    // I think (`diag(v) * M` is equal to `v *:* M(::, *)`, and I believe the former is faster)
    val MVM: BDM[Double] = (v *:* M(::, *)).t * M // faster way to compute M'VM, when V is a diagonal matrix. M+MVM is the variance matrix of marginal Z-scores when effects are random

    // Normal approximation

    // Under H1 the marginal Z-scores are N(MMU, Sigma_tilde) with Sigma_tilde = M + MVM.
    // getRTilde takes the STANDARD DEVIATION of each Z_i, so pass sqrt of the diagonal,
    // not the variance (GLOWr 0.3.3: SD_tilde <- sqrt(diag(Sigma_tilde))).
    val sigmaTilde: BDM[Double] = M + MVM
    val sdTilde: BDV[Double] = sqrt(diag(sigmaTilde))
    val rTilde: BDV[Double] = getRTilde(g, mmu, sdTilde, fisher)

    // Bahadur efficiency (BE): moments under H0, where the Z_i are standard normal with
    // correlation M -- so plain M, zero means and NO scales.
    val sigmaBETilde: BDM[Double] = covM_gXgY(g, BDV.zeros[Double](n), BDV.zeros[Double](n), M)
    val wts_BE = getWts(sigmaBETilde, rTilde, forcePositiveWeights)
    // Asymptotic power efficiency (APE): moments under H1, Z ~ N(MMU, Sigma_tilde). The Mehler
    // expansion needs a CORRELATION matrix, so Sigma_tilde is split into its correlation and its
    // standard deviations, and the scales enter through g's argument (GLOWr 0.3.3:
    // CovM_gXgY(g, MMU, MMU, M = cov2cor(Sigma_tilde), SD1 = SD_tilde, SD2 = SD_tilde)).
    // Passing Sigma_tilde itself scaled order k by Sigma^k / k!: exact for a quadratic g, but it
    // amplified the quadrature residue of the higher coefficients by up to ~881^8 on real data.
    val rTildeCorr: BDM[Double] = cov2cor(sigmaTilde)
    diag(rTildeCorr) := 1.0 // as stats::cov2cor does; this package's cov2cor leaves 1 +/- rounding
    val sigmaAPETilde: BDM[Double] = covM_gXgY(g, mmu, mmu, rTildeCorr, sd1 = sdTilde, sd2 = sdTilde)
    val wts_APE: BDV[Double] = getWts(sigmaAPETilde, rTilde, forcePositiveWeights)

    // Sparse approximation

    val r = getR(g, bStar, pi, fisher)
    // Bahadur efficiency
    val sigmaBE = getSigma(g, bStar, pi, M, h1 = false)
    val wts_BE_sparse = getWts(sigmaBE, r, forcePositiveWeights)
    // asymptotic power rate
    val sigmaAPE = getSigma(g, bStar, pi, M, h1 = true)
    val wts_APE_sparse = getWts(sigmaAPE, r, forcePositiveWeights)

    return BDM(wts_BE, wts_APE, wts_BE_sparse, wts_APE_sparse)
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
    wts_BE := wts_BE / (mean(abs(wts_BE)))
    wts_APE := wts_APE / (mean(abs(wts_APE)))
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
    * where T_i=g(Z_i), Z_i = Z_0i + Ztilde_i, Ztilde_i ~ N(mu_i,sd_i), Z_0i~N(0,1)
    *
    * @param g transfrmation function of Z
    * @param mu vector of mean mu_i, i=1,...,n.
    * @param sd vector of standard deviations of Z_i, i=1,...,n.
    */
  def getRTilde(g: (Double) => Double, mmu: BDV[Double], sd: BDV[Double], fisher: Boolean): BDV[Double] = {
    val n = mmu.size
    val e0 = if (fisher) 2.0 else E_gX_p(g, 0.0, 1.0, 1.0)
    return BDV.tabulate(n){(i) => E_gX_p(g, mmu(i),  1.0, sd(i)) - e0}
  }

  /**
    * Calculate E[g(X)^p] where X~N(mu,sigma)
    *
    * @param g function
    * @param mu mean
    * @param p power
    * @param sigma std
    */
  def E_gX_p(g: Double => Double, mu: Double, p: Double, sigma:Double=1.0): Double = {
    val integrator = new GaussKronrod(1e-12, 100)
    return integrator.integrate((x) => math.pow(g(x), p) * Normal.density((x-mu)/sigma, 0, 1.0, false), mu-8*sigma, mu+8*sigma).estimate / sigma
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
    if (normalize) w := w /:/ mean(abs(w))
    return w
  }

  /**
    * The function integrated by covM_gXgY
    *
    * @param x variable of integration
    * @param g transformation function
    * @param mu mean value
    * @param ORD order of Hermite polynomial
    */
  def covM_Integrand(x: Double, g: Double => Double, mu: Double, ORD: Int): Double = {
    return g(x) * Normal.density(x-mu, 0.0, 1.0, false) * hermite_scalar(x - mu, ORD)
  }

  /**
    * The function integrated by covM_gXgY when a scale is present: the order-ORD Hermite
    * coefficient of g(mu + s*U), U ~ N(0, 1), written in the variable x = mu + U so the density
    * and the Hermite basis stay standard and the integration window mu +/- 8 stays fixed
    * whatever s is. Only g's argument is rescaled (GLOWr 0.3.3's CovM_gXgY integrand).
    *
    * @param s standard deviation of the variable g is applied to; s = 1 reduces to covM_Integrand
    */
  def covM_IntegrandScaled(x: Double, g: Double => Double, mu: Double, s: Double, ORD: Int): Double = {
    return g(mu + s * (x - mu)) * Normal.density(x-mu, 0.0, 1.0, false) * hermite_scalar(x - mu, ORD)
  }

  /**
    * Calculate the covariance matrix with elements Cov[g(X_i), g(Y_j)], where
    *   X_i = mu1_i + sd1_i Z_i,  Y_j = mu2_j + sd2_j Z_j,  Z ~ MVN(0, M),  M a CORRELATION matrix,
    * by Mehler's expansion: sum over k of c_k(mu1_i, sd1_i) c_k(mu2_j, sd2_j) M_ij^k / k!, where
    * c_k(mu, s) is the order-k Hermite coefficient of g(mu + s U). Without sd1/sd2 the scales are
    * 1 and the result is unchanged from before they existed.
    *
    * M must be a correlation matrix: the expansion is in powers of a correlation. A covariance
    * passed in its place is rejected -- pass cov2cor(Sigma) with sd = sqrt(diag(Sigma)).
    *
    * Guards, as GLOWr 0.3.3's CovM_gXgY: M positive semi-definite (smallest eigenvalue >= -1e-10;
    * R's check_positive_definite also WARNS below +1e-10, which is not ported); mu1, mu2, sd1 and
    * sd2 of equal length; sd1, sd2 > 0; diag(M) within 1e-8 of 1. Not ported: R's identity-matrix
    * fast path, a speed-up only -- for M = I the general sum gives the same values, since the
    * off-diagonal powers are 0.
    *
    * @param g transformation function
    * @param mu1 mean vector of length n
    * @param mu2 another mean vector of length n
    * @param M an n x n correlation matrix
    * @param ORD a sequence of orders of Hermite polynomials
    * @param sd1 standard deviations of X; null (R's NULL) means all 1
    * @param sd2 standard deviations of Y; null means all 1
    * @TODO ensure that ORD starts at 1 and is sequential
    * alternatively: accept an integer that just represents number of terms to use
    */
  def covM_gXgY(
    g: Double => Double,
    mu1: BDV[Double],
    mu2: BDV[Double],
    M: BDM[Double],
    ORD: Seq[Int] = Seq(1,2,3,4,5,6,7,8),
    sd1: BDV[Double] = null,
    sd2: BDV[Double] = null
  ): BDM[Double] = {
    val n = mu1.size
    val minEig = min(eigSymD.justEigenvalues(M))
    if (minEig < -1e-10)
      fatal(f"Correlation matrix M must be positive semi-definite. Minimum eigenvalue: $minEig%.2e")
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

    // Hermite coefficients, one column per variable, computed once per distinct (mu, s) pair --
    // the symmetric APE call and the all-zero BE call repeat pairs heavily. s == 1 keeps the
    // original integrand, so every unit-scale caller gets bit-identical coefficients.
    val integrator = new GaussKronrod(1e-9, 100)
    def coefs(mu: BDV[Double], s: BDV[Double]): BDM[Double] = {
      val out = BDM.zeros[Double](ORD.size, mu.size)
      val pairs = (0 until mu.size).map(k => (mu(k), s(k))).distinct
      for (i <- 0 until ORD.size; (m, sc) <- pairs) {
        // Written inline: GaussKronrod (Java) takes a java.util.function.Function, which Scala
        // converts from a lambda literal but not from a value of Scala function type.
        val c = integrator.integrate(
          (x) => if (sc == 1.0) covM_Integrand(x, g, m, ORD(i)) else covM_IntegrandScaled(x, g, m, sc, ORD(i)),
          m - 8.0,
          m + 8.0
        ).estimate
        for (k <- 0 until mu.size if mu(k) == m && s(k) == sc) out(i, k) = c
      }
      out
    }
    val coef1 = coefs(mu1, s1)
    val coef2 = coefs(mu2, s2)

    val M_out = BDM.zeros[Double](mu1.size, mu1.size)
    for (ord <- ORD) {
      // outer product of the rows of the coefficient matrices
      // must subtract 1 from ord because scala is 0-indexed
      M_out := M_out + (coef1(ord-1,::).t * coef2(ord-1,::)) *:* (M ^:^ (ord.toDouble) ) /:/ factorialDouble(ord)
    }

    return M_out
  }


  /**
    * Calculate the covariance matrix Cov[T_i, T_j], where T_i=g(Z_i), T_j=g(Z_j),
    * Z_i = Z_0i + mu_i*C_i, C_i ~ Bern(pi_i), Z_j = Z_0j + mu_j*C_j, C_j ~ Bern(pi_j),
    * Cov[Z_0i, Z_0i] = M_ij,
    *
    * @param g transformation function
    * @param mu mean vector
    * @param pi probability vector
    * @param M a positive-definite correlation matrix
    * @param ORD a sequence of orders of Hermite polynomials (unused, as the original code does not use this)
    */
  def covMT_mix(
    g: (Double) => Double,
    mu: BDV[Double],
    pi: BDV[Double],
    M: BDM[Double],
    ORD: Seq[Int]= Seq(1,2,3,4,5,6,7,8) // never actually used (like in the R code)
  ): BDM[Double] = {
    val pp: BDM[Double] = pi * pi.t
    val q: BDV[Double] = 1.0 - pi
    val pq: BDM[Double] = pi * q.t
    val qq: BDM[Double] = q * q.t
    val covM: BDM[Double] = covM_gXgY(g, mu, mu, M, (1 to 8).toList)
    val covM0: BDM[Double] = covM_gXgY(g, mu, BDV.zeros[Double](mu.size), M, (1 to 8).toList)
    val covM00: BDM[Double] = covM_gXgY(g, BDV.zeros[Double](mu.size), BDV.zeros[Double](mu.size), M, (1 to 8).toList)
    val M_out = (pp *:* covM) + (pq *:* covM0) + (pq.t *:* covM0.t) + (qq *:* covM00)
    diag(M_out) := varT_mix(g, mu, pi) // R code did this with sapply, so the arguments to varT_mix are scalar
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
    * Calculate Var[T_i], where T_i=g(Z_i), Z_i = Z_0i + mu_i*C_i, C_i ~ Bern(pi_i)
    *
    * @param g
    * @param mu
    * @param pi
    */
  def varT_mix(g: (Double) => Double, mu: Double, pi: Double): Double = {
    return pi * E_gX_p(g, mu, 2.0) + (1.0 - pi) * E_gX_p(g, 0.0, 2.0)  - math.pow(E_T_mix(g, mu, pi, false), 2.0)
  }

  /**
    * Calculate E[T_i], where T_i=g(Z_i), Z_i = Z_0i + mu_i*C_i, C_i ~ Bern(pi_i)
    *
    * @param g
    * @param mu
    * @param pi
    */
  def E_T_mix(g: (Double) => Double, mu: Double, pi: Double, fisher: Boolean): Double = {
    val e0 = if(fisher) 2.0 else E_gX_p(g, 0.0, 1)
    if (pi == 0)
      return e0
    return pi * E_gX_p(g, mu, 1) + (1.0 - pi) * e0
  }

  def getSigma(g: (Double) => Double, mu: BDV[Double], pi: BDV[Double], M: BDM[Double], h1: Boolean = false): BDM[Double] = {
    if (h1)
      return covMT_mix(g, mu, pi, M)

    val nullMu = BDV.zeros[Double](mu.size)
    return covMT_mix(g, nullMu, nullMu, M)
  }

  def getR(g: (Double) => Double, mu: BDV[Double], pi: BDV[Double], fisher: Boolean): BDV[Double] = {
    val n = mu.size
    val e0 = E_T_mix(g, 0.0, 0.0, fisher)
    return BDV.tabulate(n){(i) => E_T_mix(g, mu(i), pi(i), fisher) -  e0}
  }



}
