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
  *   KHoar 2025-05-07: Added header to file
  */

package is.hail.methods.gfisher

import is.hail.GaussKronrod

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
    val M = cov2cor(gHG)
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

    val rTilde: BDV[Double] = getRTilde(g, mmu, diag(M + MVM), fisher) // off by average of 2.57e-5 :(

    //Bahadur efficiency (BE)
    val sigmaBETilde: BDM[Double] = covM_gXgY(g, BDV.zeros[Double](n), BDV.zeros[Double](n), M) // accurate to 1e-16
    val wts_BE = getWts(sigmaBETilde, rTilde, forcePositiveWeights) // off by average of 3.27e-4 :(
    // asymptotic power rate (APE?)
    val sigmaAPETilde: BDM[Double] = covM_gXgY(g, mmu, mmu, M + MVM)
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

  def getH_Continuous(X: BDM[Double], y: BDV[Double]): (BDM[Double], Double, BDV[Double]) = {
    // compute solution to X * beta = y, manually calculate residuals

    val yPred = lin_reg_predict(X, y, method="direct", addIntercept=true)
    val resids = y - yPred

    val sd = stddev(resids)
    val HHalf = X * (cholesky(inv(X.t * X)))
    return(HHalf, sd, resids)
  }

  def getH_Binary(X: BDM[Double], y: BDV[Double]): (BDM[Double], BDV[Double], BDV[Double]) = {
    val y0 = log_reg_predict(X, y)
    val resids = y - y0

    val XTilde = sqrt(y0 *:* (1.0-y0)) *:* X(::,*)
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
    * Calculate Covariance matrix whose elements are Cov[g(Xi),g(Yj)] where X = Z + MU1 ~ MVN(MU1, M), Y = Z + MU2 ~ MVN(MU2, M)
    *
    * @param g transformation function
    * @param mu1 mean vector of length n
    * @param mu2 another mean vector of length n
    * @param M a positive-definite correlation matrix
    * @param ORD a sequence of orders of Hermite polynomials
    * @TODO ensure that ORD starts at 1 and is sequential
    * alternatively: accept an integer that just represents number of terms to use
    */
  def covM_gXgY(g: Double => Double, mu1: BDV[Double], mu2: BDV[Double], M: BDM[Double], ORD: Seq[Int]=Seq(1,2,3,4,5,6,7,8)): BDM[Double] = {
    val mu1U = mu1.toArray.distinct
    val mu2U = mu2.toArray.distinct
    val coef1 = BDM.zeros[Double](ORD.size, mu1.size)
    val coef2 = BDM.zeros[Double](ORD.size, mu2.size)
    val integrator = new GaussKronrod(1e-9, 100)
    for (i <- 0 until ORD.size) {

      for (j <- 0 until mu1U.size) {

        val c = integrator.integrate(
          (x) => covM_Integrand(x, g, mu1U(j), ORD(i)),
          mu1U(j)-8.0,
          mu1U(j)+8.0
        ).estimate
        for (k <- 0 until mu1.size) {
          if (mu1(k) == mu1U(j)) {
            coef1(i, k) = c
          }
        }
      }
    }
    for (i <- 0 until ORD.size) {

      for (j <- 0 until mu2U.size) {

        val c = integrator.integrate(
          (x) => covM_Integrand(x, g, mu2U(j), ORD(i)),
          mu2U(j)-8.0,
          mu2U(j)+8.0
        ).estimate
        for (k <- 0 until mu2.size) {
          if (mu2(k) == mu2U(j)) {
            coef2(i, k) = c
          }
        }
      }
    }
    // val coef1 = BDM.tabulate(ORD.size, mu1.size){(i, j) =>
    //   integrator.integrate(
    //     (x) => covM_Integrand(x, g, mu1(j), ORD(i)),
    //     mu1(j)-8.0,
    //     mu1(j)+8.0
    //   ).estimate
    // }
    // val coef2 = BDM.tabulate(ORD.size, mu2.size){(i, j) =>
    //   integrator.integrate(
    //     (x) => covM_Integrand(x, g, mu2(j), ORD(i)),
    //     mu2(j)-8.0,
    //     mu2(j)+8.0
    //   ).estimate
    // }

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
