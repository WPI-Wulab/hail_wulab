package is.hail.methods.gfisher

import is.hail.GaussKronrod

import breeze.linalg.{DenseVector => BDV, DenseMatrix => BDM, _}

import net.sourceforge.jdistlib.Normal

import org.apache.commons.math3.util.CombinatoricsUtils.factorialDouble

object GFisherWeights {



  /**
    * Calculate E[g(X)^p] where X~N(mu,sigma)
    *
    * @param g function
    * @param mu mean
    * @param p power
    * @param sigma std
    */
  def E_gX_p(g: Double => Double, mu: Double, p: Double, sigma:Double=1.0): Double = {
    val integrator = new GaussKronrod(1e-9, 100)
    return integrator.integrate((x) => math.pow(g(x), p) * Normal.density((x-mu)/sigma, 0, 1.0, false), mu-8*sigma, mu+8*sigma).estimate / sigma
  }

  /**
    * Calculate the optimal weights of T = sum_i^n w_iT_i under Gaussian mixture model
    *
    * @param Sigma covariance matrix of T_1,...,T_n.
    * @param r vector of E_1(T_i) - E_0(T_i), i=1,...,n.
    * @param forcePosW if true, negative weights are set to 0.
    * @param normalize if true, weights are normalized to sum to 1.
    */
  def getWts(Sigma: BDM[Double], r: BDV[Double], forcePosW: Boolean=true, normalize:Boolean = true): BDV[Double] = {
    val w = inv(Sigma) * r
    if (forcePosW) w(w <:< 0.0) := 0.0
    if (normalize) w := w /:/ sum(w)
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
    return g(x) * Normal.density(x-mu, 0.0, 1.0, false) * hermite(x - mu, ORD)
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
    val integrator = new GaussKronrod(1e-9, 100)
    val coef1 = BDM.tabulate(ORD.size, mu1.size){(i, j) =>
      integrator.integrate(
        (x) => covM_Integrand(x, g, mu1(j), ORD(i)),
        mu1(j)-8.0,
        mu1(j)+8.0
      ).estimate
    }
    val coef2 = BDM.tabulate(ORD.size, mu2.size){(i, j) =>
      integrator.integrate(
        (x) => covM_Integrand(x, g, mu2(j), ORD(i)),
        mu2(j)-8.0,
        mu2(j)+8.0
      ).estimate
    }
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
    return pi * E_gX_p(g, mu, 2.0) + (1.0 - pi) * E_gX_p(g, 0.0, 2.0)  - math.pow(E_T_mix(g, mu, pi), 2.0)
  }

  /**
    * Calculate E[T_i], where T_i=g(Z_i), Z_i = Z_0i + mu_i*C_i, C_i ~ Bern(pi_i)
    *
    * @param g
    * @param mu
    * @param pi
    */
  def E_T_mix(g: (Double) => Double, mu: Double, pi: Double): Double = {
    return pi * E_gX_p(g, mu, 1) + (1.0 - pi) * E_gX_p(g, 0.0, 1)
  }

  /**
    * Calculate E[T_i], where T_i=g(Z_i), Z_i = Z_0i + mu_i*C_i, C_i ~ Bern(pi_i)
    *
    * @param g
    * @param mu
    * @param pi
    */
  def E_T_mix(g: (Double) => Double, mu: BDV[Double], pi: BDV[Double]): BDV[Double] = {
    return BDV.tabulate(mu.length){i => E_T_mix(g, mu(i), pi(i))}
  }

  def getSigma(g: (Double) => Double, mu: BDV[Double], pi: BDV[Double], M: BDM[Double], h1: Boolean = false): BDM[Double] = {
    if (h1)
      return covMT_mix(g, mu, pi, M)

    val nullMu = BDV.zeros[Double](mu.size)
    return covMT_mix(g, nullMu, nullMu, M)
  }

  def getR(g: (Double) => Double, mu: BDV[Double], pi: BDV[Double]): BDV[Double] = {
    val n = mu.size
    return BDV.tabulate(n){(i) => E_T_mix(g, mu(i), pi(i)) - E_T_mix(g, 0.0, 0.0) }
  }

  /**
    * Calculate the vector of E_1(T_i)-E_0(T_i) for i=1,...,n
    * where T_i=g(Z_i), Z_i = Z_0i + Ztilde_i, Ztilde_i ~ N(mu_i,sd_i), Z_0i~N(0,1)
    *
    * @param g transfrmation function of Z
    * @param mu vector of mean mu_i, i=1,...,n.
    * @param sd vector of standard deviations of Z_i, i=1,...,n.
    */
  def getRTilde(g: (Double) => Double, mu: BDV[Double], sd: BDV[Double]): BDV[Double] = {
    val n = mu.size
    return BDV.tabulate(n){(i) => E_gX_p(g, mu(i),  1.0, sd(i)) - E_gX_p(g, 0.0, 1.0, 1.0) }
  }



}
