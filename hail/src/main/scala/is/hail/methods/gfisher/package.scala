/**
  * File that contains the functions necessary to support GFisher and GLOW calculations
  * Reference: Zhang, Hong, and Zheyang Wu. "The generalized Fisher's combination and accurate p‐value
           calculation under dependence." Biometrics 79.2 (2023): 1159-1172.
  * @author Peter Howell and Kylie Hoar
  * Last update (latest update first):
  *   Claude Code (Opus 5), prompted by VPanwar, 2026-09-09: Added the zero-variance guard --
  *     zeroVarianceIdx, cov2corZeroSafe and zScoresZeroSafe -- mirroring GLOWr's
  *     .handle_zero_variance_snps and getZ_marg_score's treatment of the variants it flags. A
  *     monomorphic variant (or one collinear with the covariates) is a constant vector that lies
  *     in the intercept-bearing design matrix's column space, so its residual is zero and
  *     GHG[j,j] = 0 exactly; cov2cor then divided by it. In floating point the entry lands near
  *     zero from either side -- measured at -1.7e-13 -- so sqrt gives NaN rather than Inf, which
  *     propagates silently. The damage was not local: one NaN row and column made all 25 entries
  *     of a 5x5 getGFisherGM NaN, sent sum(GM) to NaN and made GFisher's p.GFisher fail outright,
  *     so one uninformative variant cost the whole gene's p-value. Found by the group-D fixtures.
  *   Claude Code (Opus 5), prompted by VPanwar, 2026-09-09: Two changes, both required by the
  *     group-C fixture suite (packages/glow-core in the GLOWh repo). (1) Rewrote log_reg_fit's
  *     optimiser as iteratively reweighted least squares, the algorithm R's glm.fit uses, with
  *     R's own starting value (mustart = (y + 1/2)/2), stopping rule (relative deviance change
  *     < 1e-8) and iteration cap (25). The previous L-BFGS minimisation of the same likelihood
  *     left the fitted probabilities wrong in the 5th significant digit, and the marginal
  *     Z-scores built on them wrong by up to 2.1e-04 relative; matching R's iterate path brought
  *     every binary Z-score, M_Z and M_s row inside 1e-10. (2) Added an addIntercept flag to
  *     log_reg_fit and log_reg_predict (defaulting to true, so every existing caller is
  *     unchanged), because getH_Binary must fit and project with the SAME design matrix to fix
  *     defect C1 and both helpers otherwise appended a second intercept column. The flag was
  *     already documented in log_reg_fit's docstring but had never existed.
  *     Mirrors: R's glm.fit (src/library/stats/R/glm.R) and binomial()$initialize.
  *   Claude Code (Opus 5), prompted by VPanwar, 2026-09-02: Fixed nearPD, which returned
  *     matrices that were not positive definite. eigSymD wraps LAPACK dsyevd and returns
  *     eigenvalues in ASCENDING order, but the code indexed them as if descending (R's
  *     convention): the eigTol threshold and Eps were scaled by the SMALLEST eigenvalue,
  *     and the posdefify guard tested the LARGEST against Eps -- a condition that never
  *     fires, so the clipping never ran. Found by the group-A golden-fixture suite
  *     (packages/glow-core in the GLOWh repo), where getGFisherLambda threw
  *     NotConvergedException from cholesky on an AR(1) case whose smallest eigenvalue was
  *     still -8.24e-07 after nearPD. Also rewrote the rescaling step explicitly as
  *     diag(D) * X * diag(D). Mirrors: Matrix::nearPD / Matrix::posdefify.
  *   KHoar 2025-05-07: Added header to file
  */

package is.hail.methods

import is.hail.stats.eigSymD
import is.hail.utils.fatal

import breeze.linalg.{DenseMatrix => BDM, DenseVector => BDV, _}
import breeze.numerics.{abs, sqrt, sigmoid}
import breeze.optimize.{DiffFunction, minimize}
import net.sourceforge.jdistlib.{ChiSquare, Normal}

package object gfisher {

  def vecDiff(x: BDV[Double], y: BDV[Double]): Unit = {
    val d = abs(x - y)
    printVec(d)
  }

  def printVec(x: BDV[Double], name: String = ""): Unit = {
    val s = x.data.map(i => f"$i%.2e").mkString("[", ", ", "]")
    if (name == ""){
      println(s)
    } else {
      println(s"$name: $s")
    }
  }

  def matDiff(X: BDM[Double], Y: BDM[Double]): Unit = {
    val mat = abs(X - Y)
    printMat(mat)
  }

  def printMat(mat: BDM[Double], name: String="") = {
   val rows = for (i <- 0 until mat.rows) yield {
      (0 until mat.cols).map(j => f"${mat(i, j)}%.2e").mkString("[", ", ", "]")
    }
    val linesep = if(name == "" ){
      "\n"
    } else {
      s"\n$name: "
    }
    val s = rows.mkString(linesep)
    if (name == ""){
      println(s)
    } else {
      println(s"$name: $s")
    }
  }

  def all[T](vals: Iterable[T], condition: (T) => Boolean): Boolean = {
    for (x <- vals) {
      if (!condition(x)) {
        return false
      }
    }
    return true
  }

  def all(bools: Iterable[Boolean]): Boolean = {
    for (bool <- bools) {
      if (!bool)
        return false
    }
    return true
  }

  /**
    * Calculate Pearson's correlation coefficient between two vectors
    *
    * @param x
    * @param y
    */
  def cor(x: BDV[Double], y: BDV[Double]): Double = {
    val n = x.length
    val mx = sum(x) / n.toDouble
    val my = sum(y) / n.toDouble
    val xmx = x - mx
    val ymy = y - my
    return (xmx dot ymy) / sqrt(xmx dot xmx) / sqrt(ymy dot ymy)
  }

  // this function is not correct and should not be used
  def rowCorBad(X: BDM[Double]): BDM[Double] = {
    val rows = X.rows
    val rowsD = rows.toDouble
    // val cols = X.cols
    val res = BDM.eye[Double](rows)
    val rowMeans = new Array[Double](rows)

    for (i <- 0 until rows) {
      rowMeans(i) = sum(X(i,::)) / rowsD
      //maybe: do x - xbar before
    }
    for (i <- 0 until (rows - 1)) {
      // maybe get Xi - Xbar here
      val xmx = X(i,::) - rowMeans(i)
      for (j <- (i + 1) until rows) {
        val ymy = X(j,::) - rowMeans(j)
        res(i, j) = sum(xmx *:* ymy) / sqrt(sum(xmx *:* xmx)) / sqrt(sum(ymy *:* ymy))
        res(j,i) = res(i,j)
      }
    }
    return res
  }

  /**
    * Calculate Pearson's correlation coefficient between the columns of a matrix
    *
    * Creates a copy of the matrix to avoid modifying it.
    *
    * @param x matrix
    */
  def colCorrelation(x: BDM[Double]): BDM[Double] = {
    val rows = x.rows
    val cols = x.cols
    val rowsD = rows.toDouble
    val X = x.copy
    val res = BDM.eye[Double](cols)
    val sumSqrs = new Array[Double](cols)
    for (i <- 0 until cols) {
      var sum = 0.0
      for (j <- 0 until rows)
        sum += X(j,i)
      val mean = sum / rowsD
      var sumSqr = 0.0
      for (j <- 0 until rows) {
        X(j,i) = X(j,i) - mean
        sumSqr += math.pow(X(j,i), 2.0)
      }
      sumSqrs(i) = math.sqrt(sumSqr)
    }
    for (i <- 0 until cols) {
      for (j <- (i+1) until cols) {
        var numerator = 0.0
        for (k <- 0 until rows)
          numerator += X(k, i) * X(k, j)
        val denom = sumSqrs(i) * sumSqrs(j)
        res(i,j) = numerator / denom
        res(j,i) = res(i,j)
      }
    }
    return res
  }

  /**
    * Calculate Pearson's correlation coefficient between the rows of a matrix
    *
    * This method creates a copy of the matrix to avoid modifying it.
    *
    * @param x
    */
  def rowCorrelation(x: BDM[Double]): BDM[Double] = {
    val rows = x.rows
    val cols = x.cols
    val colsD = cols.toDouble
    val X = x.copy
    val res = BDM.eye[Double](rows)
    val sumSqrs = new Array[Double](rows)
    for (i <- 0 until rows) {
      var sum = 0.0
      for (j <- 0 until cols)
        sum += X(i,j)
      val mean = sum / colsD
      var sumSqr = 0.0
      for (j <- 0 until cols) {
        X(i,j) = X(i,j) - mean
        sumSqr += math.pow(X(i,j), 2.0)
      }
      sumSqrs(i) = math.sqrt(sumSqr)
    }
    for (i <- 0 until rows) {
      for (j <- (i+1) until rows) {
        var numerator = 0.0
        for (k <- 0 until cols)
          numerator += X(i, k) * X(j, k)
        val denom = sumSqrs(i) * sumSqrs(j)
        res(i,j) = numerator / denom
        res(j,i) = res(i,j)
      }
    }
    return res
  }

  /**
    * This function is slower than the one above.
    *
    * @param X
    */
  def rowCorrelationSlow(X: BDM[Double]): BDM[Double] = {
    val rows = X.rows
    val cols = X.cols
    val rowMeans = sum(X(*,::)) / cols.toDouble
    val XMX = X(::, *) - rowMeans
    val res = BDM.eye[Double](rows)
    for (i <- 0 until (rows - 1)) {
      for (j <- (i+1) until (rows)) {
        res(i, j) = (XMX(i,::) dot XMX(j,::)) / sqrt(XMX(i,::) dot XMX(i,::)) / sqrt(XMX(j,::) dot XMX(j,::))
        res(j, i) = res(i, j)
      }
    }
    return res
  }




  /**
    * Uses method described in the _linear_skat function in statgen.py to directly compute the predicted values of the best fit model y = Xb
    *
    * @param x
    * @param y
    * @param addIntercept whether to add a column of ones to x.
    * @param method method to solve the equation. "direct" bypasses the coefficient and goes directly to the best-predicted value using the reduced QR decomposition. "qr" uses the method that Skat.scala uses to calculate beta which uses breeze's solve after using QR.
    * "breeze" simply does `X \ y`. "naive" uses the closed form equation for OLS, `inv(X.t * X) * X.t * y`
    *
    */
  def lin_reg_predict(x: BDM[Double], y: BDV[Double], method: String = "direct", addIntercept: Boolean = true): BDV[Double] = {
    val X = if (addIntercept) {
      BDM.horzcat(BDM.ones[Double](x.rows, 1), x)
    } else {
      x
    }
    if (method == "direct") {
      val QR = qr.reduced(X)
      val q = QR.q
      return q * q.t * y
    }
    val beta = lin_solve(X, y, method, false)
    return X * beta
  }

  /**
    * wrapper for solving linear equation Xb = y for b. returns the solution. Like `np.solve(X, y)` with numpy, or `X \ y` in matlab, or `solve(X, y)` in R
    *
    * @param X matrix
    * @param y vector
    * @param method how to compute it either qr, breeze, or naive.
    * `qr` uses the method that Skat.scala uses, which still uses breeze's solve.
    * `breeze` simply does `X \ y`.
    * `naive` uses the closed form equation for OLS, `inv(X.t * X) * X.t * y`
    */
  def lin_solve(x: BDM[Double], y: BDV[Double], method: String = "qr", addIntercept: Boolean = false): BDV[Double] = {
    val X = if (addIntercept) {
      BDM.horzcat(BDM.ones[Double](x.rows, 1), x)
    } else {
      x
    }
    method match {
      case "qr" =>
        val QR = qr.reduced(x)
        val Qt = QR.q.t
        val R = QR.r
        R \ (Qt * y)
      case "breeze" => X \ y
      case "naive" => inv(X.t * X) * X.t * y
      case _ => throw new IllegalArgumentException(s"unknown method: $method")
    }
  }

  def stdErrLinearRegression(X: BDM[Double], y: BDV[Double]): (BDV[Double], BDV[Double], BDV[Double]) = {
    val QR = qr.reduced(X)
    val Qt = QR.q.t
    val R = QR.r
    val beta = R \ (Qt * y)
    val yhat = X * beta
    val residuals: BDV[Double] = y - yhat
    val sigma2: Double = (residuals dot residuals) / (X.rows - X.cols)
    val se: BDV[Double] = sqrt(diag(inv(R.t * R)) * sigma2)
    return (beta, se, residuals)
  }




  /**
    * fit coefficients for a logistic regression model
    *
    * Created by Kylie Hoar
    *
    * Modified by Peter Howell, 06 Feb 2025
    *
    * @param X feature matrix
    * @param y response variable
    * @param addIntercept
    * @return the fitted coefficients
    */
  def log_reg_fit(X: BDM[Double], y: BDV[Double], addIntercept: Boolean = true): BDV[Double] = {
    // addIntercept = false lets a caller that has ALREADY built its design
    // matrix (getH_Binary) fit and form H_half from the very same matrix, as
    // R's fit_null_model does. The default keeps every existing caller's
    // behaviour unchanged.
    val XWithIntercept = if (addIntercept) BDM.horzcat(BDM.ones[Double](X.rows, 1), X) else X

    // Iteratively reweighted least squares, the algorithm R's glm.fit uses for
    // family = binomial (see R's src/library/stats/R/glm.R). This is a Newton
    // method on the log-likelihood: for the canonical logit link the expected
    // and observed information coincide, so each step solves the weighted normal
    // equations exactly and the iteration converges quadratically. The previous
    // quasi-Newton (L-BFGS) minimisation of the same likelihood stopped at the
    // optimiser's default gradient tolerance, which left the fitted
    // probabilities wrong in the 5th significant digit and the marginal
    // Z-scores derived from them wrong by up to 2.1e-04 relative -- measured
    // against the group-C fixtures, where the residual is now ~1e-15.
    //
    //   eta   = X beta                     (linear predictor)
    //   mu    = 1 / (1 + exp(-eta))        (fitted probabilities)
    //   w     = mu (1 - mu)                (IRLS weights = Bernoulli variance)
    //   z     = eta + (y - mu) / w         (working response)
    //   beta <- (X' W X)^-1 X' W z         (weighted least squares step)
    //
    // Convergence uses R's own criterion and defaults: the relative change in
    // deviance, |dev - dev_old| / (|dev| + 0.1) < 1e-8, capped at 25 iterations
    // (glm.control's epsilon and maxit).
    val n = XWithIntercept.rows
    val p = XWithIntercept.cols
    var beta = BDV.zeros[Double](p)
    var devOld = Double.PositiveInfinity
    var dev = 0.0
    var iter = 0
    var converged = false

    // R's binomial()$initialize starts the iteration from the RESPONSE, not from
    // beta = 0: mustart = (y + 1/2) / 2, eta = log(mu / (1 - mu)). Starting
    // elsewhere converges to the same optimum on a well-separated problem but
    // takes a different iterate path, and under near-separation (extreme
    // case-control imbalance) the two paths stop at measurably different points
    // -- the likelihood is flat there, so the shared stopping rule does not pin
    // the fit any tighter than the path does.
    var muCur: BDV[Double] = (y + 0.5) / 2.0
    var etaCur: BDV[Double] = breeze.numerics.log(muCur /:/ (1.0 - muCur))

    while (iter < 25 && !converged) {
      val eta = etaCur
      val mu = muCur
      // An observation whose fitted probability saturates has w = 0 and an
      // infinite working response. R's glm.fit EXCLUDES those rows from the
      // weighted fit (its `good <- mu.eta.val != 0` mask) rather than flooring
      // them; flooring biases the fit under extreme case-control imbalance,
      // measured at 1.7e-08 relative on the group-C binary_extreme case.
      val w = BDV.tabulate(n) { i => mu(i) * (1.0 - mu(i)) }
      val z = eta + (y - mu) /:/ w

      // Weighted normal equations over the retained rows only, formed as
      // (sqrt(W) X)' (sqrt(W) X). A zero weight contributes nothing to either
      // side, so zeroing sqrt(w) there is exactly the exclusion R performs.
      val sw = BDV.tabulate(n) { i => if (w(i) > 0.0 && !z(i).isNaN && !z(i).isInfinite) math.sqrt(w(i)) else 0.0 }
      val zSafe = BDV.tabulate(n) { i => if (sw(i) > 0.0) z(i) else 0.0 }
      val Xw = XWithIntercept(::, *) *:* sw
      val zw = zSafe *:* sw
      beta = (Xw.t * Xw) \ (Xw.t * zw)

      // Binomial deviance, -2 * log-likelihood, with the log arguments clamped
      // so a saturated fit cannot produce -Inf before the criterion is applied.
      etaCur = XWithIntercept * beta
      muCur = sigmoid(etaCur)
      dev = -2.0 * sum((0 until n).map { i =>
        val m = math.min(math.max(muCur(i), 1e-300), 1.0 - 1e-16)
        y(i) * math.log(m) + (1.0 - y(i)) * math.log(1.0 - m)
      })
      converged = math.abs(dev - devOld) / (math.abs(dev) + 0.1) < 1e-8
      devOld = dev
      iter += 1
    }

    if (!converged) {
      // A warning, not a failure: R's glm.fit likewise warns and returns the last
      // iterate rather than stopping.
      Console.err.println(s"WARNING: log_reg_fit: IRLS did not converge in 25 iterations " +
                          s"(deviance $dev); the fitted null model may be unreliable.")
    }
    return beta
  }

  /**
    * Train the logistic regression model using gradient descent and predict probabilities for the given feature matrix.
    * @param X Input feature matrix (rows: observations, cols: features).
    * @param y Target vector (binary labels: 0 or 1).
    * @return predicted probabilities from fitted model
    */
  def log_reg_predict(X: BDM[Double], y: BDV[Double], addIntercept: Boolean = true): BDV[Double] = {
    // Add intercept term by appending a column of ones to X
    val XWithIntercept = if (addIntercept) BDM.horzcat(BDM.ones[Double](X.rows, 1), X) else X
    val coefficients = log_reg_fit(X, y, addIntercept)
    // Predict probabilities for the given feature matrix
    return sigmoid(XWithIntercept * coefficients)
  }

  /**
    * GFisher transformation function 'g' for two sided p-values
    *
    * @param x
    * @param df
    */
  def gGFisher2(x: Double, df: Double=2.0): Double = {
    return ChiSquare.quantile(
      math.log(2.0) +
      Normal.cumulative(
        math.abs(x),
        0.0,
        1.0,
        false, //lower tail
        true //log_p
      ),
      df,
      false, //lower_tail
      true //log_p
    )
  }

  /**
    * Compute hermite polynomil for a scalar input
    *
    * @param x input
    * @param degree degree of polynomial. should be 1, 2, 3, 4, 6, or 8
    */
  def hermite_scalar(x: Double, degree: Int): Double = {
    return degree match {
      case 1 => x
      case 2 => x*x - 1.0
      case 3 => math.pow(x, 3.0) - 3.0   * x
      case 4 => math.pow(x, 4.0) - 6.0   * math.pow(x, 2.0) + 3.0
      case 5 => math.pow(x, 5.0) - 10.0  * math.pow(x, 3.0) + 15.0  * x
      case 6 => math.pow(x, 6.0) - 15.0  * math.pow(x, 4.0) + 45.0  * math.pow(x, 2.0) - 15.0
      case 7 => math.pow(x, 7.0) - 21.0  * math.pow(x, 5.0) + 105.0 * math.pow(x, 3.0) - 105.0 * x
      case _ => math.pow(x, 8.0) - 28.0  * math.pow(x, 6.0) + 210.0 * math.pow(x, 4.0) - 420.0 * math.pow(x, 2.0) + 105.0
    }
  }


  /**
    * Code to convert a covariance matrix to a correlation matrix. Should mimic R's function.
    *
    * @param X
    */
  def cov2cor(X: BDM[Double]): BDM[Double] = {
    X / (sqrt(diag(X)) * sqrt(diag(X)).t)
  }

  /**
    * Indices of variants with zero adjusted variance -- a zero entry on the
    * diagonal of GHG.
    *
    * A monomorphic variant is a constant vector, and the null model's design
    * matrix carries an intercept, so the variant lies entirely in X's column
    * space. Residualizing it gives the zero vector, hence GHG[j,j] = 0 exactly
    * and every cross-term GHG[i,j] = 0 with it. The same holds for any variant
    * collinear with the covariates. In floating point the diagonal entry lands
    * near zero rather than on it -- measured at -1.7e-13 on the group-D fixture
    * -- so the test is a threshold, and a NEGATIVE entry must be caught too:
    * sqrt of it is NaN, which propagates silently where an Inf would not.
    *
    * Mirrors GLOWr's .handle_zero_variance_snps, including its threshold of
    * machine epsilon. Note math.ulp(1.0) is machine epsilon (2.22e-16);
    * Double.MinPositiveValue is the smallest subnormal and would never fire.
    */
  def zeroVarianceIdx(GHG: BDM[Double]): IndexedSeq[Int] = {
    val d = diag(GHG)
    (0 until d.length).filter(i => !(d(i) >= math.ulp(1.0)))   // catches NaN and negatives
  }

  /**
    * cov2cor with the zero-variance variants neutralised rather than allowed to
    * divide by zero.
    *
    * Without this, one such variant does NOT merely spoil its own row: measured
    * on a 5-variant example, a single NaN row and column in M makes all 25
    * entries of getGFisherGM NaN, sends sum(GM) -- the HYB p-value's sigma2 --
    * to NaN, and makes GFisher's p.GFisher fail outright. One uninformative
    * variant would cost the whole gene's p-value.
    *
    * The treatment mirrors GLOWr's getZ_marg_score exactly: clamp the diagonal
    * to 1 before the division, then zero the variant's row and column and set
    * its diagonal back to 1. The variant becomes uncorrelated with everything,
    * with unit variance, so it contributes nothing to the combination while the
    * rest of the gene stays testable. Under optimal weights its weight is zero
    * anyway, since Bstar_j is proportional to sqrt(GHG[j,j]).
    */
  def cov2corZeroSafe(GHG: BDM[Double]): BDM[Double] = {
    val idx = zeroVarianceIdx(GHG)
    if (idx.isEmpty) return cov2cor(GHG)

    val safe = GHG.copy
    idx.foreach(i => safe(i, i) = 1.0)
    val M = cov2cor(safe)
    idx.foreach { i =>
      M(i, ::) := 0.0
      M(::, i) := 0.0
      M(i, i) = 1.0
    }
    Console.err.println(s"WARNING: ${idx.length} variant(s) with zero adjusted variance " +
      s"(indices ${idx.mkString(", ")}) -- monomorphic, or collinear with the covariates. " +
      "Their correlations are set to zero and their Z-scores to zero, so they contribute " +
      "nothing; the remaining variants are unaffected.")
    M
  }

  /**
    * Marginal Z-scores with the zero-variance variants set to 0 rather than to
    * NaN by a 0/0 division. Mirrors GLOWr's getZ_marg_score, which divides by a
    * clamped diagonal and then overwrites those entries with 0.
    *
    * Zero is the right value rather than a placeholder: every g used by this
    * package maps 0 to 0 -- the two-sided GFisher transform sends Z = 0 to
    * p = 1 and hence to a chi-square quantile of 0, and the Burden transform is
    * the identity -- so the variant contributes nothing to any statistic.
    */
  def zScoresZeroSafe(score: BDV[Double], GHG: BDM[Double]): BDV[Double] = {
    val d = diag(GHG)
    val idx = zeroVarianceIdx(GHG)
    val safe = BDV.tabulate(d.length)(i => if (idx.contains(i)) 1.0 else d(i))
    val z = score /:/ sqrt(safe)
    idx.foreach(i => z(i) = 0.0)
    z
  }

/**
  * Computes the infinity norm of a matrix, ie, the maximum of the row sums of the absolute values.
  *
  * @param X matrix
  *
  */
  def normI(X: BDM[Double]): Double = {
    val aX = abs(X)
    return max(sum(aX(*, ::)))
  }

  /**
    * Code that copies R's code for getting the nearest positive definite matrix to an approximate one
    *
    * @TODO benchmark efficiency of var vs val and reassignment with :=
    * @param M symmetric square
    */
  def nearPD(M: BDM[Double]): BDM[Double] = {
    val n: Int = M.cols
    if (M.rows != n) fatal("Matrx M must be a square matrix")

    //tolerances
    val eigTol: Double = 1e-06
    val convTol: Double = 1e-07
    val posdTol: Double = 1e-08

    var X: BDM[Double] = M // n x n
    var DS: BDM[Double] = BDM.zeros[Double](n, n)
    var R: BDM[Double] = BDM.zeros[Double](n, n)
    var iter: Int = 0
    var converg: Boolean = false

    var conv: Double = Double.PositiveInfinity
    while (iter < 100 && !converg) {
      val Y: BDM[Double] = X
      R = Y - DS
      val eigRes = eigSymD(R)
      val (eigenvalues: BDV[Double], eigenvectors: BDM[Double]) = (eigRes.eigenvalues, eigRes.eigenvectors)

      // eigSymD wraps LAPACK dsyevd, which returns eigenvalues in ASCENDING order
      // -- the reverse of R's eigen() (see GFisherLambda.scala:47). The threshold
      // is relative to the LARGEST eigenvalue, which is therefore the LAST entry.
      // val adjustedEigenvalues = eigenvalues.map(ev => if (ev > eigTol * eigenvalues(-1)) ev else 1e-10)
      val p: BitVector = eigenvalues >:> (eigTol * eigenvalues(n - 1))
      if (! any(p) ) {
        fatal("Matrix is negative definite")
      }
      val Q: BDM[Double] = eigenvectors(::, p).toDenseMatrix
      val d: BDV[Double] = eigenvalues(p).toDenseVector
      // X = (Q *:* tile(d, 1, n).t) * Q.t // equivalent to below
      X = (Q(*,::) *:* d) * Q.t // equivalent to above
      DS = X - R

      diag(X) := 1.0
      conv = normI(Y - X) / normI(X)
      iter += 1
      converg = conv <= convTol
    }
    // Final positive-definiteness projection, mirroring R's Matrix::posdefify.
    // eigSymD returns eigenvalues ASCENDING, so the smallest is index 0 and the
    // largest is index n-1. The previous code had these the other way round: it
    // scaled Eps by the SMALLEST eigenvalue and then tested the LARGEST against
    // it, a condition that is essentially never true, so the clipping below never
    // ran and nearPD returned matrices that were still not positive definite
    // (measured: smallest eigenvalue -8.24e-07 on an AR(1) case, which then made
    // cholesky throw NotConvergedException in getGFisherLambda).
    val eigRes = eigSymD(X)
    val (eigenvalues: BDV[Double], eigenvectors: BDM[Double]) = (eigRes.eigenvalues, eigRes.eigenvectors)
    val Eps = posdTol * abs(eigenvalues(n - 1))
    if (eigenvalues(0) < Eps) {
      // R's o.diag: the diagonal BEFORE reconstruction. Copied because the
      // reconstruction rebinds X.
      val oDiag: BDV[Double] = diag(X).copy
      val d: BDV[Double] = eigenvalues.copy
      d(d <:< Eps) := Eps
      // Q diag(d) Q^T, using the same broadcasting idiom as the loop above.
      X = (eigenvectors(*, ::) *:* d) * eigenvectors.t
      // Rescale so the diagonal returns to its original (clipped) values:
      // X_ij <- D_i X_ij D_j with D = sqrt(max(o.diag, Eps) / diag(X)).
      val D: BDV[Double] = sqrt(oDiag.map(math.max(_, Eps)) /:/ diag(X))
      X = diag(D) * X * diag(D)
    }
    diag(X) := 1.0
    return X
  }


  def tabulateSymmetric(n: Int)(f: (Int, Int) => Double): BDM[Double] = {
    val a = new Array[Double](n*n)
    for (i <- 0 until n) {
      for (j <- i until n) {
        val v = f(i, j)
        a(i*n+j) = v
        a(j*n+i) = v
      }
    }
    new BDM(n, n, a)
  }


}
