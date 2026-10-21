/*
This file contains main and supportive functions for computing Z-scores from a matrix of genotypes, covariates,
and a binary response
Reference: Zhang, Hong, and Zheyang Wu. "The generalized Fisher's combination and accurate p‐value
           calculation under dependence." Biometrics 79.2 (2023): 1159-1172.
           Liu, Ming. "Integrative Analysis of Large Genomic Data." WPI (2025).
Creators: Kylie Hoar and Peter Howell
Last update (latest update first):
  Claude Code (Opus 5.5), prompted by VPanwar, 2026-10-21: getZ_marg_score_binary_SPA now returns its
    two matrices under "M_Z" and "M_s" instead of "M" and "GHG". The four GLOW_* wrappers read
    "M_Z" / "M_s", so with the old names the SPA branch of GLOW.scala threw "key not found: M_Z" on
    its first group and had never run (C2 smoke test, 2026-10-07). A rename only -- no value
    changes: GLOWr itself returns list(M_Z = M, M_s = GHG), and the group-C fixtures assert these
    matrices against those oracle keys at 1e-10. Option A of GLOWh's
    0project/ai-reports/2026-10-21-spa-key-mismatch-fix-proposal.md.
    Mirrors: GLOWr's getZ_marg_score_binary_SPA return list.
  Claude Code (Opus 5), prompted by VPanwar, 2026-09-15: Ported Saddle_Prob_fast, on ZWu's decision
    that GLOWr's output is the target -- GLOWr calls SPAtest with method = "fastSPA", which keeps
    the exact CGF only for subjects with a non-zero RAW genotype and replaces the whole excluded
    block with N(NAmu, NAsigma). Implemented by PARAMETERISING the existing Korg, K1_adj, K2,
    getRootK1, getSaddleProb and SaddleProb rather than adding six _fast duplicates: SPAtest's two
    variants are the same functions under naMu = naSigma = 0 with a retained set of everything, and
    that reduction is asserted as a bit-for-bit identity in SPAGroupCSpec. SaddleProb takes one new
    argument, naSet, from which NAmu and NAsigma are derived as "the whole statistic minus the part
    kept exactly", as Saddle_Prob_fast does. Measured on the group-C fixtures: 12 fast-path rows
    improved, 0 regressed, worst per case 3.1e-04 -> 4.5e-06 (balanced), 1.2e-02 -> 6.0e-05
    (extreme), 1.3e-04 -> 9.2e-06 (rare); the plain-branch row is unchanged to the last digit, as
    predicted. What remains is the root-finder's absolute stopping rule, which is shared with
    SPAtest and cannot be closed from this side.
    Mirrors: SPAtest's Saddle_Prob_fast, Korg_fast, K1_adj_fast, K2_fast, getroot_K1_fast and
    Get_Saddle_Prob_fast.
  Claude Code (Opus 5), prompted by VPanwar, 2026-09-15: Step 1 of matching GLOWr's SPA output --
    added the per-variant allele flip and the fast/plain dispatch that SPAtest's TestSPA and
    TestSPAfast both begin with, and exposed the decision as "fast_path". Changes NO p-value today:
    the flip is provably inert for the plain path (residualizing is linear and the constant vector
    lies in X1's span, so 2 - G residualizes to -Gscale and q to -q, under which the saddlepoint
    reflects and the two tails swap), verified against SPAtest where p(G) and p(2-G) are
    bit-identical. Its real effect is the zero COUNT, which selects the branch -- measured 8.2% ->
    53.0% zeros on an AF = 0.72 variant. Saddle_Prob_fast is not ported yet, so both branches still
    run the plain path; the dispatch is pinned now so that porting it is one moving part against a
    green baseline. Measured consequence: of the three failing black-box rows, binary_balanced[8]
    takes the PLAIN branch (45.6% zeros) so its 1.1e-06 is stopping-rule only, while binary_rare[1]
    and binary_extreme[0] (98.8% and 99.2% zeros) take the FAST branch and carry the estimator
    mismatch as well. 29 of 33 binary fixture variants take the fast path.
    Mirrors: SPAtest's TestSPA and TestSPAfast.
  Claude Code (Opus 5), prompted by VPanwar, 2026-09-09: Applied the zero-variance guard at every
    site that divides by the GHG diagonal -- five M_Z constructions and four Z-score divisions --
    using package.scala's cov2corZeroSafe and zScoresZeroSafe. getZ_marg_score_binary_SPA gets its
    own guard rather than that one, because R guards it by a DIFFERENT rule: an excluded variant is
    one whose adjusted variance is non-finite or below 1e-08 times the MEDIAN positive variance (a
    relative threshold, not machine epsilon), and R reports those as NA rather than neutralising
    them. Inert on every existing fixture, which contain no degenerate variant; verified that the
    SPA outputs are bit-identical to the pre-guard run. Mirrors: GLOWr's getZ_marg_score and
    getZ_marg_score_binary_SPA.
  Claude Code (Opus 5), prompted by VPanwar, 2026-09-09: Fixed defect C2, found by the group-C SPA
    suite (packages/glow-core in the GLOWh repo). getRootK1's default convergence tolerance was
    pow(Double.MinPositiveValue, 0.25); Double.MinPositiveValue is the smallest SUBNORMAL double
    (4.9e-324), not machine epsilon, so the tolerance was ~1.5e-81 and the stopping test
    abs(tnew - t) < tol could never fire -- the Newton loop ran to maxIter and reported
    non-convergence on inputs that converge in a few iterations. Now pow(ulp(1.0), 0.25) ~ 1.22e-04,
    mirroring SPAtest's getroot_K1 default of .Machine$double.eps^0.25.
  KHoar 2025-04-23: Re-formatted docstrings
  KHoar 2025-04-21: Added comments to SPA-related functions
*/

package is.hail.methods.gfisher

import breeze.linalg.{DenseMatrix => BDM, DenseVector => BDV, _}
import breeze.numerics._
import scala.collection.mutable.ArrayBuffer
import net.sourceforge.jdistlib.{Normal, ChiSquare}
import is.hail.methods.gfisher.OptimalWeights.{getGHG_Binary, getGHG_Binary2, getGHG_Continuous, nullDesign}
import scala.util.control.Breaks._

object FuncCalcuZScores {

  /*
  Helper functions
  */

  /**
    * Calculate the Z score by the marginal t statistics for Y vs. each column of G and X
    *
    * @param g  Extracted column (BDV) of genotype matrix
    * @param X  Input feature matrix (rows: observations, cols: features) with first column of ones for intercept
    * @param Y  Target vector (binary labels: 0 or 1).
    *
    * @return   Continuous (LinReg) marginal t statistics
    */
  def contZScore (g: BDV[Double], X: BDM[Double], Y: BDV[Double]): Double = {
    // Combine column of g with X
    val XwithG = BDM.horzcat(X, g.toDenseMatrix.t)
    val (beta, se, _) = stdErrLinearRegression(XwithG, Y)
    // Z-score for g
    val tStatistic = beta(-1) / se(-1)
    tStatistic
  }

  /*
  Saddlepoint Approximation (SPA) Test Functions
  (an adaptation of the 'SPAtest' R package: https://cran.r-project.org/web/packages/SPAtest/SPAtest.pdf)
  (splineH0 and splineH are an adaptation of some R functions in the splinefun package:
  https://github.com/SurajGupta/r-source/blob/master/src/library/stats/R/splinefun.R)
  */

  /**
    * Perform log-space addition of two log-probabilities.
    * Computes log(exp(p1) + exp(p2)) in a numerically stable way.
    * This is useful for avoiding underflow when dealing with very small probabilities in log-space.
    *
    * @param p1 First log-probability
    * @param p2 Second log-probability
    *
    * @return   Logarithm of the sum of probabilities
    */
  def addLogP(p1: Double, p2: Double): Double = {
    val absp1 = -abs(p1)
    val absp2 = -abs(p2)
    val maxP = math.max(absp1, absp2)
    val minP = math.min(absp1, absp2)
    maxP + math.log(1 + math.exp(minP - maxP))
  }

  /**
    * Compute the cumulant generating function (CGF) values for a set of scalar inputs.
    * For each t_i in vector t, calculates:
    *     K(t_i) = ∑ log[(1 - mu_j) + mu_j * exp(g_j * t_i)]
    * where mu and g are vectors of the same length representing expected values and coefficients.
    * Commonly used in saddlepoint approximations or generalized linear modeling.
    *
    * @param t   Vector of scalar evaluation points
    * @param mu  Vector of expected values (e.g., probabilities)
    * @param g   Vector of coefficients (e.g., effect sizes)
    *
    * @return    Vector of CGF values evaluated at each t_i
    */
  def Korg(
    t: BDV[Double], mu: BDV[Double], g: BDV[Double],
    muNB: Option[BDV[Double]] = None, gNB: Option[BDV[Double]] = None,
    naMu: Double = 0.0, naSigma: Double = 0.0
  ): BDV[Double] = {
    // fastSPA keeps the exact CGF only for the retained (non-zero genotype)
    // subjects and replaces the whole excluded block with a GAUSSIAN CGF matched
    // on its mean and variance -- hence naMu and naSigma. With the defaults
    // (retained = everything, naMu = naSigma = 0) the extra terms vanish
    // identically and this is the plain function, bit for bit: SPAtest's plain
    // and _fast variants are the same function under that substitution.
    val mu_ = muNB.getOrElse(mu)
    val g_ = gNB.getOrElse(g)
    val out = BDV.zeros[Double](t.size)

    for (i <- 0 until t.size) {
      val t1 = t(i)
      val temp = (1.0 - mu_) + (mu_ * exp(g_ *:* t1))
      out(i) = sum(log(temp)) + naMu * t1 + 0.5 * naSigma * t1 * t1
    }

    out
  }

  /**
    * Computes the adjusted K1 statistic used in saddlepoint approximation
    *
    * @param t  Vector of evaluation points (e.g., values of the saddlepoint variable)
    * @param mu Vector of means for each component
    * @param g  Vector of effect sizes or coefficients for each component
    * @param q  Scalar constant (typically the observed test statistic or weight)
    *
    * @return   Vector where each element corresponds to the adjusted K1 value for the corresponding t(i)
    */
  def K1_adj(
    t: BDV[Double], mu: BDV[Double], g: BDV[Double], q: Double,
    muNB: Option[BDV[Double]] = None, gNB: Option[BDV[Double]] = None,
    naMu: Double = 0.0, naSigma: Double = 0.0
  ): BDV[Double] = {
    // fastSPA keeps the exact CGF only for the retained (non-zero genotype)
    // subjects and replaces the whole excluded block with a GAUSSIAN CGF matched
    // on its mean and variance -- hence naMu and naSigma. With the defaults
    // (retained = everything, naMu = naSigma = 0) the extra terms vanish
    // identically and this is the plain function, bit for bit: SPAtest's plain
    // and _fast variants are the same function under that substitution.
    val mu_ = muNB.getOrElse(mu)
    val g_ = gNB.getOrElse(g)
    val out = BDV.zeros[Double](t.size)

    for (i <- 0 until t.size) {
      val t1 = t(i)
      val temp1 = (1.0 - mu_) * exp(-g_ *:* t1) + mu_
      val temp2 = mu_ * g_
      // The saddlepoint function: sum(mu * g / ((1 - mu) * exp(-g * t1) + mu)) - q
      // This is the equation to solve for the saddlepoint (i.e., K1(t) = q)
      out(i) = sum(temp2 / temp1) + naMu + naSigma * t1 - q
    }

    out
  }

  /**
    * Computes the second derivative (K2) of the cumulant generating function (CGF)
    * Used in saddlepoint approximation to assess curvature at a given t
    *
    * @param t  Vector of evaluation points (same length as out)
    * @param mu Vector of Bernoulli means (probabilities)
    * @param g  Vector of effect sizes or weights
    *
    * @return   Vector where each element is K2(t_i)
    */
  def K2(
    t: BDV[Double], mu: BDV[Double], g: BDV[Double],
    muNB: Option[BDV[Double]] = None, gNB: Option[BDV[Double]] = None,
    naMu: Double = 0.0, naSigma: Double = 0.0
  ): BDV[Double] = {
    // fastSPA keeps the exact CGF only for the retained (non-zero genotype)
    // subjects and replaces the whole excluded block with a GAUSSIAN CGF matched
    // on its mean and variance -- hence naMu and naSigma. With the defaults
    // (retained = everything, naMu = naSigma = 0) the extra terms vanish
    // identically and this is the plain function, bit for bit: SPAtest's plain
    // and _fast variants are the same function under that substitution.
    val mu_ = muNB.getOrElse(mu)
    val g_ = gNB.getOrElse(g)
    val out = BDV.zeros[Double](t.size)

    for (i <- 0 until t.size) {
      val t1 = t(i)

      // Compute the denominator of the expression: 
      // ((1 - mu) * exp(-g * t1) + mu)^2
      val temp1 = pow(((1.0 - mu_) * exp(-g_ *:* t1) + mu_), 2)

      // Compute the numerator: 
      // (1 - mu) * mu * g^2 * exp(-g * t1)
      val temp2 = (1.0 - mu_) * mu_ * pow(g_, 2) * exp(-g_ *:* t1)

      // The second derivative of the CGF:
      // K2(t1) = sum( numerator / denominator )
      out(i) = sum(temp2 / temp1) + naSigma
    }

    out
  }

  /**
    * Constructs a matrix to be used for spline interpolation in saddlepoint approximation
    *
    * @param mu    Vector of Bernoulli means (probabilities)
    * @param g     Vector of effect sizes or weights
    * @param nodes Evaluation points for t (spline knots)
    *
    * @return      A matrix with three columns: [t, K1_adj(t) - mean, K2(t)] with any rows 
    *              where t == 0 removed (to avoid singularity or undefined values)
    */
  def Get_Saddle_Spline(mu: BDV[Double], g: BDV[Double], nodes: BDV[Double]): BDM[Double] = {
    val m1 = mu.dot(g)

    // Compute the adjusted first derivative (score function) at each node, then subtract the mean
    // This gives K'(t) - E[X] = score function centered at the mean
    val y1 = K1_adj(nodes, mu, g, 0) - m1

    // Compute the second derivative (variance function) at each node
    val y2 = K2(nodes, mu, g)

    // Assemble the full matrix with columns: [t, centered K1', K2]
    val resultMatrix = BDM.horzcat(nodes.toDenseMatrix.t, y1.toDenseMatrix.t, y2.toDenseMatrix.t)

    // Identify all rows where t != 0 (to avoid log/exp singularities at 0)
    val nonZeroRows = (0 until resultMatrix.rows).filter(i => resultMatrix(i, 0) != 0.0)
    if (nonZeroRows.isEmpty) {
      BDM.zeros[Double](0, resultMatrix.cols)  // Return an empty matrix if all rows were removed
    } else {
      BDM.vertcat(nonZeroRows.map(i => resultMatrix(i, ::).t.toDenseMatrix):_*)
    }
  }

  /**
    * Constructs a Hermite cubic spline interpolator with derivative support (0th, 1st, 2nd)
    *
    * @param x0  Vector of x-coordinates (nodes/knots), must be sorted ascending
    * @param y0  Vector of function values at x0
    * @param m   Vector of derivatives (slopes) at x0, typically estimated from K1/K2
    *
    * @return    Function that takes an input x and a derivative order (0, 1, or 2) and returns the spline value
    */
  def splineH0(x0: BDV[Double], y0: BDV[Double], m: BDV[Double]): (Double, Int) => Double = {
    val n = x0.length

    // Compute step sizes h_i = x_{i+1} - x_i
    val h = x0(1 to -1) - x0(0 until n - 1) // step sizes

    // Helper function to find which interval x belongs to in x0
    def findInterval(x: Double): Int = {
      // Mimics findInterval(x, x0, all.inside = TRUE)
      if (x <= x0(0)) 0
      else if (x >= x0(n - 1)) n - 2
      else (0 until n - 1).find(i => x0(i) <= x && x < x0(i + 1)).getOrElse(n - 2)
    }

    // The spline function returned
    (x: Double, deriv: Int) => {
      val i = findInterval(x)   // interval index
      val hi = h(i)             // step size for this interval
      val t = (x - x0(i)) / hi  // normalized position in [0,1]

      // Local values from the interval
      val yL = y0(i)
      val yR = y0(i + 1)
      val mL = m(i)
      val mR = m(i + 1)

      // Handle extrapolation cases: left and right of the domain
      if (x < x0(0)) {
        deriv match {
          case 0 => y0(0) + m(0) * (x - x0(0))  // linear extrapolation using tangent
          case 1 => m(0)                        // slope
          case 2 => 0.0                         // curvature is 0 in linear extrapolation
        }
      } else if (x > x0(n - 1)) {
        deriv match {
          case 0 => y0(n - 1) + m(n - 1) * (x - x0(n - 1))
          case 1 => m(n - 1)
          case 2 => 0.0
        }
      } 
      // Interpolation case: inside domain
      else {
        deriv match {
          // 0th derivative: cubic Hermite interpolation formula
          case 0 =>
            yL * (1 - 3 * t * t + 2 * t * t * t) +
              mL * hi * (t - 2 * t * t + t * t * t) +
              yR * (3 * t * t - 2 * t * t * t) +
              mR * hi * (-t * t + t * t * t)

          // First derivative: derivative of Hermite polynomial
          case 1 =>
            val dyL = (-6 * t + 6 * t * t)
            val dyM1 = (1 - 4 * t + 3 * t * t)
            val dyR = (6 * t - 6 * t * t)
            val dyM2 = (-2 * t + 3 * t * t)
            (yL * dyL + mL * hi * dyM1 + yR * dyR + mR * hi * dyM2) / hi

          // Second derivative: second derivative of Hermite polynomial
          case 2 =>
            val d2yL = (-6 + 12 * t)
            val d2yM1 = (-4 + 6 * t)
            val d2yR = (6 - 12 * t)
            val d2yM2 = (-2 + 6 * t)
            (yL * d2yL + mL * hi * d2yM1 + yR * d2yR + mR * hi * d2yM2) / (hi * hi)

          case _ => throw new IllegalArgumentException("deriv must be 0, 1, or 2")
        }
      }
    }
  }

  /**
    * This function wraps `splineH0` by ensuring the input data (x0, y0, m) is sorted by x0
    *
    * @param x0  Vector of x-values (not necessarily sorted)
    * @param y0  Function values at each x0
    * @param m   Derivatives (slopes) at each x0
    *
    * @return    A spline function, (Double, Int) => Double, for evaluating spline or its derivatives
    */
  def splineH(x0: BDV[Double], y0: BDV[Double], m: BDV[Double]): (Double, Int) => Double = {
    // Find indices that sort x0 in ascending order
    val idx = argsort(x0)

    // Sort all inputs based on sorted x0 order
    val xSorted = x0(idx).toDenseVector
    val ySorted = y0(idx).toDenseVector
    val mSorted = m(idx).toDenseVector

    // Delegate to the main Hermite spline constructor (which expects sorted inputs)
    splineH0(xSorted, ySorted, mSorted)
  }

  /**
    * Adaptively refines a set of spline interpolation nodes to minimize the weighted residual error between the spline
    * approximation of a transformed cumulant function (K1_adj) and its true values over a dense evaluation grid
    * @param init Initial guess for node locations where the spline will interpolate the function
    * @param mu   Mean vector used in computing the cumulant functions (K1_adj and K2)
    * @param g    Gradient vector used with mu in the cumulant function computations
    *
    * @return     spline interpolation nodes
    */
  def getNodes(init: BDV[Double], mu: BDV[Double], g: BDV[Double]): (BDV[Double], Double) = {
    val nodesInit = BDV((init.toArray :+ 0.0).distinct.sorted)
    var nodes = nodesInit // Current node set
    var rep = 0           // Iteration counter
    
    // Create exponentially spaced positive and negative test points
    val positiveT = (-2.0 to 10.0 by 0.5).map(x => math.pow(2, x))
    val negativeT = (-2.0 to 10.0 by 0.5).map(x => -math.pow(2, x))
    val tArray = (positiveT ++ negativeT :+ 0.0).distinct.sorted.toArray
    val t = BDV(tArray)   // Final test grid for evaluating function

    // Compute K1_adj at test points (true values)
    val yt = K1_adj(t, mu, g, 0)

    // Define weights: emphasize center values, downweight farther values
    val w = BDV(t.toArray.map(ti => if (ti * ti > 1) 1.0 / math.pow(math.abs(ti), 1.0 / 3.0) else 1.0))
    val jump = 0.5                            // Base jump scale for node movement
    var totres = Double.PositiveInfinity      // Previous best residual loss
    var finalNodes = BDV[Double]()            // Best node configuration
    var finalLoss = Double.PositiveInfinity  // Associated loss for the node

    breakable {
      while(true) {
        // Step 1: Generate Spline Interpolant and predictions
        val y1 = K1_adj(nodes, mu, g, 0)
        val y2 = K2(nodes, mu, g)
        val sfun = splineH(nodes, y1, y2)               // Generate spline interpolant
        val pred = BDV(t.toArray.map(x => sfun(x, 0)))  // Predict on test grid
        val res = w * abs(pred - yt)                    // Weighted residuals
        val loss = sum(res)                             // Total residual loss

        val newNodesBuffer = ArrayBuffer[Double]()      // Container for updated nodes

        // Step 2: Move each node adaptively (excluding 0.0 which is fixed)
        for (i <- 0 until nodes.length if nodes(i) != 0.0) {
          val resArray = res.toArray

          // Identify test points to the left and right of the current node
          val leftIdx = (0 until t.length).filter(j => t(j) < nodes(i) && (i == 0 || t(j) > nodes(i - 1)))
          val rightIdx = (0 until t.length).filter(j => t(j) > nodes(i) && (i == nodes.length - 1 || t(j) < nodes(i + 1)))
          
          // Get max residuals and corresponding locations for left and right
          val (r1: Double, _) = if (leftIdx.nonEmpty) {
            val maxIdx = leftIdx.maxBy(j => resArray(j))
            (resArray(maxIdx), t(maxIdx))
          } else (0.0, 0.0)
          val (r2: Double, _) = if (rightIdx.nonEmpty) {
            val maxIdx = rightIdx.maxBy(j => resArray(j))
            (resArray(maxIdx), t(maxIdx))
          } else (0.0, 0.0)

          // Adjust node location based on relative residuals
          if (r1 == r2) {
            newNodesBuffer += nodes(i)
          } else if (r1 > r2) {
            val jump1 = jump * (1 - math.max(0.1, r2 / r1)) * math.abs(nodes(i))
            newNodesBuffer += nodes(i) - jump1  // Shift left
          } else {
            val jump1 = jump * (1 - math.max(0.1, r1 / r2)) * math.abs(nodes(i))
            newNodesBuffer += nodes(i) + jump1  // Shift right
          }
        }

        // Step 3: Check stopping criteria
        if (rep > 100 || loss > totres) {
          finalNodes = nodes
          finalLoss = totres
          break
        }

        rep += 1
        totres = loss

        // Add 0.0 back and update the node set
        newNodesBuffer += 0.0
        nodes = BDV(newNodesBuffer.distinct.sorted.toArray)
      }
    }

    (finalNodes, finalLoss)
  }

  /**
    * Finds the root of the cumulant-generating function (CGF) adjustment K1_adj(t, mu, g, q) using a safeguarded Newton-Raphson method
    * with convergence and stability checks.
    * 
    * @param init     Initial guess for the root-finding procedure (starting value of t)
    * @param mu       Vector of mean parameters or location parameters used in the cumulant-generating function
    * @param g        Vector of weights or transformation coefficients applied in the cumulant-generating function
    * @param q        Target value (typically a linear constraint or statistic) that the saddlepoint approximation is solving for
    * @param tol      (optional, default = pow(ulp(1.0), 0.25) ~ 1.22e-04, mirroring SPAtest's
                      getroot_K1, whose default is .Machine$double.eps^0.25), Tolerance for
                      convergence; the root is considered found if the step size is smaller than
                      this. NOTE this is an ABSOLUTE criterion on the Newton step.
    * @param maxIter  (optional, default = 1000), Maximum number of iterations to perform before declaring non-convergence
    *
    * @return         (Double, Int, Boolean) Double: estimated root t (may be pos infinity if q is out of bounds or Null if
    *                                                computation encounters instability)
    *                                        Int: number of iterations performed before convergence or termination
    *                                        Boolean: flag indicating whether the algorithm successfully converged or not
    */
  def getRootK1(
      init: Double,
      mu: BDV[Double],
      g: BDV[Double],
      q: Double,
      // R's .Machine$double.eps^0.25. math.ulp(1.0) is machine epsilon (2.22e-16);
      // Double.MinPositiveValue, used here previously, is the smallest SUBNORMAL
      // (4.9e-324), giving a tolerance of ~1.5e-81 that a double-precision Newton
      // step can never reach -- so the loop ran to maxIter and reported failure to
      // converge on inputs that converge in a handful of iterations.
      tol: Double = pow(math.ulp(1.0), 0.25),
      maxIter: Int = 1000,
      // fastSPA's excluded block; the defaults reduce this to the plain Newton.
      // Note the support check below uses the FULL g, as SPAtest's
      // getroot_K1_fast does -- the bracket is a property of the whole statistic,
      // not of the retained subset.
      muNB: Option[BDV[Double]] = None,
      gNB: Option[BDV[Double]] = None,
      naMu: Double = 0.0,
      naSigma: Double = 0.0
  ): (Double, Int, Boolean) = {

    val gPos = sum(g(g >:> 0.0))  // Sum of positive values in g
    val gNeg = sum(g(g <:< 0.0))  // Sum of negative values in g

    // If q is outside the valid range of the cumulant domain, return immediately with an infinite root (invalid input)
    if (q >= gPos || q <= gNeg) {
      (Double.PositiveInfinity, 0, true)
    } else {
      var t = init                                    // Starting guess for root-finding
      var K1Eval = K1_adj(BDV(t), mu, g, q, muNB, gNB, naMu, naSigma).apply(0)  // Evaluate K1 at t
      var prevJump = Double.PositiveInfinity          // Used to limit oscillations
      var iter = 1

      // Begin Newton-Raphson iterations, capped by maxIter
      while (iter <= maxIter) {
        val K2Eval = K2(BDV(t), mu, g, muNB, gNB, naMu, naSigma).apply(0)  // Derivative of K1 (2nd cumulant)
        val tNew = t - K1Eval / K2Eval                // Newton-Raphson update

        // Return if update diverged
        if (tNew.isNaN) {
          return (Double.NaN, iter, false)
        }

        // If the update is smaller than the tolerance, convergence achieved
        if (math.abs(tNew - t) < tol) {
          return (tNew, iter, true)
        }

        // Evaluate K1 at new point
        val newK1 = K1_adj(BDV(tNew), mu, g, q, muNB, gNB, naMu, naSigma).apply(0)

        // Check for sign change, indicating crossing the root
        if (math.signum(K1Eval) != math.signum(newK1)) {
          if (math.abs(tNew - t) > prevJump - tol) {
            t = t + math.signum(newK1 - K1Eval) * prevJump / 2  // Backtrack slightly
            prevJump /= 2
            K1Eval = K1_adj(BDV(t), mu, g, q, muNB, gNB, naMu, naSigma).apply(0)  // Re-evaluate K1
          } else {
            prevJump = math.abs(tNew - t)                       // Accept jump
            t = tNew
            K1Eval = newK1
          }
        } else {
          t = tNew
          K1Eval = newK1
        }

        iter += 1
      }

      (t, iter, false) // did not converge within maxIter
    }
  }

  /**
    * Computes the saddlepoint approximation of a p-value (or log p-value) for a given cumulant-generating function
    * 
    * @param zeta  Candidate saddlepoint value
    * @param mu    Vector representing the means
    * @param g     Vector of weights or coefficients associated with mu (often reflecting importance or directionality of 
    *              contributions)
    * @param q     Scalar summarizing the observed test statistic
    * @param logP  If true, return the logarithm of the p-value; otherwise, return the raw p-value
    *
    * @return      Saddelpoint approximation of a p-value
    */
  def getSaddleProb(
    zeta: Double, mu: BDV[Double], g: BDV[Double], q: Double, logP: Boolean = false,
    muNB: Option[BDV[Double]] = None, gNB: Option[BDV[Double]] = None,
    naMu: Double = 0.0, naSigma: Double = 0.0
  ): Double = {
    // The r* formula is identical in SPAtest's plain and _fast variants; only the
    // K1/K2 they call differ, which is exactly what these arguments select.
    // Compute K1 (cumulant generating function) and K2 (its second derivative) at zeta
    val k1 = Korg(BDV(zeta), mu, g, muNB, gNB, naMu, naSigma)(0)
    val k2 = K2(BDV(zeta), mu, g, muNB, gNB, naMu, naSigma)(0)

    if (java.lang.Double.isFinite(k1) && java.lang.Double.isFinite(k2)) {
      val temp1 = zeta * q - k1               // exponent part of the saddlepoint approximation
      val w = signum(zeta) * sqrt(2 * temp1)  // transformed test statistic
      val v = zeta * sqrt(k2)                 // scaling factor incorporating k2.

      val zTest = w + 1 / w * log(v / w)      // final saddlepoint-based test statistic, adjusted with a correction term

      // Calculate p-value using Normal cumulative distribution
      if (zTest > 0) {
        Normal.cumulative(zTest, 0, 1, false, logP)
      } else {
        -Normal.cumulative(zTest, 0, 1, true, logP)
      }
    } else {
      if (logP) -Double.PositiveInfinity else 0.0
    }
  }

  /**
    * Computes a saddlepoint-adjusted p-value (or log p-value) for a test statistic using cumulant-based approximation, with
    * optional spline output and adaptive re-evaluation if approximation quality is poor
    *
    * @param q           Observed test statistic
    * @param mu          Vector of expected values (means)
    * @param g           Weight vector used to construct the test statistic from the mu values
    * @param Cutoff      Either a numeric cutoff for determining when to use the saddlepoint approximation, or the string "BE" 
    *                    to use the Berry-Esseen bound-based threshold
    * @param alpha       Significance level used if Cutoff is set to "BE" (e.g. 0.05); ignored otherwise
    * @param output      Specifies output type: "P" returns just p-values and convergence info, while "metaspline" returns
    *                    additional spline interpolation data
    * @param nodesFixed  Optional fixed spline nodes; used only if output == "metaspline"
    * @param nodesInit   Initial spline node guesses; required if output == "metaspline" and nodesFixed is not provided
    * @param logP        If true, returns log-transformed p-values
    * 
    * @return            A map containing:
    *                       - `"p.value"`: The computed p-value.
    *                       - `"p.value.NA"`: The unadjusted p-value (before any saddlepoint adjustments).
    *                       - `"Is.converge"`: A flag indicating whether the algorithm converged.
    *                       - `"Score"`: The score statistic (`q - m1`).
    *                       - `"splfun"`: The spline function (only returned if `output == "metaspline"`).
    *                       - `"var"`: The variance of the logistic regression model's predictions.
    *
    *                    If the p-value is small and requires further adjustment, the function will recursively call itself.
    */
  def SaddleProb(q: Double, mu: BDV[Double], g: BDV[Double], Cutoff: Any = 2.0, alpha: Double, output: String = "P", nodesFixed: Option[BDV[Double]], nodesInit: Option[BDV[Double]], logP: Boolean = false, naSet: Option[IndexedSeq[Int]] = None): Map[String, Any] = {
    val m1 = mu.dot(g)  // sum(mu * g)  // Expected value of the test statistic
    val var1 = (mu.toArray zip g.toArray).map { case (m, gi) => m * (1 - m) * gi * gi }.sum  // Variance

    // fastSPA. `naSet` holds the indices whose RAW genotype is zero. Given it,
    // the exact CGF is kept only for the complement and the excluded block is
    // replaced by a Gaussian matched on its mean and variance:
    //
    //   NAmu    = m1   - sum(gNB * muNB)
    //   NAsigma = var1 - sum(muNB * (1 - muNB) * gNB^2)
    //
    // i.e. "the whole statistic minus the part being kept exactly", which is how
    // SPAtest's Saddle_Prob_fast computes them. With naSet = None the retained
    // set is everything, both quantities are zero, and every formula below
    // reduces to the plain saddlepoint -- the case this function has always
    // computed. Mirrors: SPAtest's Saddle_Prob and Saddle_Prob_fast.
    val keep: Option[IndexedSeq[Int]] = naSet.map { na =>
      val isNA = Array.fill(g.length)(false)
      na.foreach(i => isNA(i) = true)
      (0 until g.length).filterNot(isNA)
    }
    val muNB: Option[BDV[Double]] = keep.map(k => BDV(k.map(mu(_)).toArray))
    val gNB: Option[BDV[Double]] = keep.map(k => BDV(k.map(g(_)).toArray))
    val naMu: Double = keep match {
      case None => 0.0
      case Some(k) => m1 - k.map(i => g(i) * mu(i)).sum
    }
    val naSigma: Double = keep match {
      case None => 0.0
      case Some(k) => var1 - k.map(i => mu(i) * (1.0 - mu(i)) * g(i) * g(i)).sum
    }
    
    // Initialize optional p-values and spline node vector
    var p1: Option[Double] = None
    var p2: Option[Double] = None
    var nodes: BDV[Double] = BDV()  // Empty init (won't be used unless output == "metaspline")

    // If spline output is requested, retrieve initNodes, uses NodesFixed, or compute internal spline nodes
    if (output == "metaspline") {
      val initNodes = nodesInit.getOrElse(
        throw new IllegalArgumentException("nodesInit must be defined for metaspline output.")
      )

      nodes = nodesFixed match {
        case Some(fixedNodes) => BDV(fixedNodes.toArray.sorted :+ 0.0)
        case None => getNodes(initNodes.toDenseVector, mu, g)._1
      }
    }

    // Compute test statistic deviation (score) and its "mirror" qinv for symmetric approximation
    val score = q - m1
    val qinv = -Math.signum(q - m1) * Math.abs(q - m1) + m1

    // Compute the naive (non-adjusted) chi-square p-value
    val pvalNoadj = ChiSquare.cumulative((Math.pow(q - m1, 2) / var1), 1, false, logP)

    var isConverge = true
    var pval = pvalNoadj

    // Determine whether to use saddlepoint adjustment
    val cutoffVal: Double = Cutoff match {
      // If "BE", use Berry-Esseen inequality to derive threshold
      case s: String if s == "BE" =>
        val rho = mu.toScalaVector.zip(g.toScalaVector).map {
          case (m, gi) => Math.pow(Math.abs(gi), 3) * m * (1 - m) * (Math.pow(m, 2) + Math.pow(1 - m, 2))
        }.sum
        val var1 = (mu.toArray zip g.toArray).map { case (m, gi) => m * (1 - m) * gi * gi }.sum
        val B = 0.56 * rho * Math.pow(var1, -1.5)
        val p = B + alpha / 2
        if (p >= 0.496) 0.01 else Normal.quantile(p, 0, 1, false, logP)

      // Else use user-supplied numeric cutoff, min 0.1
      case d: Double => if (d < 0.1) 0.1 else d

      case _ => throw new IllegalArgumentException("Invalid type for Cutoff. Expected Double or \"BE\".")
    }

    // P-value calculation based on cutoff and convergence
    // If the deviation is small, return naive p-value
    if (Math.abs(q - m1) / Math.sqrt(var1) < cutoffVal) {
      pval = pvalNoadj
    } 
    // Otherwise, compute saddlepoint-adjusted p-values using roots of K1
    else {
      val outUni1 = getRootK1(0, mu, g, q, pow(math.ulp(1.0), 0.25), 1000, muNB, gNB, naMu, naSigma)
      val outUni2 = getRootK1(0, mu, g, qinv, pow(math.ulp(1.0), 0.25), 1000, muNB, gNB, naMu, naSigma)

      if (outUni1._3 && outUni2._3) {
        p1 = try {
          Some(getSaddleProb(outUni1._1, mu, g, q, logP, muNB, gNB, naMu, naSigma))
        } catch {
          case e: Exception =>
            if (logP) Some(pvalNoadj - Math.log(2))
            else Some(pvalNoadj / 2)
        }
        p2 = try {
          Some(getSaddleProb(outUni2._1, mu, g, qinv, logP, muNB, gNB, naMu, naSigma))
        } catch {
          case e: Exception =>
            if (logP) Some(pvalNoadj - Math.log(2))
            else Some(pvalNoadj / 2)
        }

        if (logP) {
          pval = p1.get + p2.get
        } else {
          pval = Math.abs(p1.get) + Math.abs(p2.get)
        }
        isConverge = true
      } else {
        println("Error_Converge")
        pval = pvalNoadj
        isConverge = false
      }
    }

    // Recursively call if the p-value is small and needs further adjustment
    if (pval != 0 && pvalNoadj / pval > Math.pow(10, 3)) {
      return SaddleProb(q, mu, g, cutoffVal * 2, alpha, output, nodesFixed, nodesInit, logP, naSet)
    } else {
      if (output == "metaspline") {
        Map("p.value" -> pval, "p.value.NA" -> pvalNoadj, "Is.converge" -> isConverge, "Score" -> score, "splfun" -> Get_Saddle_Spline(mu, g, nodes), "var" -> var1)
      } else {
        Map("p.value" -> pval, "p.value.NA" -> pvalNoadj, "Is.converge" -> isConverge, "Score" -> score)
      }
    }
  }

  /*
  Main functions
  */

  /**
   * Get the standardized marginal score statistics
   * 
   * Adapted from the GLOW R package ("GLOW_R_package/GLOW/R/getZ_marg_score.R")
   * 
   * @param G         A matrix of genotype (# of individuals x # of SNPs)
   * @param X         A matrix of covariates, default is 1
   * @param Y         A single column of response variable; it has to be 0/1 for binary trait
   * @param binary    Indicator of "true" (binary, logistic regression) or "false" (continuous, linear regression).
   * @param use_lm_t  Whether to use the lm() function to get the t statistics as the Z-scores for continuous trait
   * 
   * @return Zscores  A vector of Z scores of the marginal score statistics for each SNP.
   * @return scores   A vector of the marginal score statistics for each SNP
   * @return M_Z      The correlation matrix of Z scores when effects are fixed (including H0) given X and G fixed.
   * @return M_s      The covariance matrix of the score statistics when effects are fixed (including H0) given X and G fixed.
   * @return s0       The estimated dispersion parameter. Binary trait: s0=1. Continuous trait: residual SD under the null model.
   */
  def getZMargScore(
    G: BDM[Double],
    X: BDM[Double],
    Y: BDV[Double],
    binary: Boolean = false,
    use_lm_t: Boolean = false
  ): Map[String, Any] = {

    if (binary) {
      val (ghg, _, resids) = getGHG_Binary2(G, X, Y)

      val score = G.t * resids
      val Zscore = zScoresZeroSafe(score, ghg)

      return Map(
        "Zscores" -> Zscore,
        "scores" -> score,
        "M_Z" -> cov2corZeroSafe(ghg),
        "M_s" -> ghg,
        // s0 = 1 for binary trait
        "s0" -> 1.0
      )
    }

    else {
      val (ghg, s0, resids) = getGHG_Continuous(G, X, Y)
      // Compute residuals
      val score = G.t * resids / s0

      val Zscore: BDV[Double] = if (use_lm_t) {
        // nullDesign, not an unconditional horzcat: X may already carry an
        // intercept column (R's fit_null_model returns it that way), and adding
        // a second one makes the design matrix singular.
        val XWithIntercept = nullDesign(X)
        BDV((0 until G.cols).map { j => contZScore(G(::, j), XWithIntercept, Y) }.toArray)
      } else {
        // Z scores based on the score statistics
        zScoresZeroSafe(score, ghg)
      }

      return Map(
        "Zscores" -> Zscore,
        "scores" -> score,
        "M_Z" -> cov2corZeroSafe(ghg),
        "M_s" -> ghg,
        "s0" -> s0
      )
    }
  }

  /**
    * Performs marginal score testing using Saddlepoint Approximation (SPA) for binary outcomes.
    * This is often used in genome-wide association studies (GWAS) with a logistic regression null model.
    *
    * Adapted from the GLOW R package ("GLOW_R_package/GLOW/R/getZ_marg_score_binary_SPA.R")
    *
    * @param G  Genotype matrix of shape (n_samples, n_variants)
    * @param X  Covariate matrix of shape (n_samples, n_covariates)
    * @param Y  Binary outcome vector (0 or 1), of length n_samples
    * 
    * @return   A Map containing:
    *             - "Zscores": BDV[Double] of SPA Z-scores for each variant
    *             - "M_Z": BDM[Double], correlation matrix of the Z-scores, cov2cor of "M_s"
    *                      (NaN rows/columns for zero-adjusted-variance variants, as R gives NA)
    *             - "M_s": BDM[Double], null covariance of the scores, G'[V - VX(X'VX)^-1 X'V]G
    *                      with V = diag(mu(1 - mu)) -- the covariate-adjusted GHG
    *             - "s0": Double, dispersion parameter (currently fixed at 1.0)
    */
  def getZ_marg_score_binary_SPA (
    G: BDM[Double],
    X: BDM[Double],
    Y: BDV[Double]
  ): Map[String, Any] = {
    // Logistic regression model. One design matrix, built once and used for the
    // fit, for the projection XXVX_inv and for Hhalf below -- R's fit_null_model
    // does the same. X may already carry an intercept, in which case appending
    // another makes X1 singular.
    val X1 = nullDesign(X)

    // Compute the predicted probabilities under the null model via logistic regression
    val Y0 = log_reg_predict(X1, Y, addIntercept = false)

    // Computes the variance for each sample (Bernoulli variance formula)
    val Y0_vec = Y0 *:* (1.0 - Y0)
    
    // Projects G to the orthogonal space of X, resulting in Gscale
    val XV = (X1(::, *) * Y0_vec).t
    val XVX_inv = inv(X1.t * (X1(::, *) * Y0_vec))
    val XXVX_inv = X1 * XVX_inv
    val XVG = XV * G
    val Gscale = G - XXVX_inv * XVG

    // Compute score statistic for each variant
    val score = Gscale.t * Y

    // Per-variant allele flip and sparse/dense dispatch, mirroring SPAtest's
    // TestSPA and TestSPAfast, which BOTH begin:
    //
    //   n.g <- sum(G); if (n.g / (2 * length(G)) > 0.5) G <- 2 - G
    //   NAset <- which(G == 0)
    //   if (length(NAset) / length(G) < 0.5)  Saddle_Prob(...)   # dense
    //   else                                  Saddle_Prob_fast(...)  # sparse
    //
    // Two things to know about the flip:
    //
    //  - It does NOT change the plain saddlepoint p-value. Residualizing is
    //    linear and the constant vector lies in X1's span (X1 has an intercept),
    //    so the residual of 2 - G is exactly -Gscale, and q becomes -q. Under
    //    (g, q) -> (-g, -q) the saddlepoint reflects to -zeta, w and v both
    //    negate, and the two tails swap -- leaving |p1| + |p2| unchanged.
    //    Verified against SPAtest: p(G) and p(2-G) are bit-identical.
    //  - Its real purpose is the ZERO COUNT, which decides the branch. Flipping
    //    a variant whose counted allele is the major one turns a mostly-non-zero
    //    vector into a mostly-zero one -- measured 8.2% zeros -> 53.0% on a
    //    typical AF = 0.72 variant, which crosses the 50% threshold.
    //
    // So the flip is inert today and becomes load-bearing when Saddle_Prob_fast
    // is ported. It is implemented now because the dispatch cannot be correct
    // without it.
    val nSamp = G.rows.toDouble
    val useFastPath = BDV.zeros[Double](score.size)   // 1.0 where SPAtest would use fastSPA

    val pval_spa = BDV((0 until score.size).map { x =>
      val gRaw = G(::, x)
      val flip = sum(gRaw) / (2.0 * nSamp) > 0.5
      // NAset is computed on the RAW genotype, after the flip and before
      // residualizing -- not on Gscale, whose entries are almost never zero.
      val gForMask = if (flip) 2.0 - gRaw else gRaw
      val nZero = (0 until gForMask.length).count(i => gForMask(i) == 0.0)
      useFastPath(x) = if (nZero / nSamp >= 0.5) 1.0 else 0.0

      // The flipped residualized column is -Gscale, and its q is -score; see above.
      val g1 = if (flip) -Gscale(::, x) else Gscale(::, x)
      val q  = if (flip) -score(x) else score(x)

      // The dispatch SPAtest makes: sparse variants take the fast path, which
      // keeps the exact CGF only for the carriers and approximates the rest.
      val naSet =
        if (useFastPath(x) == 1.0)
          Some((0 until gForMask.length).filter(i => gForMask(i) == 0.0))
        else None

      SaddleProb(q, Y0, g1, "BE", 5 * math.pow(10, -8), "P", None, None, false, naSet)("p.value")
        .asInstanceOf[Double]
    }.toArray)

    // Transform SPA p-values into Z-scores
    val Zscores_spa = BDV(pval_spa.map(p => Normal.quantile(p / 2.0, 0.0, 1.0, false, false)).toArray) *:* signum(score)

    // Perform a projection of genotype matrix for correlation estimation (GHG is the residualized, cross-product matrix)
    val sqrtY0 = sqrt(Y0_vec)
    // X1, not X: HHalf must project out the same space the null model was fitted
    // on, which is the defect fixed in getH_Binary (C1) appearing again here.
    val Xtilde = ((X1(::, *) * sqrtY0).t).t
    val Hhalf = Xtilde * cholesky(inv(Xtilde.t * Xtilde))
    val Gtilde = ((G(::, *) * sqrtY0).t).t
    val GHhalf = Gtilde.t * Hhalf
    val GHG = (Gtilde.t * Gtilde) - (GHhalf * GHhalf.t)

    // Zero-variance guard, mirroring GLOWr::getZ_marg_score_binary_SPA -- which
    // uses a DIFFERENT rule from getZ_marg_score's, so this is not a copy of
    // cov2corZeroSafe. R excludes a variant when its adjusted variance is not
    // finite or falls below 1e-08 times the MEDIAN positive variance -- a
    // relative threshold, not machine epsilon -- and reports the excluded
    // variants as NA rather than neutralising them, because a saddlepoint
    // p-value for a variant with no variance is meaningless rather than merely
    // uninformative.
    val varAdj = diag(GHG)
    val posVar = (0 until varAdj.length).map(varAdj(_)).filter(v => !v.isNaN && !v.isInfinite && v > 0.0)
    val varScale = if (posVar.nonEmpty) {
      val srt = posVar.sorted
      val n = srt.length
      if (n % 2 == 1) srt(n / 2) else (srt(n / 2 - 1) + srt(n / 2)) / 2.0
    } else 1.0
    val okVar = (0 until varAdj.length).map { i =>
      val v = varAdj(i)
      !v.isNaN && !v.isInfinite && v > 1e-08 * varScale
    }
    val M = if (okVar.forall(identity)) cov2cor(GHG) else {
      Console.err.println(s"WARNING: ${okVar.count(!_)} variant(s) with zero adjusted variance " +
        "in getZ_marg_score_binary_SPA; their correlations are reported as NaN, as R reports NA.")
      val out = BDM.fill[Double](GHG.rows, GHG.cols)(Double.NaN)
      val keep = (0 until varAdj.length).filter(okVar)
      val sub = BDM.tabulate(keep.length, keep.length)((a, b) => GHG(keep(a), keep(b)))
      val subCor = cov2cor(sub)
      for (a <- keep.indices; b <- keep.indices) out(keep(a), keep(b)) = subCor(a, b)
      out
    }

    // Dispersion parameter
    val s0 = 1.0

    Map(
      "Zscores" -> Zscores_spa,
      // GLOWr's names, which the four GLOW_* wrappers read: R returns
      // list(M_Z = M, M_s = GHG). M_Z is the correlation of the Z-scores
      // (cov2cor of GHG); M_s is the null covariance of the scores, GHG itself.
      "M_Z" -> M,
      "M_s" -> GHG,
      "s0" -> s0,
      // Which variants SPAtest would route through fastSPA (1.0) rather than the
      // plain saddlepoint (0.0). Additive: no existing caller reads it. Exposed
      // so the dispatch is assertable now, before Saddle_Prob_fast exists --
      // otherwise it could only be tested indirectly, once the two branches
      // produce different numbers.
      "fast_path" -> useFastPath
    )
  }

  /**
   * Get the standardized marginal score statistics
   * @param G A matrix of genotype (# of individuals x # of SNPs)
   * @return Zscores A vector of Z scores of the marginal score statistics for each SNP.
   * @return scores A vector of the marginal score statistics for each SNP
   * @return M_Z the correlation matrix of Z scores when effects are fixed (including H0) given X and G fixed.
   * @return M_s the covariance matrix of the score statistics when effects are fixed (including H0) given X and G fixed.
   * @return s0 the estimated dispersion parameter. Binary trait: s0=1. Continuous trait: residual SD under the null model.
   */
  def getZMargScoreBinary(
    G: BDM[Double],
    HHalf: BDM[Double],
    y0: BDV[Double],
    resids: BDV[Double],
  ): Map[String, Any] = {

    val GHG = getGHG_Binary(G, HHalf, y0)

    val score = G.t * resids
    val Zscore = zScoresZeroSafe(score, GHG)

    return Map(
      "Zscores" -> Zscore,
      "scores" -> score,
      "M_Z" -> cov2corZeroSafe(GHG),
      "M_s" -> GHG,
      // s0 = 1 for binary trait
      "s0" -> 1.0
    )

  }


  /**
   * Get the standardized marginal score statistics
   * @param G A matrix of genotype (# of individuals x # of SNPs)
   * @return Zscores A vector of Z scores of the marginal score statistics for each SNP.
   * @return scores A vector of the marginal score statistics for each SNP
   * @return M_Z the correlation matrix of Z scores when effects are fixed (including H0) given X and G fixed.
   * @return M_s the covariance matrix of the score statistics when effects are fixed (including H0) given X and G fixed.
   * @return s0 the estimated dispersion parameter. Binary trait: s0=1. Continuous trait: residual SD under the null model.
   */
  def getZMargScoreContinuous(
    G: BDM[Double],
    HHalf: BDM[Double],
    s0: Double,
    resids: BDV[Double],
    use_t: Boolean = false
  ): Map[String, Any] = {

    val GHG = getGHG_Continuous(G, HHalf)

    val score = G.t * resids / s0
    // TEMPORARY SOLUTION
    val Zscore: BDV[Double] = zScoresZeroSafe(score, GHG)
    // UNDO THAT LATER AND USE CODE BELOW
    // val Zscore: BDV[Double] = if (use_t) {
    //   val XWithIntercept = BDM.horzcat(BDM.ones[Double](X.rows, 1), X)
    //   BDV((0 until G.cols).map { j => contZScore(G(::, j), XWithIntercept, Y) }.toArray)
    // } else {
    //   // Z scores based on the score statistics
    //   score /:/ sqrt(diag(GHG))
    // }


    return Map(
      "Zscores" -> Zscore,
      "scores" -> score,
      "M_Z" -> cov2corZeroSafe(GHG),
      "M_s" -> GHG,
      "s0" -> s0
    )

  }

  def getZMargScoreContinuousT(
    G: BDM[Double],
    X: BDM[Double],
    Y: BDV[Double],
  ): Map[String, Any] = {

    val (ghg, s0, resids) = getGHG_Continuous(G, X, Y)
    // Compute residuals
    val score = G.t * resids / s0

    // nullDesign, not an unconditional horzcat: a duplicated intercept column
    // makes the regression design matrix singular. Same fix as in getZMargScore.
    val XWithIntercept = nullDesign(X)
    val Zscore: BDV[Double] = BDV((0 until G.cols).map { j => contZScore(G(::, j), XWithIntercept, Y) }.toArray)

    return Map(
      "Zscores" -> Zscore,
      "scores" -> score,
      "M_Z" -> cov2corZeroSafe(ghg),
      "M_s" -> ghg,
      "s0" -> s0
    )

  }
}
