/*
This file contains main and supportive functions for computing Z-scores from a matrix of genotypes, covariates,
and a binary response
Reference: Zhang, Hong, and Zheyang Wu. "The generalized Fisher's combination and accurate p‐value
           calculation under dependence." Biometrics 79.2 (2023): 1159-1172.
Creators: Kylie Hoar and Peter Howell
Last update (latest update first):
  KHoar 2025-04-21: Added comments to SPA-related functions
*/

package is.hail.methods.gfisher

import breeze.linalg.{DenseMatrix => BDM, DenseVector => BDV, _}
import breeze.numerics._
import scala.collection.mutable.ArrayBuffer
import net.sourceforge.jdistlib.{Normal, ChiSquare}
import is.hail.methods.gfisher.OptimalWeights.{getGHG_Binary, getGHG_Binary2, getGHG_Continuous}
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
  def Korg(t: BDV[Double], mu: BDV[Double], g: BDV[Double]): BDV[Double] = {
    val out = BDV.zeros[Double](t.size)

    for (i <- 0 until t.size) {
      val t1 = t(i)
      val temp = (1.0 - mu) + (mu * exp(g *:* t1))
      out(i) = sum(log(temp))
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
  def K1_adj(t: BDV[Double], mu: BDV[Double], g: BDV[Double], q: Double): BDV[Double] = {
    val out = BDV.zeros[Double](t.size)

    for (i <- 0 until t.size) {
      val t1 = t(i)
      val temp1 = (1.0 - mu) * exp(-g *:* t1) + mu
      val temp2 = mu * g
      // The saddlepoint function: sum(mu * g / ((1 - mu) * exp(-g * t1) + mu)) - q
      // This is the equation to solve for the saddlepoint (i.e., K1(t) = q)
      out(i) = sum(temp2 / temp1) -:- q
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
  def K2(t: BDV[Double], mu: BDV[Double], g: BDV[Double]): BDV[Double] = {
    val out = BDV.zeros[Double](t.size)

    for (i <- 0 until t.size) {
      val t1 = t(i)

      // Compute the denominator of the expression: 
      // ((1 - mu) * exp(-g * t1) + mu)^2
      val temp1 = pow(((1.0 - mu) * exp(-g *:* t1) + mu), 2)

      // Compute the numerator: 
      // (1 - mu) * mu * g^2 * exp(-g * t1)
      val temp2 = (1.0 - mu) * mu * pow(g, 2) * exp(-g *:* t1)

      // The second derivative of the CGF:
      // K2(t1) = sum( numerator / denominator )
      out(i) = sum(temp2 / temp1)
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
    * @param tol      (optional, default = pow(scala.Double.MinPositiveValue, 0.25)), Tolerance for convergence; the root
                      is considered found if the step size is smaller than this
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
      tol: Double = pow(scala.Double.MinPositiveValue, 0.25),
      maxIter: Int = 1000
  ): (Double, Int, Boolean) = {

    val gPos = sum(g(g >:> 0.0))  // Sum of positive values in g
    val gNeg = sum(g(g <:< 0.0))  // Sum of negative values in g

    // If q is outside the valid range of the cumulant domain, return immediately with an infinite root (invalid input)
    if (q >= gPos || q <= gNeg) {
      (Double.PositiveInfinity, 0, true)
    } else {
      var t = init                                    // Starting guess for root-finding
      var K1Eval = K1_adj(BDV(t), mu, g, q).apply(0)  // Evaluate K1 at t
      var prevJump = Double.PositiveInfinity          // Used to limit oscillations
      var iter = 1

      // Begin Newton-Raphson iterations, capped by maxIter
      while (iter <= maxIter) {
        val K2Eval = K2(BDV(t), mu, g).apply(0)       // Derivative of K1 (2nd cumulant)
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
        val newK1 = K1_adj(BDV(tNew), mu, g, q).apply(0)

        // Check for sign change, indicating crossing the root
        if (math.signum(K1Eval) != math.signum(newK1)) {
          if (math.abs(tNew - t) > prevJump - tol) {
            t = t + math.signum(newK1 - K1Eval) * prevJump / 2  // Backtrack slightly
            prevJump /= 2
            K1Eval = K1_adj(BDV(t), mu, g, q).apply(0)          // Re-evaluate K1
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
  def getSaddleProb(zeta: Double, mu: BDV[Double], g: BDV[Double], q: Double, logP: Boolean = false): Double = {
    // Compute K1 (cumulant generating function) and K2 (its second derivative) at zeta
    val k1 = Korg(BDV(zeta), mu, g)(0)
    val k2 = K2(BDV(zeta), mu, g)(0)

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
  def SaddleProb(q: Double, mu: BDV[Double], g: BDV[Double], Cutoff: Any = 2.0, alpha: Double, output: String = "P", nodesFixed: Option[BDV[Double]], nodesInit: Option[BDV[Double]], logP: Boolean = false): Map[String, Any] = {
    val m1 = mu.dot(g)  // sum(mu * g)  // Expected value of the test statistic
    val var1 = (mu.toArray zip g.toArray).map { case (m, gi) => m * (1 - m) * gi * gi }.sum  // Variance
    
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
      If "BE", use Berry-Esseen inequality to derive threshold
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
      val outUni1 = getRootK1(0, mu, g, q)
      val outUni2 = getRootK1(0, mu, g, qinv)

      if (outUni1._3 && outUni2._3) {
        p1 = try {
          Some(getSaddleProb(outUni1._1, mu, g, q, logP))
        } catch {
          case e: Exception =>
            if (logP) Some(pvalNoadj - Math.log(2))
            else Some(pvalNoadj / 2)
        }
        p2 = try {
          Some(getSaddleProb(outUni2._1, mu, g, qinv, logP))
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
      return SaddleProb(q, mu, g, cutoffVal * 2, alpha, output, nodesFixed, nodesInit, logP)
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
   * @param trait_lm  Indicator of "binary" (logistic regression) or "continuous" (linear regression).
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
      val Zscore = score /:/ sqrt(diag(ghg))

      return Map(
        "Zscores" -> Zscore,
        "scores" -> score,
        "M_Z" -> cov2cor(ghg),
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
        val XWithIntercept = BDM.horzcat(BDM.ones[Double](X.rows, 1), X)
        BDV((0 until G.cols).map { j => contZScore(G(::, j), XWithIntercept, Y) }.toArray)
      } else {
        // Z scores based on the score statistics
        score /:/ sqrt(diag(ghg))
      }

      return Map(
        "Zscores" -> Zscore,
        "scores" -> score,
        "M_Z" -> cov2cor(ghg),
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
    *             - "M": BDM[Double] of correlation matrix for variants (after projection)
    *             - "GHG": BDM[Double] of G'G adjusted for covariates
    *             - "s0": Double, dispersion parameter (currently fixed at 1.0)
    */
  def getZ_marg_score_binary_SPA (
    G: BDM[Double],
    X: BDM[Double],
    Y: BDV[Double]
  ): Map[String, Any] = {
    // Logistic regression model
    val X1 = BDM.horzcat(BDM.ones[Double](X.rows, 1), X)

    // Compute the predicted probabilities under the null model via logistic regression
    val Y0 = log_reg_predict(X, Y)

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

    // Applies the saddlepoint approximation to compute p-values for each score statistic
    val pval_spa = BDV((0 until score.size).map { x =>
      SaddleProb(score(x), Y0, Gscale(::, x), "BE", 5 * math.pow(10, -8), "P", None, None, false)("p.value").asInstanceOf[Double]
    }.toArray)

    // Transform SPA p-values into Z-scores
    val Zscores_spa = BDV(pval_spa.map(p => Normal.quantile(p / 2.0, 0.0, 1.0, false, false)).toArray) *:* signum(score)

    // Perform a projection of genotype matrix for correlation estimation (GHG is the residualized, cross-product matrix)
    val sqrtY0 = sqrt(Y0_vec)
    val Xtilde = ((X(::, *) * sqrtY0).t).t
    val Hhalf = Xtilde * cholesky(inv(Xtilde.t * Xtilde))
    val Gtilde = ((G(::, *) * sqrtY0).t).t
    val GHhalf = Gtilde.t * Hhalf
    val GHG = (Gtilde.t * Gtilde) - (GHhalf * GHhalf.t)
    val M = cov2cor(GHG)

    // Dispersion parameter
    val s0 = 1.0

    Map(
      "Zscores" -> Zscores_spa,
      "M" -> M,
      "GHG" -> GHG,
      "s0" -> s0
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
    val Zscore = score /:/ sqrt(diag(GHG))

    return Map(
      "Zscores" -> Zscore,
      "scores" -> score,
      "M_Z" -> cov2cor(GHG),
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
    val Zscore: BDV[Double] = score /:/ sqrt(diag(GHG))
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
      "M_Z" -> cov2cor(GHG),
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

    val XWithIntercept = BDM.horzcat(BDM.ones[Double](X.rows, 1), X)
    val Zscore: BDV[Double] = BDV((0 until G.cols).map { j => contZScore(G(::, j), XWithIntercept, Y) }.toArray)

    return Map(
      "Zscores" -> Zscore,
      "scores" -> score,
      "M_Z" -> cov2cor(ghg),
      "M_s" -> ghg,
      "s0" -> s0
    )

  }
}
