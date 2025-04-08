/*
This file contains main and supportive functions for computing Z-scores from a matrix of genotypes, covariates,
and a binary response
Reference: Zhang, Hong, and Zheyang Wu. "The generalized Fisher's combination and accurate p‐value
           calculation under dependence." Biometrics 79.2 (2023): 1159-1172.
Creators: Kylie Hoar
Last update (latest update first):
  KHoar 2025-01-16: sample format for future edits
*/

package is.hail.methods.gfisher

import breeze.linalg.{DenseMatrix => BDM, DenseVector => BDV, _}
import breeze.numerics._
import breeze.stats._
import breeze.interpolation._
import breeze.linalg.support.CanSlice
import scala.collection.mutable.ArrayBuffer
import breeze.optimize.{DiffFunction, minimize}
import is.hail.stats.LogisticRegressionModel
import net.sourceforge.jdistlib.{Normal, ChiSquare}
import is.hail.methods.gfisher.OptimalWeights.{getGHG_Binary, getGHG_Binary2, getGHG_Continuous}

object FuncCalcuZScores {

  /*
  Helper functions
  */

  /**
    * Calculate the Z score by the marginal t statistics for Y vs. each column of G and X
    * @param g Extracted column (BDV) of genotype matrix
    * @param X Input feature matrix (rows: observations, cols: features) with first column of ones for intercept
    * @param Y Target vector (binary labels: 0 or 1).
    */
  def contZScore (g: BDV[Double], X: BDM[Double], Y: BDV[Double]): Double = {
    // Combine column of g with X
    val XwithG = BDM.horzcat(X, g.toDenseMatrix.t)
    val (beta, se, _) = stdErrLinearRegression(XwithG, Y)
    // Z-score for g
    val tStatistic = beta(-1) / se(-1)
    tStatistic
  }

  def saddleProb(q: Double, mu: BDV[Double], g: BDV[Double]): Double = {
    // Compute variance (inner product of g * g)
    val sigmaSq = g dot g

    // Ensure sigmaSq is positive to avoid division errors
    if (sigmaSq <= 0) {
        throw new IllegalArgumentException("Variance must be positive")
    }

    // Compute adjusted mean
    val muAdj = mu dot g

    // Compute standardized test statistic
    val z = (q - muAdj) / math.sqrt(sigmaSq)

    // Compute p-value using jdistlib Normal CDF
    val pValue = 2.0 * (1.0 - Normal.cumulative(Math.abs(z), 0.0, 1.0, false, false))

    pValue
  }

  def addLogP(p1: Double, p2: Double): Double = {
    val absp1 = -abs(p1)
    val absp2 = -abs(p2)
    val maxP = math.max(absp1, absp2)
    val minP = math.min(absp1, absp2)
    maxP + math.log(1 + math.exp(minP - maxP))
  }

  def Korg(t: BDV[Double], mu: BDV[Double], g: BDV[Double]): BDV[Double] = {
    val out = BDV.zeros[Double](t.size)

    for (i <- 0 until t.size) {
      val t1 = t(i)
      val temp = (1.0 - mu) + (mu * exp(g *:* t1))
      out(i) = sum(log(temp))
    }
    
    out
  }

  def K1_adj(t: BDV[Double], mu: BDV[Double], g: BDV[Double], q: Double): BDV[Double] = {
    val out = BDV.zeros[Double](t.size)

    for (i <- 0 until t.size) {
      val t1 = t(i)
      val temp1 = (1.0 - mu) * exp(-g *:* t1) + mu
      val temp2 = mu * g
      out(i) = sum(temp2 / temp1) -:- q
    }

    out
  }

  def K2(t: BDV[Double], mu: BDV[Double], g: BDV[Double]): BDV[Double] = {
    val out = BDV.zeros[Double](t.size)

    for (i <- 0 until t.size) {
      val t1 = t(i)
      val temp1 = pow(((1.0 - mu) * exp(-g *:* t1) + mu), 2)
      val temp2 = (1.0 - mu) * mu * pow(g, 2) * exp(-g *:* t1)
      out(i) = sum(temp2 / temp1)
    }

    out
  }

  def Get_Saddle_Spline(mu: BDV[Double], g: BDV[Double], nodes: BDV[Double]): BDM[Double] = {
    val m1 = mu.dot(g)
    val y1 = K1_adj(nodes, mu, g, 0) - m1
    val y2 = K2(nodes, mu, g)
    val resultMatrix = BDM.horzcat(nodes.toDenseMatrix.t, y1.toDenseMatrix.t, y2.toDenseMatrix.t)
    val nonZeroRows = (0 until resultMatrix.rows).filter(i => resultMatrix(i, 0) != 0.0)
    if (nonZeroRows.isEmpty) {
      BDM.zeros[Double](0, resultMatrix.cols)  // Return an empty matrix if all rows were removed
    } else {
      BDM.vertcat(nonZeroRows.map(i => resultMatrix(i, ::).t.toDenseMatrix):_*)
    }
  }

  def splinefunH(nodes: BDV[Double], y1: BDV[Double], y2: BDV[Double]): (Double, Int) => Double = {
    val spline = CubicInterpolator(nodes, y1)  // Breeze cubic spline
    val h = 1e-5  // Small step size for finite differences
    
    // Find the min and max nodes for extrapolation bounds
    val minNode = nodes.min
    val maxNode = nodes.max

    (x: Double, deriv: Int) =>
      // Extrapolation outside the range of the nodes
      if (x < minNode) {
        // Linear extrapolation on the left side (using first two points)
        deriv match {
          case 0 => y1(0) + (x - nodes(0)) * (y1(1) - y1(0)) / (nodes(1) - nodes(0)) // Function value
          case 1 => (y1(1) - y1(0)) / (nodes(1) - nodes(0))  // First derivative
          case 2 => 0.0  // Second derivative (zero for linear extrapolation)
          case _ => throw new IllegalArgumentException("Only supports deriv = 0, 1, or 2")
        }
      } else if (x > maxNode) {
        // Linear extrapolation on the right side (using last two points)
        deriv match {
          case 0 => y1(nodes.length - 2) + (x - nodes(nodes.length - 2)) * (y1(nodes.length - 1) - y1(nodes.length - 2)) / (nodes(nodes.length - 1) - nodes(nodes.length - 2)) // Function value
          case 1 => (y1(nodes.length - 1) - y1(nodes.length - 2)) / (nodes(nodes.length - 1) - nodes(nodes.length - 2))  // First derivative
          case 2 => 0.0  // Second derivative (zero for linear extrapolation)
          case _ => throw new IllegalArgumentException("Only supports deriv = 0, 1, or 2")
        }
      } else {
        // Use spline interpolation within the range
        deriv match {
          case 0 => spline(x)  // Function value
          case 1 => 
            // Approximate first derivative using finite differences
            (spline(x + h) - spline(x - h)) / (2 * h)
          case 2 => 
            // Approximate second derivative using finite differences
            (spline(x + h) - 2 * spline(x) + spline(x - h)) / (h * h)
          case _ => throw new IllegalArgumentException("Only supports deriv = 0, 1, or 2")
        }
      }
  }

  def getNodes(init: BDV[Double], mu: BDV[Double], g: BDV[Double]): (BDV[Double], Double) = {
    var nodes = BDV.vertcat(init, BDV(0.0)).toArray.distinct.sorted
    var rep = 0
    val t = (Array.tabulate(25)(i => math.pow(2, i / 2.0 - 2)) ++ 
            Array.tabulate(25)(i => -math.pow(2, i / 2.0 - 2)) ++ 
            Array(0.0)).sorted
    val yt = K1_adj(BDV(t), mu, g, 0)
    val w = t.map(x => if (math.pow(x, 2) > 1) 1.0 / math.pow(math.abs(x), 1.0 / 3) else 1.0)
    var jump = 0.5
    var totres = Double.PositiveInfinity

    while (rep <= 1) {
      val y1 = K1_adj(BDV(nodes), mu, g, 0)
      val y2 = K2(BDV(nodes), mu, g)
      
      val sfun = splinefunH(BDV(nodes.toArray), BDV(y1.toArray), BDV(y2.toArray))
      val pred = t.map(x => sfun(x, 0))
      val predVec = BDV(pred: _*)
      val ytVec = BDV(yt.toArray)
      val wVec = BDV(w.toArray)

      val res = (wVec.toArray, predVec.toArray, ytVec.toArray).zipped.map {
        (wi: Double, pi: Double, yi: Double) => wi * math.abs(pi - yi)
      }

      val newNodesBuffer = ArrayBuffer[Double]()
      for (i <- nodes.indices if nodes(i) != 0) {
        val int1 = t.zipWithIndex.filter { case (ti, _) => ti < nodes(i) && (i == 0 || ti > nodes(i - 1)) }.map(_._2)
        val int2 = t.zipWithIndex.filter { case (ti, _) => ti > nodes(i) && (i == nodes.length - 1 || ti < nodes(i + 1)) }.map(_._2)

        val (r1, t1) = if (int1.nonEmpty) {
          val maxIdx = int1.maxBy(res)
          (res(maxIdx), t(maxIdx))
        } else (0.0, 0.0)

        val (r2, t2) = if (int2.nonEmpty) {
          val maxIdx = int2.maxBy(res)
          (res(maxIdx), t(maxIdx))
        } else (0.0, 0.0)

        if (r1 == r2) {
          newNodesBuffer += nodes(i)
        } else if (r1 > r2) {
          val jump1 = jump * (1 - max(0.1, r2 / r1)) * abs(nodes(i))
          newNodesBuffer += nodes(i) - jump1
        } else {
          val jump1 = jump * (1 - max(0.1, r1 / r2)) * abs(nodes(i))
          newNodesBuffer += nodes(i) + jump1
        }
      }

      newNodesBuffer += 0.0
      val newNodes = newNodesBuffer.distinct.sorted

      if (res.sum > totres) return (BDV(nodes), totres)

      rep += 1
      totres = res.sum
      nodes = newNodes.toArray
    }

    (BDV(nodes), totres)
  }

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

    if (q >= gPos || q <= gNeg) {
      (Double.PositiveInfinity, 0, true)  // Return if q is out of bounds
    } else {
      var t = init
      var K1Eval = K1_adj(BDV(t), mu, g, q).apply(0)  // Compute initial K1
      var prevJump = Double.PositiveInfinity
      var iter = 1
      var converged = false

      while (iter <= maxIter) {
        val K2Eval = K2(BDV(t), mu, g).apply(0)  // Compute K2 at t
        val tNew = t - K1Eval / K2Eval

        if (tNew.isNaN) {
          converged = false
          return (Double.NaN, iter, converged)
        }

        if (math.abs(tNew - t) < tol) {
          converged = true
          return (tNew, iter, converged)
        }

        val newK1 = K1_adj(BDV(tNew), mu, g, q).apply(0)

        if (math.signum(K1Eval) != math.signum(newK1)) {
          if (math.abs(tNew - t) > prevJump - tol) {
            val tAdjusted = t + math.signum(newK1 - K1Eval) * prevJump / 2
            t = tAdjusted
            prevJump /= 2
          } else {
            prevJump = math.abs(tNew - t)
          }
        } else {
          t = tNew
        }

        K1Eval = newK1
        iter += 1
      }

      (t, iter, converged)
    }
  }

  def getSaddleProb(zeta: Double, mu: BDV[Double], g: BDV[Double], q: Double, logP: Boolean = false): Double = {
    // Compute K1 and K2
    val k1 = Korg(BDV(zeta), mu, g)(0)
    val k2 = K2(BDV(zeta), mu, g)(0)

    if (java.lang.Double.isFinite(k1) && java.lang.Double.isFinite(k2)) {
      val temp1 = zeta * q - k1
      val w = signum(zeta) * sqrt(2 * temp1)
      val v = zeta * sqrt(k2)

      val zTest = w + 1 / w * log(v / w)

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

  def SaddleProb(q: Double, mu: BDV[Double], g: BDV[Double], Cutoff: Any = 2.0, alpha: Double, output: String = "P", nodesFixed: Option[BDV[Double]], nodesInit: Option[BDV[Double]], logP: Boolean = false): Map[String, Any] = {
    val m1 = mu.dot(g)  // sum(mu * g)
    val var1 = (mu.toArray zip g.toArray).map { case (m, gi) => m * (1 - m) * gi * gi }.sum
    var p1: Option[Double] = None
    var p2: Option[Double] = None

    //var nodes: BDV[Double] = nodesInit

    var nodes: BDV[Double] = BDV()  // Empty init (won't be used unless output == "metaspline")

    if (output == "metaspline") {
      val initNodes = nodesInit.getOrElse(
        throw new IllegalArgumentException("nodesInit must be defined for metaspline output.")
      )

      nodes = nodesFixed match {
        case Some(fixedNodes) => BDV(fixedNodes.toArray.sorted :+ 0.0)
        case None => getNodes(initNodes.toDenseVector, mu, g)._1
      }

      val splfun = Get_Saddle_Spline(mu, g, nodes)
      // Store splfun as a var if you need it later
    }

    // Handle "metaspline" output logic
    //if (output == "metaspline") {
    //  nodes = nodesFixed match {
    //    case Some(fixedNodes) => BDV(fixedNodes.toArray.sorted :+ 0.0)
    //    case None => val (newNodes, _) = getNodes(nodesInit.toDenseVector, mu, g)
    //    newNodes
    //  }
    //}

    val score = q - m1
    val qinv = -Math.signum(q - m1) * Math.abs(q - m1) + m1

    // Noadj p-value
    val pvalNoadj = ChiSquare.cumulative((Math.pow(q - m1, 2) / var1), 1, false, logP)

    var isConverge = true
    var pval = pvalNoadj

    val cutoffVal: Double = Cutoff match {
      case s: String if s == "BE" =>
        val rho = mu.toScalaVector.zip(g.toScalaVector).map {
          case (m, gi) => Math.pow(Math.abs(gi), 3) * m * (1 - m) * (Math.pow(m, 2) + Math.pow(1 - m, 2))
        }.sum
        val var1 = (mu.toArray zip g.toArray).map { case (m, gi) => m * (1 - m) * gi * gi }.sum
        val B = 0.56 * rho * Math.pow(var1, -1.5)
        val p = B + alpha / 2
        if (p >= 0.496) 0.01 else Normal.quantile(p, 0, 1, false, logP)

      case d: Double => if (d < 0.1) 0.1 else d

      case _ => throw new IllegalArgumentException("Invalid type for Cutoff. Expected Double or \"BE\".")
    }

    //if (Cutoff == "BE") {
    //  val rho = mu.toScalaVector.zip(g.toScalaVector).map { case (m, gi) => Math.pow(Math.abs(gi.toDouble), 3) * m * (1 - m) * (Math.pow(m.toDouble, 2) + Math.pow(1 - m.toDouble, 2))}.sum
    //  val B = 0.56 * rho * Math.pow(var1, -1.5)
    //  val p = B + alpha / 2
    //  cutoffValue = if (p >= 0.496) 0.01 else Normal.quantile(p, 0, 1, false, logP)
    //} else if (Cutoff < 0.1) {
    //  cutoffValue = 0.1
    //}

    // Handle "metaspline" output logic
    if (output == "metaspline") {
      val splfun = Get_Saddle_Spline(mu, g, nodes)
    }

    // p-value calculation based on cutoff and convergence
    if (Math.abs(q - m1) / Math.sqrt(var1) < cutoffVal) {
      pval = pvalNoadj
    } else {
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
  Main function
  */

  /**
   * Get the standardized marginal score statistics
   * @param G A matrix of genotype (# of individuals x # of SNPs)
   * @param X A matrix of covariates, default is 1
   * @param Y A single column of response variable; it has to be 0/1 for binary trait
   * @param trait_lm indicator of "binary" (logistic regression) or "continuous" (linear regression).
   * @param use_lm_t whether to use the lm() function to get the t statistics as the Z-scores for continuous trait
   * @return Zscores A vector of Z scores of the marginal score statistics for each SNP.
   * @return scores A vector of the marginal score statistics for each SNP
   * @return M_Z the correlation matrix of Z scores when effects are fixed (including H0) given X and G fixed.
   * @return M_s the covariance matrix of the score statistics when effects are fixed (including H0) given X and G fixed.
   * @return s0 the estimated dispersion parameter. Binary trait: s0=1. Continuous trait: residual SD under the null model.
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

  def getZ_marg_score_binary_SPA (
    G: BDM[Double],
    X: BDM[Double],
    Y: BDV[Double]
  ): Map[String, Any] = {
    // Logistic regression model
    val X1 = BDM.horzcat(BDM.ones[Double](X.rows, 1), X)
    val Y0 = log_reg_predict(X, Y)

    // Calculate XVX inverse
    val sqrtY0 = sqrt(Y0 *:* (1.0 - Y0))
    val XV = ((X1(::, *) * sqrtY0).t).t
    val XVX = X1.t * XV
    val XVX_inv = inv(XVX)

    // Compute XXVX_inv and Gscale
    val XXVX_inv = X1 * XVX_inv
    val Gscale = G - (XXVX_inv.t * (XV * G.t)).t

    // Compute score
    val score = Gscale.t * Y

    // Compute p-values (using a placeholder function)
    val pval_spa = BDV((0 until score.size).map { x =>
      saddleProb(score(x), Y0, Gscale(::, x))
    }.toArray)

    // Compute Z-scores
    val Zscores_spa = BDV(pval_spa.map(p => Normal.quantile(1.0 - p / 2.0, 0.0, 1.0, false, false)).toArray) *:* signum(score)

    // Compute GHG and M
    val Xtilde = ((X(::, *) * sqrtY0).t).t
    val Hhalf = Xtilde * cholesky(inv(Xtilde.t * Xtilde)).t
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
      // s0 = 1 for binary trait
      "s0" -> 1.0
    )

  }
}
