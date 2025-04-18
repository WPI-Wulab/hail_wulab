package is.hail.methods.gfisher

import breeze.linalg.{DenseMatrix => BDM, DenseVector => BDV, _}
import net.sourceforge.jdistlib.{Normal, ChiSquare, Cauchy}

import breeze.stats.mean
import breeze.numerics.{abs, tan}
import scala.math.Pi

object FuncCalcuCombTests {

  // Transformation function g(z) for Z-scores with two-sided input p-values
  def g_GFisher_two(x: Double, df: Double): Double = {
    val pValueLog = math.log(2) + Normal.cumulative(math.abs(x), 0.0, 1.0, false, true)
    ChiSquare.quantile(pValueLog, df, false, true)
  }

  // Transformation function g(x) for Z-scores with one-sided input p-values
  def g_GFisher_one(x: Double, df: Double): Double = {
    val pValueLog = Normal.cumulative(x, 0.0, 1.0, false, true)
    ChiSquare.quantile(pValueLog, df, false, true)
  }

    // Transformation function g(x) for burden/Laptik test (identity function)
    def g_Burden(x: Double): Double = x

  // Combines g_GFisher_two and g_GFisher_one, controlled by oneSided ("one" or "two")
  def g_GFisher(x: Double, df: Double, oneSided: Boolean = false): Double = {
    if (oneSided)  {
      g_GFisher_one(x, df)
    } else {
      g_GFisher_two(x, df)
    }
  }

  def calcu_SgZ_p (
    g: BDV[Double] => BDV[Double],  // Function g
    Zscores: BDV[Double],
    wts: BDV[Double],
    calc_p: Boolean = false,
    M: Option[BDM[Double]] = None,
    df: Option[Double] = None,  // supports positive infinity
    oneSided: Boolean = false, // never actually used?
    isPosiWts: Boolean = true
  ): Map[String, Any] = {

    var weights = if (isPosiWts) wts.map(x => math.max(x, 0)) else wts  // force the weights to be non-negative

    if (weights.size != Zscores.size) {
      // If the sizes don't match, resize weights to match Zscores size (by repeating or filling it)
      weights = BDV.fill(Zscores.size)(weights(0))  // You can use weights(0) or another logic to fill the vector
    }

    // always scale the weights
    if (sum(abs(weights)) > 0) { // this gives a weird warning
      weights = weights / sum(abs(weights))// this gives a weird warning
    }

    // calculate the statistic S using the provided g function
    val S = sum(weights.asInstanceOf[BDV[Double]] *:* g(Zscores))

    if(calc_p) {

      // why do this?!?!
      // assign proper df data types depending on the df input value
      val dfs: Double = df match {
        case Some(d: Double) => d
        case _ => 0.0
      }
      // separate the calculation for burden test (df = Inf) and GFisher test
      if (dfs.isPosInfinity) {
        // burden test
        val M_mat = M.getOrElse(BDM.zeros[Double](weights.length, weights.length))
        val S_sd = math.sqrt(weights.t * M_mat * weights)
        val normalDist = new Normal(0, S_sd)
        val p = 2 * normalDist.cumulative(-math.abs(S))
        Map("S" -> S, "p" -> p)
      } else if (dfs != 0) {
        // GFisher statistics
        // set degrees of freedom to be a vector, not an integer
        val (numRows, numCols) = M.map(m => (m.rows, m.cols)).getOrElse((0, 0))
        val degreesOfFreedom = BDV.fill(numRows)(dfs)
        // rescale the weights
        weights = weights/sum(weights.map(math.abs))
        // recalculate the S to be consistent with the new weights
        val S_ = sum(weights * g(Zscores))
        // calculate pGFisher (HYB)
        lazy val matrix: BDM[Double] = M.getOrElse(BDM.zeros[Double](numRows, numCols))
        val p = PGFisher.pGFisherHyb(S_, degreesOfFreedom, weights, matrix)
        Map("S" -> S_, "p" -> p)
      } else {
        Map("S" -> S)
      }
    } else {
      Map("S" -> S)
    }
  }

  // Multi SgZ Test with integrated burden test
  def multi_SgZ_test(
    Zscores: BDV[Double],
    DF: BDM[Double],
    W: BDM[Double],
    M: Option[BDM[Double]],
    oneSided: Boolean = false,
    calcP: Boolean = true,
    isPosiWts: Boolean = true,
    wNames: Option[Seq[String]] = None
  ): Map[String, BDM[Double]] = {

    val testN = DF.rows // Number of tests
    val STAT = BDM.zeros[Double](testN, 1)
    val PVAL = BDM.zeros[Double](testN, 1)

    for (i <- 0 until testN) {
      val dfValue = DF(i, 0)

      val result = if (dfValue.isPosInfinity) {
        // Burden test: Use identity function g(x) = x
        calcu_SgZ_p(x => x, Zscores, W(i, ::).t, calcP, M, Some(dfValue), oneSided, isPosiWts)
      } else {
        // GFisher test with transformation function
        val g = (x: BDV[Double]) => x.map(v => g_GFisher(v, dfValue, oneSided))
        calcu_SgZ_p(g, Zscores, W(i, ::).t, calcP, M, Some(dfValue), oneSided, isPosiWts)
      }

      STAT(i, 0) = result("S").asInstanceOf[Double]
      if (calcP) PVAL(i, 0) = result("p").asInstanceOf[Double]
    }

    Map("STAT" -> STAT, "PVAL" -> PVAL)
  }

  def cctTest(pvals: BDV[Double], thrLargeP: Double = 0.9, thrSmallP: Double = 1e-15): Map[String, Any] = {
    // Replace large p-values
    val clippedPvals = pvals.map(p => if (p > thrLargeP) thrLargeP else p)

    // Indicator for small p-values
    val isSmall = clippedPvals <:< thrSmallP

    // CCTSTAT initially set to PVAL values
    val CCTSTAT = clippedPvals.copy

    val cct: Double = if (!any(isSmall)) {// this gives a weird warning
      mean(tan((0.5 - CCTSTAT) * 0.5))  // If no small p-values, use regular transformation
    } else {
      for (i <- 0 until CCTSTAT.length) {
        if (!isSmall(i)) CCTSTAT(i) = tan((0.5 - CCTSTAT(i)) * Pi)
        else CCTSTAT(i) = 1.0 / (CCTSTAT(i) * Pi)
      }
      mean(CCTSTAT)  // Compute the mean of transformed values
    }

    // Compute the final CCT p-value using jdistlib.Cauchy
    val pvalCCT = Cauchy.cumulative(cct, 0.0, 1.0, false, false)

    // Return results in a Map
    Map("cct" -> cct, "pval_cct" -> pvalCCT)
  }

  def omni_SgZ_test(
    Zscores: BDV[Double],
    DF: BDM[Double],
    W: BDM[Double],
    M: BDM[Double],
    oneSided: Boolean = false,
    calcuP: Boolean = true,
    isPosiWts: Boolean = true
  ): Map[String, Any] = {

    // Call multi_SgZ_test (assuming you've translated this function to Scala)
    val multiTests = multi_SgZ_test(Zscores, DF, W, Some(M), oneSided, calcuP, isPosiWts)

    val (cct, pvalCct) = if (calcuP) {
      val PVAL_temp = multiTests("PVAL").asInstanceOf[BDM[Double]](::, 0) // Extract first column
      val cctResult = cctTest(PVAL_temp)
      (cctResult("cct").asInstanceOf[Double], cctResult("pval_cct").asInstanceOf[Double])
    } else {
      (Double.NaN, Double.NaN)
    }

    // Combine results into a Map
    Map(
      "STAT" -> BDM.vertcat(multiTests("STAT").asInstanceOf[BDM[Double]], BDM(cct.asInstanceOf[Double])),
      "PVAL" -> BDM.vertcat(multiTests("PVAL").asInstanceOf[BDM[Double]], BDM(pvalCct.asInstanceOf[Double])),
      "cct" -> cct,
      "pval_cct" -> pvalCct
    )
  }

  def BSF_test(
    Zscores: BDV[Double],
    M: BDM[Double],
    Bstar: BDV[Double],
    PI: BDV[Double],
    additionalParams: Any*
  ): Map[String, BDM[Double]] = {

    val wtsEqu = BDM.ones[Double](1, M.cols) // Equal weights as row vector

    // Burden Test
    val gBurden: Double => Double = x => x
    val statDFBurden = Double.PositiveInfinity
    val wtsOptBurden = OptimalWeights.optimalWeightsM(gBurden, Bstar, PI, M, true, true)
    val WT_opt_burden = BDM.vertcat(wtsOptBurden, wtsEqu)
    val DF_opt_burden = BDM.fill(WT_opt_burden.rows, 1)(statDFBurden)

    // SKAT Test
    val gSKAT: Double => Double = x => x * x
    val statDFSKAT = 1.0
    val wtsOptSKAT = OptimalWeights.optimalWeightsM(gSKAT, Bstar, PI, M, false, true)
    val WT_opt_skat = BDM.vertcat(wtsOptSKAT, wtsEqu)
    val DF_opt_skat = BDM.fill(WT_opt_skat.rows, 1)(statDFSKAT)

    // Fisher Test
    val gFisher: Double => Double = x => g_GFisher_two(x, 2)
    val statDFFisher = 2.0
    val wtsOptFisher = OptimalWeights.optimalWeightsM(gFisher, Bstar, PI, M, false, true)
    val WT_opt_fisher = BDM.vertcat(wtsOptFisher, wtsEqu)
    val DF_opt_fisher = BDM.fill(WT_opt_fisher.rows, 1)(statDFFisher)

    // Combine everything
    val WT_opt = BDM.vertcat(WT_opt_skat, WT_opt_burden, WT_opt_fisher)
    val DF_opt = BDM.vertcat(DF_opt_skat, DF_opt_burden, DF_opt_fisher)

    // Compute the statistics
    val omniOpt = omni_SgZ_test(Zscores, DF_opt, WT_opt, M)

    // Ensure omniOpt Map values are properly typed
    val omniStat = omniOpt("STAT").asInstanceOf[BDM[Double]]
    val omniPval = omniOpt("PVAL").asInstanceOf[BDM[Double]]

    // Return properly typed result
    Map("STAT" -> omniStat, "PVAL" -> omniPval)
  }

  def BSF_test_byP (
    Pvalues: BDV[Double],
    Zsigns: BDV[Double],
    M: BDM[Double],
    Bstar: BDV[Double],
    PI: BDV[Double]
  ): Map[String, BDM[Double]] = {
    val Zscores = Pvalues.map(p => Normal.quantile(1.0 - (p / 2.0), 0.0, 1.0, false, false)) *:* Zsigns
    BSF_test(Zscores, M, Bstar, PI)
  }

  def BSF_cctP_test (
    Zscores: BDV[Double],
    M: BDM[Double],
    Bstar: BDV[Double],
    PI: BDV[Double]
  ): Map[String, BDM[Double]] = {
    val bsf = BSF_test(Zscores, M, Bstar, PI)


    val Pvalues = Zscores.map(z =>  2.0 * Normal.cumulative(-Math.abs(z), 0.0, 1.0, true, false))
    val cct = cctTest(Pvalues)
    val cctStat = cct("cct").asInstanceOf[Double]
    val cctP = cct("pval_cct").asInstanceOf[Double]

    // Extract STAT and PVAL matrices from the map
    val STAT = bsf("STAT").asInstanceOf[BDM[Double]]
    val PVAL = bsf("PVAL").asInstanceOf[BDM[Double]]

    val PVAL_Vec = PVAL.toDenseVector

    val newPVAL = BDV.vertcat(PVAL_Vec(0 until PVAL.size - 1), BDV(cctP))
    val bsf_cctP = cctTest(newPVAL)

    val bsf_cctP_stat = bsf_cctP("cct").asInstanceOf[Double]
    val bsf_cctP_pval = bsf_cctP("pval_cct").asInstanceOf[Double]

    val cctStatMat = BDM.fill(1, 1)(cctStat)
    val bsf_cctP_statMat = BDM.fill(1, 1)(bsf_cctP_stat)
    val cctPMat = BDM.fill(1, 1)(cctP)
    val bsf_cctP_pvalMat = BDM.fill(1, 1)(bsf_cctP_pval)

    val result_stat = BDM.vertcat(STAT, cctStatMat, bsf_cctP_statMat)
    val result_pval = BDM.vertcat(PVAL.asInstanceOf[BDM[Double]], cctPMat, bsf_cctP_pvalMat)

    Map("STAT" -> result_stat, "PVAL" -> result_pval)
  }

  def BSF_cctP_test_byP (
    Pvalues: BDV[Double],
    Zsigns: BDV[Double],
    M: BDM[Double],
    Bstar: BDV[Double],
    PI: BDV[Double]
  ): Map[String, BDM[Double]] = {
    val Zscores = Pvalues.map(p => Normal.quantile(1.0 - (p / 2.0), 0.0, 1.0, false, false)) *:* Zsigns
    BSF_cctP_test(Zscores, M, Bstar, PI)
  }
}
