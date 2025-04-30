/*
This file contains functions for computing the summation-based Z-score combination statistics and related p-values. The combination statistics are defined as S = sum_i^n w_i*g(Z_i), where g() is a function for transforming Z scores, w_i's are 
scaled weights, and Z_i's are Z scores. The functions in this file are designed to be used for the combination tests in the context of genetic association studies, such as burden test, SKAT test, Fisher test, and GFisher test. Implemtation highly depends on the GFisher library/package. Further implementations for the similar task can be included into this file.
Reference: Zhang, Hong, and Zheyang Wu. "The generalized Fisher's combination and accurate p‐value
           calculation under dependence." Biometrics 79.2 (2023): 1159-1172.
           Liu, Ming. "Integrative Analysis of Large Genomic Data." WPI (2025).
Creators: Kylie Hoar
Last update (latest update first):
  KHoar 2025-04-23: Added docstrings and internal comments
*/

package is.hail.methods.gfisher

import breeze.linalg.{DenseMatrix => BDM, DenseVector => BDV, _}
import net.sourceforge.jdistlib.{Normal, ChiSquare, Cauchy}

import breeze.stats.mean
import breeze.numerics.{abs, tan}
import scala.math.Pi

object FuncCalcuCombTests {

  /*
  A Burden-SKAT-Fisher (BSF) omnibus test that combines optimal weighting and uses the Cauchy Combination Test (CCT) to combine p-values
  (adapted from the GLOW R package: "GLOW_R_package/GLOW/R/helpers_GLOWtests.R")
  */

  /**
    * Transformation function g(z) for Z-scores with two-sided input p-values
    *
    * @param x  The Z-score to be transformed. Assumed to follow a standard normal distribution
    * @param df The degrees of freedom for the resulting Chi-square distribution
    *
    * @return   The Chi-square quantile corresponding to the log-transformed two-sided p-value
    */
  def g_GFisher_two(x: Double, df: Double): Double = {
    val pValueLog = math.log(2) + Normal.cumulative(math.abs(x), 0.0, 1.0, false, true)
    ChiSquare.quantile(pValueLog, df, false, true)
  }

  /**
    * Transformation function g(x) for Z-scores with one-sided input p-values
    * 
    * @param x  The Z-score to be transformed. Assumed to follow a standard normal distribution.
    * @param df The degrees of freedom for the resulting Chi-square distribution.
    *
    * @return   The Chi-square quantile corresponding to the log-transformed one-sided p-value.
    */
  def g_GFisher_one(x: Double, df: Double): Double = {
    val pValueLog = Normal.cumulative(x, 0.0, 1.0, false, true)
    ChiSquare.quantile(pValueLog, df, false, true)
  }

  /**
    * Identity transformation function g(x) for the Burden or Laptik test.
    *
    * @param x The input value (typically a test statistic or summary score).
    *
    * @return  The same input value, unchanged.
    */
  def g_Burden(x: Double): Double = x

  /**
    * Transformation function g(x) for GFisher method, selecting one-sided or two-sided logic.
    *
    * @param x  Z-score to be transformed.
    * @param df Degrees of freedom used in the chi-square transformation.
    * @param    oneSided If true, uses the one-sided transformation; otherwise uses the two-sided.
    *
    * @return   Transformed statistic on the chi-square scale.
    */
  def g_GFisher(x: Double, df: Double, oneSided: Boolean = false): Double = {
    if (oneSided)  {
      g_GFisher_one(x, df)
    } else {
      g_GFisher_two(x, df)
    }
  }

  /**
    * Compute the combination statistic for S = sum_i^n w_i*g(Z_i), given an arbitrary function g(), the weights w_i's, the Z 
    * scores and their correlation matrix. The p-value can be calculated for burden test and GFisher test.
    *
    * @param g            Function for tranforming Z scores
    * @param Zscores      Vector of Z scores
    * @param wts          Vector of weights, allowing negative values being used in calculating S when is.posi.wts=FALSE
    * @param calc_p       Logical value indicating whether to calculate p-value. 
    *                     Currently, p-value calculation is available for 
    *                      - burden test: when g = function(x)x or df=Inf. Allow for negative weights.                   
    *                      - GFisher test: output one-sided p-value, which requires non-negative weights.
    *                     One can use the g_GFisher, g_GFisher_one, or g_GFisher_two function, or set proper df argument.
    * @param M            Correlation matrix of the Z scores. Default is NULL.
    * @param df           Constant degrees of freedom for the GFisher statistic. Default is NULL. 
    *                     Require to be consistent with the function g. If the function g contains a df argument, that value 
    *                     will be used. df=Inf indicates the burden test.
    * @param p.type       Type of input p-values of GFisher, "one" or "two" sided. Default is "two".
    *                     It is only used when the function g is for GFisher test.
    *                     If the function g contains p.type as an argument, that value will be used.
    * @param is.posi.wts  Logical value indicating whether negative weights are casted to be 0. Default is TRUE.
    *                     calc_p=TRUE for GFisher test will force the weights to be non-negative and then normalized.
    * @param method       For p.GFisher: "MR" = simulation-assisted moment ratio matching, 
    *                                    "HYB" = moment ratio matching by quadratic approximation, 
    *                                    "GB" = Brown's method with calculated variance. See details in the reference.
    * @param nsim         For p.GFisher: Number of simulation used in the "MR" method, default = 5e4.
    * @param seed         For p.GFisher: Seed for random number generation, default = NULL.
    *
    * NOTE: When calculating the statistics only (i.e., calcu_p=FALSE), the g function is pretty arbitrary, and the weights can 
    * be negative. This can be used for empirical studies (e.g., empirical type I error or power). The weights are always scaled 
    * to be wts/sum(abs(wts)).
    *
    * @return S           The combination statistic
    * @return p           The p-value. For burden test, it's two-sided in consistence with genetical studies.
    *                     For GFisher, it's one-sided calculated by p.GFisher's default "HYB" method.
    */
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

  /**
    * Performs the Multi SgZ test with optional burden test integration.
    *
    * Computes statistics and p-values for a series of tests using either the
    * burden test (when degrees of freedom are infinite) or the GFisher method (for finite degrees of freedom).
    * Each row in `DF` corresponds to a separate test, with associated weights in `W`.
    *
    * The transformation function `g(x)` is selected based on the degrees of freedom:
    *   - If df = Inf: the identity function is used (Burden test).
    *   - If df < Inf: a chi-square transformation of Z-scores is applied (GFisher test).
    *
    * @param Zscores    Vector of Z-scores for the variants being tested.
    * @param DF         Matrix where each row contains the degrees of freedom for the corresponding test.
    * @param W          Matrix of weights corresponding to each test (rows) and each Z-score (columns).
    * @param M          Optional covariance/correlation matrix (used for adjusting the score).
    * @param oneSided   Whether to use one-sided transformation for p-values (default: false = two-sided).
    * @param calcP      Whether to compute p-values in addition to test statistics (default: true).
    * @param isPosiWts  Whether to enforce positivity on weights (default: true).
    * @param wNames     Optional names of the weight sets (not used in computation, useful for annotation).
    *
    * @return           A Map object containing:
    *                     - "STAT": Column vector of computed test statistics.
    *                     - "PVAL": Column vector of p-values (if `calcP` is true).
    */
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

  /**
    * Performs the Cauchy Combination Test (CCT) to combine a vector of p-values into a single p-value.
    *
    * The CCT method is particularly robust for combining dependent p-values and handles extremely small or large
    * p-values by applying appropriate transformations. When small p-values are present (below `thrSmallP`), it uses 
    * a special approximation to avoid numerical instability.
    *
    * @param pvals      Vector of p-values to be combined.
    * @param thrLargeP  Threshold for clipping large p-values (default: 0.9). Any p-value above this is capped.
    * @param thrSmallP  Threshold below which p-values are considered extremely small (default: 1e-15).
    * @return           A Map object containing:
    *                     - "cct": The CCT statistic (combined transformed value).
    *                     - "pval_cct": The final p-value computed from the CCT statistic using the Cauchy distribution.
    */
  def cctTest(pvals: BDV[Double], thrLargeP: Double = 0.9, thrSmallP: Double = 1e-15): Map[String, Any] = {
    // Replace large p-values
    val clippedPvals = pvals.map(p => if (p > thrLargeP) thrLargeP else p)

    // Indicator for small p-values
    val isSmall: BDV[Boolean] = clippedPvals.map(_ < thrSmallP)

    // CCTSTAT initially set to PVAL values
    val CCTSTAT = clippedPvals.copy

    val cct: Double = if (!any(isSmall)) {
      mean(tan((0.5 - CCTSTAT) * Pi))  // If no small p-values, use regular transformation
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

  /**
    * Performs the Omni SgZ Test, a unified framework that combines multiple SgZ statistics
    * (including GFisher and Burden-type tests) with an overall Cauchy Combination Test (CCT).
    *
    * @param            Zscores Vector of Z-scores corresponding to genetic variants or features.
    * @param DF         Matrix specifying degrees of freedom per test. Use `Double.PositiveInfinity` for burden tests.
    * @param W          Weight matrix; each row defines a weight set used in testing.
    * @param M          Covariance matrix of decorrelated scores (used in SgZ test).
    * @param oneSided   If `true`, uses one-sided transformations; otherwise, uses two-sided.
    * @param calcuP     Whether to calculate p-values in addition to test statistics.
    * @param isPosiWts  If `true`, assumes weights are non-negative (used in weighted statistics).
    * @return           A Map object containing:
    *                     - "STAT": Matrix of test statistics (one per weight set plus CCT).
    *                     - "PVAL": Matrix of corresponding p-values.
    *                     - "cct": The CCT statistic summarizing all tests.
    *                     - "pval_cct": P-value from the CCT statistic.
    */
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

  /**
    * Performs the Burden-SKAT-Fisher (BSF) omnibus test by combining optimal weighting schemes
    * across three distinct types of tests: Burden (linear), SKAT (quadratic), and Fisher (p-value combination).
    *
    * For each test type, optimal weights are computed using the `optimalWeightsM` function and combined
    * with equal weights to ensure robustness. Test statistics are then calculated using the `omni_SgZ_test`,
    * which aggregates across the three test families using the Cauchy Combination Test (CCT).
    *
    * @param Zscores           Vector of Z-scores corresponding to genetic variants or features.
    * @param M                 Covariance matrix of decorrelated scores (used to compute test statistics).
    * @param Bstar             Vector of score magnitudes or test statistics used in weight optimization.
    * @param PI                Vector of posterior inclusion probabilities or prior weights for the variants.
    * @param additionalParams  Additional parameters (not used here, but available for interface flexibility).
    *
    * @return                  A Map object containing:
    *                           - "STAT": Matrix of computed test statistics for all weight sets.
    *                           - "PVAL": Matrix of corresponding p-values.
    */
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

  /**
    * Wrapper for the Burden-SKAT-Fisher (BSF) omnibus test that operates on p-values and Z-score signs
    * instead of raw Z-scores. Converts two-sided p-values to signed Z-scores using the inverse normal quantile function.
    *
    * This allows integration with tools or pipelines that provide p-values but not Z-scores directly.
    *
    * @param Pvalues  Vector of two-sided p-values corresponding to genetic variants or features.
    * @param Zsigns   Vector of Z-score signs (+1 or -1) to recover signed test statistics.
    * @param M        Covariance matrix of decorrelated scores (used to compute test statistics).
    * @param Bstar    Vector of score magnitudes or test statistics used in weight optimization.
    * @param PI       Vector of posterior inclusion probabilities or prior weights for the variants.
    *
    * @return         A Map object containing:
    *                   - "STAT": Matrix of computed test statistics for all weight sets.
    *                   - "PVAL": Matrix of corresponding p-values.
    */
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

  /**
    * Performs the Burden-SKAT-Fisher (BSF) omnibus test and augments it with two additional 
    * Cauchy Combination Test (CCT) p-values to enhance power and robustness.
    *
    * This test includes:
    *   1. The standard BSF test statistics and p-values.
    *   2. A global CCT p-value computed from the input Z-scores.
    *   3. A second-stage CCT p-value combining the BSF p-values and the global CCT p-value.
    *
    * @param Zscores  Vector of signed Z-scores corresponding to genetic variants or features.
    * @param M        Covariance matrix of decorrelated scores (used for test statistic calculation).
    * @param Bstar    Vector of score magnitudes used in weight optimization.
    * @param PI       Vector of posterior inclusion probabilities or prior weights for the variants.
    *
    * @return         A Map object containing:
    *                   - "STAT": Matrix of test statistics including BSF statistics, global CCT stat,
    *                             and second-stage BSF-CCT combination stat.
    *                   - "PVAL": Matrix of corresponding p-values.
    */
  def BSF_cctP_test (
    Zscores: BDV[Double],
    M: BDM[Double],
    Bstar: BDV[Double],
    PI: BDV[Double]
  ): Map[String, BDM[Double]] = {
    // conduct a BSF test
    val bsf = BSF_test(Zscores, M, Bstar, PI)

    // convert Zscore input into two-sided p-values, and then use these to calculate an overal CCT statistic and p-value
    val Pvalues = Zscores.map(z =>  2.0 * Normal.cumulative(-Math.abs(z), 0.0, 1.0, true, false))
    val cct = cctTest(Pvalues)
    val cctStat = cct("cct").asInstanceOf[Double]
    val cctP = cct("pval_cct").asInstanceOf[Double]

    // Extract STAT and PVAL matrices from the initial BSF test
    val STAT = bsf("STAT").asInstanceOf[BDM[Double]]
    val PVAL = bsf("PVAL").asInstanceOf[BDM[Double]]

    // Flatten the PVAL result, and then append the global CCT p-value
    val PVAL_Vec = PVAL.toDenseVector
    val newPVAL = BDV.vertcat(PVAL_Vec(0 until PVAL.size - 1), BDV(cctP))

    // Apply a second-stage CCT test on the combined BSF test p-values and the global CCT p-value
    val bsf_cctP = cctTest(newPVAL)
    val bsf_cctP_stat = bsf_cctP("cct").asInstanceOf[Double]
    val bsf_cctP_pval = bsf_cctP("pval_cct").asInstanceOf[Double]

    // Convert stats and p-values to 1x1 matrices for vertical concatenation
    val cctStatMat = BDM.fill(1, 1)(cctStat)
    val bsf_cctP_statMat = BDM.fill(1, 1)(bsf_cctP_stat)
    val cctPMat = BDM.fill(1, 1)(cctP)
    val bsf_cctP_pvalMat = BDM.fill(1, 1)(bsf_cctP_pval)

    // Concatenate original STAT matrix with CCT and BSF+CCT statistics
    val result_stat = BDM.vertcat(STAT, cctStatMat, bsf_cctP_statMat)
    // Concatenate original PVAL matrix with CCT p-value and BSF+CCT p-value
    val result_pval = BDM.vertcat(PVAL.asInstanceOf[BDM[Double]], cctPMat, bsf_cctP_pvalMat)

    Map("STAT" -> result_stat, "PVAL" -> result_pval)
  }

  /**
    * Performs the Burden-SKAT-Fisher (BSF) test with Cauchy Combination Test (CCT) enhancements using p-values and Z-score signs.
    *
    * @param Pvalues     Vector of two-sided p-values.
    * @param Zsigns      Vector of Z-score signs (+1 or -1), same length as `Pvalues`.
    * @param M           Matrix of genotype data or other annotation matrix used in test weight optimization.
    * @param Bstar       Vector of estimated effect sizes or related weights for variants.
    * @param PI          Vector of variant-level prior probabilities or weights.
    *
    * @return            A Map object containing:
    *                     - "STAT": a matrix of test statistics (BSF + CCT + BSF+CCT),
    *                     - "PVAL": a matrix of corresponding p-values.
    */
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