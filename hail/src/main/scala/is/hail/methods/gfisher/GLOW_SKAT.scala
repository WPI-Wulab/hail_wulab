/*
This file contains a function to conduct a SKAT test within the GLOW procedure
Reference: Zhang, Hong, and Zheyang Wu. "The generalized Fisher's combination and accurate p‐value
           calculation under dependence." Biometrics 79.2 (2023): 1159-1172.
           Liu, Ming. "Integrative Analysis of Large Genomic Data." WPI (2025).
Creators: Kylie Hoar
Last update (latest update first):
  Claude Code (Opus 5), prompted by VPanwar, 2026-09-09: Fixed defect E1, found by the group-E
    golden-fixture suite (packages/glow-core in the GLOWh repo). optimalWeightsM was called with
    fisher = true, the flag that replaces the integrated null mean E_gX_p(g, 0, 1, 1) with the
    constant 2.0. That constant is the null mean of the Fisher transform, not of SKAT's g(x) = x^2
    (null mean 1), so every SKAT weight was computed against the wrong null and the statistic was
    wrong by 1.3% to 26% across GLOWr's six bundled validation scenarios. BSF_test
    (Func_Calcu_Comb_Tests.scala:388) always made this call correctly.
    Mirrors: GLOWr::GLOW_SKAT, which passes g = function(x) x^2 to Optimal_Weights_M and has no
    such flag.
  KHoar 2025-04-23: Added comments and docstrings
*/

package is.hail.methods.gfisher

import breeze.linalg.{DenseMatrix => BDM, DenseVector => BDV, _}
import breeze.numerics._

object GLOW_SKAT {

    /**
      * SKAT test using optimal weights from GLOW procedure.
      *
      * Adapted from the GLOW R package ("GLOW_R_package/GLOW/R/GLOW_SKAT.R")
      * 
      * @param Zout  Output from "getZMargScore" function, which requires G, X, Y information.
      * @param B     Numeric vector of B (effect size) for estimating optimal weights.
      * @param PI    Numeric vector of PI (causal likelihood) for estimating optimal weights.
      *
      * @return      A Map object containing:
      *                 - "STAT": a single column matrix of the involved test statistics.
      *                 - "PVAL": the p-values of the involved tests.
      */
    def GLOW_SKAT (
        Zout: Map[String, Any],
        B: BDV[Double],
        PI: BDV[Double],
        additionalParams: Any*
    ): Map[String, BDM[Double]] = {

        // Collect outputs from the "getZMargScore" function input Zout
        val M = Zout("M_Z").asInstanceOf[BDM[Double]]
        val wtsEqu = BDM.ones[Double](1, M.cols)
        val s0 = Zout("s0").asInstanceOf[Double]
        val Bstar = (sqrt(diag(Zout("M_s").asInstanceOf[BDM[Double]])) * B.asInstanceOf[BDV[Double]]) / s0
        val Zscores = Zout("Zscores").asInstanceOf[BDV[Double]]

        // Conduct SKAT Test and get optimal weights
        val gSKAT: Double => Double = x => x * x
        val statDFSKAT = 1.0
        // fisher = FALSE. The sixth argument replaces the integrated null mean
        // E_gX_p(g, 0, 1, 1) with the constant 2.0, which is the null mean of the
        // Fisher transform (a chi-square quantile with 2 df) and NOT of SKAT's
        // g(x) = x^2, whose null mean is 1. Passing true here made every SKAT
        // weight -- and the statistic built on it -- wrong by 1.3% to 26%.
        // BSF_test (Func_Calcu_Comb_Tests.scala:388) always made this call
        // correctly; this file did not. R has no such flag: Optimal_Weights_M
        // integrates the null mean for whatever g it is given.
        val wtsOptSKAT = OptimalWeights.optimalWeightsM(gSKAT, Bstar, PI, M, false, false)
        val WT_opt_skat = BDM.vertcat(wtsOptSKAT, wtsEqu)
        val DF_opt_skat = BDM.fill(WT_opt_skat.rows, 1)(statDFSKAT)

        // Run the omnibus test on the provided Z-scores
        val omniOpt = FuncCalcuCombTests.omni_SgZ_test(Zscores, DF_opt_skat, WT_opt_skat, M)

        val omniStat = omniOpt("STAT").asInstanceOf[BDM[Double]]
        val omniPval = omniOpt("PVAL").asInstanceOf[BDM[Double]]

        Map("STAT" -> omniStat, "PVAL" -> omniPval)
    }
}