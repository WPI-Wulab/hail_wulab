/*
This file contains a function to conduct a Burden test within the GLOW procedure
Reference: Zhang, Hong, and Zheyang Wu. "The generalized Fisher's combination and accurate p‐value
           calculation under dependence." Biometrics 79.2 (2023): 1159-1172.
           Liu, Ming. "Integrative Analysis of Large Genomic Data." WPI (2025).
Creators: Kylie Hoar
Last update (latest update first):
  Claude Code (Opus 5.5), prompted by VPanwar, 2026-11-02: optimalWeightsM no longer has a `fisher`
    parameter (GLOWr 0.3.4 port); the call passes its arguments by name.
  Claude Code (Opus 5), prompted by VPanwar, 2026-09-09: Passed fisher = false to optimalWeightsM.
    The previous `true` was wrong for the same reason as defect E1 in GLOW_SKAT.scala -- the flag
    substitutes the Fisher transform's null mean of 2.0 -- but harmless here, since burden = true
    short-circuits to optimalWeightsM_Burden before the flag is read. Corrected so it cannot become
    live under a later refactor; the group-E fixtures confirm this file's output is unchanged.
  KHoar 2025-04-23: Added comments and docstrings
*/

package is.hail.methods.gfisher

import breeze.linalg.{DenseMatrix => BDM, DenseVector => BDV, _}
import breeze.numerics._

object GLOW_Burden {

    /**
      * Burden test using optimal weights from GLOW procedure.
      *
      * Adapted from the GLOW R package ("GLOW_R_package/GLOW/R/GLOW_Burden.R")
      * 
      * @param Zout  Output from "getZMargScore" function, which requires G, X, Y information.
      * @param B     Numeric vector of B (effect size) for estimating optimal weights.
      * @param PI    Numeric vector of PI (causal likelihood) for estimating optimal weights.
      *
      * @return      A Map object containing:
      *                 - "STAT": a single column matrix of the involved test statistics.
      *                 - "PVAL": the p-values of the involved tests.
      */
    def GLOW_Burden (
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

        // Conduct Burden Test and get optimal weights
        val gBurden: Double => Double = x => x
        val statDFBurden = Double.PositiveInfinity
        // burden = true short-circuits to optimalWeightsM_Burden's closed form.
        val wtsOptBurden = OptimalWeights.optimalWeightsM(gBurden, Bstar, PI, M, burden = true,
          forcePositiveWeights = true)
        val WT_opt_burden = BDM.vertcat(wtsOptBurden, wtsEqu)
        val DF_opt_burden = BDM.fill(WT_opt_burden.rows, 1)(statDFBurden)

        // Run the omnibus test on the provided Z-scores
        val omniOpt = FuncCalcuCombTests.omni_SgZ_test(Zscores, DF_opt_burden, WT_opt_burden, M)

        val omniStat = omniOpt("STAT").asInstanceOf[BDM[Double]]
        val omniPval = omniOpt("PVAL").asInstanceOf[BDM[Double]]

        Map("STAT" -> omniStat, "PVAL" -> omniPval)
    }
}