/*
This file contains a function to conduct a Fisher test within the GLOW procedure
Reference: Zhang, Hong, and Zheyang Wu. "The generalized Fisher's combination and accurate p‐value
           calculation under dependence." Biometrics 79.2 (2023): 1159-1172.
           Liu, Ming. "Integrative Analysis of Large Genomic Data." WPI (2025).
Creators: Kylie Hoar
Last update (latest update first):
  KHoar 2025-04-23: Added comments and docstrings
*/

package is.hail.methods.gfisher

import breeze.linalg.{DenseMatrix => BDM, DenseVector => BDV, _}
import breeze.numerics._

object GLOW_Fisher {

    /**
      * Fisher test using optimal weights from GLOW procedure.
      *
      * Adapted from the GLOW R package ("GLOW_R_package/GLOW/R/GLOW_Fisher.R")
      * 
      * @param Zout  Output from "getZMargScore" function, which requires G, X, Y information.
      * @param B     Numeric vector of B (effect size) for estimating optimal weights.
      * @param PI    Numeric vector of PI (causal likelihood) for estimating optimal weights.
      *
      * @return      A Map object containing:
      *                 - "STAT": a single column matrix of the involved test statistics.
      *                 - "PVAL": the p-values of the involved tests.
      */
    def GLOW_Fisher (
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

        // Conduct Fisher Test and get optimal weights
        val gFisher: Double => Double = x => FuncCalcuCombTests.g_GFisher_two(x, 2)
        val statDFFisher = 2.0
        val wtsOptFisher = OptimalWeights.optimalWeightsM(gFisher, Bstar, PI, M, false, true)
        val WT_opt_fisher = BDM.vertcat(wtsOptFisher, wtsEqu)
        val DF_opt_fisher = BDM.fill(WT_opt_fisher.rows, 1)(statDFFisher)

        // Run the omnibus test on the provided Z-scores
        val omniOpt = FuncCalcuCombTests.omni_SgZ_test(Zscores, DF_opt_fisher, WT_opt_fisher, M)

        val omniStat = omniOpt("STAT").asInstanceOf[BDM[Double]]
        val omniPval = omniOpt("PVAL").asInstanceOf[BDM[Double]]

        Map("STAT" -> omniStat, "PVAL" -> omniPval)
    }
}