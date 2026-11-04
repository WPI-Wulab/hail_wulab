/*
This file contains a function to conduct InteGrative anaLysis with Optimal Weights (GLOW)
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

object GLOW_Omni {

    /**
      * A testing procedure that contains Burden, SKAT, Fisher with their optimal & equal weights and the CCT of them. Use Z-scores as input.
      *
      * Adapted from the GLOW R package ("GLOW_R_package/GLOW/R/GLOW_Omni.R")
      *
      * @param Zout  Output from "getZMargScore" or "getZ_marg_score_binary_SPA", which requires G, X, Y information.
      * @param B     Numeric vector of B (effect size) for estimating optimal weights.
      * @param PI    Numeric vector of PI (causal likelihood) for estimating optimal weights.
      *
      * @return      A Map object containing:
      *                 - "STAT": a list of GLOW statistics.
      *                 - "PVAL": the p-values of the involved tests.
      */
    def GLOW_Omni (
        Zout: Map[String, Any],
        B: BDV[Double],
        PI: BDV[Double],
        additionalParams: Any*
    ): Map[String, BDM[Double]] = {

        // Collect outputs from the "getZMargScore" function input Zout
        val M = Zout("M_Z").asInstanceOf[BDM[Double]]
        val s0 = Zout("s0").asInstanceOf[Double]
        val Bstar = (sqrt(diag(Zout("M_s").asInstanceOf[BDM[Double]])) *:* B.asInstanceOf[BDV[Double]]) / s0
        val Zscores = Zout("Zscores").asInstanceOf[BDV[Double]]

        // Run the omnibus CCT test on the provided Z-scores
        FuncCalcuCombTests.BSF_cctP_test(Zscores, M, Bstar, PI)
    }
}
