package is.hail.methods.gfisher

import breeze.linalg.{DenseMatrix => BDM, DenseVector => BDV, _}
import breeze.numerics._

object GLOW_Fisher {
    def GLOW_Fisher (
        Zout: Map[String, Any],
        B: BDV[Double],
        PI: BDV[Double],
        additionalParams: Any*
    ): Map[String, BDM[Double]] = {

        val M = Zout("M_Z").asInstanceOf[BDM[Double]]
        val wtsEqu = BDM.ones[Double](1, M.cols)
        val s0 = Zout("s0").asInstanceOf[Double]
        val Bstar = (sqrt(diag(Zout("M_s").asInstanceOf[BDM[Double]])) * B.asInstanceOf[BDV[Double]]) / s0
        val Zscores = Zout("Zscores").asInstanceOf[BDV[Double]]

        // Fisher Test
        val gFisher: Double => Double = x => FuncCalcuCombTests.g_GFisher_two(x, 2)
        val statDFFisher = 2.0
        val wtsOptFisher = OptimalWeights.optimalWeightsM(gFisher, Bstar, PI, M, false, true)
        val WT_opt_fisher = BDM.vertcat(wtsOptFisher, wtsEqu)
        val DF_opt_fisher = BDM.fill(WT_opt_fisher.rows, 1)(statDFFisher)

        val omniOpt = FuncCalcuCombTests.omni_SgZ_test(Zscores, DF_opt_fisher, WT_opt_fisher, M)

        val omniStat = omniOpt("STAT").asInstanceOf[BDM[Double]]
        val omniPval = omniOpt("PVAL").asInstanceOf[BDM[Double]]

        Map("STAT" -> omniStat, "PVAL" -> omniPval)
    }
}