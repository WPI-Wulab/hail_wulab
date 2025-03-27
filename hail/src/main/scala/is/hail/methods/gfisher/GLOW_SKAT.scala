package is.hail.methods.gfisher

import breeze.linalg.{DenseMatrix => BDM, DenseVector => BDV, _}
import breeze.numerics._

object GLOW_SKAT {
    def GLOW_SKAT (
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

        // SKAT Test
        val gSKAT: Double => Double = x => x * x
        val statDFSKAT = 1.0
        val wtsOptSKAT = OptimalWeights.optimalWeightsM(gSKAT, Bstar, PI, M, false, true)
        val WT_opt_skat = BDM.vertcat(wtsOptSKAT, wtsEqu)
        val DF_opt_skat = BDM.fill(WT_opt_skat.rows, 1)(statDFSKAT)

        val omniOpt = FuncCalcuCombTests.omni_SgZ_test(Zscores, DF_opt_skat, WT_opt_skat, M)

        val omniStat = omniOpt("STAT").asInstanceOf[BDM[Double]]
        val omniPval = omniOpt("PVAL").asInstanceOf[BDM[Double]]

        Map("STAT" -> omniStat, "PVAL" -> omniPval)
    }
}