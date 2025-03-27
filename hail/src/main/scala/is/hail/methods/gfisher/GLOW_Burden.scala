package is.hail.methods.gfisher

import breeze.linalg.{DenseMatrix => BDM, DenseVector => BDV, _}
import breeze.numerics._

object GLOW_Burden {
    def GLOW_Burden (
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

        // Burden Test
        val gBurden: Double => Double = x => x
        val statDFBurden = Double.PositiveInfinity
        val wtsOptBurden = OptimalWeights.optimalWeightsM(gBurden, Bstar, PI, M, true, true)
        val WT_opt_burden = BDM.vertcat(wtsOptBurden, wtsEqu)
        val DF_opt_burden = BDM.fill(WT_opt_burden.rows, 1)(statDFBurden)

        val omniOpt = FuncCalcuCombTests.omni_SgZ_test(Zscores, DF_opt_burden, WT_opt_burden, M)

        val omniStat = omniOpt("STAT").asInstanceOf[BDM[Double]]
        val omniPval = omniOpt("PVAL").asInstanceOf[BDM[Double]]

        Map("STAT" -> omniStat, "PVAL" -> omniPval)
    }
}