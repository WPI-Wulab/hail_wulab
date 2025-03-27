package is.hail.methods.gfisher

import breeze.linalg.{DenseMatrix => BDM, DenseVector => BDV, _}
import breeze.numerics._

// where do the P-values come from?

object GLOW_Omni {
    def GLOW_Omni (
        Pvalues: BDV[Double],
        Zout: Map[String, Any],
        B: BDV[Double],
        PI: BDV[Double],
        additionalParams: Any*
    ): Map[String, BDM[Double]] = {

        val M = Zout("M_Z").asInstanceOf[BDM[Double]]
        val s0 = Zout("s0").asInstanceOf[Double]
        val Bstar = (sqrt(diag(Zout("M_s").asInstanceOf[BDM[Double]])) * B.asInstanceOf[BDV[Double]]) / s0
        val Zscores = Zout("Zscores").asInstanceOf[BDV[Double]]

        FuncCalcuCombTests.BSF_cctP_test(Pvalues, Zscores, M, Bstar, PI)
    }
}