package is.hail.methods.gfisher

import breeze.linalg.{DenseMatrix => BDM, DenseVector => BDV}
import breeze.numerics._
import net.sourceforge.jdistlib.Normal

object GLOW_Omni_byP {
    def GLOW_Omni_byP (
        Zscores: Option[BDM[Double]] = None,
        Pvalues: Option[BDV[Double]] = None,
        Effects: Option[BDV[Double]],
        SE: BDV[Double],
        M: BDM[Double],
        B: BDV[Double],
        PI: BDV[Double]
    ): Map[String, BDM[Double]] = {
        val finalZscores = Zscores.getOrElse {
            if (Effects.isEmpty) throw new IllegalArgumentException("Effect is required to get signs of Z scores for burden test")
            val Zsigns = signum(Effects.get)
            val PvaluesVec = Pvalues.getOrElse(throw new IllegalArgumentException("Pvalues is None"))
            PvaluesVec.map(p => Normal.quantile(1.0 - (p / 2.0), 0.0, 1.0, false, false)) *:* Zsigns
        }

        val finalZscoresVec = finalZscores match {
            case v: BDV[Double] => v  // Already a DenseVector
            case m: BDM[Double] if m.cols == 1 => m.toDenseVector  // Convert single-column matrix to vector
            case _ => throw new IllegalArgumentException("finalZscores must be a DenseVector[Double]")
        }

        val Bstar = B /:/ SE // Element-wise division

        FuncCalcuCombTests.BSF_cctP_test(Pvalues.get, finalZscoresVec, M, Bstar, PI)
    }
}