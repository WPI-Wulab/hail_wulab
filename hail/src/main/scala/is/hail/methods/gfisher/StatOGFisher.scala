package is.hail.methods.gfisher

import breeze.linalg.{DenseMatrix => BDM, DenseVector => BDV, _}
import scala.math._

object StatOGFisher {

    /*
    Helper Function
    */

    /**
    * Function to find the mean of a dense vector
    *
    * @param v n dense vector with numeric values
    * @return mean of the dense vector
    */
    def mean[T: Numeric](v: BDV[T]): Double = {
        val num = implicitly[Numeric[T]]
        sum(v.map(num.toDouble)) / v.length
    }

    /*
    Main function
    */

    /**
    * Compute the oGFisher related statistics
    *
    * @param p a vector of input p-values of the oGFisher test
    * @param DF matrix of degrees of freedom for inverse chi-square transformation for each p-value.
                Each row represents a GFisher test.
                It allows a matrix of one column, indicating the same degrees of freedom for all p-values's chi-square transformations.
    * @param W matrix of non-negative weights. Each row represents a GFisher test.
    * @param M n by n correlation matrix of the input Zscores from which the input p-values were obtained.
    * @param oneSided true = one-sided input p-values, false = two-sided input p-values
    * @param method "MR" = simulation-assisted moment ratio matching
                    "HYB" = moment ratio matching by quadratic approximation
                    "GB" = Brown's method with calculated variance. See details in the reference
    * @param nsimOpt number of simulation used in the "MR" method for pGFisher, default = 5e4
    * @param seedOpt seed for random number generation, default = None
    * @return the Cauchy combination statistics of PVAL
    * @example insert example here!
    */
    def statOGFisher(
        p: BDV[Double],
        DF: BDM[Int],
        W: BDM[Double],
        M: BDM[Double],
        oneSided: Boolean = false,
        method: String = "HYB",
        nsim: Option[Int] = None,
        seed: Option[Int] = None
    ): Map[String, Any] = {
        val STAT = (0 until DF.rows).map { i =>
            StatGFisher.statGFisher(p, DF(i, ::).t, W(i, ::).t)
        }.toArray
        val PVAL = STAT.zipWithIndex.map { case (stat, i) =>
            if (method == "HYB") PGFisher.pGFisherHyb(stat, DF(i, ::).t, W(i, ::).t, M)
            else if (method == "MR") PGFisher.pGFisherMR(stat, DF(i, ::).t, W(i, ::).t, M, oneSided)
            else PGFisher.pGFisherGB(stat, DF(i, ::).t, W(i, ::).t, M, oneSided)
        }
        val minp = PVAL.min
        val adjustedPVAL = BDV(PVAL.map(pval => if (pval > 0.9) 0.9 else pval))
        val isSmall = adjustedPVAL.map(_ < 1e-15)
        val CCTSTAT = BDV.tabulate(adjustedPVAL.length) { i =>
            if (isSmall(i)) 1 / (adjustedPVAL(i) * Pi) else tan((0.5 - adjustedPVAL(i)) * Pi)
        }
        val cct = mean(CCTSTAT)
        Map(
            "STAT" -> BDV(STAT),
            "PVAL" -> adjustedPVAL,
            "minp" -> minp,
            "cct" -> cct
        )
    }
}
