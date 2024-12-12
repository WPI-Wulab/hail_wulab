package is.hail.methods.gfisher

import breeze.linalg.{DenseMatrix => BDM, DenseVector => BDV, _}
import scala.math.{atan, Pi}
import breeze.stats.distributions.{Gaussian, CauchyDistribution, MultivariateGaussian}

object PvalOGFisher {

    def pvalOGFisher(
        p: BDV[Double],
        DF: BDM[Int],
        W: BDM[Double],
        M: BDM[Double],
        pType: String = "two",
        method: String = "HYB",
        combine: String = "cct",
        nsim: Option[Int] = None,
        seed: Option[Int] = None
    ): Map[String, Any] = {
        
        // Call statOGFisher to compute intermediate results
        val out = StatOGFisher.statOGFisher(p, DF, W, M, pType, method, nsim, seed)
        val stat = combine match {
            case "cct" =>
            val cctStat = out("cct").asInstanceOf[Double]
            val cauchy = new CauchyDistribution(median = 0.0, scale = 1.0)
            val pval = 1 - cauchy.cdf(cctStat)
            Map("stat" -> cctStat, "pval" -> pval)

            case "mvn" =>
            val minpStat = out("minp").asInstanceOf[Double]
            val nd = DF.rows
            val corGFisher = GFisherCor.getGFisherCor(DF, W, M, pType = pType)
            val regularizedCor = corGFisher + BDM.eye[Double](corGFisher.rows) * 1e-5
            val mean = BDV.zeros[Double](nd)
            val lower = BDV.fill(nd)(Double.NegativeInfinity)
            val upper = BDV.fill(nd)(Gaussian(0, 1).inverseCdf(1 - minpStat))
            val mvn = MultivariateGaussian(mean, regularizedCor)

            // Monte Carlo simulation parameters
            val numSamples = 100000
            // Generate samples from the multivariate normal distribution
            val samples = mvn.sample(numSamples)
            // Count how many samples are less than the upper quantile in each dimension
            val countInRegion = samples.count(sample => (sample <:< upper).forall(_ == true))
            // Estimate the probability
            val prob = countInRegion.toDouble / numSamples

            val mvnPval = 1.0 - prob
            Map("stat" -> minpStat, "pval" -> mvnPval)
            case _ =>
            throw new IllegalArgumentException(s"Unknown combination method: $combine")
        }

        // Return the result with individual statistics
        Map(
            "stat" -> stat("stat"),
            "pval" -> stat("pval"),
            "pval_indi" -> out("PVAL"),
            "stat_indi" -> out("STAT")
        )
    }
}