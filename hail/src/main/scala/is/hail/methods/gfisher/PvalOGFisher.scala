package is.hail.methods.gfisher

import breeze.linalg.{DenseMatrix => BDM, DenseVector => BDV}
import breeze.stats.distributions.{Gaussian, MultivariateGaussian, CauchyDistribution}

object PvalOGFisher {

  def pvalOGFisher(
    p: BDV[Double],
    DF: BDM[Int],
    W: BDM[Double],
    M: BDM[Double],
    one_sided: Boolean = false,
    method: String = "HYB",
    combine: String = "cct",
    nsim: Option[Int] = None,
    seed: Option[Int] = None
  ): Map[String, Any] = {

    // Call statOGFisher to compute intermediate results
    val out = StatOGFisher.statOGFisher(p, DF, W, M, one_sided, method, nsim, seed)
    val stat = combine match {
      case "cct" =>
        val cctStat = out("cct").asInstanceOf[Double]
        // No cdf for cauchy in jdistlib, so I need to use breeze here
        val cauchy = new CauchyDistribution(median = 0.0, scale = 1.0)
        val pval = 1 - cauchy.cdf(cctStat)
        Map("stat" -> cctStat, "pval" -> pval)

      case "mvn" =>
        val minpStat = out("minp").asInstanceOf[Double]
        val nd = DF.rows
        val corGFisher = GFisherCor.getGFisherCor(DF, W, M, one_sided)
        val regularizedCor = corGFisher + BDM.eye[Double](corGFisher.rows) * 1e-5
        val mean = BDV.zeros[Double](nd)
        // val lower = BDV.fill(nd)(Double.NegativeInfinity)
        // No Multivariate Gaussian in jdistlib so I am using the breeze implementation
        val mvn = MultivariateGaussian(mean, regularizedCor)

        // Monte Carlo simulation parameters
        val numSamples = 100000
        // Generate samples from the multivariate normal distribution
        val samples = mvn.sample(numSamples)
        // Count how many samples are less than the upper quantile in each dimension
        // No inverseCdf method in jdistlib like there is in breeze, therefore breeze package is used here
        val upper = BDV.fill(nd)(new Gaussian(0, 1).inverseCdf(1 - minpStat))
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
