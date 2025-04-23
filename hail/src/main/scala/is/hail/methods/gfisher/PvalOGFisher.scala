/*
This file contains the function for computing the oGFisher test based on a vector of p-values
Reference: Zhang, Hong, and Zheyang Wu. "The generalized Fisher's combination and accurate p‐value
           calculation under dependence." Biometrics 79.2 (2023): 1159-1172.
Creators: Kylie Hoar
Last update (latest update first):
  KHoar 2025-04-23: Added docstrings and internal comments
*/

package is.hail.methods.gfisher

import breeze.linalg.{DenseMatrix => BDM, DenseVector => BDV}
import breeze.stats.distributions.{Gaussian, MultivariateGaussian, CauchyDistribution}

object PvalOGFisher {

  /**
    * Compute the oGFisher test based on a vector of p-values
    *
    * Adapted from the GFisher R package ("~/GFisher/GFisher_v2.R"), specifically the function pval.oGFisher
    * 
    * @param p          a vector of input p-values of the oGFisher test
    * @param DF         matrix of degrees of freedom for inverse chi-square transformation for each p-value. Each row represents a GFisher test. It allows a matrix of one column, indicating the same degrees of freedom for all p-values's chi-square transformations.
    * @param W          matrix of weights. Each row represents a GFisher test.
    * @param M          correlation matrix of the input Zscores from which the input p-values were obtained.
    * @param one_sided  "false" = two-sided, "true" = one-sided input p-values.
    * @param method     "MR" = simulation-assisted moment ratio matching, 
    *                   "HYB" = moment ratio matching by quadratic approximation, 
    *                   "GB" = Brown's method with calculated variance.
    * @param combine    "cct" = oGFisher using the Cauchy combination method, 
    *                   "mvn" = oGFisher using multivariate normal distribution.
    * @param nsim       number of simulation used in the "MR" method, default = 5e4
    * @param seed       seed for random number generation, default = NULL
    *
    * @return           A Map object containing:
    *                     - "stat": the statistic of the oGFisher test, i.e., the cct or the minp statistic 
    *                     - "pval": Matrix of corresponding p-values.
    *                     - "pval_indi": the individual p-values of the involved GFisher tests
    *                     - "stat_indi": the involved GFisher statistics 
    */
  def pvalOGFisher(
    p: BDV[Double],
    DF: BDM[Double],
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
