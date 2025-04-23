/*
This file contains the function for calculating the GFisher test statistics
Reference: Zhang, Hong, and Zheyang Wu. "The generalized Fisher's combination and accurate p‐value
           calculation under dependence." Biometrics 79.2 (2023): 1159-1172.
           Liu, Ming. "Integrative Analysis of Large Genomic Data." WPI (2025).
Creators: Kylie Hoar
Last update (latest update first):
  KHoar 2025-04-23: Added docstrings and internal comments
*/

package is.hail.methods.gfisher

import breeze.linalg.{DenseVector => BDV, _}
import breeze.numerics.log
import net.sourceforge.jdistlib.ChiSquare

object StatGFisher {

  /**
    * Compute the GFisher test statistics S = sum_i w_i F^{-1}_{d_i}(1-P_i), based on a vector of p-values P_i's, degrees of freedom d_i's, and weights w_i's. F^{-1}_{d_i} is the inverse CDF of the chi-square distribution with d_i degrees of freedom.
    *
    * Adapted from the GLOW R package ("GLOW_R_package/GLOW/R/helpers_GFisher.R"), specifically the function stat.GFisher
    * 
    * @param p   Vector of input p-values of the GFisher test
    * @param df  Degrees of freedom for inverse chi-square transformation for each p-value
    *            It can be a vector of the same length as p, indicating each transformation function might have a different df, or a single number, indicating the same degrees of freedom for all.
    * @param w   Vector of non-negative weights for each p-value. Default is 1 (equal weights).
    * 
    * @return    GFisher test statistics
    */
  def statGFisher(
    p: BDV[Double],
    df: BDV[Double],
    w: BDV[Double],
  ): Double = {
    // If all degrees of freedom are 2, use the fast shortcut: -2 * log(p) (this is standard Fisher's method).
    val pT: BDV[Double] = if (df.forall(_ == 2)) {
      -2.0 * log(p)
    } 
    // Otherwise, apply the inverse chi-square CDF to each p-value with its corresponding df.
    else {
      BDV((0 until p.size).map((i) => {ChiSquare.quantile(p(i), df(i), false, false)}).toArray)
    }

    // Normalize the weights to ensure they sum to 1 (optional, depending on GFisher version).
    w := w / sum(w)

    // Compute the weighted sum of transformed p-values — the GFisher statistic.
    val fishStat: Double = w dot pT
    return fishStat
  }

}
