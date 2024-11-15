package is.hail.methods.gfisher

import breeze.linalg.{DenseVector => BDV, _}
import breeze.numerics.log
import net.sourceforge.jdistlib.ChiSquare

object StatGFisher {

  /**
    * Compute the GFisher test statistic of a set of p-values
    *
    * @param p vector of p-values
    * @param df vector of degrees of freedom
    * @param w vector of weights
    */
  def statGFisher(
    p: BDV[Double],
    df: BDV[Int],
    w: BDV[Double],
  ): Double = {
    // transformed p values
    val pT: BDV[Double] = if (all(df.map(_ == 2))) {
      -2.0 * log(p)
    } else {
      BDV((0 until p.size).map((i) => {ChiSquare.quantile(p(i), df(i), false, false)}).toArray)
    }
    w := w / sum(w)
    val fishStat: Double = w dot pT
    return fishStat
  }

}
