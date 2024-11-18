package is.hail.methods.gfisher

import breeze.linalg.{DenseMatrix => BDM, DenseVector => BDV, _}


// import net.sourceforge.jdistlib.{ChiSquare, Normal}
import net.sourceforge.jdistlib.Gamma


object PGFisher {

  /**
    * Get P-values using moment ratio matching by quadratic approximation
    *
    * @param q GFisher test statistic
    * @param df n-dimensional vector of degrees of freedom
    * @param w n-dimensional vector of weights
    * @param M n by n correlation matrix
    * @param GM n by n correlation matrix between w_1 T_1, ..., w_n T_n, which is the output of getGFisherGM
    */
  def pGFisherHyb(
    q: Double,
    df: BDV[Int],
    w: BDV[Double],
    M: BDM[Double]
  ): Double = {
    w := w / sum(w)
    val GM: BDM[Double] = GFisherGM.getGFisherGM(df, w, M, false)
    val mu: Double = w dot convert(df, Double)
    val sigma2: Double = sum(GM)
    val lam: BDV[Double] = GFisherLambda.getGFisherLambda(df, w, M, GM)
    val (c2: Double, c3: Double, c4: Double) = (sum(lam ^:^ 2.0), sum(lam ^:^ 3.0), sum(lam ^:^ 4.0))
    val gm: Double = (math.sqrt(8.0) * c3) / math.pow(c2, 3.0/2.0)
    val kp: Double = 12.0 * c4 / math.pow(c2, 2.0) + 3.0
    val a: Double = 9.0 * math.pow(gm, 2.0) / math.pow((kp - 3.0), 2.0)
    val x: Double = (q - mu) * math.sqrt(a) / math.sqrt(sigma2) + a
    return Gamma.cumulative(x, a, 1.0, false, false)
  }

}
