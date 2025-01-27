/*
This file contains main and supportive functions for computing PGFisher p-values
Reference: Zhang, Hong, and Zheyang Wu. "The generalized Fisher's combination and accurate p‐value
           calculation under dependence." Biometrics 79.2 (2023): 1159-1172.
Creators: Peter Howell and Kylie Hoar
Last update (latest update first):
  KHoar 2024-11-30: sample format for future edits
*/

package is.hail.methods.gfisher

import breeze.linalg.{DenseMatrix => BDM, DenseVector => BDV, _}
import breeze.numerics._
import breeze.stats.distributions._
import net.sourceforge.jdistlib.Gamma

object PGFisher {

  /*
  Helper functions
  */

  /**
    * Formatting function for Cholesky matrix calculated in PGFisher 'MR' method
    *
    * @param matrix n by n cholesky matrix
    * @return flipped n by n cholesky matrix
    */
  def flipLowerToUpper(matrix: BDM[Double]): BDM[Double] = {
    val rows = matrix.rows
    val cols = matrix.cols
    val flippedMatrix = BDM.zeros[Double](rows, cols)
    for (i <- 0 until rows) {
      for (j <- 0 until cols) {
        if (i == j) {
          flippedMatrix(i, j) = matrix(i, j)
        } else {
          flippedMatrix(i, j) = matrix(j, i)
          flippedMatrix(j, i) = matrix(i, j)
        }
      }
    }
    flippedMatrix
  }

  /**
    * Function that properly formats the nsim and seed Option inputs for pGFisher
    *
    * @param nsim Option[Int] = number of simulation used in the "MR" method for pGFisher, default = 5e4
    * @param seed Option[Int] = seed for random number generation, default = None
    * @return flipped array of formatted ('initialized') nsim and seed inputs
    */
  def initializeParams(nsim: Option[Int], seed: Option[Int]): (Int, Option[Int]) = {
    val initializedNsim = nsim.getOrElse(50000)
    val initializedSeed = seed match {
      case Some(value) => Some(value)
      case None => None
    }
    (initializedNsim, initializedSeed)
  }

  /**
    * Nearest positive definite matrix calculation through eigen decomposition
    *
    * @param matrix n by n cholesky matrix
    * @return n x n NPD matrix
    */
  def nearestPositiveDefinite(matrix: BDM[Double]): BDM[Double] = {
    val symMatrix = (matrix + matrix.t) / 2.0
    val eigSym.EigSym(eigenvalues, eigenvectors) = eigSym(symMatrix)
    val positiveEigenvalues = eigenvalues.map(ev => if (ev > 0) ev else 1e-10)
    val positiveDefiniteMatrix = eigenvectors * diag(positiveEigenvalues) * eigenvectors.t
    (positiveDefiniteMatrix + positiveDefiniteMatrix.t) / 2.0
  }

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

  /**
    * Function to find the standard deviation of a dense vector
    *
    * @param v n dense vector with numeric values
    * @return standard deviation of the dense vector
    */
  def stdDev[T: Numeric](v: BDV[T]): Double = {
    val num = implicitly[Numeric[T]]
    val vectorDoubles = v.map(num.toDouble)
    val vectorMean = mean(v)
    math.sqrt(sum(vectorDoubles.map(x => math.pow(x - vectorMean, 2))) / v.length)
  }

  /*
  Main functions
  */

  /**
    * Get P-values using moment ratio matching by quadratic approximation
    *
    * @param q GFisher test statistic
    * @param df n-dimensional vector of degrees of freedom
    * @param w n-dimensional vector of weights
    * @param M n by n correlation matrix
    * @return the p-value of the GFisher test
    * @example insert example here!
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

  /**
    * Get P-values using moment ratio matching by quadratic approximation
    *
    * @param q GFisher test statistic
    * @param df n-dimensional vector of degrees of freedom
    * @param w n-dimensional vector of weights
    * @param M n by n correlation matrix
    * @param pType "two" = two-sided input p-values, "one" = one-sided input p-values
    * @param nsimOpt number of simulation used in the "MR" method for pGFisher, default = 5e4
    * @param seedOpt seed for random number generation, default = None
    * @return the p-value of the GFisher test
    * @example insert example here!
    */
  def pGFisherMR(
    q: Double,
    df: BDV[Int],
    w: BDV[Double],
    M: BDM[Double],
    pType: String = "two",
    nsimOpt: Option[Int] = None,
    seedOpt: Option[Int] = None
  ) : Double = {
      val MChol = try {
        cholesky(M)
      } catch {
        case _: Exception =>
          val nearPDM = nearestPositiveDefinite(M)
          cholesky(nearPDM)
      }
      val flipped = flipLowerToUpper(MChol)
      val n = M.rows
      val (nsim, seed) = initializeParams(nsimOpt, seedOpt)
      val rand_mat = if (seed != None) {
        val intSeed: Int = seed.getOrElse(0)
        val rng = RandBasis.withSeed(intSeed)
        BDM.rand(nsim, n, Gaussian(0.0, 1.0)(rng))
      } else {
        BDM.rand(nsim, n, Gaussian(0.0, 1.0))
      }
      val znull = rand_mat * flipped
      val pnull = if (pType == "two") {
        znull.map(z => 2 * Gaussian(0.0, 1.0).cdf(-math.abs(z)))
      } else {
        znull.map(z => Gaussian(0.0, 1.0).cdf(z))
      }
      val ppTrans = if (stdDev(df) == 0.0) {
        if (df.forall(_ == 2.0)) {
          -2.0 * log(pnull)
        } else {
          BDM.tabulate(pnull.rows, pnull.cols) { (i, j) =>
            val chiSq = new ChiSquared(df(0))
            chiSq.inverseCdf(1 - pnull(i, j))
          }
        }
      } else {
        BDM.tabulate(pnull.rows, pnull.cols) { (i, j) =>
          val chiSq = new ChiSquared(df(j))
          chiSq.inverseCdf(1 - pnull(i, j))
        }
      }
      val fisherNull = ppTrans(*, ::).map(row => sum(row *:* w))
      val MM = (1 to 4).map(k => mean(fisherNull.map(math.pow(_, k))))
      val mu = MM(0)
      val sigma2 = (MM(1) - math.pow(MM(0), 2)) * nsim / (nsim - 1).toDouble
      val cmu3 = MM(2) - 3 * MM(1) * MM(0) + 2 * math.pow(MM(0), 3)
      val cmu4 = MM(3) - 4 * MM(2) * MM(0) + 6 * MM(1) * math.pow(MM(0), 2) - 3 * math.pow(MM(0), 4)
      val gm = cmu3 / math.pow(sigma2, 1.5)
      val kp = cmu4 / math.pow(sigma2, 2)
      val a = 9 * math.pow(gm, 2) / math.pow(kp - 3, 2)
      val z = (q - mu) / math.sqrt(sigma2) * math.sqrt(a) + a
      return Gamma.cumulative(z, a, 1.0, false, false)
  }

  /**
    * Get P-values using brown's method with calculated variance
    *
    * @param q GFisher test statistic
    * @param df n-dimensional vector of degrees of freedom
    * @param w n-dimensional vector of weights
    * @param M n by n correlation matrix
    * @param GM n by n correlation matrix between w_1 T_1, ..., w_n T_n, which is the output of getGFisherGM
    */
  def pGFisherGB(
    q: Double,
    df: BDV[Int],
    w: BDV[Double],
    M: BDM[Double]
  ): Double = {
    w := w / sum(w)
    val GM: BDM[Double] = GFisherGM.getGFisherGM(df, w, M, false)
    val mu: Double = w dot convert(df, Double)
    val sigma2: Double = sum(GM)
    val a: Double = math.pow(mu, 2.0) / sigma2
    val x: Double = (q - mu) * math.sqrt(a) / math.sqrt(sigma2) + a
    return Gamma.cumulative(x, a, 1.0, false, false)
  }

}
