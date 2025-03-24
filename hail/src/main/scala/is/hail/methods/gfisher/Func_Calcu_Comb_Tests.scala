package is.hail.methods.gfisher

import breeze.linalg.{DenseMatrix => BDM, DenseVector => BDV, _}
import breeze.stats._
import scala.util.Random
import net.sourceforge.jdistlib.{Normal, ChiSquare, Cauchy}
import breeze.stats.distributions.Gaussian
import breeze.numerics._
import scala.math.Pi

object FuncCalcuCombTests {

    // Transformation function g(z) for Z-scores with two-sided input p-values
    def g_GFisher_two(x: Double, df: Int): Double = {
        val pValueLog = math.log(2) + Normal.cumulative(math.abs(x), 0.0, 1.0, false, true)
        ChiSquare.quantile(pValueLog, df, false, true)
    }

    // Transformation function g(x) for Z-scores with one-sided input p-values
    def g_GFisher_one(x: Double, df: Int): Double = {
        val pValueLog = Normal.cumulative(x, 0.0, 1.0, false, true)
        ChiSquare.quantile(pValueLog, df, false, true)
    }

    // Transformation function g(x) for burden/Laptik test (identity function)
    def g_Burden(x: Double): Double = x

    // Combines g_GFisher_two and g_GFisher_one, controlled by pType ("one" or "two")
    def g_GFisher(x: Double, df: Int, pType: String = "two"): Double = {
        pType match {
        case "two" => g_GFisher_two(x, df)
        case "one" => g_GFisher_one(x, df)
        case _ => throw new IllegalArgumentException("pType must be 'one' or 'two'")
        }
    }

    def calcu_SgZ_p (
        g: BDV[Double] => BDV[Double],  // Function g
        Zscores: BDV[Double],
        wts: BDV[Double],
        calc_p: Boolean = false,
        M: Option[BDM[Double]] = None,
        df: Option[Double] = None,  // supports positive infinity
        pType: String = "two",
        isPosiWts: Boolean = true
    ): Map[String, Any] = {

        var weights = if (isPosiWts) wts.map(x => math.max(x, 0)) else wts  // force the weights to be non-negative

        if (weights.size != Zscores.size) {
        // If the sizes don't match, resize weights to match Zscores size (by repeating or filling it)
        weights = BDV.fill(Zscores.size)(weights(0))  // You can use weights(0) or another logic to fill the vector
        }

        // always scale the weights
        if (weights.map(math.abs).sum > 0) {
            weights = weights / weights.map(math.abs).sum
        }

        var S: Double = 0.0

        // calculate the statistic S using the provided g function
        S = sum(weights.asInstanceOf[BDV[Double]] *:* g(Zscores)) // no semi-colon needed?

        if(calc_p) {
            // assign proper df data types depending on the df input value
            val dfInt: Int = df match {
                case Some(d) if d.isValidInt => d.toInt
                case Some(Double.PositiveInfinity) => Int.MaxValue
                case _ => 0
            }
            // separate the calculation for burden test (df = Inf) and GFisher test
            if (dfInt == Int.MaxValue) {
                // burden test
                val M_mat = M.getOrElse(BDM.zeros[Double](weights.length, weights.length))
                val S_sd = math.sqrt(weights.t * M_mat * weights)
                val normalDist = new Normal(0, S_sd)
                val p = 2 * normalDist.cumulative(-math.abs(S))
                Map("S" -> S, "p" -> p)
            } else if (dfInt != 0) {
                // GFisher statistics
                // set degrees of freedom to be a vector, not an integer
                val (numRows, numCols) = M.map(m => (m.rows, m.cols)).getOrElse((0, 0))
                val degreesOfFreedom = BDV.fill(numRows)(dfInt.toDouble)
                // rescale the weights
                weights = weights/sum(weights.map(math.abs))
                // recalculate the S to be consistent with the new weights
                val S_ = sum(weights * g(Zscores))
                // calculate pGFisher (HYB)
                lazy val matrix: BDM[Double] = M.getOrElse(BDM.zeros[Double](numRows, numCols))
                val p = PGFisher.pGFisherHyb(S_, degreesOfFreedom, weights, matrix)
                Map("S" -> S_, "p" -> p)
            } else {
                Map("S" -> S)
            }
        } else {
            Map("S" -> S)
        }
    }

    // Multi SgZ Test with integrated burden test
    def multi_SgZ_test(
        Zscores: BDV[Double],
        DF: BDM[Double],
        W: BDM[Double],
        M: Option[BDM[Double]],
        pType: String = "two",
        calcP: Boolean = true,
        isPosiWts: Boolean = true,
        wNames: Option[Seq[String]] = None
    ): Map[String, BDM[Double]] = {

        val testN = DF.rows // Number of tests
        val STAT = BDM.zeros[Double](testN, 1)
        val PVAL = BDM.zeros[Double](testN, 1)

        for (i <- 0 until testN) {
            val dfValue = DF(i, 0)

            val result = if (dfValue.isPosInfinity) {
                // Burden test: Use identity function g(x) = x
                calcu_SgZ_p(x => x, Zscores, W(i, ::).t, calcP, M, Some(dfValue), pType, isPosiWts)
            } else {
                // GFisher test with transformation function
                val g = (x: BDV[Double]) => x.map(v => g_GFisher(v, dfValue.toInt, pType))
                calcu_SgZ_p(g, Zscores, W(i, ::).t, calcP, M, Some(dfValue), pType, isPosiWts)
            }

            STAT(i, 0) = result("S").asInstanceOf[Double]
            if (calcP) PVAL(i, 0) = result("p").asInstanceOf[Double]
            }

        Map("STAT" -> STAT, "PVAL" -> PVAL)
    }

    def cctTest(pvals: BDV[Double], thrLargeP: Double = 0.9, thrSmallP: Double = 1e-15): Map[String, Any] = {
        // Replace large p-values
        val clippedPvals = pvals.map(p => if (p > thrLargeP) thrLargeP else p)

        // Indicator for small p-values
        val isSmall = clippedPvals.map(_ < thrSmallP)

        // CCTSTAT initially set to PVAL values
        val CCTSTAT = clippedPvals.copy

        val cct: Double = if (sum(isSmall) == 0.0) {
            mean(tan(Pi * (0.5 - CCTSTAT)))  // If no small p-values, use regular transformation
        } else {
            for (i <- 0 until CCTSTAT.length) {
                if (!isSmall(i)) CCTSTAT(i) = tan((0.5 - CCTSTAT(i)) * Pi)
                else CCTSTAT(i) = 1.0 / (CCTSTAT(i) * Pi)
            }
            mean(CCTSTAT)  // Compute the mean of transformed values
        }

        // Compute the final CCT p-value using jdistlib.Cauchy
        val pvalCCT = Cauchy.cumulative(cct, 0.0, 1.0, false, false)

        // Return results in a Map
        Map("cct" -> cct, "pval_cct" -> pvalCCT)
    }

    def omni_SgZ_test(
        Zscores: BDV[Double],
        DF: BDM[Double],
        W: BDM[Double],
        M: BDM[Double],
        pType: String = "two",
        calcuP: Boolean = true,
        isPosiWts: Boolean = true
    ): Map[String, Any] = {

        // Call multi_SgZ_test (assuming you've translated this function to Scala)
        val multiTests = multi_SgZ_test(Zscores, DF, W, Some(M), pType, calcuP, isPosiWts)

        val (cct, pvalCct) = if (calcuP) {
            // val cctResult = cctTest(multiTests("PVAL").asInstanceOf[BDV[Double]])
            val PVAL_temp = multiTests("PVAL").asInstanceOf[BDM[Double]](::, 0) // Extract first column
            val cctResult = cctTest(PVAL_temp)
            (cctResult("cct").asInstanceOf[Double], cctResult("pval_cct").asInstanceOf[Double])
            // val cct = cctResult("cct").asInstanceOf[Double]
            // val pvalCct = cctResult("pval_cct").asInstanceOf[Double]
        } else {
            (Double.NaN, Double.NaN)
        }

        // Combine results into a Map
        Map(
            "STAT" -> BDM.vertcat(multiTests("STAT").asInstanceOf[BDM[Double]], BDM(cct.asInstanceOf[Double])),
            "PVAL" -> BDM.vertcat(multiTests("PVAL").asInstanceOf[BDM[Double]], BDM(pvalCct.asInstanceOf[Double])),
            "cct" -> cct,
            "pval_cct" -> pvalCct
        )
    }

    def BSF_test(
        Zscores: BDV[Double], 
        M: BDM[Double], 
        Bstar: BDV[Double], 
        PI: BDV[Double], 
        additionalParams: Any*
    ): Map[String, BDM[Double]] = {

        val wtsEqu = BDM.ones[Double](M.rows, 1) // Equal weights as column vector

        // Burden Test
        val gBurden: Double => Double = x => x
        val statDFBurden = Double.PositiveInfinity
        val wtsOptBurden = OptimalWeights.optimalWeightsM(gBurden, Bstar, PI, M, true, true)

        // Combine burden weights into a matrix
        val WT_opt_burden = BDM.vertcat(
            BDM(wtsOptBurden.toArray: _*), wtsEqu
        )

        val DF_opt_burden = BDM.fill(WT_opt_burden.rows, 1)(statDFBurden)

        // SKAT Test
        val gSKAT: Double => Double = x => x * x
        val statDFSKAT = 1.0
        val wtsOptSKAT = OptimalWeights.optimalWeightsM(gSKAT, Bstar, PI, M, true, true)

        val WT_opt_skat = BDM.vertcat(
            BDM(wtsOptSKAT.toArray: _*), wtsEqu
        )
        val DF_opt_skat = BDM.fill(WT_opt_skat.rows, 1)(statDFSKAT)

        // Fisher Test
        val gFisher: Double => Double = x => g_GFisher_two(x, 2)
        val statDFFisher = 2.0
        val wtsOptFisher = OptimalWeights.optimalWeightsM(gFisher, Bstar, PI, M, true, true)

        val WT_opt_fisher = BDM.vertcat(
            BDM(wtsOptFisher.toArray: _*), wtsEqu
        )
        val DF_opt_fisher = BDM.fill(WT_opt_fisher.rows, 1)(statDFFisher)

        // Combine everything
        val WT_opt = BDM.vertcat(WT_opt_skat, WT_opt_burden, WT_opt_fisher)
        val DF_opt = BDM.vertcat(DF_opt_skat, DF_opt_burden, DF_opt_fisher)

        // Compute the statistics
        val omniOpt = omni_SgZ_test(Zscores, DF_opt, WT_opt, M)

        // Ensure omniOpt Map values are properly typed
        val omniStat = omniOpt("STAT").asInstanceOf[BDM[Double]]
        val omniPval = omniOpt("PVAL").asInstanceOf[BDM[Double]]

        // Instead of trying to store string labels in a numeric matrix, use a separate Map
        val rowLabels = Map(
        "STAT" -> "df_1_Inf_2_wts_opt_equ_cct",
        "PVAL" -> "df_1_Inf_2_wts_opt_equ_cct"
        )

        // Return properly typed result
        Map("STAT" -> omniStat, "PVAL" -> omniPval)
    }

    def runTests(): Unit = {
        println("Running inline tests...")

        val g_Burden: BDV[Double] => BDV[Double] = x => x

        val df = 2
        val g_GF: BDV[Double] => BDV[Double] = Zscores => Zscores.map(z => g_GFisher_two(z, df))

        // val Zscores = BDV.rand(10, Gaussian(0, 1)) // Equivalent to rnorm(10)
        // val wts = BDV.rand(10) // Equivalent to runif(10)
        val Zscores = BDV(1.37095845, -0.56469817, 0.36312841, 0.63286260, 0.40426832, 
                          -0.10612452, 1.51152200, -0.09465904, 2.01842371, -0.06271410)

        val wts = BDV(0.90403139, 0.13871017, 0.98889173, 0.94666823, 0.08243756, 
                    0.51421178, 0.39020347, 0.90573813, 0.44696963, 0.83600426)

        val eps = 1e-5  // Small perturbation to improve numerical stability

        val M = BDM(
            (1.0 + eps,  0.30,  0.25,  0.20,  0.15,  0.10,  0.35,  0.40,  0.32,  0.28),
            (0.30,  1.0 + eps,  0.22,  0.18,  0.25,  0.27,  0.33,  0.29,  0.24,  0.31),
            (0.25,  0.22,  1.0 + eps,  0.21,  0.30,  0.26,  0.31,  0.28,  0.23,  0.27),
            (0.20,  0.18,  0.21,  1.0 + eps,  0.28,  0.22,  0.29,  0.24,  0.20,  0.26),
            (0.15,  0.25,  0.30,  0.28,  1.0 + eps,  0.35,  0.38,  0.36,  0.32,  0.34),
            (0.10,  0.27,  0.26,  0.22,  0.35,  1.0 + eps,  0.31,  0.33,  0.25,  0.30),
            (0.35,  0.33,  0.31,  0.29,  0.38,  0.31,  1.0 + eps,  0.40,  0.37,  0.39),
            (0.40,  0.29,  0.28,  0.24,  0.36,  0.33,  0.40,  1.0 + eps,  0.34,  0.35),
            (0.32,  0.24,  0.23,  0.20,  0.32,  0.25,  0.37,  0.34,  1.0 + eps,  0.28),
            (0.28,  0.31,  0.27,  0.26,  0.34,  0.30,  0.39,  0.35,  0.28,  1.0 + eps)
        )

        val result1 = calcu_SgZ_p(g_Burden, Zscores, wts, calc_p = false, Some(M))
        val result2 = calcu_SgZ_p(x => x, Zscores, wts, calc_p = false, Some(M))
        val result3 = calcu_SgZ_p(g_Burden, Zscores, wts, calc_p = true, Some(M), Some(Double.PositiveInfinity))
        val result4 = calcu_SgZ_p(g_GF, Zscores, wts, calc_p = true, Some(M), Some(df))

        // println("Result 1: " + result1)
        // println("Result 2: " + result2)
        // println("Result 3: " + result3)
        // println("Result 4: " + result4)

        val M2 = BDM(
        (1.0, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3),
        (0.3, 1.0, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3),
        (0.3, 0.3, 1.0, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3),
        (0.3, 0.3, 0.3, 1.0, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3),
        (0.3, 0.3, 0.3, 0.3, 1.0, 0.3, 0.3, 0.3, 0.3, 0.3),
        (0.3, 0.3, 0.3, 0.3, 0.3, 1.0, 0.3, 0.3, 0.3, 0.3),
        (0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 1.0, 0.3, 0.3, 0.3),
        (0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 1.0, 0.3, 0.3),
        (0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 1.0, 0.3),
        (0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 1.0)
        )

        // Zscores (1x10 vector)
        val Zscores2 = BDV(1.370958, -0.1274002, 0.6240275, 0.9271716, 0.8052372, 0.3962577, 1.817909, 0.5396136, 2.383442, 0.7161946)

        // Degrees of Freedom (DF)
        val DF2 = BDM(1.0, 2.0, Double.PositiveInfinity)

        // Weight matrix W (3x10)
        val W2 = BDM(
        (0.4040314, -0.36128983,  0.4888917, 0.4466682, -0.41756244,  0.01421178, -0.1097965,  0.4057381, -0.05303037, 0.3360043),
        (0.2375956,  0.31105514, -0.1118917, 0.1851697, -0.49605166,  0.33291608, -0.4926659, -0.2923410,  0.40660141, 0.1117786),
        (-0.1204408, -0.06422842, -0.4625690, 0.4735399, -0.06824875,  0.45757660,  0.3877549,  0.1399788,  0.47096661, 0.1188382)
        )

        val result5 = multi_SgZ_test(Zscores2, DF2, W2, Some(M2))
        // println("Result 5: " + result5)

        val result6 = omni_SgZ_test(Zscores2, DF2, W2, M2)
        // println("Result 6: " + result6)

        // Dense Matrix 1 (2 columns)
        val X = BDM(
        (-0.12913250, -0.16633872),
        ( 0.43144685, -0.10596090),
        ( 0.43883196,  0.20615436),
        (-0.68689981, -0.55209045),
        (-0.02442344,  1.04967967),
        (-0.64963987, -0.06202309),
        ( 0.68986570, -2.65811227),
        ( 0.74888684,  0.27944731),
        (-1.28206057, -1.04506685),
        (-0.36971191, -0.65495400),
        (-0.83325976, -0.82667493),
        (-0.02433189, -1.96054200),
        ( 0.02195152,  1.70634369),
        ( 1.80983322, -1.06236090),
        ( 0.88401487,  1.38309016),
        ( 0.20675809, -0.61060570),
        (-0.33459848, -0.92519076),
        ( 1.29299374,  0.12491036),
        ( 0.63329864,  0.80248313),
        (-0.02641578, -0.15515796)
        )

        // Dense Vector 1 (20 elements)
        val Y = BDV(
        0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 
        1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0
        )

        // Dense Matrix 2 (5 columns)
        val G = BDM(
        (1.0, 1.0, 0.0, 1.0, 0.0),
        (1.0, 0.0, 1.0, 1.0, 1.0),
        (1.0, 0.0, 0.0, 1.0, 1.0),
        (0.0, 0.0, 1.0, 0.0, 0.0),
        (1.0, 1.0, 1.0, 1.0, 1.0),
        (0.0, 1.0, 0.0, 0.0, 0.0),
        (1.0, 1.0, 0.0, 1.0, 0.0),
        (0.0, 1.0, 0.0, 0.0, 0.0),
        (0.0, 0.0, 0.0, 0.0, 0.0),
        (1.0, 1.0, 0.0, 0.0, 0.0),
        (1.0, 1.0, 1.0, 1.0, 0.0),
        (1.0, 1.0, 1.0, 1.0, 1.0),
        (1.0, 1.0, 0.0, 0.0, 1.0),
        (0.0, 1.0, 0.0, 1.0, 0.0),
        (0.0, 1.0, 1.0, 0.0, 0.0),
        (0.0, 0.0, 1.0, 0.0, 1.0),
        (1.0, 0.0, 1.0, 1.0, 0.0),
        (1.0, 1.0, 1.0, 1.0, 0.0),
        (0.0, 0.0, 0.0, 1.0, 0.0),
        (0.0, 0.0, 0.0, 1.0, 0.0)
        )

        // Dense Vector 2 (5 elements)
        val B = BDV(
        0.4812571, 0.3586995, 0.2422267, 1.1574871, 0.5287778
        )

        // Dense Vector 3 (5 elements)
        val PI = BDV(
        0.49196672, 0.22958845, 0.85594273, 0.08513217, 0.31415537
        )

        val Zout = FuncCalcuZScores.getZMargScore(G, X, Y)
        val s0 = Zout("s0").asInstanceOf[Double]
        val Bstar = (sqrt(diag(Zout("M_s").asInstanceOf[BDM[Double]])) * B.asInstanceOf[BDV[Double]]) / s0

        val result7 = BSF_test(Zout("Zscores").asInstanceOf[BDV[Double]], Zout("M_Z").asInstanceOf[BDM[Double]], Bstar, PI)
        println("Result 7: " + result7)
    }
}

// Run tests when the file is executed
object Main extends App {
  is.hail.methods.gfisher.FuncCalcuCombTests.runTests()
}