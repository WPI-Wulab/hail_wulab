// package is.hail.methods.gfisher
import breeze.stats.distributions.Gaussian
// import is.hail.{HailSuite, TestUtils}
// import is.hail.utils._
import is.hail.methods.gfisher._
import is.hail.methods.gfisher.GFisherCoefs._

import breeze.linalg.{DenseMatrix => BDM, DenseVector => BDV, _}
import breeze.numerics.abs
// import org.testng.annotations.Test


// run this with scala -cp "$hailstuff" hail/src/test/scala/is/hail/methods/GFisherSuite.scala
// it no longer works


// class AssertQ[A] extends scala.collection.mutable.Queue[A]{
//   def assert(a: A): Unit = this.enqueue(a)
// }

object GFisherSuite {


  def vecDiff(x: BDV[Double], y: BDV[Double]): Unit = {
    val d = abs(x - y)
    val s = d.data.map(i => f"$i%.2e").mkString("DenseVector(", ", ", ")")
    println(s)
  }

  def matDiff(X: BDM[Double], Y: BDM[Double]): Unit = {
    val mat = abs(X - Y)
    val rows = for (i <- 0 until mat.rows) yield {
      (0 until mat.cols).map(j => f"${mat(i, j)}%.2e").mkString("(", ", ", ")")
    }
    val s = rows.mkString("\n")
    println(s)
  }

  val defaultTolerance = 1e-7

  def assertVectorEqualityDouble(
    A: Vector[Double],
    B: Vector[Double],
    tolerance: Double = defaultTolerance,
  ): Unit = {
    assert(A.size == B.size)
    assert((0 until A.size).forall(i => D_==(A(i), B(i), tolerance)))
  }

  def assertMatrixEqualityDouble(
    A: BDM[Double],
    B: BDM[Double],
    tolerance: Double = defaultTolerance,
  ): Unit = {
    assert(A.rows == B.rows)
    assert(A.cols == B.cols)

    assert((0 until A.rows).forall(i => (0 until A.cols).forall(j => D_==(A(i, j), B(i, j), tolerance))
    ))
  }

  def D_epsilon(expected: Double, observed: Double, tolerance: Double = defaultTolerance): Double =
    math.max(java.lang.Double.MIN_NORMAL, tolerance * math.max(math.abs(expected), math.abs(observed)))

  def D_==(expected: Double, observed: Double, tolerance: Double = defaultTolerance): Boolean = {
      var x = expected == observed || math.abs(expected - observed) <= D_epsilon(expected, observed, tolerance)
      x = x || (abs(expected) < tolerance && abs(observed) < tolerance)
      if (! x) {
        println(s"\n\nexpected: $expected, observed: $observed, tol: $tolerance\n\n")
      }
      return x
  }

  // @Test def nearPDTest() = {
  def testNearPD() = {
    val X = BDM((1.000000000000, 0.477000000000, 0.644000000000, 0.478000000000, 0.651000000000, 0.826000000000),
                (0.477000000000, 1.000000000000, 0.516000000000, 0.233000000000, 0.682000000000, 0.750000000000),
                (0.644000000000, 0.516000000000, 1.000000000000, 0.599000000000, 0.581000000000, 0.742000000000),
                (0.478000000000, 0.233000000000, 0.599000000000, 1.000000000000, 0.741000000000, 0.800000000000),
                (0.651000000000, 0.682000000000, 0.581000000000, 0.741000000000, 1.000000000000, 0.798000000000),
                (0.826000000000, 0.750000000000, 0.742000000000, 0.800000000000, 0.798000000000, 1.000000000000)
              )
    val expected = BDM((1.000000000000, 0.487786106067, 0.642930897861, 0.490455409646, 0.644715065677, 0.808210054730),
                      (0.487786106067, 1.000000000000, 0.514511524676, 0.250341254148, 0.673249697807, 0.725231681854),
                      (0.642930897861, 0.514511524676, 1.000000000000, 0.597281158275, 0.581867305541, 0.744454941056),
                      (0.490455409646, 0.250341254148, 0.597281158275, 1.000000000000, 0.730895459984, 0.771398429440),
                      (0.644715065677, 0.673249697807, 0.581867305541, 0.730895459984, 1.000000000000, 0.812432091620),
                      (0.808210054730, 0.725231681854, 0.744454941056, 0.771398429440, 0.812432091620, 1.000000000000)
                    )

    val result = nearPD(X)
    assertMatrixEqualityDouble(expected, result, 1e-6)
  }


  // @Test def cov2corTest() = {
  def testCov2cor() = {
    val X = BDM((1.000000000000, 0.991589178025, 0.620633392559, 0.464744187601, 0.979163432977, 0.991149190067, 0.970898525061),
                (0.991589178025, 1.000000000000, 0.604260939890, 0.446436791893, 0.991090069458, 0.995273483765, 0.983551611180),
                (0.620633392559, 0.604260939890, 1.000000000000, -0.177420629502, 0.686551516365, 0.668256604562, 0.502498083876),
                (0.464744187601, 0.446436791893, -0.177420629502, 1.000000000000, 0.364416267189, 0.417245149835, 0.457307399976),
                (0.979163432977, 0.991090069458, 0.686551516365, 0.364416267189, 1.000000000000, 0.993952846233, 0.960390571594),
                (0.991149190067, 0.995273483765, 0.668256604562, 0.417245149835, 0.993952846233, 1.000000000000, 0.971329459192),
                (0.970898525061, 0.983551611180, 0.502498083876, 0.457307399976, 0.960390571594, 0.971329459192, 1.000000000000)
              )
    val expected = new BDM(7,7,
        Array(1.0, 0.991589178024782, 0.620633392559097, 0.464744187600675, 0.979163432977498, 0.991149190067205, 0.970898525061056,
              0.991589178024782, 1.0, 0.604260939889558, 0.446436791892627, 0.991090069458478, 0.995273483764785, 0.98355161117967,
              0.620633392559097, 0.604260939889558, 1.0, -0.177420629501878, 0.686551516365312, 0.668256604562175, 0.502498083875994,
              0.464744187600675, 0.446436791892627, -0.177420629501878, 1.0, 0.364416267189032, 0.417245149834945, 0.457307399976482,
              0.979163432977498, 0.991090069458478, 0.686551516365312, 0.364416267189032, 1.0, 0.993952846232926, 0.960390571594375,
              0.991149190067205, 0.995273483764785, 0.668256604562175, 0.417245149834945, 0.993952846232926, 1.0, 0.971329459192119,
              0.970898525061056, 0.98355161117967, 0.502498083875994, 0.457307399976482, 0.960390571594375, 0.971329459192119, 1.0))
    val result = cov2cor(X)

    assertMatrixEqualityDouble(expected, result)
  }

  // @Test def normITest() = {
  def testNormI() = {
    val X = BDM((1.000000000000, 0.500000000000, 0.333333333333, 0.250000000000, 0.200000000000, 0.166666666667, 0.142857142857, 0.125000000000, 0.111111111111),
                (0.500000000000, 0.333333333333, 0.250000000000, 0.200000000000, 0.166666666667, 0.142857142857, 0.125000000000, 0.111111111111, 0.100000000000),
                (0.333333333333, 0.250000000000, 0.200000000000, 0.166666666667, 0.142857142857, 0.125000000000, 0.111111111111, 0.100000000000, 0.090909090909),
                (0.250000000000, 0.200000000000, 0.166666666667, 0.142857142857, 0.125000000000, 0.111111111111, 0.100000000000, 0.090909090909, 0.083333333333),
                (0.200000000000, 0.166666666667, 0.142857142857, 0.125000000000, 0.111111111111, 0.100000000000, 0.090909090909, 0.083333333333, 0.076923076923),
                (0.166666666667, 0.142857142857, 0.125000000000, 0.111111111111, 0.100000000000, 0.090909090909, 0.083333333333, 0.076923076923, 0.071428571429),
                (0.142857142857, 0.125000000000, 0.111111111111, 0.100000000000, 0.090909090909, 0.083333333333, 0.076923076923, 0.071428571429, 0.066666666667),
                (0.125000000000, 0.111111111111, 0.100000000000, 0.090909090909, 0.083333333333, 0.076923076923, 0.071428571429, 0.066666666667, 0.062500000000),
                (0.111111111111, 0.100000000000, 0.090909090909, 0.083333333333, 0.076923076923, 0.071428571429, 0.066666666667, 0.062500000000, 0.058823529412)
              )
    val expected = 2.828968253968
    val result = normI(X)
    assert(D_==(expected, result))
  }


  def testGFisherCoefs() = {
    val df = BDV(1,2,3,4,5,6,7,8,9,10)

    val expectedCoef1 = BDV(1.999999999995, 2.795280984953, 3.374665277998, 3.851065409441, 4.264754011385, 4.635267241778, 4.973767157066, 5.287321966504, 5.580735943021, 5.857448100444)
    val expectedCoef2 = BDV(-0.000000000327, -1.102175237189, -1.926012279478, -2.604970524475, -3.193800498887, -3.720188756502, -4.200216489672, -4.644139233308, -5.058948055447, -5.449651711820)
    val expectedCoef3 = BDV(-0.000000018614, 3.534330441840, 6.326995615078, 8.669004207211, 10.717167867416, 12.557143631049, 14.240517801525, 15.800884376703, 17.261458054616, 18.639031251242)
    val expectedCoef4 = BDV(-0.000000980852, -18.139882715316, -33.230160095858, -46.082814378189, -57.404018837863, -67.616190532792, -76.983763040974, -85.682657381424, -93.836093535005, -101.534034691021)
    val (c1, c2, c3, c4) = GFisherCoefs.getGFisherCoefs(df, false)

    assertVectorEqualityDouble(expectedCoef1, c1, 1e-6)
    assertVectorEqualityDouble(expectedCoef2, c2, 1e-6)
    assertVectorEqualityDouble(expectedCoef3, c3, 1e-6)
    assertVectorEqualityDouble(expectedCoef4, c4, 1e-6)
  }

  def testGFisherGM() = {
    // GT <- matrix(sample(0:2,100,T),nrow=10)
    // M <- cor(t(GT))
    // w <- runif(10)
    // w <- w / sum(w)

    val w = BDV(0.06633395, 0.15953289, 0.01495265, 0.15193648, 0.02723407, 0.11551675, 0.16936508, 0.08070662, 0.07449124, 0.13993028)
    val df = BDV.ones[Int](10) * 2

    val M: BDM[Double] = new BDM(10, 10,
      Array(1.0, 0.130434782608696, 0.0564033346637209, -0.621669872160209, -0.108893101296094, 0.107896803120133, -0.160514470781026, -0.361157559257308, 0.601929265428846, -0.210675242900096,
      0.130434782608696, 1.0, -0.131607780882015, -0.466252404120157, 0.435572405184377, 0.262035093291752, 0.44141479464782, 0.39125402252875, -0.361157559257308, 0.541736338885961,
      0.0564033346637209, -0.131607780882015, 1.0, 0.201619459636378, 0.188352643739204, -0.219956013195602, -0.416463365036283, 0.0390434404721515, 0.156173761888606, -0.351390964249364,
      -0.621669872160209, -0.466252404120157, 0.201619459636378, 1.0, 0.0, -0.495884703680465, -0.430331482911935, 0.161374306091976, -0.258198889747161, -0.161374306091976,
      -0.108893101296094, 0.435572405184377, 0.188352643739204, 0.0, 1.0, 0.424650290065201, -0.201007563051842, 0.301511344577763, 0.0, 0.113066754216661,
      0.107896803120133, 0.262035093291752, -0.219956013195602, -0.495884703680465, 0.424650290065201, 1.0, 0.341431679821056, 0.288082979849016, 0.128036879932896, 0.128036879932896,
      -0.160514470781026, 0.44141479464782, -0.416463365036283, -0.430331482911935, -0.201007563051842, 0.341431679821056, 1.0, 0.166666666666667, -0.666666666666667, 0.791666666666667,
      -0.361157559257308, 0.39125402252875, 0.0390434404721515, 0.161374306091976, 0.301511344577763, 0.288082979849016, 0.166666666666667, 1.0, -0.25, 0.375,
      0.601929265428846, -0.361157559257308, 0.156173761888606, -0.258198889747161, 0.0, 0.128036879932896, -0.666666666666667, -0.25, 1.0, -0.75,
      -0.210675242900096, 0.541736338885961, -0.351390964249364, -0.161374306091976, 0.113066754216661, 0.128036879932896, 0.791666666666667, 0.375, -0.75, 1.0))
    val expected = new BDM(10,10,
      Array(0.0176007713483464, 0.000706558392307933, 1.23811316048067e-05, 0.0153710717314759, 8.40610688164359e-05, 0.000350060079076767, 0.00113609124617822, 0.00274463975271326, 0.0070621924810322, 0.0016173610211586,
      0.000706558392307933, 0.101802970299332, 0.000162146804862817, 0.0207374309517732, 0.00324267860921499, 0.00496922326277673, 0.0207119705277646, 0.0077493397756427, 0.00609250305180271, 0.0258138750435734,
      1.23811316048067e-05, 0.000162146804862817, 0.000894326878354579, 0.000362539942440261, 5.67093874644397e-05, 0.000328088034483559, 0.00172747551895729, 7.21790067717541e-06, 0.000106624426770143, 0.00101534857082544,
      0.0153710717314759, 0.0207374309517732, 0.000362539942440261, 0.0923387719931162, 0, 0.0169925431249185, 0.0187449152281872, 0.0012533335574521, 0.00296304755426192, 0.00217304752846818,
      8.40610688164359e-05, 0.00324267860921499, 5.67093874644397e-05, 0, 0.00296677725212964, 0.00223141566370573, 0.000731594603644926, 0.000784938480225133, 0, 0.000191181180408831,
      0.000350060079076767, 0.00496922326277673, 0.000328088034483559, 0.0169925431249185, 0.00223141566370573, 0.0533764771167344, 0.00896268980267862, 0.00303912580440383, 0.00055359469681181, 0.001039916446238,
      0.00113609124617822, 0.0207119705277646, 0.00172747551895729, 0.0187449152281872, 0.000731594603644926, 0.00896268980267862, 0.114738123779822, 0.00149027756946683, 0.0221501521067957, 0.0588820992485046,
      0.00274463975271326, 0.0077493397756427, 7.21790067717541e-06, 0.0012533335574521, 0.000784938480225133, 0.00303912580440383, 0.00149027756946683, 0.0260542336141661, 0.00147547716723415, 0.00624300067899126,
      0.0070621924810322, 0.00609250305180271, 0.000106624426770143, 0.00296304755426192, 0, 0.00055359469681181, 0.0221501521067957, 0.00147547716723415, 0.0221957777066808, 0.0232133199224001,
      0.0016173610211586, 0.0258138750435734, 0.00101534857082544, 0.00217304752846818, 0.000191181180408831, 0.001039916446238, 0.0588820992485046, 0.00624300067899126, 0.0232133199224001, 0.0783219366488731))

    val result: BDM[Double] = GFisherGM.getGFisherGM(df=df, w=w, M=M, one_sided=false)
    assertMatrixEqualityDouble(expected, result, 1e-6)
  }

  def testGFisherLambda() = {
    // GT <- matrix(sample(0:2,100,T),nrow=10)
    // M <- cor(t(GT))
    // w <- runif(10)
    // w <- w / sum(w)
    // val w = BDV(0.06633395, 0.15953289, 0.01495265, 0.15193648, 0.02723407, 0.11551675, 0.16936508, 0.08070662, 0.07449124, 0.13993028)

    // val df = BDV.ones[Int](10) * 2

    // val M = new BDM(10, 10,
    //   Array(1.0, 0.130434782608696, 0.0564033346637209, -0.621669872160209, -0.108893101296094, 0.107896803120133, -0.160514470781026, -0.361157559257308, 0.601929265428846, -0.210675242900096,
    //   0.130434782608696, 1.0, -0.131607780882015, -0.466252404120157, 0.435572405184377, 0.262035093291752, 0.44141479464782, 0.39125402252875, -0.361157559257308, 0.541736338885961,
    //   0.0564033346637209, -0.131607780882015, 1.0, 0.201619459636378, 0.188352643739204, -0.219956013195602, -0.416463365036283, 0.0390434404721515, 0.156173761888606, -0.351390964249364,
    //   -0.621669872160209, -0.466252404120157, 0.201619459636378, 1.0, 0.0, -0.495884703680465, -0.430331482911935, 0.161374306091976, -0.258198889747161, -0.161374306091976,
    //   -0.108893101296094, 0.435572405184377, 0.188352643739204, 0.0, 1.0, 0.424650290065201, -0.201007563051842, 0.301511344577763, 0.0, 0.113066754216661,
    //   0.107896803120133, 0.262035093291752, -0.219956013195602, -0.495884703680465, 0.424650290065201, 1.0, 0.341431679821056, 0.288082979849016, 0.128036879932896, 0.128036879932896,
    //   -0.160514470781026, 0.44141479464782, -0.416463365036283, -0.430331482911935, -0.201007563051842, 0.341431679821056, 1.0, 0.166666666666667, -0.666666666666667, 0.791666666666667,
    //   -0.361157559257308, 0.39125402252875, 0.0390434404721515, 0.161374306091976, 0.301511344577763, 0.288082979849016, 0.166666666666667, 1.0, -0.25, 0.375,
    //   0.601929265428846, -0.361157559257308, 0.156173761888606, -0.258198889747161, 0.0, 0.128036879932896, -0.666666666666667, -0.25, 1.0, -0.75,
    //   -0.210675242900096, 0.541736338885961, -0.351390964249364, -0.161374306091976, 0.113066754216661, 0.128036879932896, 0.791666666666667, 0.375, -0.75, 1.0))
    // val GM = new BDM(10,10,
    //   Array(0.0176007713483464, 0.000706558392307933, 1.23811316048067e-05, 0.0153710717314759, 8.40610688164359e-05, 0.000350060079076767, 0.00113609124617822, 0.00274463975271326, 0.0070621924810322, 0.0016173610211586,
    //   0.000706558392307933, 0.101802970299332, 0.000162146804862817, 0.0207374309517732, 0.00324267860921499, 0.00496922326277673, 0.0207119705277646, 0.0077493397756427, 0.00609250305180271, 0.0258138750435734,
    //   1.23811316048067e-05, 0.000162146804862817, 0.000894326878354579, 0.000362539942440261, 5.67093874644397e-05, 0.000328088034483559, 0.00172747551895729, 7.21790067717541e-06, 0.000106624426770143, 0.00101534857082544,
    //   0.0153710717314759, 0.0207374309517732, 0.000362539942440261, 0.0923387719931162, 0, 0.0169925431249185, 0.0187449152281872, 0.0012533335574521, 0.00296304755426192, 0.00217304752846818,
    //   8.40610688164359e-05, 0.00324267860921499, 5.67093874644397e-05, 0, 0.00296677725212964, 0.00223141566370573, 0.000731594603644926, 0.000784938480225133, 0, 0.000191181180408831,
    //   0.000350060079076767, 0.00496922326277673, 0.000328088034483559, 0.0169925431249185, 0.00223141566370573, 0.0533764771167344, 0.00896268980267862, 0.00303912580440383, 0.00055359469681181, 0.001039916446238,
    //   0.00113609124617822, 0.0207119705277646, 0.00172747551895729, 0.0187449152281872, 0.000731594603644926, 0.00896268980267862, 0.114738123779822, 0.00149027756946683, 0.0221501521067957, 0.0588820992485046,
    //   0.00274463975271326, 0.0077493397756427, 7.21790067717541e-06, 0.0012533335574521, 0.000784938480225133, 0.00303912580440383, 0.00149027756946683, 0.0260542336141661, 0.00147547716723415, 0.00624300067899126,
    //   0.0070621924810322, 0.00609250305180271, 0.000106624426770143, 0.00296304755426192, 0, 0.00055359469681181, 0.0221501521067957, 0.00147547716723415, 0.0221957777066808, 0.0232133199224001,
    //   0.0016173610211586, 0.0258138750435734, 0.00101534857082544, 0.00217304752846818, 0.000191181180408831, 0.001039916446238, 0.0588820992485046, 0.00624300067899126, 0.0232133199224001, 0.0783219366488731))

    // val expected = BDV(0.0640991815532492,0.0281892986428277,0.0160574782558939,0.0110821102692326,0.00380075491961822,0.00264800460087184,0.00118702391490419,0.000350993438876728,0.000137845464188023,1.98505997266215e-05,0.0640991815532492,0.0281892986428277,0.0160574782558939,0.0110821102692326,0.00380075491961822,0.00264800460087185,0.00118702391490419,0.00035099343887673,0.000137845464188023,1.98505997266193e-05)
    // val result = GFisherLambda.getGFisherLambda(df, w, M, GM)

    val df = BDV.ones[Int](10) * 2
    val w = BDV(0.087658852431, 0.175932408441, 0.009714572763, 0.043788683127, 0.000564845727, 0.160560462742, 0.071399446101, 0.177712093686, 0.157033221275, 0.115635413706)
    val M = BDM((1.000000000000, 0.060192926543, 0.391254022529, -0.640711607228, 0.240771706172, 0.464285714286, -0.000000000000, -0.375000000000, 0.133630620956, 0.133630620956),
                (0.060192926543, 1.000000000000, -0.275362318841, -0.254083903024, 0.159420289855, -0.670721181478, -0.120385853086, -0.361157559257, -0.353919198648, -0.032174472604),
                (0.391254022529, -0.275362318841, 1.000000000000, -0.653358607777, -0.014492753623, 0.498741391355, -0.120385853086, -0.240771706172, 0.353919198648, 0.514791561670),
                (-0.640711607228, -0.254083903024, -0.653358607777, 1.000000000000, -0.108893101296, -0.129219147676, 0.000000000000, 0.263822426506, 0.040291148201, -0.161164592805),
                (0.240771706172, 0.159420289855, -0.014492753623, -0.108893101296, 1.000000000000, -0.361157559257, 0.120385853086, -0.090289389814, 0.193046835626, 0.514791561670),
                (0.464285714286, -0.670721181478, 0.498741391355, -0.129219147676, -0.361157559257, 1.000000000000, -0.142857142857, -0.107142857143, 0.419981951577, 0.038180177416),
                (-0.000000000000, -0.120385853086, -0.120385853086, 0.000000000000, 0.120385853086, -0.142857142857, 1.000000000000, 0.500000000000, 0.267261241912, 0.267261241912),
                (-0.375000000000, -0.361157559257, -0.240771706172, 0.263822426506, -0.090289389814, -0.107142857143, 0.500000000000, 1.000000000000, -0.133630620956, -0.300668897151),
                (0.133630620956, -0.353919198648, 0.353919198648, 0.040291148201, 0.193046835626, 0.419981951577, 0.267261241912, -0.133630620956, 1.000000000000, 0.642857142857),
                (0.133630620956, -0.032174472604, 0.514791561670, -0.161164592805, 0.514791561670, 0.038180177416, 0.267261241912, -0.300668897151, 0.642857142857, 1.000000000000)
              )
    val GM = BDM(
                (0.030736297638, 0.000219246069, 0.000512536507, 0.006220867560, 0.000011270628, 0.011939702770, 0.000000000000, 0.008611628325, 0.000964671029, 0.000710360092),
                (0.000219246069, 0.123808849359, 0.000508973692, 0.001953045088, 0.000009912563, 0.050204331832, 0.000714418026, 0.016028879725, 0.013600682433, 0.000082631239),
                (0.000512536507, 0.000508973692, 0.000377491696, 0.000717104521, 0.000000004522, 0.001527644121, 0.000039448479, 0.000392973408, 0.000750997615, 0.001172456189),
                (0.006220867560, 0.001953045088, 0.000717104521, 0.007669795080, 0.000001150902, 0.000460709939, 0.000000000000, 0.002127072247, 0.000043798648, 0.000516200370),
                (0.000011270628, 0.000009912563, 0.000000004522, 0.000001150902, 0.000001276203, 0.000046495273, 0.000002293699, 0.000003211036, 0.000012975915, 0.000068171487),
                (0.011939702770, 0.050204331832, 0.001527644121, 0.000460709939, 0.000046495273, 0.103118648784, 0.000918188517, 0.001285362844, 0.017491474534, 0.000106192151),
                (0.000000000000, 0.000714418026, 0.000039448479, 0.000000000000, 0.000002293699, 0.000918188517, 0.020391523614, 0.012490198802, 0.003145205931, 0.002316052528),
                (0.008611628325, 0.016028879725, 0.000392973408, 0.002127072247, 0.000003211036, 0.001285362844, 0.012490198802, 0.126326352969, 0.001955691907, 0.007297741683),
                (0.000964671029, 0.013600682433, 0.000750997615, 0.000043798648, 0.000012975915, 0.017491474534, 0.003145205931, 0.001955691907, 0.098637730336, 0.029627872253),
                (0.000710360092, 0.000082631239, 0.001172456189, 0.000516200370, 0.000068171487, 0.000106192151, 0.002316052528, 0.007297741683, 0.029627872253, 0.053486195612)
              )
    val expected = BDV(0.055899437958, 0.042236576036, 0.023109161290, 0.009174832441, 0.005644764752, 0.002553975406, 0.001642719127, 0.000867597952, 0.000009457902, 0.000000017458, 0.055899437958, 0.042236576036, 0.023109161290, 0.009174832441, 0.005644764752, 0.002553975406, 0.001642719127, 0.000867597952, 0.000009457902, 0.000000017458)
    val result = GFisherLambda.getGFisherLambda(df, w, M, GM)
    assertVectorEqualityDouble(expected, result)
  }




  // def print2(vec: BDV[Double]): Unit = {
  //   val s = vec.data.map(x => f"$x%1.${decimals}f").mkString("DenseVector(", ", ", ")")
  //   println(s)
  // }


  def timer_fun(time_unit: String = "nano"): () => Long = {
    if (time_unit.equalsIgnoreCase("nano") || time_unit.equalsIgnoreCase("n")) {
      () => System.nanoTime
    } else {
      () => System.currentTimeMillis
    }
  }

  def elapsed(end: Long, start: Long, time_unit: String ="nano"): Double = {
    val diff = end - start
    if (time_unit.equalsIgnoreCase("nano") || time_unit.equalsIgnoreCase("n")) {
      return diff / 1.0e9
    } else {
      return diff / 1000.0
    }
  }


  def main(args: Array[String]): Unit = {
    val time_unit = "nano"
    val timer = timer_fun(time_unit)
    var start = 0L
    var end = 0L
    var t_elaps = 0.0
    val dfRange = new BDV((1 to 10).toArray)
    val dfSame = BDV.fill(10)(10)
    var (res1, res2, res3, res4) = (BDV.zeros[Double](10), BDV.zeros[Double](10), BDV.zeros[Double](10), BDV.zeros[Double](10))
    var resTup = (res1, res2, res3, res3)



    println("###################################################################")
    println(s"Original method with all df as ${dfSame(0)}")
    start = timer()
    for (i <- 0 until 100) {
      resTup = getGFisherCoefs(dfSame, false)
    }
    end = timer()
    t_elaps = elapsed(end, start, time_unit)
    println(s"Time elapsed: $t_elaps")
    res1 = resTup._1
    res2 = resTup._2
    res3 = resTup._3
    res4 = resTup._4
    println(s"coef1: $res1\ncoef2: $res2\ncoef3: $res3\ncoef4: $res4")
    println("###################################################################")
    println("Original method with range of dfs")
    start = timer()
    for (i <- 0 until 100) {
      resTup = getGFisherCoefs(dfRange, false)
    }

    end = timer()
    t_elaps = elapsed(end, start, time_unit)
    println(s"Time elapsed: $t_elaps")
    res1 = resTup._1
    res2 = resTup._2
    res3 = resTup._3
    res4 = resTup._4
    println(s"coef1: $res1\ncoef2: $res2\ncoef3: $res3\ncoef4: $res4")
    println("###################################################################")

    println("###################################################################")
    println(s"new method1 with all df as ${dfSame(0)}")
    start = timer()
    for (i <- 0 until 100) {
      resTup = getGFisherCoefs2(dfSame)
    }
    end = timer()
    t_elaps = elapsed(end, start, time_unit)
    println(s"Time elapsed: $t_elaps")
    res1 = resTup._1
    res2 = resTup._2
    res3 = resTup._3
    res4 = resTup._4
    println(s"coef1: $res1\ncoef2: $res2\ncoef3: $res3\ncoef4: $res4")

    println("###################################################################")
    println("New method1 with range of dfs")
    start = timer()
    for (i <- 0 until 100) {
      resTup = getGFisherCoefs2(dfRange)
    }

    end = timer()
    t_elaps = elapsed(end, start, time_unit)
    println(s"Time elapsed: $t_elaps")
    res1 = resTup._1
    res2 = resTup._2
    res3 = resTup._3
    res4 = resTup._4
    println(s"coef1: $res1\ncoef2: $res2\ncoef3: $res3\ncoef4: $res4")
    println("###################################################################")

    println("###################################################################")
    println(s"new method2 with all df as ${dfSame(0)}")
    start = timer()
    for (i <- 0 until 100) {
      resTup = getGFisherCoefs3(dfSame)
    }
    end = timer()
    t_elaps = elapsed(end, start, time_unit)
    println(s"Time elapsed: $t_elaps")
    res1 = resTup._1
    res2 = resTup._2
    res3 = resTup._3
    res4 = resTup._4
    println(s"coef1: $res1\ncoef2: $res2\ncoef3: $res3\ncoef4: $res4")

    println("####################################################################")
    println("New method2 with range of dfs")
    start = timer()
    for (i <- 0 until 100) {
      resTup = getGFisherCoefs3(dfRange)
    }

    end = timer()
    t_elaps = elapsed(end, start, time_unit)
    println(s"Time elapsed: $t_elaps")
    res1 = resTup._1
    res2 = resTup._2
    res3 = resTup._3
    res4 = resTup._4
    println(s"coef1: $res1\ncoef2: $res2\ncoef3: $res3\ncoef4: $res4")
    println("###################################################################")
    // val N = 5
    // val random = new scala.util.Random(42)
    // val originalMatrix = BDM(
    //   (1.0, 0.2, 0.3, 0.4, 0.5),
    //   (0.2, 1.0, 0.6, 0.7, 0.8),
    //   (0.3, 0.6, 1.0, 0.9, 0.1),
    //   (0.4, 0.7, 0.9, 1.0, 0.2),
    //   (0.5, 0.8, 0.1, 0.2, 1.0)
    // )
    // val selectedIndices = Array(3, 1, 4)
    // val input = Array(
    //   (3, 1.0, 1, 1.0, BDV(0.4, 0.7, 0.9, 1.0, 0.2)),
    //   (1, 1.0, 1, 1.0, BDV(0.2, 1.0, 0.6, 0.7, 0.8)),
    //   (0, 1.0, 1, 1.0, BDV(1.0, 0.2, 0.3, 0.4, 0.5))
    // )

    // val res = tupleArrayToVectorTuple(input)

    // val expected = BDM((0.4, 0.7, 0.9, 1.0, 0.2),
    //                    (0.2, 1.0, 0.6, 0.7, 0.8),
    //                    (1.0, 0.2, 0.3, 0.4, 0.5)).t.toDenseMatrix
    // println(res)
    // println("expected")
    // println(expected)

  //   try {
  //     testCov2cor()
  //     testNearPD()
  //     testNormI()
  //     // testGFisherCoefs()
  //     // testGFisherGM()
  //     testGFisherLambda()
  //     println("\n\nAll tests passed! Huzzah!")
  //   } catch {
  //     case e: AssertionError => {
  //       println("Assertion failed. Source:")
  //       val trace = e.getStackTrace()
  //       for (i <- 0 until math.min(4, trace.length)) {
  //         println(trace(i))
  //       }
  //     }
  //   }

  }

}
// ###################################################################
// Original method with all df as 10
// Time elapsed: 11.891212208
// coef1: DenseVector(5.857448103645337, 5.857448103645337, 5.857448103645337, 5.857448103645337, 5.857448103645337, 5.857448103645337, 5.857448103645337, 5.857448103645337, 5.857448103645337, 5.857448103645337)
// coef2: DenseVector(-5.449651722663275, -5.449651722663275, -5.449651722663275, -5.449651722663275, -5.449651722663275, -5.449651722663275, -5.449651722663275, -5.449651722663275, -5.449651722663275, -5.449651722663275)
// coef3: DenseVector(18.63903131160649, 18.63903131160649, 18.63903131160649, 18.63903131160649, 18.63903131160649, 18.63903131160649, 18.63903131160649, 18.63903131160649, 18.63903131160649, 18.63903131160649)
// coef4: DenseVector(-101.53403515317865, -101.53403515317865, -101.53403515317865, -101.53403515317865, -101.53403515317865, -101.53403515317865, -101.53403515317865, -101.53403515317865, -101.53403515317865, -101.53403515317865)
// ###################################################################
// Original method with range of dfs
// Time elapsed: 9.177541292
// coef1: DenseVector(1.9999999999946378, 2.7952809849530986, 3.3746652780619772, 3.8510654095880787, 4.264754011266045, 4.635267241582413, 4.973767157679357, 5.287321969178666, 5.580735949016274, 5.857448103645337)
// coef2: DenseVector(-3.269731152499844E-10, -1.1021752371887563, -1.9260122798802628, -2.604970526687646, -3.1938004983226973, -3.720188757646613, -4.200216490193065, -4.644139235937417, -5.058948061549234, -5.449651722663275)
// coef3: DenseVector(-1.8613784291687807E-8, 3.53433044183992, 6.326995619301124, 8.669004207879867, 10.717167867689714, 12.557143631831003, 14.24051780717368, 15.80088439381663, 17.261458090165327, 18.63903131160649)
// coef4: DenseVector(-9.784955516956018E-7, -18.139882712921732, -33.23016014357346, -46.08281438622052, -57.404018844385284, -67.61619054909958, -76.9837630984294, -85.68265752614721, -93.83609381618088, -101.53403515317865)