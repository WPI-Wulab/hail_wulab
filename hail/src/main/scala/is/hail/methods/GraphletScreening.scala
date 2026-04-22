package is.hail.methods

import is.hail.HailContext
import is.hail.backend.ExecuteContext
import is.hail.annotations._
import is.hail.expr.ir.{IntArrayBuilder, MatrixValue, TableValue}
import is.hail.expr.ir.functions.MatrixToTableFunction
import is.hail.stats.{LogisticRegressionModel, RegressionUtils, pnorm}
import is.hail.types.physical.PStruct
import is.hail.types.virtual.{MatrixType, TArray, TBoolean, TFloat64, TInt32, TStruct, TableType}
import is.hail.utils._
import is.hail.annotations.{RegionValueBuilder, WritableRegionValue}
import is.hail.io.{BufferSpec, TypedCodecSpec}
import breeze.linalg.DenseMatrix._
import breeze.linalg.DenseVector._
import breeze.linalg._
import breeze.linalg.diag
import breeze.linalg.eigSym
import breeze.linalg.sum
import breeze.linalg.svd

import breeze.stats.distributions.RandBasis
import breeze.stats.distributions.StudentsT
import breeze.numerics.{sqrt => breezeSqrt}

import scala.collection.mutable
import scala.util.Random

import org.apache.spark.sql.Row

/**
 * Graphlet Regression
 * @author: Rishit Avadhuta
 * @description: This is a Scala implementation of the Graphlet Regression algorithm as represented in Jin et al., 2014.
 * @reference: https://github.com/rishit-avadhuta/Graphlet-Regression
 * Inputs:
 * @param X: Design matrix
 * @param Y: Response vector
 * @param nm: Subgraph search depth
 * @param r: Minimum signal strength
 * @param v: Sparsity
 * Outputs: 
 * @return beta: Estimated coefficient vector
 */
object GraphletRegression {

  /**
   * Computes the Moore-Penrose pseudoinverse using SVD with a tolerance.
   * @param mat Input matrix
   * @param tol Tolerance for singular values
   * @return Pseudoinverse of mat
   */
  def ginv(mat: DenseMatrix[Double], tol: Double = 1e-6): DenseMatrix[Double] = {
    val svd.SVD(u, s, vt) = svd(mat)
    val sInv = DenseVector(s.data.map { sigma =>
      if (sigma > tol) 1.0 / sigma else 0.0
    })
    vt.t * diag(sInv) * u.t
  }

  private def solveLinearSystem(
    mat: DenseMatrix[Double],
    rhs: DenseVector[Double]
  ): DenseVector[Double] = {
    try {
      mat \ rhs
    } catch {
      case _: Throwable => ginv(mat) * rhs
    }
  }

  private def solveLinearSystem(
    mat: DenseMatrix[Double],
    rhs: DenseMatrix[Double]
  ): DenseMatrix[Double] = {
    try {
      mat \ rhs
    } catch {
      case _: Throwable => ginv(mat) * rhs
    }
  }

  private val FamilyAuto = "auto"
  private val FamilyLinear = "linear"
  private val FamilyLogistic = "logistic"

  private def selectedMask(p: Int, selectedCols: IndexedSeq[Int]): DenseVector[Boolean] = {
    val selected = DenseVector.fill[Boolean](p)(false)
    selectedCols.foreach(selected(_) = true)
    selected
  }

  private def designFromSelected(
    XVariants: DenseMatrix[Double],
    covariates: DenseMatrix[Double],
    selectedCols: IndexedSeq[Int]
  ): (DenseMatrix[Double], Int) = {
    val Xsel =
      if (selectedCols.isEmpty) DenseMatrix.zeros[Double](XVariants.rows, 0)
      else subMatrix(XVariants, (0 until XVariants.rows), selectedCols)
    if (covariates.cols > 0) {
      (DenseMatrix.horzcat(covariates, Xsel), covariates.cols)
    } else {
      (Xsel, 0)
    }
  }

  private def computeQ(covariates: DenseMatrix[Double]): DenseMatrix[Double] =
    if (covariates.cols > 0) qr.reduced.justQ(covariates)
    else DenseMatrix.zeros[Double](covariates.rows, 0)

  private def residualize(X: DenseMatrix[Double], q: DenseMatrix[Double]): DenseMatrix[Double] =
    if (q.cols == 0) X else X - (q * (q.t * X))

  private def residualize(y: DenseVector[Double], q: DenseMatrix[Double]): DenseVector[Double] =
    if (q.cols == 0) y else y - (q * (q.t * y))

  private def standardizeColumns(
    X: DenseMatrix[Double],
    minNorm: Double = 1e-8
  ): DenseMatrix[Double] = {
    val standardized = DenseMatrix.zeros[Double](X.rows, X.cols)
    var j = 0
    while (j < X.cols) {
      var norm2 = 0.0
      var i = 0
      while (i < X.rows) {
        val value = X(i, j)
        norm2 += value * value
        i += 1
      }
      val norm = math.sqrt(norm2)
      val scale =
        if (norm > minNorm) 1.0 / norm
        else 0.0
      i = 0
      while (i < X.rows) {
        standardized(i, j) = X(i, j) * scale
        i += 1
      }
      j += 1
    }
    standardized
  }

  private def logisticCovariates(covariates: DenseMatrix[Double], n: Int): DenseMatrix[Double] =
    if (covariates.cols > 0) covariates else DenseMatrix.ones[Double](n, 1)

  private def isBinaryResponse(y: DenseVector[Double]): Boolean =
    y.forall(yi => yi == 0d || yi == 1d)

  private def resolveFamily(y: DenseVector[Double], requestedFamily: String): String =
    requestedFamily match {
      case FamilyAuto =>
        if (isBinaryResponse(y)) FamilyLogistic else FamilyLinear
      case FamilyLinear | FamilyLogistic => requestedFamily
      case other => fatal(s"Unsupported graphlet screening family '$other'.")
    }

  private def validateLogisticResponse(y: DenseVector[Double]): Unit = {
    if (!isBinaryResponse(y))
      fatal("For logistic graphlet screening, y must be numeric with all present values equal to 0 or 1.")

    val sumY = sum(y)
    if (sumY == 0d || sumY == y.length.toDouble)
      fatal("For logistic graphlet screening, y must be non-constant.")
  }

  private def prepareLinearSelectionData(
    XVariants: DenseMatrix[Double],
    y: DenseVector[Double],
    covariates: DenseMatrix[Double]
  ): (DenseMatrix[Double], DenseVector[Double]) = {
    val q = computeQ(covariates)
    val xAdjusted = residualize(XVariants, q)
    (standardizeColumns(xAdjusted), residualize(y, q))
  }

  private def prepareLogisticSelectionData(
    XVariants: DenseMatrix[Double],
    y: DenseVector[Double],
    covariates: DenseMatrix[Double],
    maxIterations: Int,
    tolerance: Double
  ): (DenseMatrix[Double], DenseVector[Double]) = {
    validateLogisticResponse(y)
    val nullCovariates = logisticCovariates(covariates, y.length)
    val nullModel = new LogisticRegressionModel(nullCovariates, y)
    val nullFit = nullModel.fit(maxIter = maxIterations, tol = tolerance)
    if (!nullFit.converged)
      fatal(
        "Failed to fit graphlet screening logistic null model: " + (
          if (nullFit.exploded)
            s"exploded at Newton iteration ${nullFit.nIter}"
          else
            "Newton iteration failed to converge"
        )
      )

    val mu = breeze.numerics.sigmoid(nullCovariates * nullFit.b)
    val sqrtW = breezeSqrt(mu *:* (1d - mu))
    val weightedCovariates = nullCovariates(::, *) *:* sqrtW
    val weightedVariants = XVariants(::, *) *:* sqrtW
    val q = computeQ(weightedCovariates)
    val xAdjusted = residualize(weightedVariants, q)
    val yAdjusted = (y - mu) /:/ sqrtW
    (standardizeColumns(xAdjusted), yAdjusted)
  }

  private def refitOlsAdjusted(
    XVariants: DenseMatrix[Double],
    y: DenseVector[Double],
    covariates: DenseMatrix[Double],
    selectedCols: IndexedSeq[Int]
  ): GSResult = {
    val n = XVariants.rows
    val p = XVariants.cols
    val selected = selectedMask(p, selectedCols)
    val betaFull = DenseVector.zeros[Double](p)
    val seFull = DenseVector.fill[Double](p)(Double.NaN)
    val tFull = DenseVector.fill[Double](p)(Double.NaN)
    val pFull = DenseVector.fill[Double](p)(Double.NaN)

    if (selectedCols.isEmpty) {
      return GSResult(betaFull, seFull, tFull, pFull, selected, nTrain = n, nTest = n)
    }

    val (design, offset) = designFromSelected(XVariants, covariates, selectedCols)
    val xtx = design.t * design
    val xtxInv = ginv(xtx)
    val betaHat = xtxInv * (design.t * y)
    val residuals = y - (design * betaHat)
    val df = n - design.cols
    val sigma2 = if (df > 0) (residuals.t * residuals) / df.toDouble else Double.NaN
    val varBeta = xtxInv * sigma2
    val se = DenseVector((0 until betaHat.length).map(i => math.sqrt(varBeta(i, i))).toArray)
    val tStat = DenseVector((0 until betaHat.length).map { i =>
      if (se(i).isNaN || se(i) == 0.0) Double.NaN else betaHat(i) / se(i)
    }.toArray)
    val pVal =
      if (df > 0) {
        implicit val basis: RandBasis = RandBasis.withSeed(0)
        val dist = StudentsT(df.toDouble)
        DenseVector((0 until tStat.length).map { i =>
          val t = math.abs(tStat(i))
          if (t.isNaN) Double.NaN else 2.0 * (1.0 - dist.cdf(t))
        }.toArray)
      } else DenseVector.fill[Double](tStat.length)(Double.NaN)

    for ((colIdx, j) <- selectedCols.zipWithIndex) {
      betaFull(colIdx) = betaHat(offset + j)
      seFull(colIdx) = se(offset + j)
      tFull(colIdx) = tStat(offset + j)
      pFull(colIdx) = pVal(offset + j)
    }

    GSResult(betaFull, seFull, tFull, pFull, selected, nTrain = n, nTest = n)
  }

  private def refitLogisticAdjusted(
    XVariants: DenseMatrix[Double],
    y: DenseVector[Double],
    covariates: DenseMatrix[Double],
    selectedCols: IndexedSeq[Int],
    maxIterations: Int,
    tolerance: Double
  ): GSResult = {
    validateLogisticResponse(y)
    val n = XVariants.rows
    val p = XVariants.cols
    val selected = selectedMask(p, selectedCols)
    val betaFull = DenseVector.zeros[Double](p)
    val seFull = DenseVector.fill[Double](p)(Double.NaN)
    val zFull = DenseVector.fill[Double](p)(Double.NaN)
    val pFull = DenseVector.fill[Double](p)(Double.NaN)

    if (selectedCols.isEmpty) {
      return GSResult(betaFull, seFull, zFull, pFull, selected, nTrain = n, nTest = n)
    }

    val covForRefit = logisticCovariates(covariates, n)
    val (design, offset) = designFromSelected(XVariants, covForRefit, selectedCols)
    val nullModel = new LogisticRegressionModel(covForRefit, y)
    val nullFit = nullModel.fit(maxIter = maxIterations, tol = tolerance)
    if (!nullFit.converged) {
      return GSResult(betaFull, seFull, zFull, pFull, selected, nTrain = n, nTest = n)
    }

    val fit = new LogisticRegressionModel(design, y).fit(Some(nullFit), maxIter = maxIterations, tol = tolerance)
    if (!fit.converged || fit.fisher.isEmpty) {
      return GSResult(betaFull, seFull, zFull, pFull, selected, nTrain = n, nTest = n)
    }

    try {
      val se = breezeSqrt(diag(inv(fit.fisher.get)))
      val z = fit.b /:/ se
      val pVal = z.map(zi => 2 * pnorm(-math.abs(zi)))
      for ((colIdx, j) <- selectedCols.zipWithIndex) {
        betaFull(colIdx) = fit.b(offset + j)
        seFull(colIdx) = se(offset + j)
        zFull(colIdx) = z(offset + j)
        pFull(colIdx) = pVal(offset + j)
      }
    } catch {
      case _: breeze.linalg.MatrixSingularException =>
      case _: breeze.linalg.NotConvergedException =>
    }

    GSResult(betaFull, seFull, zFull, pFull, selected, nTrain = n, nTest = n)
  }

  /**
   * Converts a non-negative integer to its base-'base' representation of fixed length 'len'.
   * Least-significant digit is placed in the last position.
   * @param num Non-negative integer
   * @param base Base for conversion
   * @param len Output vector length
   * @return DenseVector of digits
   */
  def vectorizeBase(num: Int, base: Int, len: Int): DenseVector[Int] = {
    require(num >= 0, "num must be a non-negative integer.")
    require(base > 1, "base must be greater than 1.")
    require(len > 0,  "len must be a positive integer.")
    val digits = Array.fill(len)(0)
    var n = num
    for (i <- (len - 1) to 0 by -1) {
      digits(i) = n - base * java.lang.Math.floorDiv(n, base)
      n = java.lang.Math.floorDiv(n, base)
    }
    DenseVector(digits)
  }

  /**
   * Returns indices of a Boolean vector that are true.
   * @param vec Boolean DenseVector
   * @return IndexedSeq of true indices
   */
  def which(vec: DenseVector[Boolean]): IndexedSeq[Int] = {
    (0 until vec.length).filter(i => vec(i))
  }

  /**
   * Returns the subvector of a DenseVector[Double] containing only the elements at indices.
   * @param vec Input vector
   * @param indices Indices to select
   * @return Subvector
   */
  def subVector(vec: DenseVector[Double], indices: IndexedSeq[Int]): DenseVector[Double] = {
    DenseVector(indices.map(vec(_)).toArray)
  }

  /**
   * Returns the submatrix of a DenseMatrix[Double] given row and column index sequences.
   * @param mat Input matrix
   * @param rowInd Row indices
   * @param colInd Column indices
   * @return Submatrix
   */
  def subMatrix(mat: DenseMatrix[Double], rowInd: IndexedSeq[Int], colInd: IndexedSeq[Int]): DenseMatrix[Double] = {
    val res = DenseMatrix.zeros[Double](rowInd.length, colInd.length)
    for (i <- rowInd.indices; j <- colInd.indices) {
      res(i, j) = mat(rowInd(i), colInd(j))
    }
    res
  }

  /**
   * Returns the subvector defined by true entries in a Boolean mask.
   * @param vec Input vector
   * @param mask Boolean mask
   * @return Subvector
   */
  def maskVector(vec: DenseVector[Double], mask: DenseVector[Boolean]): DenseVector[Double] = {
    subVector(vec, which(mask))
  }

  /**
   * Returns the submatrix consisting of the rows for which mask is true.
   * @param mat Input matrix
   * @param mask Boolean mask for rows
   * @return Submatrix
   */
  def maskMatrixRows(mat: DenseMatrix[Double], mask: DenseVector[Boolean]): DenseMatrix[Double] = {
    subMatrix(mat, which(mask), (0 until mat.cols))
  }

  /**
   * Returns the submatrix consisting of the columns for which mask is true.
   * @param mat Input matrix
   * @param mask Boolean mask for columns
   * @return Submatrix
   */
  def maskMatrixCols(mat: DenseMatrix[Double], mask: DenseVector[Boolean]): DenseMatrix[Double] = {
    subMatrix(mat, (0 until mat.rows), which(mask))
  }

  /**
   * Computes row sums of a DenseMatrix[Double].
   * @param mat Input matrix
   * @return DenseVector of row sums
   */
  def rowSums(mat: DenseMatrix[Double]): DenseVector[Double] = {
    val res = DenseVector.zeros[Double](mat.rows)
    for (i <- 0 until mat.rows) {
      res(i) = sum(mat(i, ::))
    }
    res
  }

  /**
   * Checks whether two Boolean DenseVectors are elementwise equal.
   * @param v1 First vector
   * @param v2 Second vector
   * @return True if equal, false otherwise
   */
  def allEqual(v1: DenseVector[Boolean], v2: DenseVector[Boolean]): Boolean = {
    if (v1.length != v2.length) return false
    (0 until v1.length).forall(i => v1(i) == v2(i))
  }

  case class QPSolution(solution: DenseVector[Double], value: Double)

  case class GSResult(
    beta: DenseVector[Double],
    standardError: DenseVector[Double],
    tStat: DenseVector[Double],
    pValue: DenseVector[Double],
    selected: DenseVector[Boolean],
    nTrain: Int,
    nTest: Int
  )

  /**
   * Solves a quadratic programming problem using enumeration of active sets.
   * @param Q Quadratic term matrix
   * @param dvec Linear term vector
   * @param A Constraint matrix
   * @param bvec Constraint vector
   * @return QPSolution containing solution vector and objective value
   */
  def solveQP(Q: DenseMatrix[Double],
              dvec: DenseVector[Double],
              A: DenseMatrix[Double],
              bvec: DenseVector[Double]): QPSolution = {
    val n = Q.cols
    def objective(x: DenseVector[Double]): Double = {
      0.5 * (x.t * (Q * x)) - (dvec.t * x)
    }
    def satisfiesConstraints(x: DenseVector[Double]): Boolean = {
      for (j <- 0 until A.cols) {
        val aj = A(::, j)
        if (aj.t * x < bvec(j) - 1e-8) return false
      }
      true
    }
    var bestSolution: Option[DenseVector[Double]] = None
    var bestValue: Double = Double.PositiveInfinity
    val xUnconstrainedOpt = try {
      Some(solveLinearSystem(Q, dvec))
    } catch {
      case _: Throwable => None
    }
    xUnconstrainedOpt.foreach { xUnconstrained =>
      if (satisfiesConstraints(xUnconstrained)) {
        val valUnconstrained = objective(xUnconstrained)
        bestSolution = Some(xUnconstrained)
        bestValue = valUnconstrained
      }
    }
    val totalSubsets = 1 << n
    for (maskInt <- 1 until totalSubsets) {
      val activeSet = (0 until n).filter(i => ((maskInt >> i) & 1) == 1)
      val r = activeSet.length
      val A_S = subMatrix(A, (0 until A.rows), activeSet)
      val b_S = DenseVector(activeSet.map(bvec(_)).toArray)
      val K_top = DenseMatrix.horzcat(Q, -A_S)
      val K_bottom = DenseMatrix.horzcat(A_S.t, DenseMatrix.zeros[Double](r, r))
      val K = DenseMatrix.vertcat(K_top, K_bottom)
      val rhs = DenseVector.vertcat(dvec, b_S)
      val solOpt = try {
        Some(solveLinearSystem(K, rhs))
      } catch {
        case _: Throwable => None
      }
      solOpt.foreach { sol =>
        val xCandidate = sol(0 until n)
        if (satisfiesConstraints(xCandidate)) {
          val objVal = objective(xCandidate)
          if (objVal < bestValue) {
            bestValue = objVal
            bestSolution = Some(xCandidate)
          }
        }
      }
    }
    bestSolution match {
      case Some(sol) => QPSolution(sol, bestValue)
      case None => throw new Exception("QP problem infeasible or no solution found")
    }
  }

  /**
   * Penalized MLE (used in the cleaning step).
   * @param gram Gram matrix
   * @param y Response vector
   * @param lambda Penalty parameter
   * @param uu Constraint threshold
   * @return Estimated coefficient vector
   */
  def PMLE(gram: DenseMatrix[Double],
           y: DenseVector[Double],
           lambda: Double,
           uu: Double): DenseVector[Double] = {
    val n = y.length
    var bestB = DenseVector.zeros[Double](n)
    var bestL = Double.PositiveInfinity
    val total = math.pow(3, n).toInt
    for (k <- 0 until total) {
      val idx = vectorizeBase(k, 3, n).map(_ - 1)
      val cluster = idx.map(_ != 0)
      val card = cluster.data.count(x => x)
      if (card == 0) {
        val lt = 0.0
        val bt = DenseVector.zeros[Double](n)
        if (lt < bestL) {
          bestL = lt
          bestB = bt.copy
        }
      } else {
        val activeIndices = which(cluster)
        val signs = DenseVector(activeIndices.map(i => idx(i)).toArray)
        val yc = subVector(y, activeIndices)
        val gramCluster = subMatrix(gram, activeIndices, activeIndices)
        val uuCluster = DenseVector.fill[Double](card){uu}
        val amat = diag(signs.map(_.toDouble))
        val qpSol = solveQP(gramCluster, yc, amat, uuCluster)
        val bt = qpSol.solution
        val lt = qpSol.value + math.pow(lambda, 2) * card / 2.0
        if (lt < bestL) {
          val bCandidate = DenseVector.zeros[Double](n)
          for ((idxVal, j) <- activeIndices.zipWithIndex)
            bCandidate(idxVal) = bt(j)
          bestL = lt
          bestB = bCandidate
        }
      }
    }
    bestB
  }

  /**
   * Thresholds the Gram matrix.
   * @param gramFull Input Gram matrix
   * @param delta Threshold value
   * @return (thresholdedGram, gramBias)
   */
  def thresholdGram(gramFull: DenseMatrix[Double], delta: Double): (DenseMatrix[Double], DenseMatrix[Double]) = {
    require(gramFull.rows == gramFull.cols, "Input matrix must be square.")
    val gramBias = gramFull.mapValues(v => if (math.abs(v) < delta) v else 0.0)
    val gramSd = gramFull - gramBias
    (gramSd, gramBias)
  }

  /**
   * Performs the screening step.
   * @param yTilde Transformed response vector
   * @param gram Gram matrix
   * @param cgAll List of connected subgraphs
   * @param nm Maximum subgraph size
   * @param v Sparsity parameter
   * @param r Signal strength parameter
   * @param q0 Default threshold
   * @param scale Scaling factor
   * @return Boolean mask of survivors
   */
  def screeningStep(yTilde: DenseVector[Double],
                    gram: DenseMatrix[Double],
                    cgAll: List[DenseMatrix[Int]],
                    pGlobal: Int,
                    nm: Int,
                    v: Double,
                    r: Double,
                    q0: Double = 0.1,
                    scale: Double = 1.0): DenseVector[Boolean] = {
    def safeSqrt(x: Double): Double = if (x > 0.0) math.sqrt(x) else 0.0
    val p  = yTilde.length
    val pAsymptotic = math.max(2, pGlobal)
    val q1 = math.pow(v + r, 2) / (r * r) / 4.0
    val tau  = math.sqrt(2.0 * r * math.log(pAsymptotic))
    val tau1 = math.sqrt(q1) * tau * math.sqrt(scale)
    var survivor = yTilde.map(x => math.abs(x) > tau1)
    var indices  = which(survivor)
    if (nm > 1) {
      var ii = 2
      val maxKK = (2.0 / v).toInt
      while (ii <= nm && ii <= maxKK) {
        val cgIi = cgAll.lift(ii - 1).getOrElse(DenseMatrix.zeros[Int](0, 0))
        if (cgIi.rows == 0) {
          ii = nm + 1
        } else {
          val nii = cgIi.rows
          val indicatorIi = DenseVector.fill[Boolean](p)(false)
          val indicesSet = indices.toSet
          for (jj <- 0 until nii) {
            val cgRow = cgIi(jj, ::).t.toArray
            val ds = cgRow.filterNot(indicesSet.contains)
            val d  = ds.length
            if (d > 0 && d <= (1.0 / v)) {
              val es = cgRow.filter(indicesSet.contains)
              val dsIdx = ds.toIndexedSeq
              val ncmatrx: DenseMatrix[Double] =
                if (es.nonEmpty) {
                  val esIdx = es.toIndexedSeq
                  val g_ds_ds = subMatrix(gram, dsIdx, dsIdx)
                  val g_ds_es = subMatrix(gram, dsIdx, esIdx)
                  val g_es_es = subMatrix(gram, esIdx, esIdx)
                  val g_es_ds = subMatrix(gram, esIdx, dsIdx)
                  g_ds_ds - (g_ds_es * solveLinearSystem(g_es_es, g_es_ds))
                } else {
                  subMatrix(gram, dsIdx, dsIdx)
                }
              val ww: Double = d match {
                case 1 => ncmatrx(0, 0)
                case 2 =>
                  val ev = eigSym(ncmatrx).eigenvalues
                  d.toDouble * breeze.linalg.min(ev)
                case _ =>
                  val numComb = 1 << (d - 1)
                  var minVal = Double.PositiveInfinity
                  val dvecZero = DenseVector.zeros[Double](d)
                  for (kk <- 0 until numComb) {
                    val idxVec = vectorizeBase(kk, 2, d - 1)
                    val signsArr = Array(1) ++ idxVec.toArray.map(b => 2 * b - 1)
                    val amat = diag(DenseVector(signsArr.map(_.toDouble)))
                    val bvec = DenseVector.ones[Double](d)
                    try {
                      val res = solveQP(ncmatrx, dvecZero, amat, bvec)
                      if (res.value < minVal) minVal = res.value
                    } catch {
                      case _: Throwable =>
                    }
                  }
                  if (minVal.isInfinite) 0.0 else minVal
              }
              val parityFactor = if (d % 2 == 1) 1.0 else 0.0
              val wwCondition = ww > (v / r) * (2 * d - (d - math.sqrt(d * d - 1)) * parityFactor)
              val teb: Double = if (wwCondition) {
                val coor = ww * r
                val expr1 = coor * coor - 2 * coor * d * v + v * v
                val expr2 = coor * coor - 2 * coor * d * v
                val term1Sqrt = safeSqrt(expr1)
                val term2Sqrt = safeSqrt(expr2)
                val deltadelta = v * v / coor / 4.0 - term1Sqrt + term2Sqrt
                scale * 2.0 * math.log(pAsymptotic) * (
                  coor * 1.25 - v * d / 2.0 - term2Sqrt + deltadelta * parityFactor
                )
              } else {
                q0
              }
              val fullIdx = cgRow.toIndexedSeq
              val yAll = subVector(yTilde, fullIdx)
              val gramAll = subMatrix(gram, fullIdx, fullIdx)
              val testStatBase = (yAll dot (ginv(gramAll) * yAll))
              val testStat = if (es.nonEmpty) {
                val esIdx = es.toIndexedSeq
                val yEs = subVector(yTilde, esIdx)
                val gramEs = subMatrix(gram, esIdx, esIdx)
                val adjust = (yEs dot (ginv(gramEs) * yEs))
                testStatBase - adjust
              } else {
                testStatBase
              }
              if (testStat > teb) {
                dsIdx.foreach { idx => indicatorIi(idx) = true }
              }
            }
          }
          survivor = DenseVector((0 until p).map(j => survivor(j) || indicatorIi(j)).toArray)
          indices  = which(survivor)
          ii += 1
        }
      }
    }
    survivor
  }

  /**
   * Creates the full connected component starting from startIdx using BFS.
   * @param startIdx Starting index
   * @param omega Adjacency matrix
   * @param remain Boolean mask of remaining nodes
   * @return IndexedSeq of cluster indices
   */
  private def growClusterWithSize(startIdx: Int,
                                  omega: DenseMatrix[Double],
                                  remain: DenseVector[Boolean]): IndexedSeq[Int] = {
    val visited = mutable.LinkedHashSet[Int]()
    val queue   = mutable.Queue[Int]()
    val enqueued = mutable.Set[Int]()
    queue.enqueue(startIdx)
    enqueued += startIdx
    while (queue.nonEmpty) {
      val node = queue.dequeue()
      if (!visited.contains(node)) {
        visited += node
        val neighbors = (0 until omega.cols).iterator
          .filter(j => remain(j) && omega(node, j) != 0.0 && !visited.contains(j) && !enqueued.contains(j))
        neighbors.foreach { nbr =>
          queue.enqueue(nbr)
          enqueued += nbr
        }
      }
    }
    visited.toIndexedSeq
  }

  private def growClusterUpToSize(startIdx: Int,
                                  omega: DenseMatrix[Double],
                                  remain: DenseVector[Boolean],
                                  maxClusterSize: Int): IndexedSeq[Int] = {
    val visited = mutable.LinkedHashSet[Int]()
    val queue = mutable.Queue[Int]()
    val enqueued = mutable.Set[Int]()
    queue.enqueue(startIdx)
    enqueued += startIdx
    while (queue.nonEmpty && visited.size < maxClusterSize) {
      val node = queue.dequeue()
      if (!visited.contains(node)) {
        visited += node
        if (visited.size < maxClusterSize) {
          val neighbors = (0 until omega.cols).iterator
            .filter(j => remain(j) && omega(node, j) != 0.0 && !visited.contains(j) && !enqueued.contains(j))
          neighbors.foreach { nbr =>
            if (visited.size + queue.size < maxClusterSize) {
              queue.enqueue(nbr)
              enqueued += nbr
            }
          }
        }
      }
    }
    visited.toIndexedSeq
  }

  /**
   * Carries out the cleaning stage.
   * @param survivor Boolean mask of survivors
   * @param yTilde Transformed response vector
   * @param gram Gram matrix
   * @param lambda Penalty parameter
   * @param uu Constraint threshold
   * @param maxClusterSize Maximum cluster size to explore
   * @return Estimated coefficient vector
   */
  def cleaningStep(survivor: DenseVector[Boolean],
                   yTilde: DenseVector[Double],
                   gram: DenseMatrix[Double],
                   lambda: Double,
                   uu: Double,
                   maxClusterSize: Int = 20,
                   approximateLargeClusters: Boolean = false): DenseVector[Double] = {
    require(maxClusterSize >= 1, "maxClusterSize must be at least 1.")
    val p = gram.cols
    val survIndices = which(survivor)
    val nSurvivor = survIndices.length
    println(s"[GS] cleaningStep: survivors=$nSurvivor, maxClusterSize=$maxClusterSize, approximateLargeClusters=$approximateLargeClusters")
    val yt = subVector(yTilde, survIndices)
    val omega = subMatrix(gram, survIndices, survIndices)
    var beta = DenseVector.zeros[Double](nSurvivor)
    var remain = DenseVector.fill[Boolean](nSurvivor)(true)
    var nClusters = 0
    val tStart = System.nanoTime()
    while (remain.data.exists(x => x)) {
      val idxCandidates = which(remain)
      val i = idxCandidates.head
      val fullClusterIndices = growClusterWithSize(i, omega, remain)
      val clusterIndices =
        if (fullClusterIndices.length > maxClusterSize && approximateLargeClusters) {
          println(
            s"[GS] cleaningStep: truncating connected component from size ${fullClusterIndices.length} to $maxClusterSize for approximate cleaning"
          )
          growClusterUpToSize(i, omega, remain, maxClusterSize)
        } else {
          fullClusterIndices
        }
      if (fullClusterIndices.length > maxClusterSize && !approximateLargeClusters)
        fatal(
          s"Graphlet screening encountered a connected component of size ${fullClusterIndices.length}, " +
            s"which exceeds maxClusterSize=$maxClusterSize. " +
            s"This matches the original R implementation's exact-cleaning limit; " +
            s"increase maxClusterSize or sparsify the graph more aggressively."
        )
      val omegaCluster = subMatrix(omega, clusterIndices, clusterIndices)
      val ytCluster = subVector(yt, clusterIndices)
      val betaCluster = PMLE(omegaCluster, ytCluster, lambda, uu)
      for ((origIdx, j) <- clusterIndices.zipWithIndex)
        beta(origIdx) = betaCluster(j)
      for (origIdx <- clusterIndices)
        remain(origIdx) = false
      nClusters += 1
      if (nClusters % 25 == 0) {
        val nRemain = which(remain).length
        val elapsedSec = (System.nanoTime() - tStart) / 1e9
        println(f"[GS] cleaningStep: clusters=$nClusters, remaining=$nRemain, elapsed=${elapsedSec}%.2fs")
      }
    }
    val betaGS = DenseVector.zeros[Double](p)
    for ((origIdx, j) <- survIndices.zipWithIndex)
      betaGS(origIdx) = beta(j)
    val elapsedSec = (System.nanoTime() - tStart) / 1e9
    println(f"[GS] cleaningStep: complete, clusters=$nClusters, elapsed=${elapsedSec}%.2fs")
    betaGS
  }

  /**
   * Checks if a Boolean matrix is symmetric.
   * @param matrix Input Boolean matrix
   * @return True if symmetric, false otherwise
   */
  def isSymmetricBoolean(matrix: DenseMatrix[Boolean]): Boolean = {
    if (matrix.rows != matrix.cols) return false
    val n = matrix.rows
    for (i <- 0 until n; j <- 0 until n) {
      if (matrix(i, j) != matrix(j, i)) return false
    }
    true
  }

  /**
   * Computes all connected subgraphs up to level 'lc'.
   * @param adjacencyMatrix Symmetric adjacency matrix
   * @param lc Number of levels to compute
   * @return List of DenseMatrix[Int] for each level
   */
  def findAllCG(adjacencyMatrix: DenseMatrix[Boolean], lc: Int): List[DenseMatrix[Int]] = {
    if (adjacencyMatrix.rows != adjacencyMatrix.cols || !isSymmetricBoolean(adjacencyMatrix))
      throw new IllegalArgumentException("The adjacency matrix is not symmetric!")
    val p = adjacencyMatrix.rows
    val cgAll = new Array[DenseMatrix[Int]](lc)
    cgAll(0) = new DenseMatrix(p, 1, (0 until p).toArray)
    if (lc >= 2) {
      val nonZeroCoordinates = new scala.collection.mutable.ArrayBuffer[(Int, Int)]()
      for (i <- 0 until adjacencyMatrix.rows) {
        for (j <- 0 until adjacencyMatrix.cols) {
          if (adjacencyMatrix(i, j)) {
            nonZeroCoordinates += ((i, j))
          }
        }
      }
      val supportAdjacency = nonZeroCoordinates.toArray
      val filteredEdges = supportAdjacency.filter { case (row, col) => row < col }
      val numEdges = filteredEdges.length
      val edgesMatrix = DenseMatrix.zeros[Int](numEdges, 2)
      for (i <- 0 until numEdges) {
        edgesMatrix(i, 0) = filteredEdges(i)._1
        edgesMatrix(i, 1) = filteredEdges(i)._2
      }
      cgAll(1) = edgesMatrix
    }
    if (lc >= 3) {
      for (ii <- 3 to lc) {
        val next = try {
          findCG(adjacencyMatrix, cgAll(ii - 2))
        } catch {
          case _: IllegalArgumentException => DenseMatrix.zeros[Int](0, ii)
        }
        cgAll(ii - 1) = next
        if (next.rows == 0) {
          var jj = ii + 1
          while (jj <= lc) {
            cgAll(jj - 1) = DenseMatrix.zeros[Int](0, jj)
            jj += 1
          }
        }
      }
    }
    cgAll.toList
  }

  /**
   * Extends each connected subgraph by one vertex if possible.
   * @param adjacencyMatrix Symmetric adjacency matrix
   * @param cgInitial Each row is a connected subgraph
   * @return DenseMatrix whose rows are the extended subgraphs
   */
  def findCG(adjacencyMatrix: DenseMatrix[Boolean], cgInitial: DenseMatrix[Int]): DenseMatrix[Int] = {
    if (!isSymmetricBoolean(adjacencyMatrix)) {
      throw new IllegalArgumentException("The adjacency matrix is not symmetric!")
    }
    val p = adjacencyMatrix.rows
    val n = adjacencyMatrix.rows
    val diagEntries: DenseVector[Boolean] = diag(adjacencyMatrix)
    if (diagEntries.exists(identity)) {
      for (i <- 0 until n) {
        adjacencyMatrix(i, i) = false
      }
    }
    val flatEntries = cgInitial.toArray
    val sortedEntries = flatEntries.sorted
    if (sortedEntries.toSeq == (1 to p).toSeq) {
      val cgInitial = new DenseMatrix(rows = p, cols = 1, data = flatEntries)
    }
    val nCgLi = cgInitial.rows
    if (nCgLi == 0) {
      throw new IllegalArgumentException("No connected subgraphs with " + cgInitial.rows + " nodes detected.")
    } else {
      val nCg = cgInitial.rows
      val li = cgInitial.cols
      val nk = li + 1
      var cgNew = DenseMatrix.zeros[Int](0, nk)
      for (j <- 0 until nCg) {
        val cg0: Array[Int] = cgInitial(j, ::).inner.toArray
        val neighborSet: Set[Int] = cg0.flatMap { u =>
          (0 until p).collect { case v if adjacencyMatrix(u, v) => v }
        }.toSet -- cg0.toSet
        val neighbors = neighborSet.toArray.sorted
        val lenN      = neighbors.length
        if (lenN > 0) {
          val block = DenseMatrix.zeros[Int](lenN, nk)
          for {
            r <- 0 until lenN
            c <- 0 to li
          } block(r, c) = if (c < li) cg0(c) else neighbors(r)
          cgNew =
            if (cgNew.rows == 0) block
            else breeze.linalg.DenseMatrix.vertcat(cgNew, block)
        }
      }
      if (cgNew.rows == 0) {
        throw new IllegalArgumentException("No connected subgraphs with " + nk + " nodes detected.")
      } else {
        val sortedRows: Seq[Array[Int]] = (0 until cgNew.rows).map { i =>
          val rowArr = cgNew(i, ::).inner.toArray
          scala.util.Sorting.quickSort(rowArr)
          rowArr
        }
        cgNew = DenseMatrix(sortedRows: _*)
        var iii = 1
        var nrCg = cgNew.rows
        var nlCg = cgNew.cols
        while (iii < nrCg) {
          val targetRow: Array[Int] = cgNew(iii, ::).inner.toArray
          val subRows: IndexedSeq[(Int, Array[Int])] =
            (iii + 1 until nrCg).map(i => (i, cgNew(i, ::).inner.toArray))
          val idCurrent: Seq[Int] =
            subRows.collect { case (idx, row) if row.sameElements(targetRow) => idx }
          val diffCurrent: Seq[Int] = (0 until nrCg).filterNot(idCurrent.contains)
          val keptRows: Seq[Array[Int]] = diffCurrent.map(i => cgNew(i, ::).inner.toArray)
          cgNew = DenseMatrix(keptRows: _*)
          nrCg = diffCurrent.length
          iii  += 1
        }
      }
      cgNew
    }
  }

  /**
   * Iterative Graphlet Screening.
   * @param yTilde Transformed response vector
   * @param gram Thresholded Gram matrix
   * @param gramBias Gram bias matrix
   * @param cgAll List of connected subgraphs
   * @param sp Sparsity parameter
   * @param tau Signal strength parameter
   * @param nm Maximum subgraph size
   * @param q0 Default threshold
   * @param scale Scaling factor
   * @param maxIter Maximum number of iterations
   * @param stdThresh Standard deviation threshold
   * @param betaInitial Optional initial beta
   * @return (Estimated beta vector, number of iterations)
   */
  def iterGS(yTilde: DenseVector[Double],
             gram: DenseMatrix[Double],
             gramBias: DenseMatrix[Double],
             cgAll: List[DenseMatrix[Int]],
             pGlobal: Int,
             sp: Double,
             tau: Double,
             nm: Int,
             q0: Double = 0.1,
             scale: Double = 1.0,
             maxIter: Int = 3,
             stdThresh: Double = 1.05,
             betaInitial: Option[DenseVector[Double]] = None,
             maxClusterSize: Int = 7,
             approximateLargeClusters: Boolean = false
            ): (DenseVector[Double], Int) = {
    val p = gram.cols
    val pAsymptotic = math.max(2, pGlobal)
    val r = math.pow(tau, 2) / (2 * math.log(pAsymptotic))
    val v = 1.0 - math.log(sp) / math.log(pAsymptotic)
    val uu = math.sqrt(2 * r * math.log(pAsymptotic))
    val lambda = math.sqrt(2 * v * math.log(pAsymptotic))
    val betaInit = betaInitial.getOrElse {
      val signY = yTilde.map(math.signum)
      val absY  = yTilde.map(math.abs)
      val indicator = absY.map(x => if (x > uu) 1.0 else 0.0)
      (signY *:* indicator) * uu
    }
    var betaGS = betaInit.copy
    var w = yTilde.copy
    var nIteration = 0
    println(s"[GS] iterGS: start p=$p, nm=$nm, maxIter=$maxIter, maxClusterSize=$maxClusterSize, approximateLargeClusters=$approximateLargeClusters")
    for (it <- 1 to maxIter) {
      val tIterStart = System.nanoTime()
      val meanW = sum(w) / p.toDouble
      val lastWStd = math.sqrt((sum(w *:* w) - p * meanW * meanW) / (p - 1))
      val lastBeta = betaGS.copy
      val lastNonZero = lastBeta.map(x => x != 0.0)
      val nzIndices = which(lastNonZero)
      val adjustment: DenseVector[Double] =
        if (nzIndices.isEmpty) {
          DenseVector.zeros[Double](p)
        } else {
          val gramBiasSub = subMatrix(gramBias, (0 until gramBias.rows), nzIndices)
          val lastBetaSub = subVector(lastBeta, nzIndices)
          gramBiasSub * lastBetaSub
        }
      w = yTilde - adjustment
      val newMeanW = sum(w) / p.toDouble
      val newWStd = math.sqrt((sum(w *:* w) - p * newMeanW * newMeanW) / (p - 1))
      println(f"[GS] iterGS: iteration=$it, lastWStd=$lastWStd%.4f, newWStd=$newWStd%.4f")
      if (newWStd > stdThresh * lastWStd) {
        nIteration = it - 1
        val elapsedSec = (System.nanoTime() - tIterStart) / 1e9
        println(f"[GS] iterGS: iteration=$it early-stop (std threshold), elapsed=${elapsedSec}%.2fs")
        return (betaGS, nIteration)
      }
      val tScreenStart = System.nanoTime()
      val survivorMask = screeningStep(w, gram, cgAll, pGlobal, nm, v, r, q0, scale)
      val screenSec = (System.nanoTime() - tScreenStart) / 1e9
      val nSurvivor = which(survivorMask).length
      println(f"[GS] iterGS: iteration=$it screening complete, survivors=$nSurvivor, elapsed=${screenSec}%.2fs")
      val tCleanStart = System.nanoTime()
      betaGS = cleaningStep(survivorMask, w, gram, lambda, uu, maxClusterSize, approximateLargeClusters)
      val cleanSec = (System.nanoTime() - tCleanStart) / 1e9
      val nnzBeta = which(betaGS.map(_ != 0.0)).length
      println(f"[GS] iterGS: iteration=$it cleaning complete, nnzBeta=$nnzBeta, elapsed=${cleanSec}%.2fs")
      nIteration = it
      val iterSec = (System.nanoTime() - tIterStart) / 1e9
      println(f"[GS] iterGS: iteration=$it done, totalElapsed=${iterSec}%.2fs")
    }
    println(s"[GS] iterGS: finished, nIteration=$nIteration")
    (betaGS, nIteration)
  }

  /**
   * Selection-aware inference via sample-splitting:
   * - Use a training split for Graphlet Screening (selection).
   * - Refit OLS on the held-out split for unbiased (selection-aware) stats.
   */
  def executeSelectionAware(
    XVariants: DenseMatrix[Double],
    Y: DenseVector[Double],
    covariates: DenseMatrix[Double],
    family: String = FamilyAuto,
    nm: Int = 3,
    r: Double = 3.5,
    selectionAware: Boolean = true,
    splitFraction: Double = 0.5,
    seed: Int = 1,
    maxClusterSize: Int = 20,
    sparsityLevel: Double = -1.0,
    approximateLargeClusters: Boolean = false,
    pGlobal: Int = -1,
    maxIterations: Int = 25,
    tolerance: Double = 1e-6
  ): GSResult = {
    if (selectionAware)
      require(splitFraction > 0.0 && splitFraction < 1.0, "splitFraction must be in (0, 1)")

    val n = XVariants.rows
    val pLocal = XVariants.cols
    require(n > 1, "Need at least 2 samples for sample splitting.")
    require(pLocal > 0, "Need at least 1 variant for graphlet screening.")
    val effectivePGlobal = if (pGlobal > 0) pGlobal else pLocal
    val defaultSp = math.pow(effectivePGlobal.toDouble, 0.5)
    val effectiveSp =
      if (sparsityLevel > 0.0) sparsityLevel
      else defaultSp
    println(
      s"[GS] executeSelectionAware: n=$n, pLocal=$pLocal, pGlobal=$effectivePGlobal, family=$family, nm=$nm, r=$r, sp=$effectiveSp, selectionAware=$selectionAware, splitFraction=$splitFraction, maxClusterSize=$maxClusterSize, approximateLargeClusters=$approximateLargeClusters"
    )

    val rng = new Random(seed)
    val allIdx = (0 until n).toIndexedSeq
    val (trainIdx, testIdx) =
      if (selectionAware) {
        val indices = rng.shuffle(allIdx.toList)
        val nTrain = math.max(1, math.min(n - 1, (n * splitFraction).toInt))
        (indices.take(nTrain).toIndexedSeq, indices.drop(nTrain).toIndexedSeq)
      } else {
        (allIdx, allIdx)
      }

    val allCols = (0 until pLocal).toIndexedSeq
    val covCols = (0 until covariates.cols).toIndexedSeq
    val Xtrain = subMatrix(XVariants, trainIdx, allCols)
    val Ytrain = subVector(Y, trainIdx)
    val covTrain = subMatrix(covariates, trainIdx, covCols)
    val Xtest = subMatrix(XVariants, testIdx, allCols)
    val Ytest = subVector(Y, testIdx)
    val covTest = subMatrix(covariates, testIdx, covCols)

    val resolvedFamily = resolveFamily(Y, family)
    val (selectionXTrain, selectionYTrain) =
      resolvedFamily match {
        case FamilyLinear => prepareLinearSelectionData(Xtrain, Ytrain, covTrain)
        case FamilyLogistic => prepareLogisticSelectionData(Xtrain, Ytrain, covTrain, maxIterations, tolerance)
      }

    val p = XVariants.cols
    val gram = selectionXTrain.t * selectionXTrain
    val delta = 1.0 / math.log(math.max(2, effectivePGlobal))
    val (gramThresh, gramBias) = thresholdGram(gram, delta)
    val neighbor = gramThresh.map(x => x != 0.0)
    val cgAll = findAllCG(neighbor, nm)
    val yTilde = selectionXTrain.t * selectionYTrain
    val defaultTau = math.sqrt(2 * math.log(math.max(2, effectivePGlobal)) * r)
    val spPerturb = effectiveSp * (1 + 0.1 * (if (rng.nextBoolean()) 1.0 else -1.0))
    val tauPerturb = defaultTau * (1 + 0.1 * (if (rng.nextBoolean()) 1.0 else -1.0))
    val tIterStart = System.nanoTime()
    val (betaGS, _) = iterGS(
      yTilde,
      gramThresh,
      gramBias,
      cgAll,
      effectivePGlobal,
      spPerturb,
      tauPerturb,
      nm,
      maxClusterSize = maxClusterSize,
      approximateLargeClusters = approximateLargeClusters
    )
    val iterSec = (System.nanoTime() - tIterStart) / 1e9

    val selected = which(betaGS.map(_ != 0.0))
    println(s"[GS] executeSelectionAware: selected=${selected.length}, iterElapsedSec=$iterSec")
    val refit =
      resolvedFamily match {
        case FamilyLinear => refitOlsAdjusted(Xtest, Ytest, covTest, selected)
        case FamilyLogistic => refitLogisticAdjusted(Xtest, Ytest, covTest, selected, maxIterations, tolerance)
      }

    GSResult(
      refit.beta,
      refit.standardError,
      refit.tStat,
      refit.pValue,
      selectedMask(pLocal, selected),
      trainIdx.length,
      testIdx.length
    )
  }

  /**
   * Runs Graphlet Screening and returns selection-aware statistics by default.
   */
  def execute(XVariants: DenseMatrix[Double],
              Y: DenseVector[Double],
              covariates: DenseMatrix[Double],
              family: String = FamilyAuto,
              nm: Int = 3,
              r: Double = 3.5,
              selectionAware: Boolean = true,
              splitFraction: Double = 0.5,
              seed: Int = 1,
              maxClusterSize: Int = 20,
              sparsityLevel: Double = -1.0,
              approximateLargeClusters: Boolean = false,
              pGlobal: Int = -1,
              maxIterations: Int = 25,
              tolerance: Double = 1e-6): GSResult =
    executeSelectionAware(
      XVariants,
      Y,
      covariates,
      family,
      nm,
      r,
      selectionAware,
      splitFraction,
      seed,
      maxClusterSize,
      sparsityLevel,
      approximateLargeClusters,
      pGlobal,
      maxIterations,
      tolerance
    )
}
case class GraphletScreening(
  yFields: Seq[String],
  xField: String,
  covFields: Seq[String],
  rowBlockSize: Int,
  blockOverlap: Int,
  passThrough: Seq[String],
  family: String,
  mode: String,
  selectionAware: Boolean,
  splitFraction: Double,
  seed: Int,
  nm: Int,
  r: Double = 3.5,
  maxClusterSize: Int = 20,
  sparsityLevel: Double = -1.0,
  maxIterations: Int = 25,
  tolerance: Double = 1e-6,
) extends MatrixToTableFunction {

  override def typ(childType: MatrixType): TableType = {
    val passThroughType = TStruct(passThrough.map(f => f -> childType.rowType.field(f).typ): _*)
    val schema = TStruct(
      ("n", TInt32),
      ("n_total", TInt32),
      ("n_train", TInt32),
      ("n_test", TInt32),
      ("selected", TArray(TBoolean)),
      ("beta", TArray(TFloat64)),
      ("standard_error", TArray(TFloat64)),
      ("t_stat", TArray(TFloat64)),
      ("p_value", TArray(TFloat64)),
    )
    TableType(
      childType.rowKeyStruct ++ passThroughType ++ schema,
      childType.rowKey,
      TStruct.empty,
    )
  }

  def preservesPartitionCounts: Boolean = true

  def execute(ctx: ExecuteContext, mv: MatrixValue): TableValue = {
    val (y, cov, completeColIdx) =
      RegressionUtils.getPhenosCovCompleteSamples(mv, yFields.toArray, covFields.toArray)
    val n = y.rows
    val nPhenotypes = y.cols
    val k = cov.cols

    val normalizedMode = mode match {
      case "global" | "block" => mode
      case other => fatal(s"Unsupported graphlet screening mode '$other'. Expected 'global' or 'block'.")
    }

    info(
      s"graphlet_screening: running in $normalizedMode mode on $n samples for ${nPhenotypes} response ${plural(nPhenotypes, "variable")} y,\n" +
        s"    with input variable x, $k additional ${plural(k, "covariate")}, family=$family, selectionAware=$selectionAware, nm=$nm, r=$r, maxClusterSize=$maxClusterSize..."
    )

    val backend = HailContext.backend
    val completeColIdxBc = backend.broadcast(completeColIdx)
    val yBc = backend.broadcast(y)
    val covBc = backend.broadcast(cov)

    val fullRowType = mv.rvd.rowPType
    val entryArrayType = MatrixType.getEntryArrayType(fullRowType)
    val entryType = entryArrayType.elementType.asInstanceOf[PStruct]
    assert(entryType.field(xField).typ.virtualType == TFloat64)

    val entryArrayIdx = MatrixType.getEntriesIndex(fullRowType)
    val fieldIdx = entryType.fieldIdx(xField)

    val tableType = typ(mv.typ)
    val rvdType = tableType.canonicalRVDType
    val copiedFieldIndices = (mv.typ.rowKey ++ passThrough).map(fullRowType.fieldIdx(_)).toArray
    val partitionCounts = mv.rvd.countPerPartition()
    val partitionOffsets = new Array[Int](partitionCounts.length)
    var runningOffset = 0
    partitionCounts.zipWithIndex.foreach { case (count, idx) =>
      if (count > Int.MaxValue)
        fatal(s"Graphlet screening encountered a partition with $count rows, which exceeds the supported in-memory limit.")
      partitionOffsets(idx) = runningOffset
      runningOffset += count.toInt
    }
    val pGlobal = runningOffset
    if (pGlobal == 0)
      fatal("Graphlet screening requires at least one variant row.")

    val sm = ctx.stateManager

    def emitResultRow(
      rvb: RegionValueBuilder,
      rowRegion: Region,
      ptr: Long,
      rowIdx: Int,
      results: Array[GraphletRegression.GSResult]
    ): Long = {
      rvb.start(rvdType.rowType)
      rvb.startStruct()
      rvb.addFields(fullRowType, rowRegion, ptr, copiedFieldIndices)
      rvb.addInt(n)
      rvb.addInt(n)
      rvb.addInt(results.head.nTrain)
      rvb.addInt(results.head.nTest)

      rvb.startArray(nPhenotypes)
      var j = 0
      while (j < nPhenotypes) {
        rvb.addBoolean(results(j).selected(rowIdx))
        j += 1
      }
      rvb.endArray()

      rvb.startArray(nPhenotypes)
      j = 0
      while (j < nPhenotypes) {
        rvb.addDouble(results(j).beta(rowIdx))
        j += 1
      }
      rvb.endArray()

      rvb.startArray(nPhenotypes)
      j = 0
      while (j < nPhenotypes) {
        rvb.addDouble(results(j).standardError(rowIdx))
        j += 1
      }
      rvb.endArray()

      rvb.startArray(nPhenotypes)
      j = 0
      while (j < nPhenotypes) {
        rvb.addDouble(results(j).tStat(rowIdx))
        j += 1
      }
      rvb.endArray()

      rvb.startArray(nPhenotypes)
      j = 0
      while (j < nPhenotypes) {
        rvb.addDouble(results(j).pValue(rowIdx))
        j += 1
      }
      rvb.endArray()

      rvb.endStruct()
      rvb.end()
    }

    val newRVD =
      if (normalizedMode == "global") {
        val rawXData = new Array[Double](n * pGlobal)
        val missingCompleteCols = new IntArrayBuilder()
        val enc = TypedCodecSpec(ctx, fullRowType, BufferSpec.wireSpec)
        val encodedRows = mv.rvd.collectAsBytes(ctx, enc)
        val (decodedRowPType: PStruct, dec) = enc.buildDecoder(ctx, mv.rvd.rowType)

        ctx.r.pool.scopedRegion { region =>
          var globalRowIdx = 0
          RegionValue.fromBytes(ctx.theHailClassLoader, dec, region, encodedRows.iterator).foreach { ptr =>
            RegressionUtils.setMeanImputedDoubles(
              rawXData,
              globalRowIdx * n,
              completeColIdx,
              missingCompleteCols,
              ptr,
              decodedRowPType,
              entryArrayType,
              entryType,
              entryArrayIdx,
              fieldIdx,
            )
            globalRowIdx += 1
            region.clear()
          }
          if (globalRowIdx != pGlobal)
            fatal(s"Graphlet screening expected $pGlobal rows in global mode but decoded $globalRowIdx.")
        }

        val rawX = new DenseMatrix[Double](n, pGlobal, rawXData)
        val results = new Array[GraphletRegression.GSResult](nPhenotypes)
        for (phenoIdx <- 0 until nPhenotypes) {
          val yVec = y(::, phenoIdx)
          results(phenoIdx) = GraphletRegression.execute(
            rawX,
            yVec,
            cov,
            family = family,
            nm = nm,
            r = r,
            selectionAware = selectionAware,
            splitFraction = splitFraction,
            seed = seed + phenoIdx,
            maxClusterSize = maxClusterSize,
            sparsityLevel = sparsityLevel,
            approximateLargeClusters = false,
            pGlobal = pGlobal,
            maxIterations = maxIterations,
            tolerance = tolerance,
          )
        }

        val resultsBc = backend.broadcast(results)
        val partitionOffsetsBc = backend.broadcast(partitionOffsets)
        mv.rvd.mapPartitionsWithContextAndIndex(rvdType) { (partIdx, partitionCtx, it) =>
          val rvb = partitionCtx.rvb
          val offset = partitionOffsetsBc.value(partIdx)
          var localIdx = 0
          it(partitionCtx).map { ptr =>
            val out = emitResultRow(rvb, partitionCtx.r, ptr, offset + localIdx, resultsBc.value)
            localIdx += 1
            out
          }
        }
      } else {
        mv.rvd.mapPartitionsWithContext(rvdType) { (consumerCtx, it) =>
          val producerCtx = consumerCtx.freshContext
          val rvb = new RegionValueBuilder(sm)
          val overlap = blockOverlap
          val missingCompleteCols = new IntArrayBuilder()
          val maxWindowSize = math.max(1, rowBlockSize + 2 * overlap)
          val data = new Array[Double](n * maxWindowSize)
          val rowIt = it(producerCtx)
          val rowBuffer = new mutable.ArrayBuffer[WritableRegionValue](maxWindowSize)

          def appendNextRow(): Unit = {
            val ptr = rowIt.next()
            val wrv = WritableRegionValue(sm, fullRowType, producerCtx.freshRegion())
            wrv.set(ptr, true)
            rowBuffer += wrv
            producerCtx.region.clear()
          }

          new Iterator[Long] {
            private var initialized = false
            private var currentResults: Array[GraphletRegression.GSResult] = null
            private var currentWindowStart = 0
            private var currentCoreLength = 0
            private var currentCoreStartInBuffer = 0
            private var emittedInCurrentBlock = 0
            private var finished = false

            private def initializeBuffer(): Unit = {
              val initialTarget = rowBlockSize + overlap
              while (rowBuffer.length < initialTarget && rowIt.hasNext)
                appendNextRow()
              initialized = true
              if (rowBuffer.isEmpty)
                finished = true
            }

            private def prepareNextBlock(): Boolean = {
              if (!initialized)
                initializeBuffer()
              if (finished || rowBuffer.isEmpty)
                return false

              if (currentCoreStartInBuffer >= rowBuffer.length) {
                finished = true
                return false
              }

              val desiredCoreEnd = currentCoreStartInBuffer + rowBlockSize
              val desiredWindowEnd = desiredCoreEnd + overlap
              while (rowBuffer.length < desiredWindowEnd && rowIt.hasNext)
                appendNextRow()

              val coreAvailable = rowBuffer.length - currentCoreStartInBuffer
              val coreLength = math.min(rowBlockSize, coreAvailable)
              if (coreLength <= 0) {
                finished = true
                return false
              }

              val windowStart = math.max(0, currentCoreStartInBuffer - overlap)
              val windowEnd = math.min(rowBuffer.length, desiredWindowEnd)
              val windowLength = windowEnd - windowStart

              var rowIdx = 0
              while (rowIdx < windowLength) {
                val wrv = rowBuffer(windowStart + rowIdx)
                RegressionUtils.setMeanImputedDoubles(
                  data,
                  rowIdx * n,
                  completeColIdxBc.value,
                  missingCompleteCols,
                  wrv.offset,
                  fullRowType,
                  entryArrayType,
                  entryType,
                  entryArrayIdx,
                  fieldIdx,
                )
                rowIdx += 1
              }

              val rawX = new DenseMatrix[Double](n, windowLength, data)
              val results = new Array[GraphletRegression.GSResult](nPhenotypes)
              var phenoIdx = 0
              while (phenoIdx < nPhenotypes) {
                val yVec = yBc.value(::, phenoIdx)
                results(phenoIdx) = GraphletRegression.execute(
                  rawX,
                  yVec,
                  covBc.value,
                  family = family,
                  nm = nm,
                  r = r,
                  selectionAware = selectionAware,
                  splitFraction = splitFraction,
                  seed = seed + phenoIdx,
                  maxClusterSize = maxClusterSize,
                  sparsityLevel = sparsityLevel,
                  approximateLargeClusters = true,
                  pGlobal = pGlobal,
                  maxIterations = maxIterations,
                  tolerance = tolerance,
                )
                phenoIdx += 1
              }

              currentResults = results
              currentWindowStart = currentCoreStartInBuffer - windowStart
              currentCoreLength = coreLength
              emittedInCurrentBlock = 0
              true
            }

            private def advanceWindow(): Unit = {
              val nextCoreStart = currentCoreStartInBuffer + currentCoreLength
              if (nextCoreStart >= rowBuffer.length && !rowIt.hasNext) {
                rowBuffer.foreach(_.region.clear())
                rowBuffer.clear()
                finished = true
                currentResults = null
                return
              }

              val dropCount = math.max(0, nextCoreStart - overlap)
              var idx = 0
              while (idx < dropCount) {
                rowBuffer(idx).region.clear()
                idx += 1
              }
              rowBuffer.remove(0, dropCount)
              currentCoreStartInBuffer = nextCoreStart - dropCount
              currentResults = null
            }

            override def hasNext: Boolean = {
              if (finished)
                return false
              if (currentResults != null && emittedInCurrentBlock < currentCoreLength)
                return true
              if (currentResults != null)
                advanceWindow()
              if (finished)
                false
              else
                prepareNextBlock()
            }

            override def next(): Long = {
              if (!hasNext)
                throw new java.util.NoSuchElementException("graphlet screening block iterator exhausted")

              val windowIdx = currentWindowStart + emittedInCurrentBlock
              emittedInCurrentBlock += 1
              val wrv = rowBuffer(windowIdx)
              rvb.set(wrv.region)
              emitResultRow(rvb, wrv.region, wrv.offset, windowIdx, currentResults)
            }
          }
        }
      }

    TableValue(ctx, tableType, BroadcastRow.empty(ctx), newRVD)
  }
}
