package is.hail.methods

import is.hail.annotations._
import is.hail.backend.ExecuteContext
import is.hail.expr.ir.{IntArrayBuilder, MatrixValue, TableValue}
import is.hail.expr.ir.functions.MatrixToTableFunction
import is.hail.stats.RegressionUtils
import is.hail.types.virtual.{MatrixType, TArray, TFloat64, TStruct, TableType}
import is.hail.utils._
import org.apache.spark.sql.Row
import is.hail.types.virtual._
import is.hail.types.physical.PStruct

import is.hail.HailContext._
import is.hail.types.virtual.{TableType, TStruct, TArray, TFloat64}
import is.hail.utils.FastSeq
import breeze.linalg.{DenseMatrix, DenseVector}
import breeze.linalg._
import breeze.linalg.DenseMatrix._
import breeze.linalg.DenseVector._

import breeze.linalg.{DenseMatrix, DenseVector, norm}
import breeze.linalg.DenseMatrix.eye

import org.apache.spark.mllib.linalg.distributed.{IndexedRow, IndexedRowMatrix, CoordinateMatrix, MatrixEntry, BlockMatrix}
import org.apache.spark.mllib.linalg.{Vector => MLVector, Vectors => MLVectors, DenseMatrix => MLDenseMatrix}

import scala.util.Random

object GraphletRegression {

  ///////////////////////////////
  // Helper functions
  ///////////////////////////////

  def vectorizeBase(num: Int, base: Int, len: Int): DenseVector[Int] = {
    val digits = new Array[Int](len)
    var n = num
    for (i <- (len - 1) to 0 by -1) {
      digits(i) = n % base
      n = n / base
    }
    DenseVector(digits)
  }

  def which(vec: DenseVector[Boolean]): IndexedSeq[Int] = {
    (0 until vec.length).filter(i => vec(i))
  }

  def subVector(vec: DenseVector[Double], indices: IndexedSeq[Int]): DenseVector[Double] = {
    val result = new Array[Double](indices.length)
    var i = 0
    while (i < indices.length) {
      result(i) = vec(indices(i))
      i += 1
    }
    new DenseVector(result)
  }

  def subMatrix(mat: DenseMatrix[Double], rowInd: IndexedSeq[Int], colInd: IndexedSeq[Int]): DenseMatrix[Double] = {
    val r = rowInd.length
    val c = colInd.length
    val res = DenseMatrix.zeros[Double](r, c)
    for (j <- colInd.indices) {
      val col = colInd(j)
      for (i <- rowInd.indices) {
        res(i, j) = mat(rowInd(i), col)
      }
    }
    res
  }

  def allEqual(v1: DenseVector[Boolean], v2: DenseVector[Boolean]): Boolean = {
    if (v1.length != v2.length) return false
    (0 until v1.length).forall(i => v1(i) == v2(i))
  }

  def isSymmetricBoolean(matrix: DenseMatrix[Boolean]): Boolean = {
    if (matrix.rows != matrix.cols) return false
    val n = matrix.rows
    for (i <- 0 until n; j <- 0 until n) {
      if (matrix(i, j) != matrix(j, i)) return false
    }
    true
  }

  ///////////////////////////////
  // Quadratic programming solveQP function
  ///////////////////////////////

  case class QPSolution(solution: DenseVector[Double], value: Double)

  // In GraphletRegression.scala
  def solveQP(Q: DenseMatrix[Double],
              c: DenseVector[Double],
              tolerance: Double = 1e-6,
              maxIter: Int = 1000): QPSolution = {

    val reg = 1e-6
    val Q_sym = (Q + Q.t) * 0.5
    val Q_reg = Q_sym + eye[Double](Q_sym.rows) * reg

    var x = DenseVector.zeros[Double](Q_reg.cols)

    val alphaInit = 1.0
    val beta = 0.5
    val c1 = 1e-4

    var iter = 0
    var grad = Q_reg * x + c
    var gradNorm = norm(grad)

    while (iter < maxIter && gradNorm > tolerance) {
      val descentDir = -grad
      var step = alphaInit
      val fx = 0.5 * (x.t * Q_reg * x) + (c.t * x)
      while (0.5 * ((x + step * descentDir).t * Q_reg * (x + step * descentDir)) + (c.t * (x + step * descentDir)) >
             fx + c1 * step * (grad.t * descentDir)) {
        step *= beta
        if (step < 1e-10)
          throw new IllegalArgumentException("Line search failed to converge")
      }
      x = x + step * descentDir
      grad = Q_reg * x + c
      gradNorm = norm(grad)
      iter += 1
    }

    val objectiveValue = 0.5 * (x.t * Q_reg * x) + (c.t * x)
    QPSolution(x, objectiveValue)
  }

  ///////////////////////////////
  // PMLE: Penalized Maximum Likelihood Estimation
  ///////////////////////////////
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
        val bvec = DenseVector.ones[Double](card)
        val qpSol = solveQP(gramCluster, yc)
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

  ///////////////////////////////
  // ThresholdGram: threshold the Gram matrix
  ///////////////////////////////
  def thresholdGram(gram: DenseMatrix[Double], delta: Double): (DenseMatrix[Double], DenseMatrix[Double]) = {
    val gramThresh = gram.map(x => if (math.abs(x) < delta) 0.0 else x)
    val gramBias = gram - gramThresh
    (gramThresh, gramBias)
  }

  ///////////////////////////////
  // ScreeningStep of GS function
  ///////////////////////////////
  def screeningStep(w: DenseVector[Double],
                    gram: DenseMatrix[Double],
                    cgAll: List[DenseMatrix[Int]],
                    nm: Int,
                    v: Double,
                    r: Double,
                    q0: Double,
                    scale: Double): IndexedSeq[Int] = {

    var bestCandidate: Option[IndexedSeq[Int]] = None
    var bestScore = Double.PositiveInfinity

    val baseIndices: IndexedSeq[Int] = 0 until w.length

    for (cg <- cgAll; i <- 0 until cg.rows) {
      val subgraph = cg(i, ::).t.toArray.toSeq.distinct
      val candidateIndices: IndexedSeq[Int] = subgraph.intersect(baseIndices).distinct.toIndexedSeq
      if (candidateIndices.nonEmpty) {
        val wSub = subVector(w, candidateIndices)
        val gramSub = subMatrix(gram, candidateIndices, candidateIndices)
        if (wSub.length == gramSub.rows && wSub.length == gramSub.cols) {
          val candidateScore = (gramSub * wSub).dot(wSub)
          if (candidateScore < bestScore) {
            bestScore = candidateScore
            bestCandidate = Some(candidateIndices)
          }
        }
      }
    }

    bestCandidate.getOrElse(IndexedSeq.empty[Int])
  }

  ///////////////////////////////
  // CleaningStep of GS function
  ///////////////////////////////
  def cleaningStep(survivor: DenseVector[Boolean],
                   yTilde: DenseVector[Double],
                   gram: DenseMatrix[Double],
                   lambda: Double,
                   uu: Double): DenseVector[Double] = {
    val p = gram.cols
    val survIndices = which(survivor)
    val nSurvivor = survIndices.length
    val yt = subVector(yTilde, survIndices)
    val omega = subMatrix(gram, survIndices, survIndices)
    val beta = DenseVector.zeros[Double](nSurvivor)
    val remain = DenseVector.fill[Boolean](nSurvivor)(true)

    while (remain.data.exists(x => x)) {
      val idxCandidates = which(remain)
      val i = idxCandidates.head
      val clusterArr = Array.fill[Boolean](nSurvivor)(false)
      clusterArr(i) = true
      val newClusterArr = new Array[Boolean](nSurvivor)
      for (j <- 0 until nSurvivor) {
        newClusterArr(j) = omega(j, i) != 0.0
      }
      while (!clusterArr.sameElements(newClusterArr)) {
        // Copy newClusterArr into clusterArr
        Array.copy(newClusterArr, 0, clusterArr, 0, nSurvivor)
        val selectedCols = (0 until nSurvivor).filter(j => clusterArr(j))
        for (j <- 0 until nSurvivor) {
          var sumVal = 0.0
          for (col <- selectedCols) {
            sumVal += math.abs(omega(j, col))
          }
          newClusterArr(j) = (sumVal != 0.0)
        }
      }
      val cluster = new DenseVector(clusterArr)
      if (which(cluster).length > 20) {
        throw new Exception(s"cluster too long. The cluster length is ${which(cluster).length}")
      }
      val clusterIndices = which(cluster)
      val omegaCluster = subMatrix(omega, clusterIndices, clusterIndices)
      val ytCluster = subVector(yt, clusterIndices)
      val betaCluster = PMLE(omegaCluster, ytCluster, lambda, uu)
      for ((origIdx, j) <- clusterIndices.zipWithIndex)
        beta(origIdx) = betaCluster(j)
      for (origIdx <- clusterIndices)
        remain(origIdx) = false
    }
    val betaGS = DenseVector.zeros[Double](p)
    for ((origIdx, j) <- survIndices.zipWithIndex)
      betaGS(origIdx) = beta(j)
    betaGS
  }

  ///////////////////////////////
  // FindAllCG: Find all connected subgraphs
  ///////////////////////////////
  def findAllCG(adjacencyMatrix: DenseMatrix[Boolean], lc: Int): List[DenseMatrix[Int]] = {

    if (adjacencyMatrix.rows != adjacencyMatrix.cols || !isSymmetricBoolean(adjacencyMatrix))
      throw new IllegalArgumentException("The adjacency matrix is not symmetric!")

    val p = adjacencyMatrix.rows
    val cgAll = new Array[DenseMatrix[Int]](lc)

    cgAll(0) = new DenseMatrix(p, 1, (1 to p).toArray)

    if (lc >= 2) {
      val edgesBuffer = scala.collection.mutable.ArrayBuffer[Array[Int]]()
      for (i <- 0 until p; j <- i + 1 until p) {
        if (adjacencyMatrix(i, j)) {
          edgesBuffer.append(Array(i + 1, j + 1))
        }
      }
      val numEdges = edgesBuffer.length
      val edgesMatrix =
        if (numEdges > 0) DenseMatrix.tabulate(numEdges, 2) { (r, c) => edgesBuffer(r)(c) }
        else new DenseMatrix[Int](0, 2, Array.empty[Int])
      cgAll(1) = edgesMatrix
    }

    if (lc >= 3) {
      for (level <- 3 to lc) {
        cgAll(level - 1) = findCG(adjacencyMatrix, cgAll(level - 2))
      }
    }
    cgAll.toList
  }

  ///////////////////////////////
  // FindCG: Subgraph processing operations
  ///////////////////////////////
  def findCG(adjacencyMatrix: DenseMatrix[Boolean], cgInitial: DenseMatrix[Int]): DenseMatrix[Int] = {
    val p = adjacencyMatrix.rows

    val A = DenseMatrix.tabulate[Boolean](p, p) { (i, j) => adjacencyMatrix(i, j) }
    for (i <- 0 until p) A(i, i) = false

    if (cgInitial.size == p) {
      val arr = cgInitial.toArray.sorted
      if (arr.sameElements((1 to p).toArray[Int])) {
        if (cgInitial.rows == 1) {
          val colMat = new DenseMatrix(p, 1, cgInitial.toArray)
          return findCG(adjacencyMatrix, colMat)
        }
      }
    }

    val currentSubgraphSize = cgInitial.cols
    val nextSize = currentSubgraphSize + 1
    if (cgInitial.rows == 0)
      throw new IllegalArgumentException(s"No connected subgraphs with ${nextSize} nodes detected.")
    val newSubgraphsBuffer = scala.collection.mutable.ArrayBuffer[Seq[Int]]()
    for (rowIndex <- 0 until cgInitial.rows) {
      val cg0: Seq[Int] = (0 until cgInitial.cols).map(j => cgInitial(rowIndex, j))
      val candidateNeighbors = cg0.flatMap { v =>
        (1 to p).filter { w => A(v - 1, w - 1) }
      }.distinct.filterNot(n => cg0.contains(n))
      if (candidateNeighbors.nonEmpty) {
        candidateNeighbors.foreach { n =>
          val newSub = (cg0 :+ n).sorted
          newSubgraphsBuffer.append(newSub)
        }
      }
    }

    if (newSubgraphsBuffer.isEmpty)
      throw new IllegalArgumentException(s"No connected subgraphs with ${nextSize} nodes detected.")

    val uniqueSubgraphs: Seq[Seq[Int]] = newSubgraphsBuffer.toSeq.distinct

    DenseMatrix.tabulate(uniqueSubgraphs.size, nextSize) { (i, j) =>
      uniqueSubgraphs(i)(j)
    }
  }

  ///////////////////////////////
  // IterGS: Iterative Graphlet Screening.
  ///////////////////////////////
  def iterGS(yTilde: DenseVector[Double],
             gram: DenseMatrix[Double],
             gramBias: DenseMatrix[Double],
             cgAll: List[DenseMatrix[Int]],
             sp: Double,
             tau: Double,
             nm: Int,
             q0: Double = 0.1,
             scale: Double = 1.0,
             maxIter: Int = 3,
             stdThresh: Double = 1.05,
             betaInitial: Option[DenseVector[Double]] = None
            ): (DenseVector[Double], Int) = {
    val p = gram.cols
    val r = math.pow(tau, 2) / (2 * math.log(p))
    val v = 1.0 - math.log(sp) / math.log(p)
    val uu = math.sqrt(2 * r * math.log(p))
    val lambda = math.sqrt(2 * v * math.log(p))
    val betaInit = betaInitial.getOrElse {
      val signY = yTilde.map(x => if (x >= 0) 1.0 else -1.0)
      val absY = yTilde.map(math.abs)
      val indicator = absY.map(x => if (x > uu) 1.0 else 0.0)
      signY *:* indicator *:* uu
    }
    var betaGS = betaInit.copy
    var w = yTilde.copy
    var nIteration = 0
    for (it <- 1 to maxIter) {
      val meanW = sum(w) / p.toDouble
      val lastWStd = math.sqrt((sum(w *:* w) - p * meanW * meanW) / (p - 1))
      val lastBeta = betaGS.copy
      val lastNonZero = lastBeta.map(x => x != 0.0)
      val nzIndices = which(lastNonZero)
      val gramBiasSub = subMatrix(gramBias, (0 until gramBias.rows), nzIndices)
      val lastBetaSub = subVector(lastBeta, nzIndices)
      val adjustment = gramBiasSub * lastBetaSub
      w = yTilde - adjustment
      val newMeanW = sum(w) / p.toDouble
      val newWStd = math.sqrt((sum(w *:* w) - p * newMeanW * newMeanW) / (p - 1))
      if (newWStd > stdThresh * lastWStd) {
        nIteration = it - 1
        return (betaGS, nIteration)
      }
      val survivor = screeningStep(w, gram, cgAll, nm, v, r, q0, scale)
      val survivorMask = DenseVector.fill(p)(false)
      for (i <- survivor) survivorMask(i) = true
      betaGS = cleaningStep(survivorMask, w, gram, lambda, uu)
      nIteration = it
    }
    (betaGS, nIteration)
  }

  ///////////////////////////////
  // The main Graphlet Screening routines
  ///////////////////////////////

  def execute(X: DenseMatrix[Double],
              Y: DenseVector[Double],
              nm: Int = 3): DenseVector[Double] = {
    val p = X.cols
    val gram = X.t * X
    val delta = 1.0 / math.log(p)
    val (gramThresh, gramBias) = thresholdGram(gram, delta)
    val neighbor = gramThresh.map(x => x != 0.0)
    val cgAll = findAllCG(neighbor, nm)
    val yTilde = X.t * Y

    val defaultTau = math.sqrt(2 * math.log(p))
    val defaultSp  = math.pow(p.toDouble, 0.5)

    val spPerturb = defaultSp * (1 + 0.1 * (if (Random.nextBoolean()) 1.0 else -1.0))
    val tauPerturb = defaultTau * (1 + 0.1 * (if (Random.nextBoolean()) 1.0 else -1.0))

    val (betaGS, nIter) = iterGS(yTilde, gramThresh, gramBias, cgAll, spPerturb, tauPerturb, nm)
    betaGS
  }
}

case class GraphletScreening(
                              yField: String,
                              xField: String,
                              covFields: IndexedSeq[String],
                              nm: Int = 3
                            ) extends MatrixToTableFunction {

  override def typ(childType: MatrixType): TableType = {
    TableType(TStruct("beta" -> TArray(TFloat64)), FastSeq("beta"), TStruct.empty)
  }

  override def preservesPartitionCounts: Boolean = false

  override def execute(ctx: ExecuteContext, mv: MatrixValue): TableValue = {

  val (y, cov, completeColIdx) =
    RegressionUtils.getPhenoCovCompleteSamples(mv, yField, covFields.toArray)
  val fullRowType = mv.rvd.rowPType
  val entryArrayType = MatrixType.getEntryArrayType(fullRowType)
  val entryType = entryArrayType.elementType.asInstanceOf[PStruct]
  val entryArrayIdx = MatrixType.getEntriesIndex(fullRowType)
  val fieldIdx = entryType.fieldIdx(xField)

  val indexedRows = mv.rvd.mapPartitionsWithIndex { (partitionIndex: Int, ctx: is.hail.rvd.RVDContext, it: Iterator[Long]) =>
    it.zipWithIndex.map { case (rvOffset, localIndex) =>
      val data = new Array[Double](completeColIdx.length)
      RegressionUtils.setMeanImputedDoubles(
        data,
        0,
        completeColIdx,
        new IntArrayBuilder(),
        rvOffset,
        fullRowType,
        entryArrayType,
        entryType,
        entryArrayIdx,
        fieldIdx
      )
      new IndexedRow(partitionIndex.toLong * 1000000L + localIndex, MLVectors.dense(data))
    }
  }
  val indexedRowMatrix = new IndexedRowMatrix(indexedRows)

  val coordEntries = indexedRowMatrix.rows.flatMap { row =>
    val i = row.index
    row.vector.toArray.zipWithIndex.map { case (v, j) =>
      new MatrixEntry(j, i, v)
    }
  }
  val coordinateMatrix = new CoordinateMatrix(coordEntries)
  val transposedCoord = coordinateMatrix.transpose
  // Convert the transposed CoordinateMatrix into a BlockMatrix.
  val blockMatrixX = transposedCoord.toBlockMatrix().cache()
  blockMatrixX.validate()

  val X_localMat = blockMatrixX.toLocalMatrix().asInstanceOf[MLDenseMatrix]
  val X_breeze = new DenseMatrix[Double](X_localMat.numRows, X_localMat.numCols, X_localMat.toArray)

  val X_breeze_subset = X_breeze(::, completeColIdx.toIndexedSeq).toDenseMatrix

  val X_for_regression = if (X_breeze.cols == completeColIdx.length) {
    X_breeze
  } else {
    X_breeze(::, completeColIdx.toIndexedSeq).toDenseMatrix
  }

  val gram = X_breeze_subset.t * X_breeze_subset

  val p = gram.cols
  val delta = 1.0 / math.log(p)
  val (gramThresh, gramBias) = GraphletRegression.thresholdGram(gram, delta)
  val neighbor = gramThresh.map(x => x != 0.0)
  val cgAll = GraphletRegression.findAllCG(neighbor, nm)

  val yTilde = X_breeze_subset.t * y

  val beta: DenseVector[Double] = GraphletRegression.execute(X_breeze_subset, y, nm)

  val row = org.apache.spark.sql.Row(beta.toArray.toFastSeq)
  val rdd = mv.rvd.sparkContext.parallelize(Seq(row))
  TableValue(ctx, this.typ(mv.typ).rowType, this.typ(mv.typ).key, rdd)
}
}
