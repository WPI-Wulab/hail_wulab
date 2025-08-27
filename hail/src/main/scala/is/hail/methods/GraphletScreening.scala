package is.hail.methods

import is.hail.HailContext
import is.hail.backend.ExecuteContext
import is.hail.annotations._
import is.hail.expr.ir.{IntArrayBuilder, MatrixValue, TableValue}
import is.hail.expr.ir.functions.MatrixToTableFunction
import is.hail.stats.RegressionUtils
import is.hail.types.physical.PStruct
import is.hail.types.virtual.{MatrixType, TArray, TFloat64, TInt32, TStruct, TableType}
import is.hail.utils._
import is.hail.annotations.{RegionValueBuilder, WritableRegionValue}
import org.apache.spark.sql.Row
import breeze.linalg.DenseMatrix._
import breeze.linalg.DenseVector._
import breeze.linalg._
import breeze.linalg.diag
import breeze.linalg.eigSym
import breeze.linalg.inv
import breeze.linalg.sum
import breeze.linalg.svd
import breeze.linalg.csvwrite
import breeze.linalg.CSCMatrix

import breeze.numerics.sqrt

import breeze.stats.distributions.Gaussian
import breeze.stats.distributions.Uniform
import breeze.stats.distributions.RandBasis

import scala.util.Random
import java.io.File
import java.io.PrintWriter

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
    val xUnconstrained = inv(Q) * dvec
    if (satisfiesConstraints(xUnconstrained)) {
      val valUnconstrained = objective(xUnconstrained)
      bestSolution = Some(xUnconstrained)
      bestValue = valUnconstrained
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
      try {
        val sol = inv(K) * rhs
        val xCandidate = sol(0 until n)
        if (satisfiesConstraints(xCandidate)) {
          val objVal = objective(xCandidate)
          if (objVal < bestValue) {
            bestValue = objVal
            bestSolution = Some(xCandidate)
          }
        }
      } catch {
        case _: Exception =>
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
                    nm: Int,
                    v: Double,
                    r: Double,
                    q0: Double = 0.1,
                    scale: Double = 1.0): DenseVector[Boolean] = {
    def safeSqrt(x: Double): Double = if (x > 0.0) math.sqrt(x) else 0.0
    val p  = yTilde.length
    val q1 = math.pow(v + r, 2) / (r * r) / 4.0
    val tau  = math.sqrt(2.0 * r * math.log(p))
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
                  g_ds_ds - (g_ds_es * (inv(g_es_es) * g_es_ds))
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
                scale * 2.0 * math.log(p) * (
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
   * Carries out the cleaning stage.
   * @param survivor Boolean mask of survivors
   * @param yTilde Transformed response vector
   * @param gram Gram matrix
   * @param lambda Penalty parameter
   * @param uu Constraint threshold
   * @return Estimated coefficient vector
   */
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
    var beta = DenseVector.zeros[Double](nSurvivor)
    var remain = DenseVector.fill[Boolean](nSurvivor)(true)
    while (remain.data.exists(x => x)) {
      val idxCandidates = which(remain)
      val i = idxCandidates.head
      var cluster = DenseVector.fill[Boolean](nSurvivor)(false)
      cluster(i) = true
      var newCluster = DenseVector.zeros[Boolean](nSurvivor)
      for (j <- 0 until nSurvivor) {
        newCluster(j) = omega(j, i) != 0.0
      }
      while (!allEqual(cluster, newCluster)) {
        cluster = newCluster.copy
        val selectedCols = which(cluster)
        newCluster = DenseVector((0 until nSurvivor).map { j =>
          var sumVal = 0.0
          for (col <- selectedCols) {
            sumVal += math.abs(omega(j, col))
          }
          sumVal != 0.0
        }.toArray)
      }
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
        cgAll(ii-1) = findCG(adjacencyMatrix, cgAll(ii-2))
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
      val signY = yTilde.map(math.signum)
      val absY  = yTilde.map(math.abs)
      val indicator = absY.map(x => if (x > uu) 1.0 else 0.0)
      (signY *:* indicator) * uu
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
      if (newWStd > stdThresh * lastWStd) {
        nIteration = it - 1
        return (betaGS, nIteration)
      }
      val survivorMask = screeningStep(w, gram, cgAll, nm, v, r, q0, scale)
      betaGS = cleaningStep(survivorMask, w, gram, lambda, uu)
      nIteration = it
    }
    (betaGS, nIteration)
  }

  /**
   * Runs Iterative Graphlet Screening on the given data.
   * @param X Design matrix
   * @param Y Response vector
   * @param nm Maximum subgraph size
   * @param r Signal strength parameter
   * @return Estimated beta vector
   */
  def execute(X: DenseMatrix[Double],
              Y: DenseVector[Double],
              nm: Int = 3,
              r: Double = 3.5): DenseVector[Double] = {
    println("Graphlet Screening has begun")
    println("X: " + X)
    println("Y: " + Y)
    println("nm: " + nm)
    println("r: " + r)
    val p = X.cols
    val gram = X.t * X
    val delta = 1.0 / math.log(p)
    val (gramThresh, gramBias) = thresholdGram(gram, delta)
    val neighbor = gramThresh.map(x => x != 0.0)
    val cgAll = findAllCG(neighbor, nm)
    val yTilde = X.t * Y
    val defaultTau = math.sqrt(2 * math.log(p) * r)
    val defaultSp  = math.pow(p.toDouble, 0.5)
    val spPerturb = defaultSp * (1 + 0.1 * (if (Random.nextBoolean()) 1.0 else -1.0))
    val tauPerturb = defaultTau * (1 + 0.1 * (if (Random.nextBoolean()) 1.0 else -1.0))
    val (betaGS, nIter) = iterGS(yTilde, gramThresh, gramBias, cgAll, spPerturb, tauPerturb, nm)
    println("beta: " + betaGS)
    betaGS
  }
}
case class GraphletScreening(
  yFields: Seq[String],
  xField: String,
  covFields: Seq[String],
  rowBlockSize: Int,
  passThrough: Seq[String],
  nm: Int,
  r: Double = 3.5,  // Add signal strength parameter with default
) extends MatrixToTableFunction {

  override def typ(childType: MatrixType): TableType = {
    val passThroughType = TStruct(passThrough.map(f => f -> childType.rowType.field(f).typ): _*)
    val schema = TStruct(
      ("n", TInt32),
      ("beta", TArray(TFloat64)),
    )
    TableType(
      childType.rowKeyStruct ++ passThroughType ++ schema,
      childType.rowKey,
      TStruct.empty,
    )
  }

  def preservesPartitionCounts: Boolean = true

  def execute(ctx: ExecuteContext, mv: MatrixValue): TableValue = {
    // Extract y matrix and covariates similar to LinearRegression
    val (y, cov, completeColIdx) =
      RegressionUtils.getPhenosCovCompleteSamples(mv, yFields.toArray, covFields.toArray)
    
    val n = y.rows // n_complete_samples  
    val nPhenotypes = y.cols // number of response variables
    val k = cov.cols // nCovariates
    
    info(s"graphlet_screening: running on $n samples for ${nPhenotypes} response ${plural(nPhenotypes, "variable")} y,\n"
      + s"    with input variable x, $k additional ${plural(k, "covariate")}, nm=$nm, r=$r...")
    
    // Get backend and broadcast needed data
    val backend = HailContext.backend
    val completeColIdxBc = backend.broadcast(completeColIdx)
    val yBc = backend.broadcast(y)
    val covBc = backend.broadcast(cov)
    
    // Get row and entry types for extracting x values
    val fullRowType = mv.rvd.rowPType
    val entryArrayType = MatrixType.getEntryArrayType(fullRowType)
    val entryType = entryArrayType.elementType.asInstanceOf[PStruct]
    assert(entryType.field(xField).typ.virtualType == TFloat64)
    
    val entryArrayIdx = MatrixType.getEntriesIndex(fullRowType)
    val fieldIdx = entryType.fieldIdx(xField)
    
    val tableType = typ(mv.typ)
    val rvdType = tableType.canonicalRVDType
    val copiedFieldIndices = (mv.typ.rowKey ++ passThrough).map(fullRowType.fieldIdx(_)).toArray
    
    val sm = ctx.stateManager
    
    // Process rows in blocks similar to LinearRegression
    val newRVD = mv.rvd.mapPartitionsWithContext(rvdType) { (consumerCtx, it) =>
      val producerCtx = consumerCtx.freshContext
      val rvb = new RegionValueBuilder(sm)
      
      val missingCompleteCols = new IntArrayBuilder()
      val data = new Array[Double](n * rowBlockSize)
      
      val blockWRVs = new Array[WritableRegionValue](rowBlockSize)
      var i = 0
      while (i < rowBlockSize) {
        blockWRVs(i) = WritableRegionValue(sm, fullRowType, producerCtx.freshRegion())
        i += 1
      }
      
      it(producerCtx).trueGroupedIterator(rowBlockSize)
        .flatMap { git =>
          var i = 0
          while (git.hasNext) {
            val ptr = git.next()
            RegressionUtils.setMeanImputedDoubles(
              data,
              i * n,
              completeColIdxBc.value,
              missingCompleteCols,
              ptr,
              fullRowType,
              entryArrayType,
              entryType,
              entryArrayIdx,
              fieldIdx
            )
            blockWRVs(i).set(ptr, true)
            producerCtx.region.clear()
            i += 1
          }
          val blockLength = i
          
          // Build X matrix for this block of rows
          val X = new DenseMatrix[Double](n, blockLength, data)
          
          // Add covariates to X if present
          val XWithCov = if (k > 0) {
            DenseMatrix.horzcat(X, covBc.value)
          } else {
            X
          }
          
          // Process each phenotype and get beta coefficients
          val betaResults = new Array[DenseVector[Double]](nPhenotypes)
          for (phenoIdx <- 0 until nPhenotypes) {
            val yVec = yBc.value(::, phenoIdx)
            
            // Call GraphletRegression for this phenotype
            val beta = GraphletRegression.execute(XWithCov, yVec, nm, r)
            
            // Extract only the coefficients for genetic variants in this block (not covariates)
            betaResults(phenoIdx) = beta(0 until blockLength)
          }
          
          // Generate output rows
          (0 until blockLength).iterator.map { i =>
            val wrv = blockWRVs(i)
            rvb.set(wrv.region)
            rvb.start(rvdType.rowType)
            rvb.startStruct()
            
            // Add row key and pass-through fields
            rvb.addFields(fullRowType, wrv.region, wrv.offset, copiedFieldIndices)
            
            // Add n (number of samples)
            rvb.addInt(n)
            
            // Add beta array for all phenotypes for this row
            rvb.startArray(nPhenotypes)
            var j = 0
            while (j < nPhenotypes) {
              rvb.addDouble(betaResults(j)(i))
              j += 1
            }
            rvb.endArray()
            
            rvb.endStruct()
            
            producerCtx.region.addReferenceTo(wrv.region)
            rvb.end()
          }
        }
    }
    
    TableValue(ctx, tableType, BroadcastRow.empty(ctx), newRVD)
  }
}