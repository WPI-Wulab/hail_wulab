/**
  * @author Peter Howell
  */
package is.hail.methods.gfisher

import is.hail.annotations.{Annotation, UnsafeRow, Region}
import is.hail.expr.ir.{IntArrayBuilder, MatrixValue}

import is.hail.rvd.RVDContext
import is.hail.HailContext

import is.hail.stats.RegressionUtils

import is.hail.types.physical.{PStruct, PCanonicalArray, PArray}
// import is.hail.types.virtual.{TFloat64}

import is.hail.utils._

import org.apache.spark.rdd.RDD

import breeze.linalg.{DenseMatrix => BDM, DenseVector => BDV}

abstract class GFisherTuple(pval: Double)

case class GFisherTupleCorr(rowIdx: Int, pval: Double, df: Int, weight: Double, corrArr: Array[Double]) extends GFisherTuple(pval)

case class GFisherTupleGeno(pval: Double, df: Int, weight: Double, genoArr: Array[Double]) extends GFisherTuple(pval)

case class OGFisherTupleCorr(rowIdx: Int, pval: Double, df: Array[Int], weight: Array[Double], corrArr: Array[Double]) extends GFisherTuple(pval)

case class OGFisherTupleGeno(pval: Double, df: Array[Int], weight: Array[Double], genoArr: Array[Double]) extends GFisherTuple(pval)

trait FieldIndexes{
  def key: Int
  def pval: Int
  def df: Int
  def weight: Int
}

case class GFisherCorrFieldIdxs(key:Int, rowIdx: Int, pval: Int, df:Int, weight: Int, corrArr: Int) extends FieldIndexes

case class GFisherGenoFieldIdxs(key:Int, pval: Int, df:Int, weight: Int, geno: Int, entryArray: Int) extends FieldIndexes

abstract class GRowProcessor(
  fullRowType: PStruct,
  ptr: Long,
  fieldIds: FieldIndexes
) {

  def checkFieldsDefined(): Boolean = {
    return fullRowType.isFieldDefined(ptr, fieldIds.key) &&
      fullRowType.isFieldDefined(ptr, fieldIds.pval) &&
      fullRowType.isFieldDefined(ptr, fieldIds.df) &&
      fullRowType.isFieldDefined(ptr, fieldIds.weight)
  }
  def getDouble(idx: Int):Double = {
    Region.loadDouble(fullRowType.loadField(ptr, idx))
  }

  def getInt(idx: Int): Int = {
    Region.loadInt(fullRowType.loadField(ptr, idx))
  }

  def getKey(ctx: RVDContext): Annotation = {
    val keyType = fullRowType.fields(fieldIds.key).typ
    Annotation.copy(
      keyType.virtualType,
      UnsafeRow.read(keyType, ctx.r, fullRowType.loadField(ptr, fieldIds.key)),
    )
  }

  def getDoubleArray(idx: Int): Array[Double] = {
    val arrOffset = fullRowType.loadField(ptr, idx)// Long
    val arrType = (fullRowType.types(idx)).asInstanceOf[PCanonicalArray]
    val n = arrType.loadLength(arrOffset)
    val data = new Array[Double](n)
    GFisherDataPrep.setArrayMeanImputedDoubles(
      data,
      ptr,
      fullRowType,
      idx
    )
    return data
  }

  def getIntArray(idx: Int): Array[Int] = {
    val arrOffset = fullRowType.loadField(ptr, idx)// Long
    val arrType = (fullRowType.types(idx)).asInstanceOf[PCanonicalArray]
    val n = arrType.loadLength(arrOffset)
    val data = new Array[Int](n)
    GFisherDataPrep.setArrayInt(
      data,
      ptr,
      fullRowType,
      idx
    )
    return data
  }



  def getData(ctx: RVDContext): Option[(Annotation, GFisherTuple)]
}

abstract class GRowProcessorEntry(
  fullRowType: PStruct,
  ptr: Long,
  fieldIds: FieldIndexes,
  entryArrayType:PArray,
  entryType: PStruct,
  nCols: Int
) extends GRowProcessor(fullRowType, ptr, fieldIds) {
  def getEntryArray(idx: Int, entryArrayIdx: Int, n: Int): Array[Double] = {

    val data = new Array[Double](n)// array that will hold the genotype data

    // get the correlation values and make sure they aren't all missing/NaN
    RegressionUtils.setMeanImputedDoubles(
      data,
      0,
      (0 until n).toArray,
      new IntArrayBuilder(),
      ptr,
      fullRowType,
      entryArrayType,
      entryType,
      entryArrayIdx,
      idx
    )
    return data
  }
}

class GFisherCorrRowProcessor(
  fullRowType: PStruct,
  ptr: Long,
  fieldIds: GFisherCorrFieldIdxs,
  nRows: Int
) extends GRowProcessor(fullRowType, ptr, fieldIds) {
  override def checkFieldsDefined(): Boolean = {
    super.checkFieldsDefined() &&
      fullRowType.isFieldDefined(ptr, fieldIds.rowIdx) &&
      fullRowType.isFieldDefined(ptr, fieldIds.corrArr)
  }

  override def getData(ctx: RVDContext): Option[(Annotation, GFisherTupleCorr)] = {
    val key = getKey(ctx)
    val rowIdx = getInt(fieldIds.rowIdx)
    val pval = getDouble(fieldIds.pval)
    val df = getInt(fieldIds.df)
    val weight = getDouble(fieldIds.weight)
    val corrArr = getDoubleArray(fieldIds.corrArr)
    if (all(corrArr, (i: Double) => i == 0.0))
      return None
    return Some((key, new GFisherTupleCorr(rowIdx=rowIdx,pval=pval,df=df,weight=weight,corrArr=corrArr)))
  }
}

class GFisherGenoRowProcessor(
  fullRowType: PStruct,
  ptr: Long,
  fieldIds: GFisherGenoFieldIdxs,
  entryArrayType:PArray,
  entryType: PStruct,
  nCols: Int
) extends GRowProcessorEntry(fullRowType, ptr, fieldIds, entryArrayType, entryType, nCols) {

  override def getData(ctx: RVDContext): Option[(Annotation, GFisherTupleGeno)] = {
    val key = getKey(ctx)
    val pval = getDouble(fieldIds.pval)
    val df = getInt(fieldIds.df)
    val weight = getDouble(fieldIds.weight)
    val genoArr = getEntryArray(fieldIds.geno, fieldIds.entryArray, nCols)
    if (genoArr.distinct.length == 1)
      return None
    return Some((key, new GFisherTupleGeno(pval=pval, df=df, weight=weight, genoArr=genoArr)))
  }
}

class OGFisherCorrRowProcessor(
  fullRowType: PStruct,
  ptr: Long,
  fieldIds: GFisherCorrFieldIdxs,
  nRows: Int,
  nTests: Int
) extends GRowProcessor(fullRowType, ptr, fieldIds) {

  override def checkFieldsDefined(): Boolean = {
    super.checkFieldsDefined() &&
      fullRowType.isFieldDefined(ptr, fieldIds.rowIdx) &&
      fullRowType.isFieldDefined(ptr, fieldIds.corrArr)
  }

  override def getData(ctx: RVDContext): Option[(Annotation, OGFisherTupleCorr)] = {
    val key = getKey(ctx)
    val rowIdx = getInt(fieldIds.rowIdx)
    val pval = getDouble(fieldIds.pval)
    val df = getIntArray(fieldIds.df)
    val weight = getDoubleArray(fieldIds.weight)
    val corrArr = getDoubleArray(fieldIds.corrArr)
    if (all(corrArr, (i: Double) => i == 0.0))
      return None
    return Some((key,  new OGFisherTupleCorr(rowIdx=rowIdx, pval=pval, df=df, weight=weight, corrArr=corrArr)))
  }
}

class OGFisherGenoRowProcessor(
  fullRowType: PStruct,
  ptr: Long,
  fieldIds: GFisherGenoFieldIdxs,
  entryArrayType:PArray,
  entryType: PStruct,
  nCols: Int,
  nTests: Int
) extends GRowProcessorEntry(fullRowType, ptr, fieldIds, entryArrayType, entryType, nCols) {

  override def getData(ctx: RVDContext): Option[(Annotation, OGFisherTupleGeno)] = {
    val key = getKey(ctx)
    val pval = getDouble(fieldIds.pval)
    val df = getIntArray(fieldIds.df)
    val weight = getDoubleArray(fieldIds.weight)
    val genoArr = getEntryArray(fieldIds.geno, fieldIds.entryArray, nCols)
    if (genoArr.distinct.length == 1)
      return None
    return Some((key, new OGFisherTupleGeno(pval=pval, df=df, weight=weight, genoArr=genoArr)))
  }
}



object GFisherDataPrep {

  def prepGFisherCorrRDD(
    mv: MatrixValue,
    keyField: String,
    pField: String,
    dfField: String,
    weightField: String,
    corrField: String,
    rowIdxField: String
  ): RDD[(Annotation, Iterable[GFisherTupleCorr])] = {
    val fullRowType = mv.rvRowPType //PStruct

    val ids = getFieldIds(fullRowType, keyField, rowIdxField, pField, dfField, weightField, corrField)
    val fieldIds = new GFisherCorrFieldIdxs(ids(0), ids(1), ids(2), ids(3), ids(4), ids(5))
    val fieldIdsBC = HailContext.backend.broadcast(fieldIds)
    val nRows = mv.rvd.count().asInstanceOf[Int]
    mv.rvd.mapPartitions { (ctx, it) =>
      it.flatMap { ptr =>
        val rowProcessor = new GFisherCorrRowProcessor(fullRowType, ptr, fieldIdsBC.value, nRows)
        //check fields defined
        val fieldsDefined = rowProcessor.checkFieldsDefined
        if (fieldsDefined) {
          None
        }

        rowProcessor.getData(ctx)
      }
    }.groupByKey()
  }

  def prepGFisherGenoRDD(
    mv: MatrixValue,
    keyField: String,
    pField: String,
    dfField: String,
    weightField: String,
    genoField: String
  ): RDD[(Annotation, Iterable[GFisherTupleGeno])] = {
    val fullRowType = mv.rvRowPType //PStruct

    val ids = getFieldIds(fullRowType, keyField, pField, dfField, weightField)
    val entryArrayType = mv.entryArrayPType
    val entryType = mv.entryPType
    val entryArrayIdx = mv.entriesIdx
    val genoFieldIdx = entryType.fieldIdx(genoField)
    val fieldIds = new GFisherGenoFieldIdxs(ids(0), ids(1), ids(2), ids(3), genoFieldIdx, entryArrayIdx)


    val fieldIdsBC = HailContext.backend.broadcast(fieldIds)
    val nCols = mv.nCols

    mv.rvd.mapPartitions { (ctx, it) =>
      it.flatMap { ptr =>
        val rowProcessor = new GFisherGenoRowProcessor(fullRowType, ptr, fieldIdsBC.value, entryArrayType, entryType, nCols)

        //check fields defined
        val fieldsDefined = rowProcessor.checkFieldsDefined
        if (fieldsDefined) {
          None
        }

        rowProcessor.getData(ctx)
      }
    }.groupByKey()
  }

  def prepOGFisherCorrRDD(
    mv: MatrixValue,
    keyField: String,
    pField: String,
    dfField: String,
    weightField: String,
    corrField: String,
    rowIdxField: String,
    nTests: Int
  ): RDD[(Annotation, Iterable[OGFisherTupleCorr])] ={
    val fullRowType = mv.rvRowPType //PStruct

    val ids = getFieldIds(fullRowType, keyField, rowIdxField, pField, dfField, weightField, corrField)
    val fieldIds = new GFisherCorrFieldIdxs(ids(0), ids(1), ids(2), ids(3), ids(4), ids(5))
    val fieldIdsBC = HailContext.backend.broadcast(fieldIds)
    val nRows = mv.rvd.count().asInstanceOf[Int]
    mv.rvd.mapPartitions { (ctx, it) =>
      it.flatMap { ptr =>
        val rowProcessor = new OGFisherCorrRowProcessor(fullRowType, ptr, fieldIdsBC.value, nRows, nTests)

        //check fields defined
        val fieldsDefined = rowProcessor.checkFieldsDefined
        if (fieldsDefined) {
          None
        }

        rowProcessor.getData(ctx)
      }
    }.groupByKey()
  }

  def prepOGFisherGenoRDD(
    mv: MatrixValue,
    keyField: String,
    pField: String,
    dfField: String,
    weightField: String,
    genoField: String,
    nTests: Int
  ): RDD[(Annotation, Iterable[OGFisherTupleGeno])] = {
    val fullRowType = mv.rvRowPType //PStruct

    val ids = getFieldIds(fullRowType, keyField, pField, dfField, weightField)
    val entryArrayType = mv.entryArrayPType
    val entryType = mv.entryPType
    val entryArrayIdx = mv.entriesIdx
    val genoFieldIdx = entryType.fieldIdx(genoField)
    val fieldIds = new GFisherGenoFieldIdxs(ids(0), ids(1), ids(2), ids(3), genoFieldIdx, entryArrayIdx)

    val fieldIdsBC = HailContext.backend.broadcast(fieldIds)
    val nCols = mv.nCols

    mv.rvd.mapPartitions { (ctx, it) =>
      it.flatMap { ptr =>
        val rowProcessor = new OGFisherGenoRowProcessor(fullRowType, ptr, fieldIdsBC.value, entryArrayType, entryType, nCols, nTests)

        //check fields defined
        val fieldsDefined = rowProcessor.checkFieldsDefined
        if (fieldsDefined) {
          None
        }

        rowProcessor.getData(ctx)
      }
    }.groupByKey()
  }

    /**
    * Collects values from a row-indexed hail array expression into a scala array, replacing missing values with a mean. returns false if every value is missing/NaN
    *
    * This function is used to collect the correlation values from the correlation array field in the hail MatrixTable.
    *
    * @param data array to fill
    * @param ptr pointer created in MatrixValue.rvd.mapPartitions.flatMat
    * @param fullRowType the MatrixValue.rowPType
    * @param arrFieldName name of the array field in the MatrixValue
    */
  def setArrayMeanImputedDoubles(
    data: Array[Double],
    ptr: Long,
    fullRowType: PStruct,
    arrFieldIdx: Int
  ): Boolean = {

    val arrOffset = fullRowType.loadField(ptr, arrFieldIdx)// Long
    val arrType = (fullRowType.types(arrFieldIdx)).asInstanceOf[PCanonicalArray] // PCanonicalArray hopefully

    // stores indices of elements that are missing or NaN, and builds sum of non-missing values
    val nanElts = new IntArrayBuilder()
    val n = data.length
    var sum = 0.0
    var i = 0
    while (i < n) {
      if (arrType.isElementDefined(arrOffset, i)) {
        val entryOffset = arrType.loadElement(arrOffset, i)// Long
        val elt = Region.loadDouble(entryOffset)
        if (! elt.isNaN) {
          sum += elt
          data(i) = elt
        } else
            nanElts += i
      } else
          nanElts += i
      i += 1
    }

    val nMissing = nanElts.size

    //if they were also missing
    if (nMissing == n) return false

    val mean = sum / (n - nMissing)

    i = 0
    // replace the missing values with the mean
    while (i < nMissing) {
      data(nanElts(i)) = mean
      i += 1
    }
    return true
  }

  /**
    * Collects values from a row-indexed hail array expression into a scala array, replacing missing values with a default value.
    * returns false if every value is missing/NaN
    *
    * @param data array to fill
    * @param ptr
    * @param fullRowType
    * @param arrFieldName
    * @param defaultVal value to replace missing values with
    */
  def setArrayInt(
    data: Array[Int],
    ptr: Long,
    fullRowType: PStruct,
    arrFieldIdx: Int,
    defaultVal: Int = 2
  ): Boolean = {

    val arrOffset = fullRowType.loadField(ptr, arrFieldIdx)// Long
    val arrType = (fullRowType.types(arrFieldIdx)).asInstanceOf[PCanonicalArray] // PCanonicalArray? hopefully?

    // stores indices of elements that are missing or NaN
    val nanElts = new IntArrayBuilder()
    val n = data.length
    var i = 0
    while (i < n) {
      if (arrType.isElementDefined(arrOffset, i)) {
        val entryOffset = arrType.loadElement(arrOffset, i)// Long
        val elt = Region.loadInt(entryOffset)
        if (! elt.isNaN) {
          data(i) = elt
        } else
            nanElts += i
      } else
          nanElts += i
      i += 1
    }

    val nMissing = nanElts.size

    //if they were also missing
    if (nMissing == n) return false

    i = 0
    // replace the missing values with the mean
    while (i < nMissing) {
      data(nanElts(i)) = defaultVal
      i += 1
    }
    return true
  }

}


object GFisherArrayToVectors {

  // def oGFisherTups[T](vals: Array[T]): (BDV[Double], BDM[Int], BDM[Double], BDM[Double]) = {
  //   vals match {
  //     case a: Array[OGFisherTupleGeno] => oGFisherTupsGenoToVectors(vals)
  //     case a: Array[OGFisherTupleCorr] => oGFisherTupsCorrToVectors(vals)
  //     case _ => throw new IllegalArgumentException("Unknown type")
  //   }
  // }

  def gFisherGeno(tups: Array[GFisherTupleGeno]): (BDV[Double], BDV[Int], BDV[Double], BDM[Double]) = {
    require(tups.nonEmpty)
    val g0 = tups(0).genoArr
    // require(g0.offset == 0 && g0.stride == 1)
    val n = g0.size
    val m: Int = tups.length // number of rows that were put in this group

    val pvalArr = new Array[Double](m)
    val weightArr = new Array[Double](m)
    val dfArr = new Array[Int](m)
    // val corrArr = new Array[Double](m*n)

    var i = 0
    while (i < m) {
      pvalArr(i) = tups(i).pval
      dfArr(i) = tups(i).df
      weightArr(i) = tups(i).weight
      i += 1
    }
    i = 0

    // fill in the correlation matrix
    val genoArr = new Array[Double](n*m)
    while (i < m) {
      for (j <- (0 until n)) {
        genoArr(i + j*m) = tups(i).genoArr(j)
      }
      i += 1
    }
    val genoMat = new BDM(m, n, genoArr)
    val corrMat = rowCorrelation(genoMat)
    return (BDV(pvalArr), BDV(dfArr), BDV(weightArr), corrMat)
  }
  /**
    * Used to convert the iterable of rows in a group to a set of vectors and a matrix.
    *
    * @param tups array containing
    */
  def gFisherCorr(
    tups: Array[GFisherTupleCorr]
  ): (BDV[Double], BDV[Int], BDV[Double], BDM[Double]) = {
    require(tups.nonEmpty)
    // require(c0.offset == 0 && c0.stride == 1)
    val m: Int = tups.length // number of rows that were put in this group

    val rowIdxArr = new Array[Int](m)
    val pvalArr = new Array[Double](m)
    val weightArr = new Array[Double](m)
    val dfArr = new Array[Int](m)

    var i = 0
    while (i < m) {
      rowIdxArr(i) = tups(i).rowIdx
      pvalArr(i) = tups(i).pval
      dfArr(i) = tups(i).df
      weightArr(i) = tups(i).weight
      i += 1
    }
    i = 0
    val corrArr = new Array[Double](m*m)
    // fill in the correlation matrix
    while (i < m) {
      for (j <- (0 until m)) {
        corrArr(i*m+j) = tups(i).corrArr(rowIdxArr(j))
      }
      i += 1
    }
    val corrMatrix = new BDM[Double](m, m, corrArr)
    return (BDV(pvalArr), BDV(dfArr), BDV(weightArr), corrMatrix)
  }



  /**
    * Used to convert the iterable of rows in a group to a set of vectors and matrices for OGFisher.
    *
    * @param tups array containing
    */
  def oGFisherCorr(
    tups: Array[OGFisherTupleCorr],
    nTests: Int
  ): (BDV[Double], BDM[Int], BDM[Double], BDM[Double]) = {
    require(tups.nonEmpty)
    val m: Int = tups.length // number of rows that were put in this group
    val rowIdxArr = new Array[Int](m)
    val pvalArr = new Array[Double](m)
    val dfArr = new Array[Int](m * nTests)
    val weightArr = new Array[Double](m * nTests)

    var i = 0
    while (i < m) {
      rowIdxArr(i) = tups(i).rowIdx
      pvalArr(i) = tups(i).pval
      if (tups(i).weight.length != nTests || tups(i).df.length != nTests)
        fatal(s"Number of tests in each row must be the same. Either weights or degrees of freedom in a row were not equal to $nTests")
      // again note that breeze matrices are column-major
      System.arraycopy(tups(i).df, 0, dfArr, i*nTests, nTests)
      System.arraycopy(tups(i).weight, 0, weightArr, i*nTests, nTests)
      // System.arraycopy(tups(i)._5.data, 0, corrArr, i*n, n)
      i += 1
    }
    i = 0
    val corrArr = new Array[Double](m*m)

    // important! breeze matrices are column-major, so we need to fill in the matrix by columns.
    // this makes no difference for correlation matrices, but we do need to be careful for the df and weight matrices

    // fill in the correlation matrix
    while (i < m) {
      for (j <- (0 until m)) {
        corrArr(i*m+j) = tups(i).corrArr(rowIdxArr(j))
      }
      i += 1
    }

    val pval = new BDV(pvalArr)
    // again note that breeze matrices are column-major
    val df = new BDM[Int](nTests, m, dfArr)
    val weight = new BDM[Double](nTests, m, weightArr)
    val corr = new BDM[Double](m, m, corrArr)
    return (pval, df, weight, corr)
  }

  def oGFisherGeno(
    tups: Array[OGFisherTupleGeno],
    nTests: Int
  ): (BDV[Double], BDM[Int], BDM[Double], BDM[Double]) = {
    require(tups.nonEmpty)
    val m: Int = tups.length // number of rows that were put in this group
    val pvalArr = new Array[Double](m)
    val g0 = tups(0).genoArr
    val n = g0.size // number of samples
    var i = 0

    // important! breeze matrices are column-major, so we need to fill in the matrix by columns.
    // this makes no difference for correlation matrices, but we do need to be careful for the genotype, df, and weight matrices

    val weightArr = new Array[Double](m * nTests)
    val dfArr = new Array[Int](m * nTests)
    // fill in the genotype matrix
    val genoArr = new Array[Double](n*m)
    while (i < m) {
      pvalArr(i) = tups(i).pval
      if (tups(i).weight.length != nTests || tups(i).df.length != nTests)
        fatal(s"Number of tests in each row must be the same. Either weights or degrees of freedom in a row were not equal to $nTests")
      // again note that breeze matrices are column-major
      System.arraycopy(tups(i).df, 0, dfArr, i*nTests, nTests)
      System.arraycopy(tups(i).weight, 0, weightArr, i*nTests, nTests)
      for (j <- (0 until n)) {
        genoArr(i + j*m) = tups(i).genoArr(j)
      }
      i += 1
    }
    i=0

    val pval = new BDV(pvalArr)
    // again note that breeze matrices are column-major
    val df = new BDM[Int](nTests, m, dfArr)
    val weight = new BDM[Double](nTests, m, weightArr)
    val G = new BDM(m, n, genoArr)
    val M = rowCorrelation(G)
    return (pval, df, weight, M)
  }
}
