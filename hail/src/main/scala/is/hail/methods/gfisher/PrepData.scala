/**
  * @author Peter Howell
  */
package is.hail.methods.gfisher

import is.hail.annotations.{Annotation, UnsafeRow, Region}
import is.hail.expr.ir.{IntArrayBuilder, MatrixValue}

import is.hail.stats.RegressionUtils

import is.hail.types.physical.{PStruct, PCanonicalArray}
import is.hail.types.virtual.{TFloat64}

import is.hail.utils._

import org.apache.spark.rdd.RDD

import breeze.linalg.{DenseMatrix => BDM, DenseVector => BDV}

case class GFisherTupleCorr(rowIdx: Int, pval: Double, df: Int, weight: Double, corrArr: Array[Double])

case class GFisherTupleGeno(pval: Double, df: Int, weight: Double, genoArr: Array[Double])

case class OGFisherTupleCorr(rowIdx: Int, pval: Double, df: Array[Int], weight: Array[Double], corrArr: Array[Double])

case class OGFisherTupleGeno(pval: Double, df: Array[Int], weight: Array[Double], genoArr: Array[Double])

object GFisherDataPrep {



  /**
    * Collects the required fields and groups by key.
    *
    * Take a MatrixTable containing row-indexed p-values, degrees of freedom, weights, correlation arrays, and correlation indexes
    *
    * Returns an RDD of (key, Iterable[(rowIdx, pval, df, weight, correlationVector)]).
    * This means that each group in the MatrixTable will be a single row in the resulting RDD, represented as the tuple containing the key and an iterable of the rows in the group.
    * The Iterable contains the p-values, degrees of freedom, weights, correlation vectors, and row indices of the rows in the group.
    *
    * @param mv the MatrixValue of our hail MatrixTable
    * @param keyField what we are grouping by
    * @param pField field the p-values are in
    * @param dfField field storing degrees of freedom
    * @param weightField field storing weights
    * @param corrField field storing correlation
    * @param rowIdxField field storing the row index
    */
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

    val keyStructField = fullRowType.field(keyField) //:PField
    val keyIndex: Int = keyStructField.index
    val keyType = keyStructField.typ // PType

    // get the field the p-value is in
    val pStructField = fullRowType.field(pField)
    val pIndex = pStructField.index
    assert(pStructField.typ.virtualType == TFloat64)

    //get the field the weight is in
    val weightStructField = fullRowType.field(weightField)
    val weightIndex = weightStructField.index
    assert(weightStructField.typ.virtualType == TFloat64)

    // get the field the degree of freedom is in
    val dfStructField = fullRowType.field(dfField)
    val dfIndex = dfStructField.index

    val rowIdxIndex = fullRowType.field(rowIdxField).index

    val n = mv.rvd.count().asInstanceOf[Int]

    mv.rvd.mapPartitions { (ctx, it) =>
      it.flatMap { ptr =>

        val keyIsDefined = fullRowType.isFieldDefined(ptr, keyIndex)
        val weightIsDefined = fullRowType.isFieldDefined(ptr, weightIndex)
        val pIsDefined = fullRowType.isFieldDefined(ptr, pIndex)
        val dfIsDefined = fullRowType.isFieldDefined(ptr, dfIndex)
        val rowIdxIsDefined = fullRowType.isFieldDefined(ptr, rowIdxIndex)

        if (keyIsDefined && weightIsDefined && pIsDefined && dfIsDefined && rowIdxIsDefined) {
          val weight = Region.loadDouble(fullRowType.loadField(ptr, weightIndex))
          val pval = Region.loadDouble(fullRowType.loadField(ptr, pIndex))
          val df = Region.loadInt(fullRowType.loadField(ptr, dfIndex))
          val rowIdx = Region.loadInt(fullRowType.loadField(ptr, rowIdxIndex))
          if (weight < 0)
            fatal(s"Row weights must be non-negative, got $weight")
          val key = Annotation.copy(
            keyType.virtualType,
            UnsafeRow.read(keyType, ctx.r, fullRowType.loadField(ptr, keyIndex)),
          )

          val corrArr = new Array[Double](n)// array that will hold the correlation vector of this row

          // get the correlation values and make sure they aren't all missing/NaN
          val notAllNA = setArrayMeanImputedDoubles(
            corrArr,
            ptr,
            fullRowType,
            corrField
          )
          if (notAllNA){
            corrArr(rowIdx) = 1.0
            Some((key, GFisherTupleCorr(rowIdx, pval, df, weight, corrArr)))
          } else None

        } else None
      }
    }.groupByKey()
  }


  /**
    * Collects the required fields and groups by key.
    *
    * Take a MatrixTable containing row-indexed p-values, degrees of freedom, weights, correlation arrays, and correlation indexes
    *
    * Returns an RDD of (key, Iterable[(rowIdx, pval, df, weight, correlationVector)]).
    * This means that each group in the MatrixTable will be a single row in the resulting RDD, represented as the tuple containing the key and an iterable of the rows in the group.
    * The Iterable contains the p-values, degrees of freedom, weights, correlation vectors, and row indices of the rows in the group.
    *
    * @param mv the MatrixValue of our hail MatrixTable
    * @param nTests number of involved tests for each group. This is the number of rows in the DF matrix in R's implementation of OGFisher
    * @param keyField what we are grouping by
    * @param pField field the p-values are in
    * @param dfField field storing degrees of freedom
    * @param weightField field storing weights
    * @param corrField field storing correlation
    * @param rowIdxField field storing the row index
    */
  def prepOGFisherCorrRDD(
    mv: MatrixValue,
    nTests: Int,
    keyField: String,
    pField: String,
    dfField: String,
    weightField: String,
    corrField: String,
    rowIdxField: String
  ): RDD[(Annotation, Iterable[OGFisherTupleCorr])] = {

    val fullRowType = mv.rvRowPType //PStruct

    val keyStructField = fullRowType.field(keyField) //:PField
    val keyIndex: Int = keyStructField.index
    val keyType = keyStructField.typ // PType

    // get the field the p-value is in
    val pStructField = fullRowType.field(pField)
    val pIndex = pStructField.index
    assert(pStructField.typ.virtualType == TFloat64)

    //get the field the weight is in
    val weightStructField = fullRowType.field(weightField)
    val weightIndex = weightStructField.index
    // assert(weightStructField.typ.virtualType == TFloat64) // not true anymore because it will be an array

    // get the field the degree of freedom is in
    val dfStructField = fullRowType.field(dfField)
    val dfIndex = dfStructField.index

    val rowIdxIndex = fullRowType.field(rowIdxField).index

    val n = mv.rvd.count().asInstanceOf[Int]

    mv.rvd.mapPartitions { (ctx, it) =>
      it.flatMap { ptr =>

        val keyIsDefined = fullRowType.isFieldDefined(ptr, keyIndex)
        val weightIsDefined = fullRowType.isFieldDefined(ptr, weightIndex)
        val pIsDefined = fullRowType.isFieldDefined(ptr, pIndex)
        val dfIsDefined = fullRowType.isFieldDefined(ptr, dfIndex)
        val rowIdxIsDefined = fullRowType.isFieldDefined(ptr, rowIdxIndex)

        if (keyIsDefined && weightIsDefined && pIsDefined && dfIsDefined && rowIdxIsDefined) {
          val weightsArr = new Array[Double](nTests)
          var notAllNA = setArrayMeanImputedDoubles(
            weightsArr,
            ptr,
            fullRowType,
            weightField
          )
          if (!notAllNA) None
          val dfArr = new Array[Int](nTests)
          notAllNA = setArrayInt(
            dfArr,
            ptr,
            fullRowType,
            dfField
          )
          if (!notAllNA) None
          val pval = Region.loadDouble(fullRowType.loadField(ptr, pIndex))
          val rowIdx = Region.loadInt(fullRowType.loadField(ptr, rowIdxIndex))
          if (!weightsArr.forall(_ >= 0.0))
            fatal(s"Row weights must be non-negative, got at least one negative weight")
          val key = Annotation.copy(
            keyType.virtualType,
            UnsafeRow.read(keyType, ctx.r, fullRowType.loadField(ptr, keyIndex)),
          )

          val corrArr = new Array[Double](n)// array that will hold the correlation vector of this row

          // get the correlation values and make sure they aren't all missing/NaN
          notAllNA = setArrayMeanImputedDoubles(
            corrArr,
            ptr,
            fullRowType,
            corrField
          )
          if (notAllNA){
            corrArr(rowIdx) = 1.0
            Some((key, OGFisherTupleCorr(rowIdx, pval, dfArr, weightsArr, corrArr)))
          } else None

        } else None
      }
    }.groupByKey()
  }


  def prepGFisherRDD_Genotype(
    mv: MatrixValue,
    keyField: String,
    pField: String,
    dfField: String,
    weightField: String,
    genotypeField: String
  ): RDD[(Annotation, Iterable[GFisherTupleGeno])] = {

    val fullRowType = mv.rvRowPType //PStruct

    val keyStructField = fullRowType.field(keyField) //:PField
    val keyIndex: Int = keyStructField.index
    val keyType = keyStructField.typ // PType

    // get the field the p-value is in
    val pStructField = fullRowType.field(pField)
    val pIndex = pStructField.index
    assert(pStructField.typ.virtualType == TFloat64)

    // for the genotype entry-array
    val entryArrayType = mv.entryArrayPType
    val entryType = mv.entryPType
    val fieldType = entryType.field(genotypeField).typ
    assert(fieldType.virtualType == TFloat64)

    val entryArrayIdx = mv.entriesIdx
    val fieldIdx = entryType.fieldIdx(genotypeField)

    //get the field the weight is in
    val weightStructField = fullRowType.field(weightField)
    val weightIndex = weightStructField.index
    assert(weightStructField.typ.virtualType == TFloat64)

    // get the field the degree of freedom is in
    val dfStructField = fullRowType.field(dfField)
    val dfIndex = dfStructField.index

    val nCols = mv.nCols

    mv.rvd.mapPartitions { (ctx, it) =>
      it.flatMap { ptr =>

        val keyIsDefined = fullRowType.isFieldDefined(ptr, keyIndex)
        val weightIsDefined = fullRowType.isFieldDefined(ptr, weightIndex)
        val pIsDefined = fullRowType.isFieldDefined(ptr, pIndex)
        val dfIsDefined = fullRowType.isFieldDefined(ptr, dfIndex)

        if (keyIsDefined && weightIsDefined && pIsDefined && dfIsDefined) {
          val weight = Region.loadDouble(fullRowType.loadField(ptr, weightIndex))
          val pval = Region.loadDouble(fullRowType.loadField(ptr, pIndex))
          val df = Region.loadInt(fullRowType.loadField(ptr, dfIndex))
          if (weight < 0)
            fatal(s"Row weights must be non-negative, got $weight")
          val key = Annotation.copy(
            keyType.virtualType,
            UnsafeRow.read(keyType, ctx.r, fullRowType.loadField(ptr, keyIndex)),
          )

          val data = new Array[Double](nCols)// array that will hold the genotype data

          // get the correlation values and make sure they aren't all missing/NaN
          RegressionUtils.setMeanImputedDoubles(
            data,
            0,
            (0 until nCols).toArray,
            new IntArrayBuilder(),
            ptr,
            fullRowType,
            entryArrayType,
            entryType,
            entryArrayIdx,
            fieldIdx
          )
          Some((key, GFisherTupleGeno(pval, df, weight, data)))

        } else None
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
    * @param rvRowType the MatrixValue.rowPType
    * @param arrFieldName name of the array field in the MatrixValue
    */
  def setArrayMeanImputedDoubles(
    data: Array[Double],
    ptr: Long,
    rvRowType: PStruct,
    arrFieldName: String
  ): Boolean = {

    val arrField = rvRowType.field(arrFieldName)// PField
    val arrFieldIdx = arrField.index // Int
    val arrOffset = rvRowType.loadField(ptr, arrFieldIdx)// Long
    val arrType = (rvRowType.types(arrFieldIdx)).asInstanceOf[PCanonicalArray] // PCanonicalArray hopefully

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
    * @param rvRowType
    * @param arrFieldName
    * @param defaultVal value to replace missing values with
    */
  def setArrayInt(
    data: Array[Int],
    ptr: Long,
    rvRowType: PStruct,
    arrFieldName: String,
    defaultVal: Int = 2
  ): Boolean = {

    val arrField = rvRowType.field(arrFieldName)// PField
    val arrFieldIdx = arrField.index // Int
    val arrOffset = rvRowType.loadField(ptr, arrFieldIdx)// Long
    val arrType = (rvRowType.types(arrFieldIdx)).asInstanceOf[PCanonicalArray] // PCanonicalArray? hopefully?

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

  def gFisherTupsGenoToVectors(tups: Array[GFisherTupleGeno]): (BDV[Double], BDV[Int], BDV[Double], BDM[Double]) = {
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
      // System.arraycopy(tups(i)._5.data, 0, corrArr, i*n, n)
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
    return (BDV(pvalArr), BDV(dfArr), BDV(weightArr), new BDM(m, n, genoArr))
  }
  /**
    * Used to convert the iterable of rows in a group to a set of vectors and a matrix.
    *
    * @param tups array containing
    */
  def gFisherTupsCorrToVectors(
    tups: Array[GFisherTupleCorr]
  ): (BDV[Double], BDV[Int], BDV[Double], BDM[Double]) = {
    require(tups.nonEmpty)
    // require(c0.offset == 0 && c0.stride == 1)
    val m: Int = tups.length // number of rows that were put in this group

    val rowIdxArr = new Array[Int](m)
    val pvalArr = new Array[Double](m)
    val weightArr = new Array[Double](m)
    val dfArr = new Array[Int](m)
    // val corrArr = new Array[Double](m*n)

    var i = 0
    while (i < m) {
      rowIdxArr(i) = tups(i).rowIdx
      pvalArr(i) = tups(i).pval
      dfArr(i) = tups(i).df
      weightArr(i) = tups(i).weight
      // System.arraycopy(tups(i)._5.data, 0, corrArr, i*n, n)
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
  def oGFisherTupsCorrToVectors(
    tups: Array[OGFisherTupleCorr],
    nTests: Int
  ): (BDV[Double], BDM[Int], BDM[Double], BDM[Double]) = {
    require(tups.nonEmpty)
    val c0 = tups(0).corrArr
    // require(c0.offset == 0 && c0.stride == 1)
    val m: Int = tups.length // number of rows that were put in this group
    val rowIdxArr = new Array[Int](m)
    val pvalArr = new Array[Double](m)

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
    i = 0
    // fill in the df array and the weight array
    val dfArr = new Array[Int](m * nTests)
    val weightArr = new Array[Double](m * nTests)


    // i = 0
    // while (i < m) {
    //   if (tups(i)._4.length != nTests)
    //       fatal(s"Number of tests in each row must be the same, got ${tups(i)._3.length} weights in row $i, expected $nTests")
    //   for (j <- (0 until nTests)) {
    //     // again note that breeze matrices are column-major
    //     weightArr(i*nTests + j) = tups(i).weight(j)
    //   }
    //   i += 1
    // }

    val pval = new BDV(pvalArr)
    // again note that breeze matrices are column-major
    val df = new BDM[Int](nTests, m, dfArr)
    val weight = new BDM[Double](nTests, m, weightArr)
    val corr = new BDM[Double](m, m, corrArr)
    return (pval, df, weight, corr)
  }

  def oGFisherTupsGenoToVectors(
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
    // fill in the df array and the weight array
    // i = 0

    // while (i < m) {
    //   if (tups(i)._4.length != nTests)
    //       fatal(s"Number of tests in each row must be the same, got ${tups(i)._3.length} weights in row $i, expected $nTests")
    //   for (j <- (0 until nTests)) {
    //     // again note that breeze matrices are column-major
    //     weightArr(i*nTests + j) = tups(i)._4(j)
    //   }
    //   i += 1
    // }

    val pval = new BDV(pvalArr)
    // again note that breeze matrices are column-major
    val df = new BDM[Int](nTests, m, dfArr)
    val weight = new BDM[Double](nTests, m, weightArr)
    return (pval, df, weight, new BDM(m,n, genoArr))
  }

}
