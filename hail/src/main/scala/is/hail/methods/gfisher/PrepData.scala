/**
  * @author Peter Howell
  */
package is.hail.methods.gfisher

import is.hail.annotations.{Annotation, UnsafeRow, Region}
import is.hail.expr.ir.{IntArrayBuilder, MatrixValue}


import is.hail.types.physical.{PStruct, PCanonicalArray}
import is.hail.types.virtual.{TFloat64}

import is.hail.utils._

import org.apache.spark.rdd.RDD

import breeze.linalg.{DenseMatrix => BDM, DenseVector => BDV}

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
  def prepGFisherRDD(
    mv: MatrixValue,
    keyField: String,
    pField: String,
    dfField: String,
    weightField: String,
    corrField: String,
    rowIdxField: String
  ): RDD[(Annotation, Iterable[(Int, Double, Int, Double, BDV[Double])])] = {

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
            Some((key, (rowIdx, pval, df, weight, BDV(corrArr))))
          } else None

        } else None
      }
    }.groupByKey()
  }

  def prepOGFisherRDD(
    mv: MatrixValue,
    nTests: Int,
    keyField: String,
    pField: String,
    dfField: String,
    weightField: String,
    corrField: String,
    rowIdxField: String
  ): RDD[(Annotation, Iterable[(Int, Double, BDV[Int], BDV[Double], BDV[Double])])] = {

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
            Some((key, (rowIdx, pval, BDV(dfArr), BDV(weightsArr), BDV(corrArr))))
          } else None

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

  /**
    * Used to convert the iterable of rows in a group to a set of vectors and a matrix.
    *
    * @param a array containing
    */
  def arrayTupleToVectorTuple(
    a: Array[(Int, Double, Int, Double, BDV[Double])]
  ): (BDV[Int], BDV[Double], BDV[Int], BDV[Double], BDM[Double]) = {
    require(a.nonEmpty)
    val c0 = a(0)._5
    require(c0.offset == 0 && c0.stride == 1)
    val m: Int = a.length // number of rows that were put in this group

    val rowIdxArr = new Array[Int](m)
    val pvalArr = new Array[Double](m)
    val weightArr = new Array[Double](m)
    val dfArr = new Array[Int](m)
    // val corrArr = new Array[Double](m*n)

    var i = 0
    while (i < m) {
      rowIdxArr(i) = a(i)._1
      pvalArr(i) = a(i)._2
      dfArr(i) = a(i)._3
      weightArr(i) = a(i)._4
      // System.arraycopy(a(i)._5.data, 0, corrArr, i*n, n)
      i += 1
    }
    i = 0
    val corrArr = new Array[Double](m*m)
    // fill in the correlation matrix
    while (i < m) {
      for (j <- (0 until m)) {
        corrArr(i*m+j) = a(i)._5(rowIdxArr(j))
      }
      i += 1
    }
    val corrMatrix = new BDM[Double](m, m, corrArr)
    return (BDV(rowIdxArr), BDV(pvalArr), BDV(dfArr), BDV(weightArr), corrMatrix)
  }



/**
    * Used to convert the iterable of rows in a group to a set of vectors and matrices for OGFisher.
    *
    * @param a array containing
    */
  def arrayTupleToVectorTuple2(
    a: Array[(Int, Double, BDV[Int], BDV[Double], BDV[Double])]
  ): (BDV[Int], BDV[Double], BDM[Int], BDM[Double], BDM[Double]) = {
    require(a.nonEmpty)
    val c0 = a(0)._5
    require(c0.offset == 0 && c0.stride == 1)
    val m: Int = a.length // number of rows that were put in this group
    val n: Int = a(0)._3.length
    val rowIdxArr = new Array[Int](m)
    val pvalArr = new Array[Double](m)

    var i = 0
    while (i < m) {
      rowIdxArr(i) = a(i)._1
      pvalArr(i) = a(i)._2
      // System.arraycopy(a(i)._5.data, 0, corrArr, i*n, n)
      i += 1
    }
    i = 0
    val corrArr = new Array[Double](m*m)

    // fill in the correlation matrix
    while (i < m) {
      for (j <- (0 until m)) {
        corrArr(i*m+j) = a(i)._5(rowIdxArr(j))
      }
      i += 1
    }
    i = 0

    // fill in the df array and the weight array
    val nTests = a(0)._3.length
    val dfArr = new Array[Int](m * n)
    while (i < m) {
        if (a(i)._3.length != nTests)
          fatal(s"Number of tests in each row must be the same, got ${a(i)._3.length} in row $i, expected $nTests")
      for (j <- (0 until nTests)) {
        dfArr(i*m+j) = a(i)._3(j)
      }
      i += 1
    }
    i = 0
    val weightArr = new Array[Double](m * n)
    while (i < m) {
      for (j <- (0 until nTests)) {
        if (a(i)._4.length != nTests)
          fatal(s"Number of tests in each row must be the same, got ${a(i)._3.length} in row $i, expected $nTests")
        weightArr(i*m+j) = a(i)._4(j)
      }
      i += 1
    }
    val corrMatrix = new BDM[Double](m, m, corrArr)
    return (new BDV(rowIdxArr), new BDV(pvalArr), new BDM[Int](m, nTests, dfArr), new BDM[Double](m, nTests, weightArr), new BDM[Double](m, m, corrArr))
  }

}
