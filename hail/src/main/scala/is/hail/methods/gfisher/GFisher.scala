/**
  * @author Peter Howell
  */
package is.hail.methods.gfisher

import is.hail.annotations.{Annotation, UnsafeRow, Region}
import is.hail.backend.ExecuteContext
import is.hail.expr.ir.{IntArrayBuilder, MatrixValue, TableValue}

import is.hail.expr.ir.functions.MatrixToTableFunction

import is.hail.types.physical.{PStruct, PCanonicalArray}
import is.hail.types.virtual.{MatrixType, TFloat64, TStruct, TableType}

import is.hail.utils._

import org.apache.spark.rdd.RDD
import org.apache.spark.sql.Row

import breeze.linalg.{DenseMatrix => BDM, DenseVector => BDV}
/**
  * Generalized Fisher's combination testing.
  *
  * @author Peter Howell
  *
  * @param keyField name of field to group by
  * @param pField name of field containing p-values
  * @param dfField name of field containing degrees of freedom
  * @param weightField name of field containing weights
  * @param corrField name of field containing correlation arrays
  * @param rowIDXField name of field containing indices of the rows in the correlation matrix
  * @param method which method to use. Either HYB, MR, or GB
  * @param oneSided whether the input p-values are one-sided
  */
case class GFisher(
  keyField: String,
  pField: String,
  dfField: String,
  weightField: String,
  corrField: String,
  rowIDXField: String,
  method: String,
  oneSided: Boolean
) extends MatrixToTableFunction {

  override def typ(childType: MatrixType): TableType = {
    val keyType = childType.rowType.fieldType(keyField)
    val mySchema = TStruct(
      ("id", keyType),
      ("stat", TFloat64),
      ("p_value", TFloat64)
    )
    TableType(mySchema, FastSeq("id"), TStruct.empty)
  }
  def preservesPartitionCounts: Boolean = false

  def execute(ctx: ExecuteContext, mv: MatrixValue): TableValue = {
    val groupedRDD = GFisherPrepareMatrix.prepMatrixTable(mv, keyField, pField, dfField, weightField, corrField, rowIDXField)
    val newrdd = groupedRDD.map{case(key, vals) =>
      val valArr = vals.toArray// array of the rows in this group. each element is a tuple with all the fields.

      val (_, pvals: BDV[Double], df: BDV[Int], w: BDV[Double], corrMat: BDM[Double]) = tupleArrayToVectorTuple(valArr)
      val gFishStat: Double = StatGFisher.statGFisher(pvals, df, w)
      val gFishPVal: Double = if (method == "HYB") PGFisher.pGFisherHyb(gFishStat, df, w, corrMat)
                              else if (method == "MR") 0.01  //*@TODO PGFisher.pGFisherMR
                              else 0.01  //*@TODO PGFisher.pGFisherGB
      Row(key, gFishStat, gFishPVal)
    }
    TableValue(ctx, typ(mv.typ).rowType, typ(mv.typ).key, newrdd)
  }
}

object GFisherPrepareMatrix {

  /**
    * Collects the required fields and groups by key.
    *
    * @param mv the MatrixValue of our hail MatrixTable
    * @param keyField what we are grouping by
    * @param pField field the p-values are in
    * @param dfField field storing degrees of freedom
    * @param weightField field storing weights
    * @param corrField field storing correlation
    * @param rowIdxField field storing the row index
    */
  def prepMatrixTable(
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

  /**
    * Collects values from a hail array into a scala array, replacing missing values with a mean. returns false if every value is missing/NaN
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
    val arrType = (rvRowType.types(arrFieldIdx)).asInstanceOf[PCanonicalArray] // PCanonicalArray? hopefully?

    // stores indices of elements that are missing or NaN
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

}
