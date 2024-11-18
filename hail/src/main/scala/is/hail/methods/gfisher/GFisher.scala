package is.hail.methods.gfisher

import is.hail.HailContext
import is.hail.annotations.{Annotation, UnsafeRow, Region}
import is.hail.backend.ExecuteContext
import is.hail.expr.ir.{IntArrayBuilder, MatrixValue, TableValue}

import is.hail.expr.ir.functions.MatrixToTableFunction

import is.hail.types.physical.{PStruct, PCanonicalArray}
import is.hail.types.virtual.{MatrixType, TFloat64, TStruct, TableType}

import is.hail.utils._

import org.apache.spark.rdd.RDD
import org.apache.spark.sql.Row

import breeze.linalg.{DenseMatrix => BDM, DenseVector => BDV, _}

case class GFisher(
  keyField: String,
  pField: String,
  dfField: String,
  weightField: String,
  corrField: String,
  rowIDXField: String,
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
      val valArr = vals.toArray

      val (_, pval, df, w, corrMat) = tupleArrayToVectorTuple(valArr)
      println(s"key $key\n$corrMat\n")
      Row(key, 0.0, 3.0)
    }
    TableValue(ctx, typ(mv.typ).rowType, typ(mv.typ).key, newrdd)
  }
}

object GFisherPrepareMatrix {

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

    println("corrStructField")
    val n = mv.rvd.count().asInstanceOf[Int]
    val completeColIdx = (0 until n).toArray
    val completeColIdxBc = HailContext.backend.broadcast(completeColIdx)

    mv.rvd.mapPartitions { (ctx, it) =>
      it.flatMap { ptr =>

        // println(s"ptr class: ${ptr.getClass}, ptr: $ptr")

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

          val data = new Array[Double](n)

          setArrayMeanImputedDoubles(
            data,
            completeColIdxBc.value,
            ptr,
            fullRowType,
            corrField
          )

          // val data = Array(rowIdx * 1.0, math.pow(rowIdx,2.0), math.pow(rowIdx, 3.0))
          // val data = Array.fill(n)(0.3)
          data(rowIdx) = 1.0
          Some((key, (rowIdx, pval, df, weight, BDV(data))))
        } else None
      }
    }.groupByKey()
    // println("corrStructField")
  }

  def setArrayMeanImputedDoubles(
    data: Array[Double],
    completeColIdx: Array[Int],
    ptr: Long,
    rvRowType: PStruct,
    arrFieldName: String
  ): Unit = {
    val arrField = rvRowType.field(arrFieldName)// PField
    val arrFieldIdx = arrField.index // Int
    val arrOffset = rvRowType.loadField(ptr, arrFieldIdx)// Long
    val arrType = (rvRowType.types(arrFieldIdx)).asInstanceOf[PCanonicalArray] // PCanonicalArray? hopefully?

    val missingCompleteCols = new IntArrayBuilder()
    val n = completeColIdx.length
    var sum = 0.0
    var j = 0
    while (j < n) {
      val k = completeColIdx(j)
      if (arrType.isElementDefined(arrOffset, k)) {
        val entryOffset = arrType.loadElement(arrOffset, k)// Long
        val e = Region.loadDouble(entryOffset)
        sum += e
        data(j) = e
      } else
          missingCompleteCols += j
      j += 1
    }

    val nMissing = missingCompleteCols.size
    val mean = sum / (n - nMissing)
    var i = 0
    while (i < nMissing) {
      data(missingCompleteCols(i)) = mean
      i += 1
    }
  }

}
