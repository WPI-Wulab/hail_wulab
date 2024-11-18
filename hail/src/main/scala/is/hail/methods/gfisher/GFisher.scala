package is.hail.methods.gfisher

import is.hail.HailContext
import is.hail.annotations.{Annotation, UnsafeRow, Region}
import is.hail.backend.ExecuteContext
import is.hail.expr.ir.{IntArrayBuilder, MatrixValue, TableValue}

import is.hail.expr.ir.functions.MatrixToTableFunction
// import is.hail.stats.RegressionUtils
import is.hail.types.physical.{PArray, PStruct, PCanonicalNDArray, PType, PField}
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
      println("vals")
      val valArr = vals.toArray
      val (_, pval, df, w, M) = tupleArrayToVectorTuple(valArr)
      println(valArr(0))
      Row(key, 0.0, 3.0)
    }
    TableValue(ctx, typ(mv.typ).rowType, typ(mv.typ).key, newrdd)
  }
}

object GFisherPrepareMatrix {

  def setMeanImputedDoubles(
    data: Array[Double],
    offset: Int,
    completeColIdx: Array[Int],
    missingCompleteCols: IntArrayBuilder,
    ptr: Long,
    fullRowType: PStruct,
    entryArrayType: PType,
    // entryArrayType: PType,
    // entryType: PStruct,
    entryArrayIdx: Int,
    fieldIdx: Int,
  ): Unit = {

    missingCompleteCols.clear()
    val n = completeColIdx.length
    var sum = 0.0
    val entryArrayOffset = fullRowType.loadField(ptr, entryArrayIdx)

    var j = 0
    while (j < n) {
      val k = completeColIdx(j)
      // if (entryArrayType.isElementDefined(entryArrayOffset, k)) {
      // val entryOffset = entryArrayType.loadElement(entryArrayOffset, k)
        // if (entryType.isFieldDefined(entryOffset, fieldIdx)) {
          // val fieldOffset = entryType.loadField(entryOffset, fieldIdx)
      // val e = Region.loadDouble(entryOffset)
      val e = 0.0
      sum += e
      data(offset + j) = e
        // } else
          // missingCompleteCols += j
      // } else
        // missingCompleteCols += j
      j += 1
    }

    val nMissing = missingCompleteCols.size
    val mean = sum / (n - nMissing)
    var i = 0
    while (i < nMissing) {
      data(offset + missingCompleteCols(i)) = mean
      i += 1
    }
  }

  def prepMatrixTable(
    mv: MatrixValue,
    keyField: String,
    pField: String,
    dfField: String,
    weightField: String,
    corrField: String,
    rowIdxField: String
  ): RDD[(Annotation, Iterable[(Int, Double, Int, Double, BDV[Double])])] = {

    val fullRowType: PStruct = mv.rvRowPType
    val keyStructField:PField = fullRowType.field(keyField)
    val keyIndex: Int = keyStructField.index
    val keyType: PType = keyStructField.typ

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

    // get the field the correlation is in
    val corrStructField = fullRowType.field(corrField)
    val corrIndex = corrStructField.index
    val corrArrayType = fullRowType.types(corrIndex)//.asInstanceOf[PArray]
    // val entryType = corrArrayType.elementType.asInstanceOf[PStruct]
    // val fieldIdx = entryType.fieldIdx(corrField)
    val fieldIdx = corrIndex

    println("corrStructField")
    val n = mv.rvd.count().asInstanceOf[Int]
    val completeColIdx = (0 until n).toArray
    val completeColIdxBc = HailContext.backend.broadcast(completeColIdx)

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


          // val data = new Array[Double](n)

          // setMeanImputedDoubles(
          //   data,
          //   0,
          //   completeColIdxBc.value,
          //   new IntArrayBuilder(),
          //   ptr,
          //   fullRowType,
          //   corrArrayType,
          //   // corrArrayType,
          //   // entryType,
          //   corrIndex,
          //   fieldIdx,
          // )

          // val data = Array(rowIdx * 1.0, math.pow(rowIdx,2.0), math.pow(rowIdx, 3.0))
          val data = Array.fill(n)(0.3)
          data(rowIdx) = 1.0
          Some((key, (rowIdx, pval, df, weight, BDV(data))))
        } else None
      }
    }.groupByKey()
    // println("corrStructField")
  }
}
