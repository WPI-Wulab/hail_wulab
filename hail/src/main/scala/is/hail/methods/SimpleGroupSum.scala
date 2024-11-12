package is.hail.methods

import is.hail.HailContext
import is.hail.annotations.{Annotation, UnsafeRow}
import is.hail.backend.ExecuteContext
import is.hail.expr.ir.{IntArrayBuilder, MatrixValue, TableValue}
import is.hail.expr.ir.functions.MatrixToTableFunction
import is.hail.stats.RegressionUtils
import is.hail.types.virtual.{MatrixType, TFloat64, TStruct, TableType}
import is.hail.utils._

import breeze.linalg.{DenseMatrix => BDM, DenseVector => BDV, _}
// import breeze.numerics._
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.Row


/**
  * A simple example of grouping by something and then summing the values.
  *
  * @param keyField
  * @param xField
  */
case class SimpleGroupSum(
  keyField: String,
  xField: String
) extends MatrixToTableFunction {

  override def typ(childType: MatrixType): TableType = {
    val keyType = childType.rowType.fieldType(keyField)
    val mySchema = TStruct(
      ("id", keyType),
      ("my_sum", TFloat64)
    )
    TableType(mySchema, FastSeq("id"), TStruct.empty)
  }

  def preservesPartitionCounts: Boolean = false

  def execute(ctx: ExecuteContext, mv: MatrixValue): TableValue = {
    // val backend = HailContext.backend

    val groupedObj = groupAndImputeMean(mv)
    val rddObj = groupedObj.map{case (key, vals) =>
      val valArr = vals.toArray
      val n = valArr(0).length
      val X = new BDM(valArr.length, n, valArr.flatMap(_.toList))
      Row(key, sum(X))
    }
    val mtyp = typ(mv.typ)
    TableValue(ctx, mtyp.rowType, mtyp.key, rddObj)

  }

  /**
    * groups a MatrixTable, collecting the x variable into an iterable object (CondensedBuffer)
    *
    * @param mv hail MatrixTable
    */
  def groupAndImputeMean(mv: MatrixValue): RDD[(Annotation, Iterable[Array[Double]])] = {
    val ncol = mv.nCols
    val completeColIdx = (0 until ncol).toArray
    val completeColIdxBc = HailContext.backend.broadcast(completeColIdx)
    val fullRowType = mv.rvRowPType
    val keyStructField = fullRowType.field(keyField)
    val keyIndex = keyStructField.index
    val keyType = keyStructField.typ

    val entryArrayType = mv.entryArrayPType
    val entryArrayIdx = mv.entriesIdx

    val entryType = mv.entryPType
    val fieldIdx = entryType.fieldIdx(xField)
    mv.rvd.mapPartitions { (ctx, it) =>
      it.flatMap { ptr =>
        val keyIsDefined = fullRowType.isFieldDefined(ptr, keyIndex)

        if (keyIsDefined) {
          val key = Annotation.copy(
            keyType.virtualType,
            UnsafeRow.read(keyType, ctx.r, fullRowType.loadField(ptr, keyIndex)),
          )
          val data = new Array[Double](ncol)

          RegressionUtils.setMeanImputedDoubles(
            data,
            0,
            completeColIdxBc.value,
            new IntArrayBuilder(),
            ptr,
            fullRowType,
            entryArrayType,
            entryType,
            entryArrayIdx,
            fieldIdx,
          )
          Some(key -> data)
        } else None
      }
    }.groupByKey()
  }

}
