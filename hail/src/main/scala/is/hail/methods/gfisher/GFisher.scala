/**
  * @author Peter Howell
  */
package is.hail.methods.gfisher

import is.hail.backend.ExecuteContext
import is.hail.expr.ir.{MatrixValue, TableValue}

import is.hail.expr.ir.functions.MatrixToTableFunction

import is.hail.types.virtual.{MatrixType, TFloat64, TStruct, TableType, TArray}

import is.hail.utils._

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
  keyFieldOut: String,
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
      (keyFieldOut, keyType),
      ("stat", TFloat64),
      ("p_value", TFloat64)
    )
    TableType(mySchema, FastSeq(keyFieldOut), TStruct.empty)
  }
  def preservesPartitionCounts: Boolean = false

  def execute(ctx: ExecuteContext, mv: MatrixValue): TableValue = {
    val groupedRDD = GFisherDataPrep.prepGFisherCorrRDD(mv, keyField, pField, dfField, weightField, corrField, rowIDXField)
    val newrdd = groupedRDD.map{case(key, vals) =>
      val valArr = vals.toArray// array of the rows in this group. each element is a tuple with all the fields.

      val (pvals: BDV[Double], df: BDV[Int], w: BDV[Double], corrMat: BDM[Double]) = GFisherDataPrep.gFisherTupsCorrToVectors(valArr)
      val gFishStat: Double = StatGFisher.statGFisher(pvals, df, w)
      val gFishPVal: Double = if (method == "HYB") PGFisher.pGFisherHyb(gFishStat, df, w, corrMat)
                              else if (method == "MR") PGFisher.pGFisherMR(gFishStat, df, w, corrMat)
                              else PGFisher.pGFisherGB(gFishStat, df, w, corrMat)
      Row(key, gFishStat, gFishPVal)
    }
    TableValue(ctx, typ(mv.typ).rowType, typ(mv.typ).key, newrdd)
  }
}


/**
  * Omnibus version of Generalized Fisher's combination testing.
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
case class OGFisher(
  nTests: Int,
  keyField: String,
  keyFieldOut: String,
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
    // val arrayType = childType.rowType.fieldType(weightField)//do this to get the array type
    // println("arrayType: " + arrayType)
    // println(arrayType)
    val mySchema = TStruct(
      (keyFieldOut, keyType),
      ("stat", TFloat64),
      ("p_value", TFloat64),
      ("stat_ind", TArray(TFloat64)),
      ("p_value_ind",  TArray(TFloat64))
      // ("stat_ind", arrayType),
      // ("p_value_ind", arrayType)
    )
    TableType(mySchema, FastSeq(keyFieldOut), TStruct.empty)
  }
  def preservesPartitionCounts: Boolean = false

  def execute(ctx: ExecuteContext, mv: MatrixValue): TableValue = {
    val groupedRDD = GFisherDataPrep.prepOGFisherCorrRDD(mv, nTests, keyField, pField, dfField, weightField, corrField, rowIDXField)
    val newrdd = groupedRDD.map{case(key, vals) =>
      val valArr = vals.toArray// array of the rows in this group. each element is a tuple with all the fields.

      val (pvals: BDV[Double], df: BDM[Int], w: BDM[Double], corrMat: BDM[Double]) = GFisherDataPrep.oGFisherTupsCorrToVectors(valArr, nTests)
      val res = PvalOGFisher.pvalOGFisher(pvals, df, w, corrMat, method = method)
      Row(key,
        res("stat"),
        res("pval"),
        res("stat_indi").asInstanceOf[BDV[Double]].toArray.toFastSeq,
        res("pval_indi").asInstanceOf[BDV[Double]].toArray.toFastSeq
        // res("stat_indi"),
        // res("pval_indi")
      )
    }
    TableValue(ctx, typ(mv.typ).rowType, typ(mv.typ).key, newrdd)
  }
}
