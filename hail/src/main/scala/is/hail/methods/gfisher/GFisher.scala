/**
  * This file contains the supporting functions for the GFisher and oGFisher pipelines, allowing for connectivity between the
  * Scala code base and the GFisher python user interface.
  * Reference: Zhang, Hong, and Zheyang Wu. "The generalized Fisher's combination and accurate p‐value
           calculation under dependence." Biometrics 79.2 (2023): 1159-1172.
  * @author Peter Howell
  * Last update (latest update first):
  *   KHoar 2025-05-07: Added header to file
  */

package is.hail.methods.gfisher

import is.hail.backend.ExecuteContext
import is.hail.expr.ir.{MatrixValue, TableValue}

import is.hail.expr.ir.functions.MatrixToTableFunction

import is.hail.types.virtual.{MatrixType, TFloat64, TStruct, TableType, TArray, TInt32}

import is.hail.utils._

import org.apache.spark.sql.Row

import breeze.linalg.{DenseMatrix => BDM, DenseVector => BDV}

/**
  * To connect this sort of scala class with python, you must change:
  *   - hail/python/hail/ir/table_ir.py
  *     - to update the schema of the return type
  *   - hail/python/hail/methods/statgen.py
  *     - make python function, make correct config dictionary
  *   - hail/python/hail/methods/__init__.py
  *     - import and the function in statgen, and put it in __all__
  *   - hail/src/main/scala/is/hail/expr/ir/functions/RelationalFunctions.scala
  *     - import it and add a line with classOf[<YourClass>] to the list
  */

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
  useGenotype: Boolean,
  genoField: String,
  corrField: String,
  rowIDXField: String,
  method: String,
  oneSided: Boolean
) extends MatrixToTableFunction {

  override def typ(childType: MatrixType): TableType = {
    val keyType = childType.rowType.fieldType(keyField)
    val mySchema = TStruct(
      (keyFieldOut, keyType),
      ("n", TInt32),
      ("stat", TFloat64),
      ("p_value", TFloat64)
    )
    TableType(mySchema, FastSeq(keyFieldOut), TStruct.empty)
  }
  def preservesPartitionCounts: Boolean = false

  def execute(ctx: ExecuteContext, mv: MatrixValue): TableValue = {

    def genotypeData() = {
      val groupedRDD = GFisherDataPrep.prepGFisherGenoRDD(mv, keyField, pField, dfField, weightField, genoField)
      val newrdd = groupedRDD.map{case(key, vals) =>
        val valArr = vals.toArray// array of the rows in this group. each element is a tuple with all the fields.

        val (pvals: BDV[Double], df: BDV[Double], w: BDV[Double], corrMat: BDM[Double]) = GFisherArrayToVectors.gFisherGeno(valArr)
        val gFishStat: Double = StatGFisher.statGFisher(pvals, df, w)
        val gFishPVal: Double = if (method == "HYB") PGFisher.pGFisherHyb(gFishStat, df, w, corrMat)
                                else if (method == "MR") PGFisher.pGFisherMR(gFishStat, df, w, corrMat, oneSided)
                                else PGFisher.pGFisherGB(gFishStat, df, w, corrMat, oneSided)
        Row(key, valArr.length, gFishStat, gFishPVal)
      }
      newrdd
    }
    def corrData() = {
      val groupedRDD = GFisherDataPrep.prepGFisherCorrRDD(mv, keyField, pField, dfField, weightField, corrField, rowIDXField)
      val newrdd = groupedRDD.map{case(key, vals) =>
        val valArr = vals.toArray// array of the rows in this group. each element is a tuple with all the fields.

        val (pvals: BDV[Double], df: BDV[Double], w: BDV[Double], corrMat: BDM[Double]) = GFisherArrayToVectors.gFisherCorr(valArr)
        val gFishStat: Double = StatGFisher.statGFisher(pvals, df, w)
        val gFishPVal: Double = if (method == "HYB") PGFisher.pGFisherHyb(gFishStat, df, w, corrMat)
                                else if (method == "MR") PGFisher.pGFisherMR(gFishStat, df, w, corrMat, oneSided)
                                else PGFisher.pGFisherGB(gFishStat, df, w, corrMat, oneSided)
        Row(key, valArr.length, gFishStat, gFishPVal)
      }
      newrdd
    }

    val newrdd = if (useGenotype) {
      genotypeData()
    } else {
      corrData()
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
  useGenotype: Boolean,
  genoField: String,
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
      ("n", TInt32),
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
    def genotypeData() = {
      val groupedRDD = GFisherDataPrep.prepOGFisherGenoRDD(mv, keyField, pField, dfField, weightField, genoField, nTests)
      val newrdd = groupedRDD.map{case(key, vals) =>
        val valArr = vals.toArray// array of the rows in this group. each element is a tuple with all the fields.

        val (pvals: BDV[Double], df: BDM[Double], w: BDM[Double], corrMat: BDM[Double]) = GFisherArrayToVectors.oGFisherGeno(valArr, nTests)
        val res = PvalOGFisher.pvalOGFisher(pvals, df, w, corrMat, method = method)
        Row(key,
          valArr.length,
          res("stat"),
          res("pval"),
          res("stat_indi").asInstanceOf[BDV[Double]].toArray.toFastSeq,
          res("pval_indi").asInstanceOf[BDV[Double]].toArray.toFastSeq
          // res("stat_indi"),
          // res("pval_indi")
        )
      }
      newrdd
    }

    def corrData() = {
      val groupedRDD = GFisherDataPrep.prepOGFisherCorrRDD(mv, keyField, pField, dfField, weightField, corrField, rowIDXField, nTests)
      val newrdd = groupedRDD.map{case(key, vals) =>
        val valArr = vals.toArray// array of the rows in this group. each element is a tuple with all the fields.

        val (pvals: BDV[Double], df: BDM[Double], w: BDM[Double], corrMat: BDM[Double]) = GFisherArrayToVectors.oGFisherCorr(valArr, nTests)
        val res = PvalOGFisher.pvalOGFisher(pvals, df, w, corrMat, method = method)
        Row(key,
          valArr.length,
          res("stat"),
          res("pval"),
          res("stat_indi").asInstanceOf[BDV[Double]].toArray.toFastSeq,
          res("pval_indi").asInstanceOf[BDV[Double]].toArray.toFastSeq
          // res("stat_indi"),
          // res("pval_indi")
        )
      }
      newrdd
    }
    val newrdd = if (useGenotype) {
      genotypeData()
    } else {
      corrData()
    }
    TableValue(ctx, typ(mv.typ).rowType, typ(mv.typ).key, newrdd)
  }
}
