/**
  * @author Peter Howell
  */
package is.hail.methods.gfisher

import is.hail.HailContext
import is.hail.stats.RegressionUtils
import is.hail.backend.ExecuteContext
import is.hail.expr.ir.{MatrixValue, TableValue}
import is.hail.expr.ir.functions.MatrixToTableFunction
import is.hail.types.virtual.{MatrixType, TFloat64, TStruct, TableType, TArray, TInt32}
import is.hail.utils._

import is.hail.methods.gfisher.OptimalWeights.{getH_Binary, getH_Continuous}
import is.hail.methods.gfisher.FuncCalcuZScores.{getZMargScoreBinary, getZMargScoreContinuous}
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
case class GLOW(
  keyField: String,
  keyFieldOut: String,
  bField: String,
  piField: String,
  genoField: String,
  covFields: Seq[String],
  phenoField: String,
  logistic: Boolean,
) extends MatrixToTableFunction {

  override def typ(childType: MatrixType): TableType = {
    val keyType = childType.rowType.fieldType(keyField)
    val mySchema = TStruct(
      (keyFieldOut, keyType),
      ("n", TInt32),
      ("Zstat", TArray(TFloat64)),
    )
    TableType(mySchema, FastSeq(keyFieldOut), TStruct.empty)
  }
  def preservesPartitionCounts: Boolean = false

  def execute(ctx: ExecuteContext, mv: MatrixValue): TableValue = {

    val (y, cov, completeColIdx) =
      RegressionUtils.getPhenoCovCompleteSamples(mv, phenoField, covFields.toArray)

    val n = y.size
    val k = cov.cols
    val d = n - k

    if (d < 1)
      fatal(
        s"$n samples and $k ${plural(k, "covariate")} (including intercept) implies $d degrees of freedom."
      )
    if (logistic) {
      val badVals = y.findAll(yi => yi != 0d && yi != 1d)
      if (badVals.nonEmpty)
        fatal(s"For logistic SKAT, phenotype must be Boolean or numeric with value 0 or 1 for each complete " +
          s"sample; found ${badVals.length} ${plural(badVals.length, "violation")} starting with ${badVals(0)}")
    }

    val groupedRDD = GFisherDataPrep.prepGlowRDD(mv, keyField, bField, piField, genoField, completeColIdx)
    printMat(cov, "cov")
    def linearGlow() = {
      val (hH, s0, resids) = getH_Continuous(cov, y)
      val HhalfBC = HailContext.backend.broadcast(hH)
      val s0BC = HailContext.backend.broadcast(s0)
      val residsBC = HailContext.backend.broadcast(resids)
      // printVec(resids, "resids")
      groupedRDD.map{case(key, vals) =>
        val valArr = vals.toArray// array of the rows in this group. each element is a tuple with all the fields.
        val n = valArr.length
        val (b: BDV[Double], pi: BDV[Double], geno: BDM[Double]) = GFisherArrayToVectors.glow(valArr)
        // printVec(b, s"$key b")
        // printVec(pi, s"$key pi")
        // printMat(geno, s"$key geno")
        // val GHG = OptimalWeights.getGHG_Continuous(geno, HhalfBC.value)
        // printMat(GHG, s"$key: GHG")
        // val score = geno.t * residsBC.value / s0BC.value
        // printVec(score, s"$key: score")

        val zstats = getZMargScoreContinuous(geno, HhalfBC.value, s0BC.value, residsBC.value)
        // printVec(zstats("Zscores").asInstanceOf[BDV[Double]], s"$key: zscores")

        Row(key, n, zstats("Zscores").asInstanceOf[BDV[Double]].data.toFastSeq)
      }
    }

    def logisticGlow() = {
      val (hH, y0, resids) = getH_Binary(cov, y)
      val HhalfBC = HailContext.backend.broadcast(hH)
      val y0BC = HailContext.backend.broadcast(y0)
      val residsBC = HailContext.backend.broadcast(resids)
      groupedRDD.map{case(key, vals) =>
        val valArr = vals.toArray// array of the rows in this group. each element is a tuple with all the fields.
        val n = valArr.length
        val (b: BDV[Double], pi: BDV[Double], geno: BDM[Double]) = GFisherArrayToVectors.glow(valArr)

        val zstats = getZMargScoreBinary(geno, HhalfBC.value, y0BC.value, residsBC.value)
        Row(key, n, zstats("Zscores").asInstanceOf[BDV[Double]].toArray.toFastSeq)
      }
    }




    val newrdd = if (logistic) {
      logisticGlow()
    } else {
      linearGlow()
    }
    TableValue(ctx, typ(mv.typ).rowType, typ(mv.typ).key, newrdd)
  }
}
