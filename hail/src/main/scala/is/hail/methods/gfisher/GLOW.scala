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
import is.hail.methods.gfisher.FuncCalcuZScores.{getZMargScoreBinary, getZMargScoreContinuous, getZMargScoreContinuousT}
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
  * @param method what test to conduct. Either BURDEN, SKAT, OMNI, or FISHER
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
  method: String,
  useT: Boolean,
) extends MatrixToTableFunction {

  // define the return type of the function
  override def typ(childType: MatrixType): TableType = {
    val keyType = childType.rowType.fieldType(keyField)
    val mySchema = TStruct(
      (keyFieldOut, keyType),
      ("n", TInt32),
      ("stat", TArray(TFloat64)),
      ("pval", TArray(TFloat64)),
    )
    TableType(mySchema, FastSeq(keyFieldOut), TStruct.empty)
  }
  def preservesPartitionCounts: Boolean = false

  // the real code goes in here
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
        fatal(s"For logistic GLOW, phenotype must be Boolean or numeric with value 0 or 1 for each complete " +
          s"sample; found ${badVals.length} ${plural(badVals.length, "violation")} starting with ${badVals(0)}")
    }

    val groupedRDD = GFisherDataPrep.prepGlowRDD(mv, keyField, bField, piField, genoField, completeColIdx)

    def linearGlow() = {

      if (useT) {
        val XBC = HailContext.backend.broadcast(cov)
        val yBC = HailContext.backend.broadcast(y)

        groupedRDD.map{case(key, vals) =>
          val valArr = vals.toArray// array of the rows in this group. each element is a tuple with all the fields.
          val n = valArr.length
          val (b: BDV[Double], pi: BDV[Double], geno: BDM[Double]) = GFisherArrayToVectors.glow(valArr)

          val zstats = getZMargScoreContinuousT(geno, XBC.value, yBC.value)
          val result = method match {
            case "BURDEN" => GLOW_Burden.GLOW_Burden(zstats, b, pi)
            case "SKAT" => GLOW_SKAT.GLOW_SKAT(zstats, b, pi)
            case "OMNI" => GLOW_Omni.GLOW_Omni(zstats, b, pi)
            case "FISHER" => GLOW_Fisher.GLOW_Fisher(zstats, b, pi)
            case _ => fatal(s"Unknown method: $method")
          }

          Row(key, n, result("STAT").toArray.toFastSeq, result("PVAL").toArray.toFastSeq)
        }
      }
      val (hH, s0, resids) = getH_Continuous(cov, y)
      val HhalfBC = HailContext.backend.broadcast(hH)
      val s0BC = HailContext.backend.broadcast(s0)
      val residsBC = HailContext.backend.broadcast(resids)

      groupedRDD.map{case(key, vals) =>
        val valArr = vals.toArray// array of the rows in this group. each element is a tuple with all the fields.
        val n = valArr.length
        val (b: BDV[Double], pi: BDV[Double], geno: BDM[Double]) = GFisherArrayToVectors.glow(valArr)

        val zstats = getZMargScoreContinuous(geno, HhalfBC.value, s0BC.value, residsBC.value)
        val result = method match {
          case "BURDEN" => GLOW_Burden.GLOW_Burden(zstats, b, pi)
          case "SKAT" => GLOW_SKAT.GLOW_SKAT(zstats, b, pi)
          case "OMNI" => GLOW_Omni.GLOW_Omni(zstats, b, pi)
          case "FISHER" => GLOW_Fisher.GLOW_Fisher(zstats, b, pi)
          case _ => fatal(s"Unknown method: $method")
        }

        Row(key, n, result("STAT").toArray.toFastSeq, result("PVAL").toArray.toFastSeq)
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
        val result = method match {
          case "BURDEN" => GLOW_Burden.GLOW_Burden(zstats, b, pi)
          case "SKAT" => GLOW_SKAT.GLOW_SKAT(zstats, b, pi)
          case "OMNI" => GLOW_Omni.GLOW_Omni(zstats, b, pi)
          case "FISHER" => GLOW_Fisher.GLOW_Fisher(zstats, b, pi)
          case _ => fatal(s"Unknown method: $method")
        }

        Row(key, n, result("STAT").toArray.toFastSeq, result("PVAL").toArray.toFastSeq)
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
