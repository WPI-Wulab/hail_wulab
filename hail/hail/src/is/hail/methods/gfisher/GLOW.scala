/**
  * This file contains the supporting functions for the GLOW pipeline, allowing for connectivity between the
  * Scala code base and the GLOW python user interface.
  * Reference: Liu, Ming. "Integrative Analysis of Large Genomic Data." WPI (2025).
  * @author Peter Howell
  * Last update (latest update first):
  *   Claude Code (Opus 5.5), prompted by VPanwar, 2026-11-04: ported to the Hail 0.2.138 API (stage 3c of
  *     GLOWh's 0project/glow-hail-refactor-plan.md): the data-prep calls take the ExecuteContext
  *     for broadcasts, as HailContext is gone; FastSeq and toFastSeq from is.hail.collection; `override` on execute and
  *     preservesPartitionCounts (Release builds treat WartRemover's MissingOverride as an error). No change to what is computed.
  *   KHoar 2025-05-07: Added header to file
  */

package is.hail.methods.gfisher

import is.hail.stats.RegressionUtils
import is.hail.backend.ExecuteContext
import is.hail.collection.FastSeq
import is.hail.collection.implicits.toRichIterable
import is.hail.expr.ir.{MatrixValue, TableValue}
import is.hail.expr.ir.functions.MatrixToTableFunction
import is.hail.types.virtual.{MatrixType, TFloat64, TStruct, TableType, TArray, TInt32}
import is.hail.utils._

import is.hail.methods.gfisher.OptimalWeights.{getH_Binary, getH_Continuous}
import is.hail.methods.gfisher.FuncCalcuZScores.{getZMargScoreBinary, getZMargScoreContinuous, getZMargScoreContinuousT, getZ_marg_score_binary_SPA}
import org.apache.spark.sql.Row

import breeze.linalg.{DenseMatrix => BDM, DenseVector => BDV}

/**
  * Generalized Fisher's combination testing.
  *
  * @author Peter Howell
  *
  * @param keyField name of field to group by
  * @param oneSided whether the input p-values are one-sided
  * @param method what test to conduct. Either BURDEN, SKAT, OMNI, or FISHER
  *
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
  useSPA: Boolean,
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
  override def preservesPartitionCounts: Boolean = false

  // the real code goes in here
  override def execute(ctx: ExecuteContext, mv: MatrixValue): TableValue = {

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

    val groupedRDD = GFisherDataPrep.prepGlowRDD(ctx, mv, keyField, bField, piField, genoField, completeColIdx)

    def linearGlow() = {

      if (useT) {
        val XBC = ctx.backend.broadcast(cov)
        val yBC = ctx.backend.broadcast(y)

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
      val HhalfBC = ctx.backend.broadcast(hH)
      val s0BC = ctx.backend.broadcast(s0)
      val residsBC = ctx.backend.broadcast(resids)

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
      if (useSPA) {
        val y0BC = ctx.backend.broadcast(y)
        val XBC = ctx.backend.broadcast(cov)
        groupedRDD.map{case(key, vals) =>
          val valArr = vals.toArray// array of the rows in this group. each element is a tuple with all the fields.
          val n = valArr.length
          val (b: BDV[Double], pi: BDV[Double], geno: BDM[Double]) = GFisherArrayToVectors.glow(valArr)

          val zstats = getZ_marg_score_binary_SPA(geno, XBC.value, y0BC.value)
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
      else {
        val (hH, y0, resids) = getH_Binary(cov, y)
        val HhalfBC = ctx.backend.broadcast(hH)
        val y0BC = ctx.backend.broadcast(y0)
        val residsBC = ctx.backend.broadcast(resids)
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
    }




    val newrdd = if (logistic) {
      logisticGlow()
    } else {
      linearGlow()
    }
    TableValue(ctx, typ(mv.typ).rowType, typ(mv.typ).key, newrdd)
  }
}
