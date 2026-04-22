from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

import hail as hl

try:
    from testing.graphlet_screening_compare import (
        DEFAULT_LOCUS_WINDOW_BP,
        ExpectedLocus,
        ObservedHit,
        load_expected_loci,
        summarize_expected_matches,
        write_comparison_report,
    )
except ModuleNotFoundError:
    from graphlet_screening_compare import (
        DEFAULT_LOCUS_WINDOW_BP,
        ExpectedLocus,
        ObservedHit,
        load_expected_loci,
        summarize_expected_matches,
        write_comparison_report,
    )

TraitType = Literal["auto", "quantitative", "binary"]
PhenotypeSource = Literal["fam", "external"]


@dataclass
class GWASGSConfig:
    bed: str
    bim: str
    fam: str
    output_dir: str
    phenotype_source: PhenotypeSource = "fam"
    trait_type: TraitType = "auto"
    phenotype_table_path: str | None = None
    phenotype_sample_column: str = "s"
    phenotype_column: str | None = None
    covariate_table_path: str | None = None
    covariate_sample_column: str = "s"
    covariate_columns: list[str] = field(default_factory=list)
    include_is_female_covariate: bool = True
    expected_loci_path: str | None = None
    default_locus_window_bp: int = DEFAULT_LOCUS_WINDOW_BP
    significance_threshold: float = 5e-8
    missing_pheno_token: str = "NA"
    fam_delimiter: str = " +"
    quant_pheno: bool | None = None
    reference_genome: str | None = "GRCh38"
    a2_reference: bool = True
    skip_invalid_loci: bool = False
    n_partitions: int | None = None
    import_block_size: int | None = None
    checkpoint_path: str | None = None
    spark_driver_memory: str | None = None
    spark_executor_memory: str | None = None
    spark_conf: dict[str, str] = field(default_factory=dict)
    min_maf: float | None = 0.01
    min_row_call_rate: float | None = None
    min_col_call_rate: float | None = None
    gs_nm: int = 3
    gs_r: float = 3.5
    gs_family: str = "auto"
    gs_mode: str = "global"
    gs_selection_aware: bool = True
    gs_split_fraction: float = 0.5
    gs_seed: int = 1
    gs_block_size: int = 32
    gs_block_overlap: int = 32
    gs_max_cluster_size: int = 20
    gs_sparsity_level: float | None = None
    gs_max_iterations: int = 25
    gs_tolerance: float = 1e-6
    gs_ld_prune: bool = False
    gs_ld_prune_r2: float = 0.2
    gs_ld_prune_window_bp: int = 500_000
    gs_ld_prune_memory_per_core: int = 256
    gs_ld_prune_keep_higher_maf: bool = True
    smoke_test: bool = False
    smoke_row_hash_modulus: int = 32
    smoke_col_hash_modulus: int = 128
    max_hits_to_report: int = 200
    export_results: bool = True
    quiet: bool = True
    init_hail: bool = True
    fail_on_missing_expected: bool = True

    @classmethod
    def from_json_file(cls, path: str | Path) -> "GWASGSConfig":
        with Path(path).open(encoding="utf-8") as config_file:
            data = json.load(config_file)
        return cls(**data)


class StageTimer:
    def __init__(self) -> None:
        self.timings: dict[str, float] = {}

    def __call__(self, stage_name: str):
        timer = self

        class _StageContext:
            def __enter__(self_inner) -> None:
                self_inner.start = time.perf_counter()

            def __exit__(self_inner, exc_type, exc, tb) -> None:
                timer.timings[stage_name] = time.perf_counter() - self_inner.start

        return _StageContext()


def _ensure_output_dir(path: str | Path) -> Path:
    output_dir = Path(path)
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def _path_exists(path: str | Path) -> bool:
    return Path(path).exists()


def _possible_contig_labels(contig: str | None) -> set[str]:
    if contig is None:
        return set()
    if contig.startswith("chr"):
        return {contig, contig[3:]}
    return {contig, f"chr{contig}"}


def _load_fam_summary(fam_path: str | Path, missing_pheno_token: str) -> dict[str, Any]:
    sample_ids: set[str] = set()
    phenotype_counts: dict[str, int] = {}
    sex_counts: dict[str, int] = {}
    duplicate_ids: list[str] = []
    for line_number, line in enumerate(Path(fam_path).read_text(encoding="utf-8").splitlines(), start=1):
        fields = line.split()
        if len(fields) < 6:
            raise ValueError(f"Malformed FAM line {line_number}: expected 6 columns, found {len(fields)}")
        sample_id = fields[1]
        if sample_id in sample_ids:
            duplicate_ids.append(sample_id)
        sample_ids.add(sample_id)
        phenotype_counts[fields[5]] = phenotype_counts.get(fields[5], 0) + 1
        sex_counts[fields[4]] = sex_counts.get(fields[4], 0) + 1

    observed_pheno_values = {
        value
        for value in phenotype_counts
        if value not in {"0", "-9", "N/A", missing_pheno_token}
    }
    inferred_binary = observed_pheno_values.issubset({"1", "2"})
    return {
        "sample_count": len(sample_ids),
        "duplicate_sample_ids": duplicate_ids,
        "phenotype_counts": phenotype_counts,
        "sex_counts": sex_counts,
        "inferred_trait_type": "binary" if inferred_binary else "quantitative",
        "inferred_quant_pheno": not inferred_binary,
    }


def _resolve_spark_conf(config: GWASGSConfig) -> dict[str, str]:
    spark_conf = dict(config.spark_conf)
    if config.spark_driver_memory:
        spark_conf["spark.driver.memory"] = config.spark_driver_memory
    if config.spark_executor_memory:
        spark_conf["spark.executor.memory"] = config.spark_executor_memory

    # Global mode needs enough heap to materialize the full design matrix on driver.
    if config.gs_mode == "global" and not config.smoke_test:
        spark_conf.setdefault("spark.driver.memory", "4g")
        spark_conf.setdefault("spark.executor.memory", "4g")

    return spark_conf


def _ensure_hail_initialized(config: GWASGSConfig) -> None:
    try:
        hl.current_backend()
        return
    except Exception:
        spark_conf = _resolve_spark_conf(config)
        hl.init(
            master="local[*]",
            min_block_size=0,
            quiet=config.quiet,
            global_seed=0,
            spark_conf=spark_conf or None,
        )


def _bool_to_float64(expr):
    return hl.or_missing(hl.is_defined(expr), hl.if_else(expr, 1.0, 0.0))


def _to_float64(expr):
    if expr.dtype == hl.tbool:
        return _bool_to_float64(expr)
    return hl.float64(expr)


def _bool_like_expr(expr):
    if expr.dtype == hl.tbool:
        return expr
    return hl.or_missing(hl.is_defined(expr), expr != 0)


def _materialize(mt: hl.MatrixTable) -> tuple[int, int]:
    return mt.count()


def _persist_or_checkpoint(mt: hl.MatrixTable, config: GWASGSConfig, output_dir: Path, suffix: str) -> hl.MatrixTable:
    if config.checkpoint_path:
        checkpoint_path = Path(config.checkpoint_path)
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        return mt.checkpoint(str(checkpoint_path.with_name(f"{checkpoint_path.stem}_{suffix}{checkpoint_path.suffix or '.mt'}")), overwrite=True)
    return mt.persist()


def _annotate_qc(mt: hl.MatrixTable) -> hl.MatrixTable:
    return hl.sample_qc(hl.variant_qc(mt))


def _collect_row_metadata(mt: hl.MatrixTable) -> hl.Table:
    field_names = [field for field in ("rsid", "cm_position") if field in mt.row_value]
    rows = mt.rows()
    if field_names:
        return rows.select(*field_names)
    return rows.select()


def _contig_expr(dataset):
    return hl.str(dataset.locus.contig)


def _prepare_plink_matrix_table(config: GWASGSConfig) -> tuple[hl.MatrixTable, dict[str, Any]]:
    fam_summary = _load_fam_summary(config.fam, config.missing_pheno_token)
    if fam_summary["duplicate_sample_ids"]:
        duplicates = sorted(set(fam_summary["duplicate_sample_ids"]))[:10]
        raise ValueError(f"Duplicate sample IDs in FAM file, first duplicates: {duplicates}")

    quant_pheno = config.quant_pheno
    trait_type = config.trait_type
    if config.phenotype_source == "fam":
        if trait_type == "auto":
            trait_type = fam_summary["inferred_trait_type"]
        if quant_pheno is None:
            quant_pheno = fam_summary["inferred_quant_pheno"]
    else:
        quant_pheno = bool(config.quant_pheno) if config.quant_pheno is not None else False

    mt = hl.import_plink( # is phenotype mismatched?
        bed=config.bed,
        bim=config.bim,
        fam=config.fam,
        delimiter=config.fam_delimiter,
        missing=config.missing_pheno_token,
        quant_pheno=quant_pheno,
        a2_reference=config.a2_reference,
        reference_genome=config.reference_genome,
        skip_invalid_loci=config.skip_invalid_loci,
        n_partitions=config.n_partitions,
        block_size=config.import_block_size,
    )

    if config.smoke_test:
        mt = mt.add_row_index("_smoke_row_idx").add_col_index("_smoke_col_idx")
        mt = mt.filter_rows((mt._smoke_row_idx % config.smoke_row_hash_modulus) == 0)
        mt = mt.filter_cols((mt._smoke_col_idx % config.smoke_col_hash_modulus) == 0)
        mt = mt.drop("_smoke_row_idx", "_smoke_col_idx")

    mt = _persist_or_checkpoint(mt, config, Path(config.output_dir), "imported")
    _materialize(mt)

    return mt, {
        "fam_summary": fam_summary,
        "trait_type": trait_type,
        "quant_pheno": quant_pheno,
    }


def _import_metadata_table(path: str, key_column: str) -> hl.Table:
    return hl.import_table(path, impute=True).key_by(key_column)


def _prepare_trait_and_covariates(
    mt: hl.MatrixTable,
    config: GWASGSConfig,
    inferred_trait_type: str,
) -> tuple[hl.MatrixTable, Any, Any, list[Any], str]:
    trait_type = inferred_trait_type

    if config.phenotype_source == "external":
        if not config.phenotype_table_path or not config.phenotype_column:
            raise ValueError("External phenotype source requires phenotype_table_path and phenotype_column")
        phenotype_ht = _import_metadata_table(config.phenotype_table_path, config.phenotype_sample_column)
        mt = mt.annotate_cols(_phenotype=phenotype_ht[mt.s])
        phenotype_expr = mt._phenotype[config.phenotype_column]
        if trait_type == "auto":
            trait_type = "binary" if phenotype_expr.dtype == hl.tbool else "quantitative"
    else:
        if inferred_trait_type == "binary":
            phenotype_expr = mt.is_case
        else:
            phenotype_expr = mt.quant_pheno

    if trait_type == "binary":
        baseline_y = _bool_like_expr(phenotype_expr)
        gs_y = _bool_to_float64(baseline_y)
    else:
        baseline_y = _to_float64(phenotype_expr)
        gs_y = baseline_y

    covariates: list[Any] = [hl.float64(1.0)]
    if config.include_is_female_covariate and "is_female" in mt.col_value:
        covariates.append(_bool_to_float64(mt.is_female))

    if config.covariate_table_path:
        covariate_ht = _import_metadata_table(config.covariate_table_path, config.covariate_sample_column)
        mt = mt.annotate_cols(_covariates=covariate_ht[mt.s])
        for covariate in config.covariate_columns:
            covariates.append(_to_float64(mt._covariates[covariate]))
    elif config.phenotype_source == "external" and config.covariate_columns:
        for covariate in config.covariate_columns:
            covariates.append(_to_float64(mt._phenotype[covariate]))

    return mt, gs_y, baseline_y, covariates, trait_type


def _summarize_matrix_table(mt: hl.MatrixTable, phenotype_expr, trait_type: str, covariates: list[Any] | None = None) -> dict[str, Any]:
    complete_case = hl.is_defined(phenotype_expr)
    for covariate in covariates or []:
        complete_case = complete_case & hl.is_defined(covariate)
    summary_mt = mt.annotate_cols(_summary_phenotype=phenotype_expr, _summary_complete_case=complete_case).key_cols_by()
    row_count, col_count = _materialize(summary_mt)
    summary: dict[str, Any] = {
        "variant_count": row_count,
        "sample_count": col_count,
        "samples_with_complete_model_data": summary_mt.aggregate_cols(hl.agg.count_where(summary_mt._summary_complete_case)),
        "contig_counts": summary_mt.aggregate_rows(hl.agg.counter(_contig_expr(summary_mt))),
        "samples_with_defined_phenotype": summary_mt.aggregate_cols(
            hl.agg.count_where(hl.is_defined(summary_mt._summary_phenotype))
        ),
        "sex_distribution": summary_mt.aggregate_cols(
            hl.agg.counter(
                hl.case()
                .when(hl.is_missing(summary_mt.is_female), "missing")
                .when(summary_mt.is_female, "female")
                .default("male")
            )
        )
        if "is_female" in summary_mt.col_value
        else {},
    }

    if trait_type == "binary":
        summary["phenotype_distribution"] = summary_mt.aggregate_cols(
            hl.agg.counter(
                hl.case()
                .when(hl.is_missing(summary_mt._summary_phenotype), "missing")
                .when(summary_mt._summary_phenotype, "case")
                .default("control")
            )
        )
    else:
        stats = summary_mt.aggregate_cols(hl.agg.stats(summary_mt._summary_phenotype))
        summary["phenotype_distribution"] = dict(stats)

    qc_mt = _annotate_qc(summary_mt.select_entries(GT=summary_mt.GT))
    summary["sample_call_rate_stats"] = dict(qc_mt.aggregate_cols(hl.agg.stats(qc_mt.sample_qc.call_rate)))
    summary["variant_call_rate_stats"] = dict(qc_mt.aggregate_rows(hl.agg.stats(qc_mt.variant_qc.call_rate)))
    summary["maf_stats"] = dict(
        qc_mt.aggregate_rows(hl.agg.stats(hl.min(qc_mt.variant_qc.AF[1], 1.0 - qc_mt.variant_qc.AF[1])))
    )
    summary["monomorphic_variant_count"] = qc_mt.aggregate_rows(
        hl.agg.count_where((qc_mt.variant_qc.AF[1] == 0.0) | (qc_mt.variant_qc.AF[1] == 1.0))
    )
    return summary


def _apply_filters(mt: hl.MatrixTable, config: GWASGSConfig) -> tuple[hl.MatrixTable, dict[str, Any]]:
    qc_mt = _annotate_qc(mt)
    filter_report = {
        "pre_filter_variant_count": qc_mt.count_rows(),
        "pre_filter_sample_count": qc_mt.count_cols(),
        "min_maf": config.min_maf,
        "min_row_call_rate": config.min_row_call_rate,
        "min_col_call_rate": config.min_col_call_rate,
    }

    if config.min_maf is not None:
        qc_mt = qc_mt.filter_rows(hl.min(qc_mt.variant_qc.AF[1], 1.0 - qc_mt.variant_qc.AF[1]) >= config.min_maf)
    if config.min_row_call_rate is not None:
        qc_mt = qc_mt.filter_rows(qc_mt.variant_qc.call_rate >= config.min_row_call_rate)
    if config.min_col_call_rate is not None:
        qc_mt = qc_mt.filter_cols(qc_mt.sample_qc.call_rate >= config.min_col_call_rate)

    qc_mt = qc_mt.drop("sample_qc", "variant_qc")
    post_rows, post_cols = _materialize(qc_mt)
    filter_report["post_filter_variant_count"] = post_rows
    filter_report["post_filter_sample_count"] = post_cols
    filter_report["dropped_variant_count"] = filter_report["pre_filter_variant_count"] - post_rows
    filter_report["dropped_sample_count"] = filter_report["pre_filter_sample_count"] - post_cols
    return qc_mt.persist(), filter_report


def _prepare_gs_matrix(mt: hl.MatrixTable, config: GWASGSConfig) -> tuple[hl.MatrixTable, dict[str, Any]]:
    report = {
        "ld_pruned": config.gs_ld_prune,
        "r2": config.gs_ld_prune_r2,
        "bp_window_size": config.gs_ld_prune_window_bp,
        "memory_per_core_mb": config.gs_ld_prune_memory_per_core,
        "keep_higher_maf": config.gs_ld_prune_keep_higher_maf,
        "pre_prune_variant_count": mt.count_rows(),
        "pre_prune_sample_count": mt.count_cols(),
    }
    if not config.gs_ld_prune:
        report["post_prune_variant_count"] = report["pre_prune_variant_count"]
        report["post_prune_sample_count"] = report["pre_prune_sample_count"]
        report["dropped_variant_count"] = 0
        report["dropped_sample_count"] = 0
        return mt, report

    pruned_variant_ht = hl.ld_prune(
        mt.GT,
        r2=config.gs_ld_prune_r2,
        bp_window_size=config.gs_ld_prune_window_bp,
        memory_per_core=config.gs_ld_prune_memory_per_core,
        keep_higher_maf=config.gs_ld_prune_keep_higher_maf,
    )
    pruned_mt = mt.filter_rows(hl.is_defined(pruned_variant_ht[mt.row_key])).persist()
    post_rows, post_cols = _materialize(pruned_mt)
    if post_rows == 0:
        raise ValueError("GS LD pruning removed all variants; relax the LD-prune parameters.")

    report["post_prune_variant_count"] = post_rows
    report["post_prune_sample_count"] = post_cols
    report["dropped_variant_count"] = report["pre_prune_variant_count"] - post_rows
    report["dropped_sample_count"] = report["pre_prune_sample_count"] - post_cols
    report["variant_retention_fraction"] = (
        post_rows / report["pre_prune_variant_count"] if report["pre_prune_variant_count"] else None
    )
    return pruned_mt, report


def _annotate_row_metadata(ht: hl.Table, row_metadata: hl.Table) -> hl.Table:
    extra_fields = {field: row_metadata[ht.key][field] for field in row_metadata.row_value}
    return ht.annotate(**extra_fields)


def _remove_output_path(path: Path) -> None:
    if path.exists():
        if path.is_dir():
            for child in path.iterdir():
                if child.is_file():
                    child.unlink()
        else:
            path.unlink()


def _export_result_table(ht: hl.Table, path: Path) -> None:
    _remove_output_path(path)
    ht.export(str(path))


def _extract_scalar(row, field: str):
    return row[field] if field in row else None


def _row_to_hit(row, method: str) -> ObservedHit:
    return ObservedHit(
        locus_label=f"{row.locus.contig}:{row.locus.position}",
        contig=str(row.locus.contig),
        position=int(row.locus.position),
        p_value=float(row.p_value),
        method=method,
        rsid=_extract_scalar(row, "rsid"),
        alleles=tuple(row.alleles),
        beta=float(row.beta) if "beta" in row and row.beta is not None and not math.isnan(row.beta) else None,
        n=int(row.n) if "n" in row and row.n is not None else None,
    )


def _collect_significant_hits(ht: hl.Table, *, method: str, threshold: float, limit: int) -> list[ObservedHit]:
    filtered = ht.filter(hl.is_defined(ht.p_value) & (ht.p_value <= threshold))
    rows = filtered.order_by(filtered.p_value).take(limit)
    return [_row_to_hit(row, method) for row in rows]


def _filter_for_expected(ht: hl.Table, expected: ExpectedLocus, default_window_bp: int) -> hl.Table:
    contigs = _possible_contig_labels(expected.contig)
    contig_filter = hl.literal(contigs).contains(_contig_expr(ht)) if contigs else hl.literal(True)

    if expected.match_type == "exact" and expected.rsid:
        return ht.filter(hl.is_defined(ht.rsid) & (ht.rsid == expected.rsid))

    if expected.position is None:
        return ht.filter(contig_filter)

    if expected.match_type == "exact":
        return ht.filter(contig_filter & (ht.locus.position == expected.position))

    window_bp = expected.resolved_window_bp(default_window_bp)
    return ht.filter(
        contig_filter
        & (ht.locus.position >= expected.position - window_bp)
        & (ht.locus.position <= expected.position + window_bp)
    )


def _collect_best_expected_contexts(
    ht: hl.Table,
    expected_loci: list[ExpectedLocus],
    *,
    method: str,
    default_window_bp: int,
) -> dict[str, ObservedHit]:
    contexts: dict[str, ObservedHit] = {}
    for expected in expected_loci:
        filtered = _filter_for_expected(ht, expected, default_window_bp)
        rows = filtered.order_by(filtered.p_value).take(1)
        if rows:
            contexts[expected.label] = _row_to_hit(rows[0], method)
    return contexts


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as out:
        json.dump(payload, out, indent=2, sort_keys=True)
        out.write("\n")


def _write_text(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8")


def _render_summary_text(report: dict[str, Any]) -> str:
    analyzed_samples = report["post_filter_stats"].get(
        "samples_with_complete_model_data",
        report["post_filter_stats"]["sample_count"],
    )
    gs_preprocessing = report.get("gs_preprocessing", {})
    gs_matrix_stats = report.get("gs_matrix_stats", report["post_filter_stats"])
    lines = [
        f"Trait type: {report['trait_type']}",
        f"Smoke mode: {report['config']['smoke_test']}",
        f"Samples analyzed: {analyzed_samples}",
        f"GS significant hits reported: {report['gs']['significant_hit_count']}",
        f"Baseline significant hits reported: {report['baseline']['significant_hit_count']}",
        f"Total wall time (s): {report['timings']['total_wall_time_s']:.3f}",
    ]
    if gs_preprocessing.get("ld_pruned"):
        lines.insert(3, f"Baseline variants analyzed: {report['post_filter_stats']['variant_count']}")
        lines.insert(4, f"GS matrix variants analyzed: {gs_matrix_stats['variant_count']}")
        lines.insert(5, f"GS LD-prune dropped variants: {gs_preprocessing['dropped_variant_count']}")
    else:
        lines.insert(3, f"Variants analyzed: {report['post_filter_stats']['variant_count']}")
    comparison = report.get("comparison")
    if comparison:
        summary = comparison["summary"]
        lines.extend(
            [
                f"Expected loci matched: {summary['matched_expected_count']}/{summary['expected_locus_count']}",
                f"Missing expected loci: {summary['missing_expected_count']}",
                f"Extra loci not in truth: {summary['extra_not_in_truth_count']}",
            ]
        )
    return "\n".join(lines) + "\n"


def run_graphlet_gwas_suite(config: GWASGSConfig) -> dict[str, Any]:
    output_dir = _ensure_output_dir(config.output_dir)
    timer = StageTimer()
    start_time = time.perf_counter()

    if config.init_hail:
        _ensure_hail_initialized(config)

    with timer("import"):
        mt, import_metadata = _prepare_plink_matrix_table(config)

    with timer("phenotype_and_covariates"):
        mt, _, baseline_y_for_summary, covariates_for_summary, trait_type = _prepare_trait_and_covariates(
            mt,
            config,
            import_metadata["trait_type"],
        )

    with timer("pre_filter_summary"):
        pre_filter_stats = _summarize_matrix_table(
            mt,
            baseline_y_for_summary,
            trait_type,
            covariates_for_summary,
        )

    with timer("filtering"):
        mt, filter_report = _apply_filters(mt, config)

    with timer("post_filter_trait_binding"):
        mt, gs_y, baseline_y, covariates, trait_type = _prepare_trait_and_covariates(
            mt,
            config,
            trait_type,
        )

    with timer("post_filter_summary"):
        post_filter_stats = _summarize_matrix_table(mt, baseline_y, trait_type, covariates)

    gs_mt = mt
    gs_covariates = covariates
    gs_preprocessing = {
        "ld_pruned": False,
        "pre_prune_variant_count": post_filter_stats["variant_count"],
        "post_prune_variant_count": post_filter_stats["variant_count"],
        "dropped_variant_count": 0,
        "pre_prune_sample_count": post_filter_stats["sample_count"],
        "post_prune_sample_count": post_filter_stats["sample_count"],
        "dropped_sample_count": 0,
    }
    gs_matrix_stats = post_filter_stats
    if config.gs_ld_prune:
        with timer("gs_matrix_preparation"):
            gs_mt, gs_preprocessing = _prepare_gs_matrix(mt, config)
            gs_mt, gs_y, gs_baseline_y, gs_covariates, gs_trait_type = _prepare_trait_and_covariates(
                gs_mt,
                config,
                trait_type,
            )
            if gs_trait_type != trait_type:
                raise ValueError("GS LD-pruned matrix changed inferred trait type unexpectedly.")
        with timer("gs_matrix_summary"):
            gs_matrix_stats = _summarize_matrix_table(gs_mt, gs_baseline_y, trait_type, gs_covariates)

    baseline_row_metadata = _collect_row_metadata(mt)
    gs_row_metadata = _collect_row_metadata(gs_mt)
    baseline_x_expr = mt.GT.n_alt_alleles()
    gs_x_expr = gs_mt.GT.n_alt_alleles()

    with timer("graphlet_screening"):
        gs_ht = hl.graphlet_screening(
            gs_mt,
            y=gs_y,
            x=gs_x_expr,
            covariates=gs_covariates,
            family=config.gs_family,
            mode=config.gs_mode,
            selection_aware=config.gs_selection_aware,
            split_fraction=config.gs_split_fraction,
            seed=config.gs_seed,
            nm=config.gs_nm,
            r=config.gs_r,
            block_size=config.gs_block_size,
            block_overlap=config.gs_block_overlap,
            max_cluster_size=config.gs_max_cluster_size,
            sparsity_level=config.gs_sparsity_level,
            max_iterations=config.gs_max_iterations,
            tolerance=config.gs_tolerance,
            pass_through=[field for field in ("rsid", "cm_position") if field in gs_mt.row_value],
        )
        gs_ht = _annotate_row_metadata(gs_ht, gs_row_metadata).persist()
        gs_row_count = gs_ht.count()

    with timer("baseline_gwas"):
        if trait_type == "binary":
            baseline_ht = hl.logistic_regression_rows(
                test="wald",
                y=baseline_y,
                x=baseline_x_expr,
                covariates=covariates,
            )
            baseline_method = "logistic_regression_rows"
        else:
            baseline_ht = hl.linear_regression_rows(
                y=baseline_y,
                x=baseline_x_expr,
                covariates=covariates,
            )
            baseline_method = "linear_regression_rows"
        baseline_ht = _annotate_row_metadata(baseline_ht, baseline_row_metadata).persist()
        baseline_row_count = baseline_ht.count()

    with timer("collect_significant_hits"):
        gs_hits = _collect_significant_hits(
            gs_ht,
            method="graphlet_screening",
            threshold=config.significance_threshold,
            limit=config.max_hits_to_report,
        )
        baseline_hits = _collect_significant_hits(
            baseline_ht,
            method=baseline_method,
            threshold=config.significance_threshold,
            limit=config.max_hits_to_report,
        )

    comparison_report = None
    if config.expected_loci_path:
        expected_loci = load_expected_loci(config.expected_loci_path)
        with timer("expected_loci_comparison"):
            gs_context = _collect_best_expected_contexts(
                gs_ht,
                expected_loci,
                method="graphlet_screening",
                default_window_bp=config.default_locus_window_bp,
            )
            baseline_context = _collect_best_expected_contexts(
                baseline_ht,
                expected_loci,
                method=baseline_method,
                default_window_bp=config.default_locus_window_bp,
            )
            comparison_report = summarize_expected_matches(
                expected_loci,
                gs_hits,
                baseline_hits,
                default_window_bp=config.default_locus_window_bp,
                gs_context=gs_context,
                baseline_context=baseline_context,
            )
            write_comparison_report(comparison_report, output_dir / "comparison_report.json")
            if config.fail_on_missing_expected and not comparison_report["summary"]["passes_expected_loci_check"]:
                missing_labels = [
                    record["expected"]["label"] for record in comparison_report["missing_expected"]
                ]
                raise AssertionError(f"Graphlet Screening missed expected loci: {missing_labels}")

    if config.export_results:
        with timer("export_results"):
            _export_result_table(gs_ht, output_dir / "graphlet_screening_results.tsv")
            _export_result_table(baseline_ht, output_dir / "baseline_gwas_results.tsv")

    total_wall_time = time.perf_counter() - start_time
    report = {
        "config": asdict(config),
        "trait_type": trait_type,
        "import_metadata": import_metadata,
        "pre_filter_stats": pre_filter_stats,
        "filter_report": filter_report,
        "post_filter_stats": post_filter_stats,
        "gs_preprocessing": gs_preprocessing,
        "gs_matrix_stats": gs_matrix_stats,
        "gs": {
            "row_count": gs_row_count,
            "method": "graphlet_screening",
            "significant_hit_count": len(gs_hits),
            "significant_hits": [asdict(hit) for hit in gs_hits],
        },
        "baseline": {
            "row_count": baseline_row_count,
            "method": baseline_method,
            "significant_hit_count": len(baseline_hits),
            "significant_hits": [asdict(hit) for hit in baseline_hits],
        },
        "comparison": comparison_report,
        "timings": {
            **{f"{stage}_s": duration for stage, duration in timer.timings.items()},
            "total_wall_time_s": total_wall_time,
        },
    }

    _write_json(output_dir / "analysis_report.json", report)
    _write_text(output_dir / "analysis_summary.txt", _render_summary_text(report))
    return report


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a Graphlet Screening GWAS analysis on PLINK inputs.")
    parser.add_argument("--bed", required=True)
    parser.add_argument("--bim", required=True)
    parser.add_argument("--fam", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--config-json")
    parser.add_argument("--reference-genome", default="GRCh38")
    parser.add_argument("--phenotype-source", choices=["fam", "external"], default="fam")
    parser.add_argument("--trait-type", choices=["auto", "quantitative", "binary"], default="auto")
    parser.add_argument("--phenotype-table-path")
    parser.add_argument("--phenotype-column")
    parser.add_argument("--phenotype-sample-column", default="s")
    parser.add_argument("--covariate-table-path")
    parser.add_argument("--covariate-sample-column", default="s")
    parser.add_argument("--covariate-columns", nargs="*", default=[])
    parser.add_argument("--expected-loci-path")
    parser.add_argument("--significance-threshold", type=float, default=5e-8)
    parser.add_argument("--default-locus-window-bp", type=int, default=DEFAULT_LOCUS_WINDOW_BP)
    parser.add_argument("--spark-driver-memory")
    parser.add_argument("--spark-executor-memory")
    parser.add_argument("--min-maf", type=float, default=0.01)
    parser.add_argument("--min-row-call-rate", type=float)
    parser.add_argument("--min-col-call-rate", type=float)
    parser.add_argument("--gs-block-size", type=int, default=32)
    parser.add_argument("--gs-block-overlap", type=int, default=32)
    parser.add_argument("--gs-max-cluster-size", type=int, default=20)
    parser.add_argument("--gs-sparsity-level", type=float)
    parser.add_argument("--gs-family", choices=["auto", "linear", "logistic"], default="auto")
    parser.add_argument("--gs-mode", choices=["global", "block"], default="global")
    parser.add_argument("--gs-selection-aware", dest="gs_selection_aware", action="store_true")
    parser.add_argument("--no-gs-selection-aware", dest="gs_selection_aware", action="store_false")
    parser.set_defaults(gs_selection_aware=True)
    parser.add_argument("--gs-split-fraction", type=float, default=0.5)
    parser.add_argument("--gs-seed", type=int, default=1)
    parser.add_argument("--gs-max-iterations", type=int, default=25)
    parser.add_argument("--gs-tolerance", type=float, default=1e-6)
    parser.add_argument("--gs-ld-prune", action="store_true")
    parser.add_argument("--gs-ld-prune-r2", type=float, default=0.2)
    parser.add_argument("--gs-ld-prune-window-bp", type=int, default=500_000)
    parser.add_argument("--gs-ld-prune-memory-per-core", type=int, default=256)
    parser.add_argument("--no-gs-ld-prune-keep-higher-maf", dest="gs_ld_prune_keep_higher_maf", action="store_false")
    parser.set_defaults(gs_ld_prune_keep_higher_maf=True)
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--smoke-row-hash-modulus", type=int, default=32)
    parser.add_argument("--smoke-col-hash-modulus", type=int, default=128)
    parser.add_argument("--max-hits-to-report", type=int, default=200)
    parser.add_argument("--no-export-results", action="store_true")
    return parser.parse_args()


def _config_from_args(args: argparse.Namespace) -> GWASGSConfig:
    if args.config_json:
        config = GWASGSConfig.from_json_file(args.config_json)
        config.output_dir = args.output_dir or config.output_dir
        return config
    return GWASGSConfig(
        bed=args.bed,
        bim=args.bim,
        fam=args.fam,
        output_dir=args.output_dir,
        reference_genome=args.reference_genome,
        phenotype_source=args.phenotype_source,
        trait_type=args.trait_type,
        phenotype_table_path=args.phenotype_table_path,
        phenotype_column=args.phenotype_column,
        phenotype_sample_column=args.phenotype_sample_column,
        covariate_table_path=args.covariate_table_path,
        covariate_sample_column=args.covariate_sample_column,
        covariate_columns=list(args.covariate_columns),
        expected_loci_path=args.expected_loci_path,
        significance_threshold=args.significance_threshold,
        default_locus_window_bp=args.default_locus_window_bp,
        spark_driver_memory=args.spark_driver_memory,
        spark_executor_memory=args.spark_executor_memory,
        min_maf=args.min_maf,
        min_row_call_rate=args.min_row_call_rate,
        min_col_call_rate=args.min_col_call_rate,
        gs_block_size=args.gs_block_size,
        gs_block_overlap=args.gs_block_overlap,
        gs_max_cluster_size=args.gs_max_cluster_size,
        gs_sparsity_level=args.gs_sparsity_level,
        gs_family=args.gs_family,
        gs_mode=args.gs_mode,
        gs_selection_aware=args.gs_selection_aware,
        gs_split_fraction=args.gs_split_fraction,
        gs_seed=args.gs_seed,
        gs_max_iterations=args.gs_max_iterations,
        gs_tolerance=args.gs_tolerance,
        gs_ld_prune=args.gs_ld_prune,
        gs_ld_prune_r2=args.gs_ld_prune_r2,
        gs_ld_prune_window_bp=args.gs_ld_prune_window_bp,
        gs_ld_prune_memory_per_core=args.gs_ld_prune_memory_per_core,
        gs_ld_prune_keep_higher_maf=args.gs_ld_prune_keep_higher_maf,
        smoke_test=args.smoke_test,
        smoke_row_hash_modulus=args.smoke_row_hash_modulus,
        smoke_col_hash_modulus=args.smoke_col_hash_modulus,
        max_hits_to_report=args.max_hits_to_report,
        export_results=not args.no_export_results,
    )


def main() -> None:
    args = _parse_args()
    report = run_graphlet_gwas_suite(_config_from_args(args))
    print(_render_summary_text(report), end="")


if __name__ == "__main__":
    main()
