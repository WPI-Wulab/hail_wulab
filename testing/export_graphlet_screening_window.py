from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import hail as hl
from hail.linalg import BlockMatrix

try:
    from testing.graphlet_screening_gwas_runner import (
        GWASGSConfig,
        _apply_filters,
        _ensure_hail_initialized,
        _prepare_plink_matrix_table,
        _prepare_trait_and_covariates,
    )
except ModuleNotFoundError:
    from graphlet_screening_gwas_runner import (
        GWASGSConfig,
        _apply_filters,
        _ensure_hail_initialized,
        _prepare_plink_matrix_table,
        _prepare_trait_and_covariates,
    )


def _variant_label(row) -> str:
    alleles = "_".join(row.alleles)
    return f"{row.locus.contig}:{row.locus.position}:{alleles}"


def _window_bounds(row_count: int, block_size: int, block_overlap: int, block_index: int) -> dict[str, int]:
    core_start = block_index * block_size
    if core_start >= row_count:
        raise ValueError(
            f"Requested block_index={block_index} starts at row {core_start}, but only {row_count} filtered rows exist."
        )

    core_end = min(row_count, core_start + block_size)
    window_start = max(0, core_start - block_overlap)
    window_end = min(row_count, core_end + block_overlap)
    return {
        "core_start": core_start,
        "core_end": core_end,
        "window_start": window_start,
        "window_end": window_end,
    }


def _write_x_csv(path: Path, matrix: np.ndarray, sample_ids: list[str], variant_labels: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as out:
        writer = csv.writer(out)
        writer.writerow(["sample_id", *variant_labels])
        for sample_id, values in zip(sample_ids, matrix, strict=True):
            writer.writerow([sample_id, *values.tolist()])


def _write_y_csv(path: Path, sample_rows: list) -> None:
    n_covariates = len(sample_rows[0].covariates) if sample_rows else 0
    with path.open("w", encoding="utf-8", newline="") as out:
        writer = csv.writer(out)
        writer.writerow(["sample_id", "y_gs", "y_baseline", *[f"covariate_{i}" for i in range(n_covariates)]])
        for row in sample_rows:
            writer.writerow([row.sample_id, row.y_gs, row.y_baseline, *row.covariates])


def _write_variant_csv(path: Path, variant_rows: list, bounds: dict[str, int]) -> None:
    with path.open("w", encoding="utf-8", newline="") as out:
        writer = csv.writer(out)
        writer.writerow(
            [
                "window_col_idx",
                "filtered_row_idx",
                "role",
                "contig",
                "position",
                "rsid",
                "alleles",
                "variant_label",
            ]
        )
        for window_col_idx, row in enumerate(variant_rows):
            if row.filtered_row_idx < bounds["core_start"]:
                role = "left_overlap"
            elif row.filtered_row_idx >= bounds["core_end"]:
                role = "right_overlap"
            else:
                role = "core"
            writer.writerow(
                [
                    window_col_idx,
                    row.filtered_row_idx,
                    role,
                    row.locus.contig,
                    row.locus.position,
                    row.rsid,
                    "/".join(row.alleles),
                    _variant_label(row),
                ]
            )


def export_window(config: GWASGSConfig, block_index: int) -> dict[str, object]:
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    _ensure_hail_initialized(config)

    mt, import_metadata = _prepare_plink_matrix_table(config)
    mt, _, _, _, trait_type = _prepare_trait_and_covariates(mt, config, import_metadata["trait_type"])
    mt, _ = _apply_filters(mt, config)
    mt, gs_y, baseline_y, covariates, trait_type = _prepare_trait_and_covariates(mt, config, trait_type)

    complete_case = hl.is_defined(gs_y)
    for covariate in covariates:
        complete_case = complete_case & hl.is_defined(covariate)

    mt = mt.annotate_cols(
        _y_gs=gs_y,
        _y_baseline=hl.float64(baseline_y),
        _covariates=hl.array(covariates),
        _complete_case=complete_case,
    )
    mt = mt.filter_cols(mt._complete_case)
    mt = mt.key_cols_by()
    mt = mt.add_row_index("_row_idx").add_col_index("_col_idx")

    filtered_row_count = mt.count_rows()
    filtered_col_count = mt.count_cols()
    bounds = _window_bounds(filtered_row_count, config.gs_block_size, config.gs_block_overlap, block_index)

    mt = mt.filter_rows((mt._row_idx >= bounds["window_start"]) & (mt._row_idx < bounds["window_end"]))
    mt = mt.annotate_rows(_mean_dosage=hl.or_else(hl.agg.mean(hl.float64(mt.GT.n_alt_alleles())), 0.0))
    mt = mt.select_entries(_dosage=hl.or_else(hl.float64(mt.GT.n_alt_alleles()), mt._mean_dosage))

    col_ht = mt.cols()
    sample_rows = (
        col_ht
        .select(
            sample_id=col_ht.s,
            filtered_col_idx=col_ht._col_idx,
            y_gs=col_ht._y_gs,
            y_baseline=col_ht._y_baseline,
            covariates=col_ht._covariates,
        )
        .order_by("filtered_col_idx")
        .collect()
    )
    row_ht = mt.rows().key_by()
    variant_rows = (
        row_ht
        .select(
            filtered_row_idx=row_ht._row_idx,
            locus=row_ht.locus,
            alleles=row_ht.alleles,
            rsid=row_ht.rsid if "rsid" in row_ht.row_value else hl.missing(hl.tstr),
        )
        .order_by("filtered_row_idx")
        .collect()
    )

    bm = BlockMatrix.from_entry_expr(mt._dosage)
    x_variant_by_sample = bm.to_numpy()
    x_sample_by_variant = x_variant_by_sample.T

    sample_ids = [row.sample_id for row in sample_rows]
    variant_labels = [_variant_label(row) for row in variant_rows]

    x_path = output_dir / "X.csv"
    y_path = output_dir / "Y.csv"
    variants_path = output_dir / "variants.csv"
    metadata_path = output_dir / "window_metadata.json"

    _write_x_csv(x_path, x_sample_by_variant, sample_ids, variant_labels)
    _write_y_csv(y_path, sample_rows)
    _write_variant_csv(variants_path, variant_rows, bounds)

    metadata = {
        "trait_type": trait_type,
        "filtered_row_count": filtered_row_count,
        "filtered_col_count": filtered_col_count,
        "block_index": block_index,
        "block_size": config.gs_block_size,
        "block_overlap": config.gs_block_overlap,
        **bounds,
        "window_variant_count": len(variant_rows),
        "window_sample_count": len(sample_rows),
        "x_csv": str(x_path),
        "y_csv": str(y_path),
        "variants_csv": str(variants_path),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return metadata


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export one Graphlet Screening ALS window to CSV.")
    parser.add_argument("--bed", required=True)
    parser.add_argument("--bim", required=True)
    parser.add_argument("--fam", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--block-index", type=int, default=1)
    parser.add_argument("--block-size", type=int, default=32)
    parser.add_argument("--block-overlap", type=int, default=8)
    parser.add_argument("--min-maf", type=float, default=0.01)
    parser.add_argument("--reference-genome", default="GRCh38")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    config = GWASGSConfig(
        bed=args.bed,
        bim=args.bim,
        fam=args.fam,
        output_dir=args.output_dir,
        phenotype_source="fam",
        trait_type="auto",
        reference_genome=args.reference_genome,
        min_maf=args.min_maf,
        gs_block_size=args.block_size,
        gs_block_overlap=args.block_overlap,
    )
    metadata = export_window(config, args.block_index)
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
