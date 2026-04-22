from __future__ import annotations

from pathlib import Path

import hail as hl
import pytest

try:
    from testing.graphlet_screening_compare import ObservedHit, load_expected_loci, summarize_expected_matches
    from testing.graphlet_screening_gwas_runner import GWASGSConfig, _load_fam_summary, run_graphlet_gwas_suite
except ModuleNotFoundError:
    from graphlet_screening_compare import ObservedHit, load_expected_loci, summarize_expected_matches
    from graphlet_screening_gwas_runner import GWASGSConfig, _load_fam_summary, run_graphlet_gwas_suite


@pytest.fixture(scope="module", autouse=True)
def hail_session():
    hl.init(master="local[2]", min_block_size=0, quiet=True, global_seed=0)
    yield
    hl.stop()


def test_load_fam_summary_infers_binary_trait(tmp_path: Path):
    fam_path = tmp_path / "mini.fam"
    fam_path.write_text(
        "\n".join(
            [
                "fam1 sample1 0 0 1 1",
                "fam1 sample2 0 0 2 2",
                "fam1 sample3 0 0 2 -9",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    summary = _load_fam_summary(fam_path, "NA")

    assert summary["sample_count"] == 3
    assert summary["duplicate_sample_ids"] == []
    assert summary["inferred_trait_type"] == "binary"
    assert summary["inferred_quant_pheno"] is False


def test_expected_locus_comparison_flags_missing_and_extra(tmp_path: Path):
    expected_path = tmp_path / "expected.tsv"
    expected_path.write_text(
        "\n".join(
            [
                "label\tmatch_type\tcontig\tposition\twindow_bp\trsid\talleles\tsource\tnotes",
                "KnownWindow\twindow\tchr21\t100000\t250\t\t\tpaper_a\twindow match",
                "KnownExact\texact\tchr21\t200000\t0\trsExact\tA/G\tpaper_b\texact match",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    expected_loci = load_expected_loci(expected_path)
    gs_hits = [
        ObservedHit(
            locus_label="chr21:100050",
            contig="chr21",
            position=100050,
            p_value=1e-10,
            method="graphlet_screening",
            rsid="rsWindowHit",
            alleles=("A", "T"),
        ),
        ObservedHit(
            locus_label="chr21:900000",
            contig="chr21",
            position=900000,
            p_value=2e-9,
            method="graphlet_screening",
            rsid="rsNovel",
            alleles=("C", "T"),
        ),
    ]
    baseline_hits = [
        ObservedHit(
            locus_label="chr21:900100",
            contig="chr21",
            position=900100,
            p_value=4e-8,
            method="linear_regression_rows",
            rsid="rsBaselineSupport",
            alleles=("C", "T"),
        )
    ]

    report = summarize_expected_matches(expected_loci, gs_hits, baseline_hits, default_window_bp=250)

    assert report["summary"]["matched_expected_count"] == 1
    assert report["summary"]["missing_expected_count"] == 1
    assert report["summary"]["extra_known_or_plausible_count"] == 1
    assert report["summary"]["extra_not_in_truth_count"] == 0
    assert report["missing_expected"][0]["expected"]["label"] == "KnownExact"


def test_run_graphlet_gwas_suite_smoke_on_local_plink_fixture(tmp_path: Path):
    fixture_prefix = Path("/home/nvim/hail_wulab/testing/ALS_GWAS_chr21_hg38_rename_first800")
    output_dir = tmp_path / "als_gs_smoke"
    config = GWASGSConfig(
        bed=str(fixture_prefix.with_suffix(".bed")),
        bim=str(fixture_prefix.with_suffix(".bim")),
        fam=str(fixture_prefix.with_suffix(".fam")),
        output_dir=str(output_dir),
        reference_genome="GRCh38",
        phenotype_source="fam",
        trait_type="auto",
        smoke_test=True,
        smoke_row_hash_modulus=32,
        smoke_col_hash_modulus=128,
        min_maf=0.01,
        max_hits_to_report=25,
        export_results=False,
        init_hail=False,
    )

    report = run_graphlet_gwas_suite(config)

    assert report["trait_type"] == "binary"
    assert report["config"]["smoke_test"] is True
    assert report["pre_filter_stats"]["sample_count"] > 0
    assert report["pre_filter_stats"]["variant_count"] > 0
    assert report["post_filter_stats"]["sample_count"] > 0
    assert report["post_filter_stats"]["variant_count"] > 0
    assert report["gs"]["row_count"] == report["post_filter_stats"]["variant_count"]
    assert report["baseline"]["method"] == "logistic_regression_rows"
    assert (output_dir / "analysis_report.json").exists()
    assert (output_dir / "analysis_summary.txt").exists()
