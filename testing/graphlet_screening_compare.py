from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable


DEFAULT_LOCUS_WINDOW_BP = 250_000


@dataclass(frozen=True)
class ExpectedLocus:
    label: str
    match_type: str = "window"
    contig: str | None = None
    position: int | None = None
    window_bp: int | None = None
    rsid: str | None = None
    alleles: tuple[str, ...] = ()
    source: str | None = None
    notes: str | None = None

    def resolved_window_bp(self, default_window_bp: int = DEFAULT_LOCUS_WINDOW_BP) -> int:
        return self.window_bp if self.window_bp is not None else default_window_bp


@dataclass(frozen=True)
class ObservedHit:
    locus_label: str
    contig: str
    position: int
    p_value: float
    method: str
    rsid: str | None = None
    alleles: tuple[str, ...] = ()
    beta: float | None = None
    n: int | None = None


@dataclass
class LocusMatch:
    expected: ExpectedLocus
    gs_hit: ObservedHit | None
    baseline_hit: ObservedHit | None
    status: str
    gs_best_hit: ObservedHit | None = None
    baseline_best_hit: ObservedHit | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "expected": asdict(self.expected),
            "gs_hit": asdict(self.gs_hit) if self.gs_hit else None,
            "baseline_hit": asdict(self.baseline_hit) if self.baseline_hit else None,
            "status": self.status,
            "gs_best_hit": asdict(self.gs_best_hit) if self.gs_best_hit else None,
            "baseline_best_hit": asdict(self.baseline_best_hit) if self.baseline_best_hit else None,
        }


def normalize_contig(contig: str | None) -> str | None:
    if contig is None:
        return None
    contig = str(contig).strip()
    if not contig:
        return None
    if contig.lower().startswith("chr"):
        return f"chr{contig[3:]}"
    return contig


def _split_alleles(raw_value: str | None) -> tuple[str, ...]:
    if raw_value is None:
        return ()
    stripped = raw_value.strip()
    if not stripped:
        return ()
    delimiter = "/" if "/" in stripped else ","
    return tuple(part.strip() for part in stripped.split(delimiter) if part.strip())


def _infer_match_type(row: dict[str, str]) -> str:
    explicit = row.get("match_type", "").strip().lower()
    if explicit:
        return explicit
    if row.get("rsid"):
        return "exact"
    if row.get("alleles") and row.get("position"):
        return "exact"
    return "window"


def load_expected_loci(path: str | Path) -> list[ExpectedLocus]:
    file_path = Path(path)
    delimiter = "\t" if file_path.suffix.lower() in {".tsv", ".bgz"} else ","
    expected_loci: list[ExpectedLocus] = []
    with file_path.open(newline="", encoding="utf-8") as input_file:
        reader = csv.DictReader(input_file, delimiter=delimiter)
        for row in reader:
            expected_loci.append(
                ExpectedLocus(
                    label=row.get("label") or row.get("rsid") or f"{row.get('contig')}:{row.get('position')}",
                    match_type=_infer_match_type(row),
                    contig=normalize_contig(row.get("contig")),
                    position=int(row["position"]) if row.get("position") else None,
                    window_bp=int(row["window_bp"]) if row.get("window_bp") else None,
                    rsid=row.get("rsid") or None,
                    alleles=_split_alleles(row.get("alleles")),
                    source=row.get("source") or None,
                    notes=row.get("notes") or None,
                )
            )
    return expected_loci


def hit_matches_expected(hit: ObservedHit, expected: ExpectedLocus, default_window_bp: int = DEFAULT_LOCUS_WINDOW_BP) -> bool:
    if expected.match_type == "exact":
        if expected.rsid and hit.rsid:
            return hit.rsid == expected.rsid
        if expected.contig is None or expected.position is None:
            return False
        if normalize_contig(hit.contig) != normalize_contig(expected.contig):
            return False
        if hit.position != expected.position:
            return False
        if expected.alleles:
            return tuple(hit.alleles) == expected.alleles
        return True

    if expected.contig is None or expected.position is None:
        return False
    if normalize_contig(hit.contig) != normalize_contig(expected.contig):
        return False
    return abs(hit.position - expected.position) <= expected.resolved_window_bp(default_window_bp)


def pick_best_hit(hits: Iterable[ObservedHit], expected: ExpectedLocus, default_window_bp: int = DEFAULT_LOCUS_WINDOW_BP) -> ObservedHit | None:
    matched_hits = [hit for hit in hits if hit_matches_expected(hit, expected, default_window_bp)]
    if not matched_hits:
        return None
    return min(matched_hits, key=lambda hit: hit.p_value)


def summarize_expected_matches(
    expected_loci: list[ExpectedLocus],
    gs_hits: list[ObservedHit],
    baseline_hits: list[ObservedHit],
    *,
    default_window_bp: int = DEFAULT_LOCUS_WINDOW_BP,
    gs_context: dict[str, ObservedHit] | None = None,
    baseline_context: dict[str, ObservedHit] | None = None,
) -> dict[str, Any]:
    gs_context = gs_context or {}
    baseline_context = baseline_context or {}
    matched: list[LocusMatch] = []
    missing: list[LocusMatch] = []

    for expected in expected_loci:
        gs_hit = pick_best_hit(gs_hits, expected, default_window_bp)
        baseline_hit = pick_best_hit(baseline_hits, expected, default_window_bp)
        record = LocusMatch(
            expected=expected,
            gs_hit=gs_hit,
            baseline_hit=baseline_hit,
            status="matched_expected" if gs_hit else "missing_expected",
            gs_best_hit=gs_context.get(expected.label),
            baseline_best_hit=baseline_context.get(expected.label),
        )
        if gs_hit:
            matched.append(record)
        else:
            missing.append(record)

    extra_known_or_plausible: list[dict[str, Any]] = []
    extra_not_in_truth: list[dict[str, Any]] = []
    for hit in gs_hits:
        if any(hit_matches_expected(hit, expected, default_window_bp) for expected in expected_loci):
            continue
        hit_dict = asdict(hit)
        if any(
            normalize_contig(hit.contig) == normalize_contig(baseline_hit.contig)
            and abs(hit.position - baseline_hit.position) <= default_window_bp
            for baseline_hit in baseline_hits
        ):
            extra_known_or_plausible.append(hit_dict)
        else:
            extra_not_in_truth.append(hit_dict)

    return {
        "summary": {
            "expected_locus_count": len(expected_loci),
            "matched_expected_count": len(matched),
            "missing_expected_count": len(missing),
            "extra_known_or_plausible_count": len(extra_known_or_plausible),
            "extra_not_in_truth_count": len(extra_not_in_truth),
            "passes_expected_loci_check": not missing,
        },
        "matched_expected": [record.to_dict() for record in matched],
        "missing_expected": [record.to_dict() for record in missing],
        "extra_known_or_plausible": extra_known_or_plausible,
        "extra_not_in_truth": extra_not_in_truth,
    }


def write_comparison_report(report: dict[str, Any], output_path: str | Path) -> None:
    with Path(output_path).open("w", encoding="utf-8") as out:
        json.dump(report, out, indent=2, sort_keys=True)
        out.write("\n")
