import math

import numpy as np

import hail as hl

from ..helpers import test_timeout


def _make_graphlet_test_mt(n_variants=6, n_samples=80):
    mt = hl.utils.range_matrix_table(n_variants, n_samples, n_partitions=2)
    mt = mt.key_rows_by(row_idx=mt.row_idx)
    mt = mt.annotate_entries(
        xval=hl.float64(
            hl.case()
            .when(mt.row_idx == 0, mt.col_idx % 3)
            .when(mt.row_idx == 1, hl.if_else(((mt.col_idx + 1) % 4) == 0, 2, 0))
            .when(mt.row_idx == 2, hl.if_else(((mt.col_idx + 2) % 5) == 0, 2, 0))
            .when(mt.row_idx == 3, hl.if_else(((mt.col_idx + 3) % 6) == 0, 2, 0))
            .default(hl.if_else(((mt.col_idx + mt.row_idx) % 7) == 0, 2, 0))
        )
    )
    x0 = hl.float64(mt.col_idx % 3)
    cov = hl.float64(mt.col_idx % 2)
    mt = mt.annotate_cols(
        covariate=cov,
        y_linear=4.0 * x0 + 0.5 * cov + 0.01 * hl.float64(mt.col_idx % 5),
        y_binary=hl.float64(((mt.col_idx * 11) % 7) < (1 + hl.int32(x0) + hl.int32(cov))),
    )
    return mt


@test_timeout(180)
def test_graphlet_screening_global_mode_block_invariant():
    mt = _make_graphlet_test_mt()

    ht_small = hl.graphlet_screening(
        mt,
        y=mt.y_linear,
        x=mt.xval,
        covariates=[1.0, mt.covariate],
        family='linear',
        mode='global',
        block_size=2,
        seed=13,
    )
    ht_large = hl.graphlet_screening(
        mt,
        y=mt.y_linear,
        x=mt.xval,
        covariates=[1.0, mt.covariate],
        family='linear',
        mode='global',
        block_size=5,
        seed=13,
    )

    rows_small = {row.row_idx: row for row in ht_small.select('selected', 'beta', 'p_value').collect()}
    rows_large = {row.row_idx: row for row in ht_large.select('selected', 'beta', 'p_value').collect()}

    assert rows_small[0].selected is True
    assert rows_large[0].selected is True
    assert np.isclose(rows_small[0].beta, rows_large[0].beta)
    assert np.isclose(rows_small[0].p_value, rows_large[0].p_value)
    assert all(rows_small[idx].selected == rows_large[idx].selected for idx in rows_small)


@test_timeout(180)
def test_graphlet_screening_block_overlap_emits_complete_results():
    mt = _make_graphlet_test_mt()

    ht_overlap = hl.graphlet_screening(
        mt,
        y=mt.y_linear,
        x=mt.xval,
        covariates=[1.0, mt.covariate],
        family='linear',
        mode='block',
        block_size=2,
        block_overlap=4,
        seed=23,
    )

    rows_overlap = {row.row_idx: row for row in ht_overlap.select('selected', 'beta', 'p_value').collect()}

    assert set(rows_overlap) == set(range(6))
    assert rows_overlap[0].selected is True
    assert np.isfinite(rows_overlap[0].beta)
    assert np.isfinite(rows_overlap[0].p_value)


@test_timeout(180)
def test_graphlet_screening_reports_selection_and_split_counts():
    mt = _make_graphlet_test_mt()

    ht = hl.graphlet_screening(
        mt,
        y=mt.y_linear,
        x=mt.xval,
        covariates=[1.0, mt.covariate],
        family='linear',
        mode='global',
        selection_aware=True,
        split_fraction=0.6,
        seed=7,
    )

    lead = ht.filter(ht.row_idx == 0).take(1)[0]
    assert lead.selected is True
    assert lead.n_total == 80
    assert lead.n == lead.n_total
    assert lead.n_train + lead.n_test == 80
    assert math.isfinite(lead.p_value)


@test_timeout(180)
def test_graphlet_screening_logistic_family_returns_finite_statistics():
    mt = _make_graphlet_test_mt()

    ht = hl.graphlet_screening(
        mt,
        y=mt.y_binary,
        x=mt.xval,
        covariates=[1.0, mt.covariate],
        family='logistic',
        mode='global',
        selection_aware=False,
        seed=19,
    )

    lead = ht.order_by(ht.p_value).take(1)[0]
    assert lead.row_idx == 0
    assert lead.selected is True
    assert lead.n_train == lead.n_total
    assert lead.n_test == lead.n_total
