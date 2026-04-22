# Note: no longer using this file

import math
import hail as hl
import pandas as pd
import numpy as np
from pprint import pprint

hl.init(local='local[*]', quiet=True)

hl.utils.get_1kg('data/')
mt = hl.read_matrix_table('data/1kg.mt')
table = hl.import_table('data/1kg_annotations.txt', impute=True).key_by('Sample')
mt = mt.annotate_cols(pheno=table[mt.s])
pprint(table.aggregate(hl.agg.stats(table.CaffeineConsumption)))
mt.aggregate_cols(hl.agg.counter(mt.pheno.CaffeineConsumption))
mt = hl.sample_qc(mt)
mt = mt.filter_cols((mt.sample_qc.dp_stats.mean >= 4) & (mt.sample_qc.call_rate >= 0.97))

ab = mt.AD[1] / hl.sum(mt.AD)

filter_condition_ab = (
    (mt.GT.is_hom_ref() & (ab <= 0.1))
    | (mt.GT.is_het() & (ab >= 0.25) & (ab <= 0.75))
    | (mt.GT.is_hom_var() & (ab >= 0.9))
)

fraction_filtered = mt.aggregate_entries(hl.agg.fraction(~filter_condition_ab))
print(f'Filtering {fraction_filtered * 100:.2f}% entries out of downstream analysis.')
mt = mt.filter_entries(filter_condition_ab)

mt = hl.variant_qc(mt)

mt = mt.filter_rows(mt.variant_qc.AF[1] > 0.01)

mt = mt.filter_rows(mt.variant_qc.p_value_hwe > 1e-6)

print('Samples: %d  Variants: %d' % (mt.count_cols(), mt.count_rows()))

# eigenvalues, pcs, _ = hl.hwe_normalized_pca(mt.GT)
# This appears to be failing because of a breeze dependency mismatch for now -- will fix later

# mt = mt.annotate_cols(scores=pcs[mt.s].scores)

output = hl.methods.graphlet_screening(
    mt,
    y=mt.pheno.CaffeineConsumption,
    x=mt.GT.n_alt_alleles(),
    covariates=[1.0, mt.pheno.isFemale],  # Include intercept
    nm=3,
    r=3.5,  # Signal strength parameter
    block_size=32,
    max_cluster_size=7,
    pass_through=[])  # No additional row fields to pass through

p = hl.plot.manhattan(output.p_value)
hl.plot.show(p)

# print(output.show())

# gwas = hl.linear_regression_rows(
#     y=mt.pheno.CaffeineConsumption,
#     x=mt.GT.n_alt_alleles(),
#     covariates=[1.0, mt.pheno.isFemale],
# )

# p = hl.plot.manhattan(gwas.p_value)
# hl.plot.show(p)