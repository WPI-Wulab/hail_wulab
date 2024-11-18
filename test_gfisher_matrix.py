import hail as hl

hl.init(log='test.log', append=False)

hl.reset_global_randomness()
mt = hl.balding_nichols_model(1, n_samples=20, n_variants=50)
mt = mt.annotate_rows(gene=mt.locus.position % 12)
# mt = mt.annotate_rows(gene = mt.locus.position // 4)
mt = mt.annotate_entries(X=mt.GT.n_alt_alleles())
mt = mt.annotate_rows(weight=1.0)
mt = mt.annotate_cols(phenotype=hl.agg.sum(mt.GT.n_alt_alleles()) - 20 + hl.rand_norm(0, 1))
mt = mt.select_rows(mt.gene, mt.weight)
mt = mt.select_cols(mt.phenotype)
mt = mt.select_globals()
# mt.describe()

mt = mt.add_row_index()
# mt.describe()

pval_ht = hl.linear_regression_rows(y=mt.phenotype, x=mt.X, covariates=[1.0])
# pval_ht.show()

row_cor = hl.row_correlation(mt.X)

rce = row_cor.to_table_row_major()
rce = rce.annotate(entries=hl.nd.array(rce.entries))
# rce.show()

mt = mt.annotate_rows(row_cor=rce[mt.row_idx].entries, pval=pval_ht[mt.row_key].p_value)
# mt.rows().show()

mt = mt.annotate_rows(df=hl.literal(list(range(1, 1 + mt.count_rows())))[hl.int32(mt.row_idx)])
# mt.rows().show()

hl.gfisher_thing(mt.gene, mt.pval, mt.df, mt.weight, mt.row_cor, mt.row_idx)
