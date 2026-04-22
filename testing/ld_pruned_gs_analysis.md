# LD-Pruned Graphlet Screening Analysis on `ALS_GWAS_chr21_hg38_rename_first800` report
## Rishit Avadhuta
### 4 / 22 / 26

This note is a validation memo for the local artifacts in `testing/out/codex_ldprune_full_export`. Every quantitative claim below was checked against the provided BIM/FAM inputs, `analysis_report.json`, the exported GS and baseline tables, and the runner implementation in `testing/graphlet_screening_gwas_runner.py`.

## Verified run configuration

The validated run was:

```bash
python3 testing/graphlet_screening_gwas_runner.py \
  --bed testing/ALS_GWAS_chr21_hg38_rename_first800.bed \
  --bim testing/ALS_GWAS_chr21_hg38_rename_first800.bim \
  --fam testing/ALS_GWAS_chr21_hg38_rename_first800.fam \
  --output-dir testing/out/codex_ldprune_full_export \
  --gs-mode block \
  --gs-block-size 32 \
  --gs-block-overlap 32 \
  --gs-ld-prune \
  --gs-ld-prune-r2 0.1 \
  --gs-ld-prune-window-bp 500000
```

1. LD Prune is included because of highly correlated clusters creating computationally impossible problems for PMLE. LD Prune is a method of eliminating highly correlated clusters by assigning them to a single SNP tag.
2. Block size is used to avoid memory issues which are still ongoing

Verified from `analysis_report.json` and the runner code:

- Baseline method: `logistic_regression_rows(test="wald")` on the full post-filter matrix.
- GS method: `hl.graphlet_screening(..., mode="block", selection_aware=True, split_fraction=0.5, family="auto")`.
- Trait type: binary.
- Covariates actually used: intercept plus sex only.
- Pre-filtering before both methods: `min_maf=0.01`.
- GS-only preprocessing: `hl.ld_prune(..., r2=0.1, bp_window_size=500000, keep_higher_maf=True)`.

That last point matters for interpretation: the retained LD-pruned variants are high-MAF proxy tags, not association-ranked lead SNPs. A GS-selected tag should therefore be read as a representative variant within the retained tag set, not automatically as the strongest or causal SNP in the locus.

## Panel facts that are directly verified

- The provided BIM contains exactly `800` variants, all on chromosome `21`.
- The BIM span is `21:13,464,068-18,350,513` in the input file. After import with `reference_genome="GRCh38"`, Hail labels these loci on `chr21`.
- The provided FAM contains `48,224` samples with phenotype counts:
  - `39,973` controls (`1`)
  - `8,225` cases (`2`)
  - `26` missing phenotypes (`-9`)
- After `min_maf >= 0.01`, the analyzed matrix contains `688` variants and `48,198` analyzable samples.
- GS-only LD pruning reduces the GS matrix from `688` to `211` variants, dropping `477` and retaining `30.67%` of the post-filter variants.

One wording correction is important here: this is not a genome-wide panel. The runner uses the conventional GWAS threshold `5e-8`, but phrases like "genome-wide significant" should be read as "passing the conventional GWAS threshold used by the runner", not as a claim that the panel itself is genome-wide.

The provided BIM has `800` chr21 variants over the span above.

## What the LD-pruned run actually produced

Directly from `analysis_report.json` and the exported tables:

- Baseline tested variants: `688`
- GS tested tag variants after pruning: `211`
- GS-selected tag variants: `9`
- GS rows with finite `p_value`: `9`
- GS rows with `NaN` `p_value`: `202`
- GS variants passing `5e-8`: `2`
- Baseline variants passing `5e-8`: `55`

Note that non selected variants by GS return NaN for p_value to maintain row matching in a cohesive manner.

The selected tags from the run by Graphlet Screening were:

- `chr21:14227527`
- `chr21:15148329`
- `chr21:15185976`
- `chr21:16021595`
- `chr21:16044073`
- `chr21:16845630`
- `chr21:17267800`
- `chr21:18286757`
- `chr21:18339886`

The two GS tags passing the p-value threshold of `5*10^-8` (standard GWAS threshold) are:

- `chr21:17267800` with GS `p = 3.2309e-15`
- `chr21:15185976` with GS `p = 4.9635e-09`

## Exact overlap with the baseline single-SNP scan

For each selected GS tag, I checked the exact same locus in `baseline_gwas_results.tsv`:

Formatted using AI: 
| Locus | GS p | Baseline p | Same beta sign |
| --- | ---: | ---: | :---: |
| `chr21:17267800` | 3.231e-15 | 9.954e-25 | yes |
| `chr21:15185976` | 4.964e-09 | 1.981e-11 | yes |
| `chr21:18339886` | 1.038e-06 | 1.612e-12 | yes |
| `chr21:14227527` | 0.001882 | 4.305e-07 | yes |
| `chr21:16845630` | 0.01488 | 4.477e-09 | yes |
| `chr21:16021595` | 0.04198 | 0.0002008 | yes |
| `chr21:16044073` | 0.05062 | 2.791e-05 | yes |
| `chr21:18286757` | 0.06199 | 0.00282 | yes |
| `chr21:15148329` | 0.08387 | 3.374e-06 | yes |

- all `9/9` selected GS tags have baseline `p < 0.05`
- `6/9` have baseline `p < 1e-5`
- `4/9` themselves pass the runner's `5e-8` threshold in baseline GWAS
- all `9/9` have the same sign in GS and baseline
- the top GS hit, `chr21:17267800`, is also the top baseline hit in this panel

That is strong internal consistency. It does not prove causal correctness, but it does argue that the LD-pruned GS run is not selecting arbitrary noise.

## What can and cannot be said about the unpruned setting

What is directly supported:

- Older unpruned artifacts in `testing/out/run1`, `testing/out/run2`, `testing/out/run3`, and `testing/out/codex_block_overlap_full8` report `0` GS hits.
- Those artifacts are not a clean like-for-like control for the validated LD-pruned run because settings changed across runs, including at least `block_overlap`, `max_cluster_size`, and the output schema itself.
- The implementation and the block-mode code path are consistent with dense local correlation being a real difficulty for GS: the original theory assumes a sparse Gram graph, and block mode is explicitly an approximation rather than a global exact solve.

TLDR:
- in this dataset, the unpruned setting has repeatedly been operationally problematic
- LD pruning materially changes the problem and is likely central to obtaining a non-degenerate GS solution
- **LD pruning appears to be a viable method for computationally avoiding incredibly dense clusters and is a reasonable method backed up by literature and is in consistency with other GWAS methods**

## Literature cross-check

This panel is not suitable for testing recovery of the best-known chr21 ALS loci cited in the literature.

Verified external references (concatenated and compiled by AI, human-reviewed):

- `SOD1` on GRCh38 is at `21:31,659,693-31,668,931`  
  NCBI Gene: <https://www.ncbi.nlm.nih.gov/gene/6647>

- `CFAP410` (`C21orf2`) on GRCh38 is at `21:44,328,944-44,339,402`  
  Ensembl: <https://www.ensembl.org/Homo_sapiens/Gene/Summary?g=ENSG00000160226>

- van Rheenen et al. 2016 reported that they fine-mapped a chromosome 21 ALS risk locus and identified `C21orf2`  
  Nature Genetics: <https://www.nature.com/articles/ng.3622>

- van Rheenen et al. 2021 identified `15` ALS risk loci and explicitly discussed both `CFAP410` and a low-frequency `SOD1` signal  
  Nature Genetics: <https://www.nature.com/articles/s41588-021-00973-1>

The current test panel ends at `18,350,513`, which is:

- `13,309,180` bp upstream of `SOD1`
- `25,978,431` bp upstream of `CFAP410`

So this dataset simply doesn't appear to contain those loci. Because of that, this run cannot be used as a direct replication check for the best-known chr21 ALS associations. 

## Scientific and implementation caveats

- The original GS theory paper is framed around a linear model with Gaussian noise and a sparse Gram graph, not around a binary trait with logistic refits:
  <https://www.jmlr.org/papers/v15/jin14a.html>
- This implementation extends GS to binary traits by using logistic-family inference with sample splitting. That is a reasonable engineering extension, but it should not be presented as a direct restatement of the 2014 theory.
- With `selection_aware=True` and `split_fraction=0.5`, the selected model is built on one half of the samples and the reported GS coefficients and p-values are refit on the held-out half. This is a major reason GS and baseline should not be expected to have the same hit count.
- The baseline and GS comparisons here use only sex as a non-intercept covariate. No ancestry PCs, batch terms, or richer GWAS covariates were included, so biological interpretation should remain conservative.
- Block mode is an approximation to a full global solve. Useful here, but not identical to a true global run. The global run was not used due to computational infeasibility, although this is currently being developed.

## Conclusions

Ultimately, the GS method is operational in the GWAS context but requires more evidence to state it as biologically useful given the current testing data. This is mainly because the primary genes associated with ALS are not included in the given data.

- Under the validated LD-pruned block-mode configuration, Graphlet Screening returns a sparse, non-degenerate set of `9` selected tag SNPs.
- Those selected tags show strong internal agreement with the baseline single-SNP logistic scan: all are nominally supported in baseline, all have matching effect directions, and the strongest exact overlap is `chr21:17267800`, which is the top hit in both analyses.
- That supports the claim that the implementation behaves coherently as a tag-level screening method on this panel after LD pruning.

CRUCIAL NOTES:
- this is NOT a biological validation of ALS chr21 loci
- this is not evidence that the selected tag SNPs are causal or lead SNPs
- this is not enough to claim that GS has been validated on raw GWAS matrices in general

The correct reading is that this run proves that GS can work in the GWAS context, specifically on the provided chr21 subset. The evidence suggests that an LD-pruned configuration could be effective.
