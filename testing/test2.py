# Note: no longer using this file

import hail as hl
import numpy as np
import math
import matplotlib.pyplot as plt

hl.init()

# 1. Create a synthetic dataset
n_samples = 100
n_variants = 200
mt = hl.balding_nichols_model(n_populations=1, n_samples=n_samples, n_variants=n_variants)

# Add sample annotations (phenotypes and covariates)
# We'll simulate a continuous phenotype 'CaffeineConsumption' and a binary phenotype 'is_case'
# Covariates: 'isFemale' (binary), and an intercept (1.0)

mt = mt.annotate_cols(
    isFemale = hl.rand_bool(0.5),
    noise = hl.rand_norm(0, 1)
)

# Create phenotypes with some genetic signal from the first variant
# To do this, we need to realize the random values, but for a simple test, random is fine.
# We'll just make random phenotypes for simplicity of the API test.
mt = mt.annotate_cols(
    CaffeineConsumption = hl.rand_norm(10, 2) + hl.if_else(mt.isFemale, 1.0, 0.0),
    is_case = hl.rand_bool(0.2)
)

# 2. Run Graphlet Screening
# Graphlet screening is often used for detecting associations with structure, but here we use it as a regression method.
print("Running Graphlet Screening...")
gs_results = hl.graphlet_screening(
    mt,
    y=mt.CaffeineConsumption,
    x=mt.GT.n_alt_alleles(),
    covariates=[1.0, mt.isFemale],
    nm=3,
    r=3.5,
    block_size=100
)
gs_results.show()

# 3. Run Linear Regression (OLS) using Hail
# This is the standard GWAS method for continuous traits.
print("Running Linear Regression (OLS)...")
ols_results = hl.linear_regression_rows(
    y=mt.CaffeineConsumption,
    x=mt.GT.n_alt_alleles(),
    covariates=[1.0, mt.isFemale]
)
ols_results.show()

# 4. Run Logistic Regression
# This is the standard GWAS method for binary traits (case/control).
print("Running Logistic Regression...")
logreg_results = hl.logistic_regression_rows(
    test='wald',
    y=mt.is_case,
    x=mt.GT.n_alt_alleles(),
    covariates=[1.0, mt.isFemale]
)
logreg_results.show()

# 5. Compare results (Basic comparison)
# We can join the tables to compare p-values for the first few variants.
print("Comparison of P-values (First 5 variants):")
# Join OLS, Graphlet Screening, and Logistic results
comparison = ols_results.select(ols_p=ols_results.p_value).join(
    gs_results.select(gs_p=gs_results.p_value, gs_beta=gs_results.beta),
    how='inner'
).join(
    logreg_results.select(logit_p=logreg_results.p_value),
    how='inner'
)

comparison.select("ols_p", "gs_p", "logit_p", "gs_beta").show(5)

print("Graphlet Screening Fields:", list(gs_results.row_value))
print("OLS Fields:", list(ols_results.row_value))
print("Logistic Regression Fields:", list(logreg_results.row_value))

# 6. Build a GWAS-like Manhattan plot.
# We plot -log10(p) across genomic position for each method.
def safe_neg_log10(p):
    if p is None or np.isnan(p) or p <= 0.0:
        return np.nan
    return -math.log10(p)

sorted_comparison = comparison.order_by(comparison.locus)
plot_rows = sorted_comparison.select(
    pos=sorted_comparison.locus.position,
    ols_p=sorted_comparison.ols_p,
    gs_p=sorted_comparison.gs_p,
    logit_p=sorted_comparison.logit_p
).collect()

positions = [row.pos for row in plot_rows]
ols_y = [safe_neg_log10(row.ols_p) for row in plot_rows]
gs_y = [safe_neg_log10(row.gs_p) for row in plot_rows]
logit_y = [safe_neg_log10(row.logit_p) for row in plot_rows]

genome_wide_line = -math.log10(5e-8)

series = [
    ("OLS", ols_y, "steelblue"),
    ("Graphlet Screening", gs_y, "darkorange"),
    ("Logistic Regression", logit_y, "seagreen"),
]

fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

for ax, (title, yvals, color) in zip(axes, series):
    finite_vals = [v for v in yvals if not np.isnan(v)]
    panel_y_max = max([genome_wide_line] + finite_vals) if finite_vals else genome_wide_line
    panel_y_max = max(panel_y_max * 1.08, 1.0)

    ax.scatter(positions, yvals, s=14, alpha=0.75, c=color, edgecolors="none")
    ax.axhline(genome_wide_line, color="crimson", linestyle="--", linewidth=1.0, label="5e-8")
    ax.set_ylim(0, panel_y_max)
    ax.set_ylabel("-log10(p)")
    ax.set_title(f"{title} Manhattan-like Plot")
    ax.grid(alpha=0.2, linestyle=":")

axes[-1].set_xlabel("Genomic Position")
plt.tight_layout()

plt.show()

hl.stop()

