import pandas as pd
from scipy.stats import ttest_ind


def rank_genes_by_ttest(df):
	"""
	Rank gene columns by independent two-sample t-test p-value.

	Only training rows are used so test data cannot influence feature selection.
	"""
	train = df[df["split"] == "train"]
	all_group = train[train["cancer"] == "ALL"]
	aml_group = train[train["cancer"] == "AML"]
	gene_columns = [column for column in df.columns if column not in {"patient", "cancer", "split"}]

	p_values = {}
	for gene in gene_columns:
		_, p_value = ttest_ind(all_group[gene], aml_group[gene])
		p_values[gene] = p_value

	return sorted(gene_columns, key=lambda gene: (pd.isna(p_values[gene]), p_values[gene]))
