from pathlib import Path

import pandas as pd


RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")


def build_dataset():
	train_path = RAW_DIR / "data_set_ALL_AML_train.csv"
	independent_path = RAW_DIR / "data_set_ALL_AML_independent.csv"
	labels_path = RAW_DIR / "actual.csv"

	train = pd.read_csv(train_path)
	independent = pd.read_csv(independent_path)

	print(train.columns.tolist()[:10])
	print(independent.columns.tolist()[:10])

	def prepare_expression_data(dataframe):
		columns_to_drop = [
			column
			for column in dataframe.columns
			if column == "Gene Description"
			or column == "call"
			or column.startswith("call.")
		]
		expression = dataframe.drop(columns=columns_to_drop)
		expression = expression.set_index("Gene Accession Number").transpose()
		expression.index = pd.to_numeric(expression.index).astype(int)
		expression.index.name = "patient"
		return expression.reset_index()

	combined = pd.concat(
		[prepare_expression_data(train), prepare_expression_data(independent)],
		ignore_index=True,
	).sort_values("patient", ignore_index=True)

	gene_columns = [column for column in combined.columns if column != "patient"]
	combined[gene_columns] = combined[gene_columns].apply(
		pd.to_numeric, errors="coerce"
	)
	nan_counts = combined[gene_columns].isna().sum()
	print(f"NaN values after numeric conversion: {int(nan_counts.sum())}")
	if nan_counts.any():
		print(nan_counts[nan_counts > 0])

	labels = pd.read_csv(labels_path)
	combined = combined.merge(labels[["patient", "cancer"]], on="patient", how="left")
	split = pd.Series(
		["train" if patient <= 38 else "test" for patient in combined["patient"]],
		name="split",
	)
	combined = pd.concat([combined, split], axis=1)

	output_path = PROCESSED_DIR / "golub_full.csv"
	output_path.parent.mkdir(parents=True, exist_ok=True)
	combined.to_csv(output_path, index=False)

	print(combined.shape)
	print(combined["split"].value_counts())
	print(combined.groupby("split")["cancer"].value_counts())
	return combined


if __name__ == "__main__":
	build_dataset()
