from pathlib import Path

import pandas as pd


DATA_PATH = Path("data/processed/golub_full.csv")
OUTPUT_DIR = Path("sample_patients")


def make_sample_patients():
	"""Create one test-split ALL and AML patient CSV for upload testing."""
	df = pd.read_csv(DATA_PATH)
	gene_columns = [column for column in df.columns if column not in {"patient", "cancer", "split"}]
	test_patients = df[df["split"] == "test"]

	for label in ("ALL", "AML"):
		patient = test_patients[test_patients["cancer"] == label].iloc[0]
		output_path = OUTPUT_DIR / f"patient_{label}_example.csv"
		output_path.parent.mkdir(parents=True, exist_ok=True)
		patient[gene_columns].to_frame().T.to_csv(output_path, index=False)
		print(f"{output_path}: patient {int(patient['patient'])}, true label {label}")


if __name__ == "__main__":
	make_sample_patients()
