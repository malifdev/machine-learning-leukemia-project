from pathlib import Path

import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


DATA_DIR = Path("data/processed")
FULL_DF = pd.read_csv(DATA_DIR / "golub_full.csv")
GENE_RANKING = pd.read_csv(DATA_DIR / "gene_ranking.csv")["gene"].tolist()
_model_cache = {}


def validate_patient_csv(file_obj):
	"""
	Validate and clean a Gradio patient CSV upload.

	Returns ``(is_valid, message, patient_row)``. The patient row contains the
	full ranked gene set in the order used by the prediction pipeline.
	"""
	if file_obj is None:
		return False, "No file uploaded.", None

	try:
		file_path = file_obj.name
		dataframe = pd.read_csv(file_path)
	except Exception as error:
		return False, f"Could not read file: {error}", None

	if dataframe.empty:
		return False, "The uploaded CSV has no rows.", None

	message_parts = []
	if len(dataframe) > 1:
		message_parts.append("Only the first row was used because multiple rows were uploaded.")
	present_genes = set(dataframe.columns).intersection(GENE_RANKING)
	missing_genes = [gene for gene in GENE_RANKING if gene not in present_genes]

	if len(missing_genes) > len(GENE_RANKING) * 0.05:
		examples = ", ".join(missing_genes[:5])
		return (
			False,
			f"Too many genes are missing: {len(missing_genes)} of {len(GENE_RANKING)}. "
			f"Examples: {examples}",
			none,
		)

	row = dataframe.iloc[0].reindex(GENE_RANKING)
	cleaned_row = pd.to_numeric(row, errors="coerce")
	nan_genes = cleaned_row[cleaned_row.isna()].index.tolist()
	if nan_genes:
		examples = ", ".join(nan_genes[:5])
		return (
			False,
			f"{len(nan_genes)} values are missing or non-numeric: {examples}",
			none,
		)

	message = (
		f"✅ **{Path(file_path).name}** looks valid.\n\n"
		f"- Columns detected: {len(present_genes)}\n"
		f"- Missing values: 0\n"
		"- Ready to predict."
	)
	if message_parts:
		message += "\n\n" + " ".join(message_parts)
	return True, message, cleaned_row


def predict_svm(patient_series, k):
	"""
	Predict ALL or AML using the top ``k`` ranked genes.

	Models are trained on all labeled patients in ``FULL_DF`` and cached by k.
	Returns ``(label, confidence_pct)``.
	"""
	try:
		k = int(k)
	except (TypeError, ValueError) as error:
		raise ValueError("k must be an integer.") from error

	if k < 1 or k > len(GENE_RANKING):
		raise ValueError(f"k must be between 1 and {len(GENE_RANKING)}.")
	if not isinstance(patient_series, pd.Series):
		raise TypeError("patient_series must be a pandas Series.")

	top_genes = GENE_RANKING[:k]
	if k not in _model_cache:
		model = Pipeline(
			[
				("scaler", StandardScaler()),
				("svc", SVC(kernel="linear", probability=True, random_state=42)),
			]
		)
		model.fit(FULL_DF[top_genes], FULL_DF["cancer"])
		_model_cache[k] = model

	model = _model_cache[k]
	patient_values = pd.to_numeric(patient_series.reindex(top_genes), errors="coerce")
	if patient_values.isna().any():
		raise ValueError("patient_series contains missing or non-numeric gene values.")

	patient_array = pd.DataFrame(
		[patient_values.to_numpy().reshape(1, -1)[0]],
		columns=top_genes,
	)
	prediction = model.predict(patient_array)[0]
	confidence_pct = round(float(model.predict_proba(patient_array)[0].max()) * 100, 1)
	return str(prediction), confidence_pct
