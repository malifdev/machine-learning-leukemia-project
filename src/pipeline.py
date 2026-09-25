from pathlib import Path
import os
import re

from dotenv import load_dotenv
import google.generativeai as genai
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
GEMMA_MODEL_NAME = "gemma-4-26b-a4b-it"

if GOOGLE_API_KEY:
	genai.configure(api_key=GOOGLE_API_KEY)


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


def _build_few_shot_prompt(patient_series, k, n_examples_per_class=2):
	"""
	Build a reproducible few-shot Gemma prompt using the SVM's top-k genes.

	The labeled examples come only from the training split. Each expression
	value is rounded to two decimal places so the prompt stays readable while
	both models receive the same selected features.
	"""
	top_genes = GENE_RANKING[:k]
	train_df = FULL_DF[FULL_DF["split"] == "train"]
	all_examples = train_df[train_df["cancer"] == "ALL"].sample(
		n=n_examples_per_class, random_state=42
	)
	aml_examples = train_df[train_df["cancer"] == "AML"].sample(
		n=n_examples_per_class, random_state=42
	)

	def format_values(values):
		return ", ".join(
			f"{gene}={float(values[gene]):.2f}" for gene in top_genes
		)

	example_lines = []
	for example_number, (_, patient) in enumerate(all_examples.iterrows(), start=1):
		example_lines.append(
			f"Example {example_number} (ALL): {format_values(patient)}"
		)
	for example_number, (_, patient) in enumerate(
		aml_examples.iterrows(), start=n_examples_per_class + 1
	):
		example_lines.append(
			f"Example {example_number} (AML): {format_values(patient)}"
		)

	return (
		"You are a medical machine learning assistant classifying leukemia subtype "
		"from gene expression data. Below are labeled example patients, each showing "
		f"expression values for the same {k} genes. Learn the pattern from these "
		"examples, then classify the new unlabeled patient.\n\n"
		+ "\n".join(example_lines)
		+ "\n\nNew patient (unlabeled): "
		+ format_values(patient_series)
		+ "\n\nRespond in EXACTLY this format and nothing else:\n"
		"Diagnosis: <ALL or AML>\n"
		"Confidence: <a number from 0 to 100>\n"
		"Explanation: <one or two plain-language sentences a non-expert could "
		"understand, comparing the new patient to the example patients>"
	)


def predict_gemma(patient_series, k):
	"""
	Classify a patient with Gemma using reproducible labeled examples.

	Returns ``(label, confidence_pct, explanation)``. A ``RuntimeError`` is
	raised for missing configuration, API failures, or an unparseable response.
	"""
	if not GOOGLE_API_KEY:
		raise RuntimeError("GOOGLE_API_KEY is missing — add it to your .env file.")

	prompt = _build_few_shot_prompt(patient_series, k)
	try:
		response = genai.GenerativeModel(GEMMA_MODEL_NAME).generate_content(prompt)
		response_text = response.text
	except Exception as error:
		raise RuntimeError(f"Gemma API call failed: {error}") from error

	diagnosis_match = re.search(r"Diagnosis:\s*(ALL|AML)", response_text, re.IGNORECASE)
	confidence_match = re.search(
		r"Confidence:\s*(\d+(?:\.\d+)?)", response_text, re.IGNORECASE
	)
	explanation_match = re.search(
		r"Explanation:\s*(.+)", response_text, re.IGNORECASE | re.DOTALL
	)
	if not diagnosis_match or not confidence_match or not explanation_match:
		raise RuntimeError(
			f"Gemma response did not match the expected format. Raw response: {response_text}"
		)

	return (
		diagnosis_match.group(1).upper(),
		float(confidence_match.group(1)),
		explanation_match.group(1).strip(),
	)
