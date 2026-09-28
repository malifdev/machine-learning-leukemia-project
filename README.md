# Leukemia Slayer 2.0

Leukemia classification (ALL vs. AML) from gene expression data, comparing two approaches side by side:

- **SVM**: our main machine learning model, trained on the Golub et al. (1999) dataset with t-test gene selection.
- **Gemma**: Google's language model, prompted with a few labeled examples (few-shot prompting). It is not trained.

The app is built with Gradio. You upload one patient's gene-expression CSV, the app validates it, both models predict, and the results are compared with a short explanation.

> **Disclaimer:** This is a class project built on a small research dataset (72 samples). It is not a medical tool and must not be used for real diagnosis.

---

## Project structure

```
leukemia-slayer/
├── app.py                    # Gradio web app (run this)
├── requirements.txt          # Python dependencies
├── .env                      # YOUR API key (you create this, not in the repo)
├── data/
│   ├── raw/                  # Original Golub CSV files
│   └── processed/            # golub_full.csv, gene_ranking.csv
├── sample_patients/          # Ready-made test CSVs for the upload box
└── src/
    ├── preprocess.py
    ├── feature_selection.py
    ├── train_evaluate.py
    ├── pipeline.py           # Validation, SVM prediction, Gemma prediction
    └── make_sample_patient.py
```

The dataset, processed files, gene ranking, and sample patients are all included in the repo. **You do not need to run any preprocessing or training scripts** just to use the app.

---

## Prerequisites

- **Windows** with PowerShell or Command Prompt
- **Python 3.10 or newer** (check with `python --version`)
- **Git**
- A free **Google AI Studio API key** (Step 4 below)

---

## Setup (Windows)

### 1. Clone the repository

```
git clone <REPO-URL>
cd leukemia-slayer
```

Replace `<REPO-URL>` with the repository link. If the folder created by `git clone` has a different name than `leukemia-slayer`, `cd` into that folder instead.

### 2. Confirm the data came through (optional)

```
dir data\processed
```

You should see `golub_full.csv` and `gene_ranking.csv`.

### 3. Create your own virtual environment and install dependencies

Do not reuse anyone else's `venv` folder. It is tied to their machine and is not part of the repo.

```
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

When it works, you will see `(venv)` at the start of your terminal line. You need to run `venv\Scripts\activate` again every time you open a new terminal for this project.

**PowerShell blocked the activate script?** Run this once in that terminal window, then activate again:

```
Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
venv\Scripts\activate
```

This only affects the current terminal session.

### 4. Get your own Gemma / Gemini API key

1. Go to https://aistudio.google.com/apikey
2. Sign in with a Google account and create a free API key.

Use your own key. Do not share keys between people: rate limits are per key, and different keys can have access to different Gemma model versions.

### 5. Create the `.env` file

The `.env` file is intentionally not in the repo, because it holds your secret key.

**Option A: manually.** In the project root (next to `app.py`), create a file named exactly `.env` containing one line:

```
GOOGLE_API_KEY=paste_your_real_key_here
```

**Option B: with Copilot.** Paste this into Copilot chat, replacing the placeholder with your real key first:

```text
Create a file named .env in the project root with exactly this content,
using my real Google API key in place of the placeholder:

GOOGLE_API_KEY=paste_your_real_key_here

Do not commit this file or print the key back to me in chat.
```

The file is already listed in `.gitignore`, so it will not be pushed to the repo. Never commit it.

### 6. Run the app

```
python app.py
```

Open the local address shown in the terminal (usually http://127.0.0.1:7860).

---

## Using the app

1. On the **Predict** tab, upload `sample_patients/patient_ALL_example.csv`.
2. Click **Validate CSV**. You should see a green check and "Ready to predict."
3. Optionally adjust the **Top-K Genes** slider (how many top-ranked genes feed both models).
4. Click **Run Prediction**.
5. The SVM card, Gemma card, and comparison card will fill in. Try `patient_AML_example.csv` too.

### Patient CSV format

One row per patient, with columns being the same gene accession numbers used in the training data (about 7,129 columns). The provided files in `sample_patients/` show the exact format.

---

## Troubleshooting

> **Skip this section if everything is working.** Only come here if you see an error message. Find the error below that matches yours and follow only that fix.

**A warning about `google.generativeai` being deprecated appears in the terminal**
This is only a warning, not an error. The app still works, so you can ignore it.

**`ModuleNotFoundError` (for example sklearn or gradio)**
Your virtual environment is not active, or dependencies were not installed. Run `venv\Scripts\activate`, then `pip install -r requirements.txt`. To confirm you are using the right Python:

```
python -c "import sklearn; print(sklearn.__file__)"
```

The printed path should contain your project's `venv` folder.

**Gemma card shows `404 ... model not found`**
Your API key may not have access to the model named in `src/pipeline.py`. Different keys can see different Gemma versions. To list what your key can use, create and run a small script (or ask Copilot to do it):

```text
Create a temporary file check_models.py in the project root that:
1. Loads GOOGLE_API_KEY from .env using python-dotenv, same way
   pipeline.py does.
2. Calls genai.configure(api_key=...) then genai.list_models().
3. Loops through the results and prints model.name for every model
   where "generateContent" is in model.supported_generation_methods.
Run it as a script.
```

Then set `GEMMA_MODEL_NAME` in `src/pipeline.py` to one of the printed names, without the `models/` prefix (for example, `models/gemma-4-26b-a4b-it` becomes `gemma-4-26b-a4b-it`), and restart `python app.py`.

**Gemma card shows "GOOGLE_API_KEY is missing"**
The `.env` file is missing, misnamed, or not in the project root. It must be named exactly `.env` (not `.env.txt`) and sit next to `app.py`.

**Gemma card shows a rate-limit error (429)**
You sent too many requests in a short time. Wait a minute and try again. The SVM side keeps working regardless, since Gemma errors are handled without crashing the app.

---

## Optional: regenerate the data files from scratch

Only needed if you change the raw data or the feature-selection logic. Run these in order with the venv active:

```
python src/preprocess.py
python src/train_evaluate.py
python src/make_sample_patient.py
```

`train_evaluate.py` also prints the accuracy table (Leave-One-Out CV on the training set, plus held-out test accuracy) used to choose the default number of genes.

---

## Team

Group: **Leukemia Slayer 2.0**
