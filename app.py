"""
Leukemia Slayer 2.0
--------------------
UI-only build. Predicts ALL vs AML from gene-expression CSV data using:
  1. SVM  (trained on top-K genes selected via t-test)
  2. Gemma (few-shot LLM prediction)
then compares both predictions with a plain-language explanation.

NOTE: This file wires up the interface only. All prediction / pipeline
functions below are placeholders (`# TODO: connect real pipeline`) that
return mock data so the UI can be demoed end-to-end before the model
and backend logic are implemented.
"""

import time
import gradio as gr
from src import pipeline

# --------------------------------------------------------------------------
# THEME & STYLING
# --------------------------------------------------------------------------

theme = gr.themes.Soft(
    primary_hue=gr.themes.colors.violet,
    secondary_hue=gr.themes.colors.cyan,
    neutral_hue=gr.themes.colors.slate,
    font=[gr.themes.GoogleFont("Inter"), "ui-sans-serif", "system-ui", "sans-serif"],
    font_mono=[gr.themes.GoogleFont("JetBrains Mono"), "ui-monospace", "monospace"],
).set(
    body_background_fill="*neutral_50",
    block_background_fill="white",
    block_border_width="1px",
    block_border_color="*neutral_200",
    block_shadow="0 1px 3px 0 rgba(0,0,0,0.06), 0 1px 2px -1px rgba(0,0,0,0.06)",
    block_radius="16px",
    button_primary_background_fill="linear-gradient(90deg, #7C3AED 0%, #6D28D9 100%)",
    button_primary_background_fill_hover="linear-gradient(90deg, #6D28D9 0%, #5B21B6 100%)",
    button_primary_text_color="white",
    button_secondary_background_fill="white",
    button_secondary_border_color="*neutral_300",
    input_background_fill="white",
    input_border_color="*neutral_200",
)

CUSTOM_CSS = """
#hero {
    background: linear-gradient(135deg, #6D28D9 0%, #7C3AED 45%, #0891B2 100%);
    border-radius: 20px;
    padding: 36px 40px;
    color: white !important;
    margin-bottom: 8px;
}
#hero h1 { color: white !important; font-size: 30px; margin-bottom: 4px; font-weight: 700; }
#hero p { color: rgba(255,255,255,0.9) !important; font-size: 15px; margin: 2px 0; }
#hero .badges { margin-top: 14px; }

.badge {
    display: inline-block;
    background: rgba(255,255,255,0.18);
    border: 1px solid rgba(255,255,255,0.35);
    color: white;
    padding: 4px 12px;
    border-radius: 999px;
    font-size: 12px;
    margin-right: 6px;
    font-weight: 500;
}

.section-title {
    font-size: 17px;
    font-weight: 700;
    color: #1e1b2e;
    margin-bottom: 2px;
}
.section-sub {
    font-size: 13px;
    color: #6b7280;
    margin-bottom: 10px;
}

.result-card {
    border-radius: 14px;
    padding: 18px 20px;
    border: 1px solid #e5e7eb;
    background: white;
}
.result-card.svm { border-left: 4px solid #7C3AED; }
.result-card.gemma { border-left: 4px solid #0891B2; }
.result-card.compare { border-left: 4px solid #059669; }

.pred-label {
    font-size: 26px;
    font-weight: 800;
    margin: 4px 0;
}
.pred-label.all { color: #DB2777; }
.pred-label.aml { color: #EA580C; }

.conf-text { font-size: 13px; color: #6b7280; }

.pipeline-step {
    background: #FAF5FF;
    border: 1px solid #E9D5FF;
    border-radius: 10px;
    padding: 10px 14px;
    font-size: 13px;
    font-weight: 600;
    color: #6D28D9;
    text-align: center;
}

footer { display: none !important; }
"""

# --------------------------------------------------------------------------
# PLACEHOLDER / MOCK LOGIC  (swap these out for the real pipeline later)
# --------------------------------------------------------------------------

REQUIRED_GENES_MSG = (
    "Expecting ~7,000 gene-expression columns matching the Golub dataset "
    "feature set (or the reduced top-K set after feature selection)."
)


def validate_csv(file_obj):
    """Validate a patient CSV and enable prediction when it is ready."""
    is_valid, message, _ = pipeline.validate_patient_csv(file_obj)
    return message, gr.update(interactive=is_valid)


def run_prediction(file_obj, top_k, progress=gr.Progress()):
    """Run the real SVM prediction and compare it with the mock Gemma result."""
    if file_obj is None:
        empty = "—"
        return (
            gr.update(value="Upload a CSV to see the SVM result.", elem_classes=["pred-label"]),
            "—",
            gr.update(value="Upload a CSV to see the Gemma result.", elem_classes=["pred-label"]),
            "—",
            "Run a prediction to see the comparison and explanation here.",
        )

    progress(0.2, desc="Validating and selecting top-K genes...")
    is_valid, validation_message, patient_series = pipeline.validate_patient_csv(file_obj)
    if not is_valid:
        return (
            gr.update(value=validation_message, elem_classes=["pred-label"]),
            "—",
            gr.update(value="—", elem_classes=["pred-label"]),
            "—",
            "Run a prediction to see the comparison and explanation here.",
        )

    progress(0.6, desc="Running SVM prediction...")
    svm_pred, svm_conf = pipeline.predict_svm(patient_series, top_k)

    progress(0.8, desc="Asking Gemma (few-shot)...")
    try:
        gemma_pred, gemma_conf, gemma_explanation = pipeline.predict_gemma(patient_series, top_k)
        gemma_conf = round(gemma_conf, 1)
    except Exception as error:
        gemma_pred = "ERROR"
        gemma_conf = 0.0
        gemma_explanation = str(error)

    progress(1.0, desc="Comparing results...")

    svm_class = "all" if svm_pred == "ALL" else "aml"
    gemma_class = "" if gemma_pred == "ERROR" else ("all" if gemma_pred == "ALL" else "aml")

    if gemma_pred == "ERROR":
        verdict = (
            f"⚠️ SVM predicted **{svm_pred}** with {svm_conf}% confidence. "
            f"Gemma comparison unavailable: {gemma_explanation}"
        )
    else:
        agree = svm_pred == gemma_pred
        verdict = (
            f"✅ **Both models agree: {svm_pred}**\n\nSVM and Gemma reached the same conclusion "
            f"using the top {top_k} selected genes, which increases confidence in this result."
            if agree
            else
            f"⚠️ **Models disagree** — SVM says **{svm_pred}**, Gemma says **{gemma_pred}**.\n\n"
            f"This can happen with a small training set (72 samples). Consider reviewing the "
            f"top {top_k} selected genes or requesting a second opinion."
        )
        verdict += f"\n\n**Gemma's reasoning:** {gemma_explanation}"

    return (
        gr.update(value=svm_pred, elem_classes=["pred-label", svm_class]),
        f"Confidence: {svm_conf}%",
        gr.update(value=gemma_pred, elem_classes=["pred-label", gemma_class]),
        f"Confidence: {gemma_conf}%",
        verdict,
    )


def reset_all():
    return (
        None,
        "",
        gr.update(value="—", elem_classes=["pred-label"]),
        "—",
        gr.update(value="—", elem_classes=["pred-label"]),
        "—",
        "Run a prediction to see the comparison and explanation here.",
    )


# --------------------------------------------------------------------------
# UI LAYOUT
# --------------------------------------------------------------------------

with gr.Blocks(theme=theme, css=CUSTOM_CSS, title="Leukemia Slayer 2.0") as demo:

    # ---------- HERO ----------
    gr.HTML(
        """
        <div id="hero">
            <h1>🧬 Leukemia Slayer 2.0</h1>
            <p>Leukemia Classification Using Gene Expression Data</p>
            <p>Machine Learning (SVM) + Large Language Model (Gemma) Comparison</p>
            <div class="badges">
                <span class="badge">ALL vs AML</span>
                <span class="badge">t-test Feature Selection</span>
                <span class="badge">SVM</span>
                <span class="badge">Gemma Few-Shot</span>
                <span class="badge">Golub et al. Dataset</span>
            </div>
        </div>
        """
    )

    with gr.Tabs():

        # =====================================================
        # TAB 1 — PREDICT
        # =====================================================
        with gr.Tab("🔬 Predict"):

            with gr.Row():
                # ---------- LEFT: Upload & controls ----------
                with gr.Column(scale=4):
                    gr.HTML(
                        '<div class="section-title">1. Upload Patient Data</div>'
                        '<div class="section-sub">Upload one patient\'s gene-expression profile as a CSV file.</div>'
                    )
                    file_input = gr.File(
                        label="Patient Gene-Expression CSV",
                        file_types=[".csv"],
                        height=140,
                    )
                    validate_btn = gr.Button("✅ Validate CSV", variant="secondary")
                    validation_output = gr.Markdown(REQUIRED_GENES_MSG)

                    gr.HTML('<div style="height:8px"></div>')
                    gr.HTML(
                        '<div class="section-title">2. Feature Selection</div>'
                        '<div class="section-sub">Number of top genes (by t-test p-value) to feed into SVM.</div>'
                    )
                    top_k_slider = gr.Slider(
                        minimum=5, maximum=200, value=10, step=5,
                        label="Top-K Genes",
                    )

                    gr.HTML('<div style="height:4px"></div>')
                    with gr.Row():
                        predict_btn = gr.Button("🚀 Run Prediction", variant="primary", interactive=False)
                        reset_btn = gr.Button("↺ Reset", variant="secondary")

                # ---------- RIGHT: Results ----------
                with gr.Column(scale=5):
                    gr.HTML(
                        '<div class="section-title">3. Results</div>'
                        '<div class="section-sub">SVM and Gemma predictions, side by side.</div>'
                    )

                    with gr.Row():
                        with gr.Column():
                            gr.HTML('<div class="result-card svm">'
                                    '<div style="font-size:13px;font-weight:700;color:#7C3AED;">🧠 SVM PREDICTION</div>')
                            svm_label = gr.HTML('<div class="pred-label">—</div>')
                            svm_conf = gr.Markdown("—", elem_classes=["conf-text"])
                            gr.HTML('</div>')
                        with gr.Column():
                            gr.HTML('<div class="result-card gemma">'
                                    '<div style="font-size:13px;font-weight:700;color:#0891B2;">✨ GEMMA PREDICTION</div>')
                            gemma_label = gr.HTML('<div class="pred-label">—</div>')
                            gemma_conf = gr.Markdown("—", elem_classes=["conf-text"])
                            gr.HTML('</div>')

                    gr.HTML('<div style="height:6px"></div>')
                    gr.HTML('<div class="result-card compare">'
                            '<div style="font-size:13px;font-weight:700;color:#059669;margin-bottom:6px;">🔎 COMPARISON &amp; EXPLANATION</div>')
                    comparison_output = gr.Markdown("Run a prediction to see the comparison and explanation here.")
                    gr.HTML('</div>')

        # =====================================================
        # TAB 2 — PIPELINE
        # =====================================================
        with gr.Tab("🧭 Pipeline"):
            gr.HTML(
                '<div class="section-title">Whole Project Pipeline</div>'
                '<div class="section-sub">From dataset to final explained result.</div>'
            )
            with gr.Row():
                for step in ["Golub\nDataset", "Preprocess\nData", "t-test\nFeature Selection",
                             "Train\nSVM", "Upload\nPatient CSV", "Validate\nCSV"]:
                    gr.HTML(f'<div class="pipeline-step">{step.replace(chr(10), "<br>")}</div>')
            with gr.Row():
                for step in ["SVM\nPrediction", "Gemma\nPrediction", "Compare\nResults",
                             "Show\nExplanation"]:
                    gr.HTML(f'<div class="pipeline-step">{step.replace(chr(10), "<br>")}</div>')

            gr.Markdown(
                """
&nbsp;
### In simple words
We train the computer using old labeled patient data. Later, a user uploads one patient's
gene-expression CSV. The system checks it, predicts ALL/AML with SVM, asks Gemma for another
prediction, and compares both — then shows an easy explanation.

### Why feature selection first?
We first find the most useful genes from the ~7,000 available using a **t-test**, ranking
genes by p-value. Only the top-K genes are then used to train and query the **SVM** model.
                """
            )

        # =====================================================
        # TAB 3 — ABOUT / DATASET / MODELS
        # =====================================================
        with gr.Tab("📚 About the Project"):
            with gr.Row():
                with gr.Column():
                    gr.Markdown(
                        """
### 🎯 The Main Idea
We're building a web application that predicts whether a leukemia sample is:

- **ALL** — Acute Lymphoblastic Leukemia
- **AML** — Acute Myeloid Leukemia

The prediction is made from **gene expression data**, using two independent approaches
that are then compared side by side, along with a plain-language explanation.
                        """
                    )
                    gr.Markdown(
                        """
### 🧫 Dataset — Golub et al. (1999)
- 72 patient samples
- ~7,000 gene-expression features
- Two classes: ALL and AML
- Originally used for leukemia classification research

Gene expression data tells us how active different genes are in a sample — think of it as
a large table of numbers describing the activity of thousands of genes. With only 72 samples
but ~7,000 features, this is a classic small-data / high-dimensionality ML challenge.
                        """
                    )
                with gr.Column():
                    gr.Markdown(
                        """
### 🧠 SVM — Support Vector Machine
Our main machine learning model. It learns the difference between ALL and AML from labeled
training data, then predicts one of the two classes for a new patient. SVM is a strong choice
when there are many features but relatively few samples.

### ✨ Gemma — Large Language Model
Google's family of AI language models. We are **not training** Gemma — instead we give it a
few labeled examples in the prompt and ask it to predict the new patient's class
(**few-shot prompting**).
                        """
                    )
                    gr.Markdown(
                        """
### 📎 Why CSV Upload?
A patient profile contains thousands of gene-expression values — typing ~7,000 numbers into
a form isn't practical. So the user uploads a CSV with one complete patient profile, which the
system validates (columns, missing values, ordering) before running both predictions.
                        """
                    )

            gr.Markdown("### 📖 Glossary")
            with gr.Row():
                gr.Markdown("**Gene expression**\nNumbers showing how active genes are in a sample.")
                gr.Markdown("**Feature**\nOne input value used by the model — here, a gene-expression measurement.")
                gr.Markdown("**SVM**\nA machine learning method that learns to separate two groups.")
            with gr.Row():
                gr.Markdown("**Few-shot prompting**\nGiving an AI a few examples before asking it to classify a new case.")
                gr.Markdown("**Gemma**\nGoogle's family of large language models.")
                gr.Markdown("**CSV**\nA simple table file used to store the patient's gene-expression values.")

    # ---------- FOOTER ----------
    gr.HTML(
        '<div style="text-align:center; color:#9ca3af; font-size:12px; margin-top:18px;">'
        'Leukemia Slayer 2.0 — Project Update UI (backend pipeline not yet connected)'
        '</div>'
    )

    # ---------- EVENTS ----------
    validate_btn.click(
        fn=validate_csv,
        inputs=[file_input],
        outputs=[validation_output, predict_btn],
    )

    predict_btn.click(
        fn=run_prediction,
        inputs=[file_input, top_k_slider],
        outputs=[svm_label, svm_conf, gemma_label, gemma_conf, comparison_output],
    )

    reset_btn.click(
        fn=reset_all,
        inputs=[],
        outputs=[file_input, validation_output, svm_label, svm_conf, gemma_label, gemma_conf, comparison_output],
    )


if __name__ == "__main__":
    demo.launch()
