# Mobile Price Prediction

Applied Machine Learning assignment using the supplied Mobile Price Prediction datasets.

**Group 2**

| Member | BITS ID |
|---|---|
| Jannatul Firdousi | 2025NS04069 |
| Dyutik Chaudhary Suryadevara | 2025NS04090 |
| YUVARAJ S S | 2025NS04059 |
| KOTHAPALLI BHARGAVA AVINASH | 2025NS04077 |

**Deployed application:** https://mobile-price-prediction-aml.streamlit.app/

## Report

The two submission files are:

- [Executed notebook](submission/2.ipynb): preprocessing, exploration, training, evaluation and the complete application source.
- [PDF report](submission/2.pdf): explanations, tables and plots from the executed notebook.

The [HTML review copy](output/html/2.html) contains the same analysis and outputs. It stays outside the submission folder so that only the notebook and PDF are uploaded.

## Results

| Classifier | Development CV macro-F1 | CV accuracy |
|---|---:|---:|
| Logistic Regression | 0.9612 | 0.9613 |
| Soft Voting | 0.9443 | 0.9444 |
| RBF SVM | 0.9156 | 0.9156 |
| Random Forest | 0.8917 | 0.8913 |
| Decision Tree | 0.8404 | 0.8400 |
| Dummy Classifier | 0.1000 | 0.2500 |

Five-fold stratified cross-validation uses 1,600 development rows, with 400 labeled rows held out for evaluation. Logistic Regression (`C=100`) is selected by CV macro-F1. Its holdout accuracy is 0.9800, macro precision 0.9802, macro recall 0.9800 and macro-F1 0.9800. All eight errors are between adjacent price classes.

The chosen pipeline is then refitted on all 2,000 labeled rows. It predicts the 1,000 supplied test rows while retaining their IDs and original row order. The test data has no labels, so its output is used for prediction only.

## Run locally

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
streamlit run app.py
```

The saved pipeline and input settings are in [artifacts/](artifacts/). Dependencies are pinned in [requirements.txt](requirements.txt); the original execution environment is also recorded in [requirements-lock.txt](requirements-lock.txt).

The app provides a single-phone form, class probabilities, batch CSV prediction and a RAM/battery what-if comparison. In downloaded results, `price_range` is the predicted class and `p_0` to `p_3` are probabilities for Low, Medium, High and Very high.

## Regenerate the report

Keep the original supplied CSVs in the project directory. The notebook contains the implementation needed to regenerate preprocessing, experiments and the app.

```bash
source .venv/bin/activate
python run_notebook.py
python -m pip install -r requirements-pdf.txt
python export_pdf.py
```

[run_notebook.py](run_notebook.py) starts a clean kernel and exports the executed analysis to HTML. [export_pdf.py](export_pdf.py) converts that HTML into a paginated PDF, using landscape pages for the dense charts. The notebook contains all code; the HTML and PDF focus on explanations and executed outputs.

For source-driven notebook edits, update [notebook_editorial.py](notebook_editorial.py), then run [build_notebook.py](build_notebook.py) before the execution/export commands. Rebuilding replaces manual notebook edits, so preserve any group details entered directly in the notebook.

## Checks

```bash
python -m pytest -q tests
python verify_app_server.py
```

Tests cover the input schema, class labels, saved-pipeline reloads, probability totals, batch order, invalid inputs, and Streamlit's prediction/what-if forms. The report revision passed 15 tests and 9 subtests.

The deployed app follows the `main` branch. After testing changes, commit and push the app source to update the deployment. Changes to group details and report wording do not require retraining a different model configuration.
