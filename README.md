# Mobile Price Prediction — completed assignment

The authoritative PDF rubric is covered in nine ordered notebook sections. The only Taxila upload files are:

- `submission/GROUP_NO.ipynb` — fully executed notebook, including readable app/module/test source regeneration.
- `submission/GROUP_NO.html` — offline HTML export with the same outputs.

Do not zip these files or upload any other project artifact to Taxila. The group number and member/student-ID details remain conspicuous placeholders at your request. Replace them in the notebook title and first code cell, rerun/export, and rename both files to the genuine group number before submission. Grading and the live viva remain the examiner's decision.

## Verified results

Logistic Regression (`C=100`, original predictors, no engineered areas) was selected using five-fold stratified development CV macro-F1. The seed-42 split contains 1,600 development rows and 400 holdout rows, with 100 holdout phones per class.

| Model | Development CV macro-F1 | Fold std | CV accuracy |
|---|---:|---:|---:|
| Logistic Regression | 0.9612 | 0.0127 | 0.9613 |
| Soft Voting | 0.9443 | 0.0174 | 0.9444 |
| SVM (RBF) | 0.9156 | 0.0172 | 0.9156 |
| Random Forest | 0.8917 | 0.0092 | 0.8912 |
| Decision Tree | 0.8404 | 0.0071 | 0.8400 |
| Dummy | 0.1000 | 0.0000 | 0.2500 |

Holdout accuracy is **0.9800** (392/400), macro-F1 **0.980049**, macro precision **0.980196**, macro recall **0.9800**. Weighted metrics equal the corresponding macro metrics because class support is equal. All eight errors are between adjacent classes. Accuracy's Wilson 95% interval is [0.9610, 0.9898]. The strongest raw-feature permutation drivers are RAM, battery capacity and pixel height. These are predictive associations.

The complete chosen pipeline was then refitted on all 2,000 labeled rows for deployment. Holdout metrics were measured before that full-data refit. `test_predictions.csv` contains exactly 1,000 `id,price_range` rows in original supplied-test order. The test file has no labels; no accuracy, F1 or confusion matrix is claimed for it.

Area-feature ablation improved LR macro-F1 by only 0.000639, below the predeclared 0.002 retention threshold; it reduced SVM macro-F1. Neither area variant was retained. No balancing, balanced weights and fold-local RandomOverSampler gave identical scores: every training fold already has 320 examples per class, so weights are 1 and oversampling adds zero rows.

## Run the verified local app

From this project directory:

```bash
source .venv/bin/activate
streamlit run app.py
```

Or use `.venv/bin/python -m streamlit run app.py`. The application loads paths relative to `app.py`, so an absolute app path also works from another directory. The temporary verification server has been stopped; run the command to start your demonstration.

1. Adjust the grouped specifications and click **Predict price category**.
2. Inspect the relative 0–3 category and genuine probability bars.
3. Open **What-if simulator**, alter RAM/battery and click **Compare scenarios**.
4. Open **Batch CSV**, upload the original test CSV, and download ordered predictions.

The app validates all twenty raw predictors, retains optional opaque IDs, strips known Unnamed export columns, and rejects invalid categories, missing fields and malformed numbers. It supports recorded zeros. The scenario is predictive, not causal; there are no invented rupee amounts.

## Reproduce

Verified interpreter: Python 3.14.8 on Linux. Direct versions are pinned in `requirements.txt`; the complete environment is recorded in `requirements-lock.txt`. All originals remain unchanged and SHA-256 fingerprints are recorded in `reports/reproducibility.json`.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-lock.txt
python run_notebook.py
```

On this machine PyPI's file host timed out; installation succeeded using the Tsinghua PyPI mirror. The requirements are ordinary package/version pins, with no mirror hard-coded. Use your normal trusted package index when reproducing.

`run_notebook.py` starts a clean kernel with the active interpreter, executes the entire notebook, and exports HTML with embedded images/styles. It accepts another filename, for example:

```bash
python run_notebook.py submission/42.ipynb
```

`build_notebook.py` is a developer convenience for rebuilding the original placeholder notebook from local sources; it will overwrite notebook edits, so do not use it after customizing your group metadata. The submitted notebook itself contains all indispensable source for model/app/schema/test regeneration. To rerun from only the academic submission, place the original two CSVs alongside a working copy, install the listed dependencies, and run the notebook. The original supplied CSVs are required inputs and are not additional Taxila uploads.

Checks can also be run explicitly:

```bash
python -m pytest -q tests
python verify_app_server.py
```

Verified: 15 tests and 9 subtests passed; single-phone and what-if forms executed through Streamlit AppTest; batch/persistence/schema checks passed; all test rows scored; actual Streamlit HTTP health and page returned 200 from a different working directory. AppTest tests application behavior, not visual browser rendering of the app. The HTML title page and representative figures were visually inspected. All 25 notebook code cells have consecutive execution counts and no error/warning outputs; ten figure outputs and 37 HTML table/style outputs are embedded.

## Evidence and supporting artifacts

- `RUBRIC_AUDIT.md` and `reports/rubric_compliance.csv`: all nine criteria / 30 marks mapped to evidence.
- `reports/results.json`: metrics, confusion matrix, per-class report and methodology.
- `reports/execution_audit.json`: clean execution and export audit.
- `reports/app_verification.json`, `reports/server_verification.json`: actual app checks.
- `artifacts/model.joblib`, `artifacts/schema.json`: complete deployment pipeline and input schema.
- `artifacts/development_models.joblib`: development-fitted models, distinct from the full-data deployment model.
- `app.py`, `mobile_pipeline.py`, `model_experiments.py`, `tests/`: readable tested implementation.
- `test_predictions.csv`: supplied unlabeled test predictions; local artifact only.
- `reports/notebook_preview.png`, `reports/figures/`: local presentation-check artifacts.

## Public hosting and limitations

No public Streamlit URL was created or claimed. To publish, use your own authorized repository/account: put `app.py`, `mobile_pipeline.py`, `requirements.txt`, `.streamlit/config.toml`, `artifacts/model.joblib` and `artifacts/schema.json` in the repository; in Streamlit Community Cloud choose that repository/branch and `app.py`, select a Python runtime compatible with the pinned packages, and deploy. Verify the returned URL using single and batch inference. Public-cloud runtime compatibility has not been tested. Official guide: https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/deploy

Other limits: one supplied dataset and split, undocumented sampling provenance and zero-dimension semantics, coarse weight categories, selection optimism in non-nested development CV, no external/current-market validation, and no known currency boundaries. The notebook includes 12 viva questions and a timed 2–3 minute demo narrative. Avoid changing model choices based on the already-used holdout and presenting that as a fresh unbiased evaluation.
