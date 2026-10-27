# Assignment rubric audit

Authoritative source: `S1_26_AML_Assignment.pdf`, read in full on 22 October 2026. The user also explicitly requested every instruction in `Codex_Full_Marks_AML_Prompt.md`; its additional scientific and delivery requirements are tracked below. This document is a local development artifact and must **not** be uploaded to Taxila.

Final verification on 22 October 2026: the full prompt and three-page PDF were read again against the completed artifacts. The notebook executed from a clean kernel, all 25 code cells have consecutive execution counts, there are zero error/warning outputs, ten figure outputs and 37 HTML table/style outputs. The HTML contains embedded images/styles, no external script/style dependencies, and was visually opened in headless Firefox. Group/member placeholders are intentional at the user's request. Public hosting is not claimed. Coverage is not a guarantee of awarded marks.

## Implementation and execution review

The experiment function accepts development data only. Fold-local preprocessing, nested SVM probability calibration, balancing and the area ablation are implemented and executed. The voting estimator contains complete per-member pipelines. Reusing development folds for tuning and voting is disclosed as non-nested selection; a fixed untouched holdout supplies the final estimate. The selection rule, grids and holdout split were not changed in response to holdout scores.

Logistic Regression was selected by CV macro-F1 **0.961188 ± 0.012726**, ahead of soft voting (0.944294), calibrated RBF SVM (0.915572), Random Forest (0.891722), Decision Tree (0.840429) and Dummy (0.100000). The 400-row holdout accuracy is **0.980000**, macro-F1 **0.980049**, with 100 observations per class and eight adjacent-class errors. Full-data deployment refitting follows the locked metrics. Both area additions were rejected by the predeclared rule; balancing methods were identical because folds already contain equal class counts.

All **15 tests and 9 subtests passed**. Streamlit AppTest exercised real prediction and what-if forms; batch validation, 1,000 ordered IDs, complete-pipeline reload equivalence, class schema and probability normalization were checked. A separate real server returned HTTP 200 for both health and page while launched from a different working directory; that temporary server was stopped. The app itself has not been visually browser-tested or publicly hosted.

The seven support sources embedded in the notebook match their executed on-disk counterparts. Only `GROUP_NO.ipynb` and `GROUP_NO.html` occur in `submission/`; all other artifacts remain outside. `reports/execution_audit.json`, `reports/results.json`, `reports/app_verification.json`, and `reports/server_verification.json` provide machine-readable evidence.

## Authoritative marking map

| PDF component | Marks | Required notebook evidence | Verification |
|---|---:|---|---|
| 1. Import Libraries / Dataset | 1 | numpy, pandas, sklearn, matplotlib; supplied training CSV loaded; initial records and target explicitly shown | Verified — executed notebook |
| 2. Data and Data Models | 1 | Shape, feature names, dtypes; exactly two complete sanity rows | Verified — executed notebook |
| 3. Data Visualization and Exploration | 5 | Consolidated attributes dashboard; class imbalance graph/comment; descriptive statistics; histograms/countplots; scatterplots/boxplots; numeric correlation heatmap; evidence-based influential-feature and nonlinearity discussion | Verified — executed notebook |
| 4. Data Pre-processing and Cleaning | 5 | Missingness/outlier/skew audit and justified handling; binary/label encoding and one-hot encoding; comparison of balancing techniques and metrics; feature engineering, scaling/transformation justification; complementary feature-importance techniques | Verified — executed notebook |
| 5. Model Building | 5 | Features/target/unlabeled-test separation with shapes; justified classifiers. PDF objective lists DT, RF, SVM, Ensemble; implement all four to cover the objective and the section's lower minimum of two | Verified — executed notebook |
| 6. Performance Evaluation | 2 | Accuracy, confusion matrix, classification report for all four classes, justified selection/metrics and under/overfitting discussion; honest labeled holdout evaluation | Verified — executed notebook |
| 7. Model Deployment (Streamlit) | 4 | Persisted complete pipeline; app takes new feature values, queries saved model, returns result; working local evidence and explicit public-hosting status | Verified — notebook and AppTest |
| 8. Novelty and Innovation | 2 | Implemented, tested substantive addition plus polished usable interface | Verified — notebook and AppTest |
| 9. Presentation and Viva | 5 | Clear narrative and outputs, executive summary, viva questions, demo narrative, limitations, genuine group/member metadata | Verified content — requested identity placeholders remain |
| **Total** | **30** | **Nine ordered sections** | **Verified locally — metadata/public hosting limitations disclosed** |

## PDF ambiguities resolved against the actual supplied files

- The PDF asks for metrics on “test” data. The supplied test CSV has **no target labels**, so its ground-truth accuracy, confusion matrix and F1 cannot be computed. A stratified labeled holdout drawn once from the supplied training file supplies evaluation; the 1,000 supplied unlabeled rows receive inference only.
- The PDF says “download” but also “dataset provided.” Use the supplied CSVs without downloading a different dataset variant.
- The PDF objective names DT, RF, SVM and an ensemble, while section 5 asks for any two models. The requested solution covers all four families.
- The page-3 shorthand “PDF/HTML with all outputs; source code” is subordinate to page-1 instruction 9: upload **only two unzipped files**, a notebook and its HTML/PDF output. App source and regeneration logic must therefore be visible and runnable inside the notebook; supporting artifacts stay outside `submission/`.
- Group number and identities are unknown. Use conspicuous placeholders, never invented names. The final student action includes replacing placeholders and renaming both deliverables.

## Independently rechecked source-data facts

Read directly with Python's CSV parser, preserving duplicate empty header positions:

| Check | Training CSV | Unlabeled test CSV |
|---|---:|---:|
| Rows | 2,000 | 1,000 |
| Raw columns | 22 | 24 |
| Predictors after excluding target/id/empty trailing columns | 20 | 20 |
| `price_range` present | Yes | No |
| Class counts | 0: 500; 1: 500; 2: 500; 3: 500 | Unavailable |
| Empty trailing header columns | 0 | 3 (pandas names `Unnamed: 21`–`Unnamed: 23`) |
| Whitespace-only cell in trailing columns | 0 | One in raw column position 22 (zero-based) |
| Duplicate complete rows | 0 | 0 |
| Duplicate predictor vectors | 0 | 0 |
| Blank/missing valid predictor values | 0 | 0 |
| `sc_w == 0` | 180 | 112 |
| `px_height == 0` | 2 | 2 |
| `fc == 0` | 474 | 210 |
| `pc == 0` | 101 | 40 |
| `fc > pc` | 0 | 0 |
| `four_g == 1` with `three_g == 0` | 0 | 0 |
| `sc_w > sc_h` | 0 | 0 |

The five string predictors are `blue`, `dual_sim`, `mobile_wt`, `touch_screen`, `wifi`. Four are Yes/No; mobile weight is already a Low/Med/High category, not grams. `id` must never enter a fitted predictor matrix. Zero-valued camera specifications can represent absent cameras; zero physical dimensions deserve explicit qualification because their semantics are not documented. Neither dropping them nor converting them to missingness is automatically justified.

## Scientific and engineering acceptance checks

- [x] Split exactly once with stratification and seed 42 into 1,600 development and 400 holdout rows; all supervised exploration/decisions use development data only.
- [x] Numeric imputation/scaling/selection and categorical encoding are fit within CV pipelines; held-out permutation importance is interpretive only, after selection.
- [x] No balancing, balanced class weights and training-fold-only oversampling are compared on identical development folds; full-data balance and any per-fold rounding effects are explained.
- [x] Derived pixel/screen characteristics are zero-safe and accepted/rejected using development CV ablation, with no holdout-based feature choice.
- [x] DT, RF, RBF SVM and ensemble CV metrics/tuning rationale are visible; selected model is fixed by CV macro-F1 before holdout access.
- [x] Holdout evaluation includes accuracy, macro/weighted precision/recall/F1, per-class support 0–3, raw and normalized confusion matrices, train/CV/holdout fit discussion and honest confidence limitations.
- [x] Full-data deployment refit happens only after holdout scores are fixed; reported holdout scores belong to the pre-refit model.
- [x] Predictions contain exactly 1,000 rows, columns `id,price_range`, labels in 0–3 and original test ID/order.
- [x] The persisted model includes preprocessing, reload predictions match, and genuine probability rows sum to one where supported.
- [x] Raw-input app schema matches model order; no invented currency ranges; all predictor widgets, clear validation, batch inference and what-if comparison are tested.
- [x] The notebook contains runnable model/module/schema/app regeneration code, not only references to external files; app source is readable rather than opaque binary data.
- [x] Notebook executed from a clean kernel, every intended code cell has an execution count, zero error outputs, all charts/tables present.
- [x] HTML exported from the executed notebook, embeds figure outputs, retains all nine sections and measured results.
- [x] `submission/` contains exactly `GROUP_NO.ipynb` and `GROUP_NO.html` pending real group number.
- [x] Final response separates verified local deployment from unavailable/unverified public hosting and flags real identity placeholders.

## Current authoritative technical references

Official documentation checked on 22 October 2026. Methods are established approaches maintained in current tooling; describing them as newly invented would be inaccurate.

1. [scikit-learn: common pitfalls and data leakage](https://scikit-learn.org/stable/common_pitfalls.html): fit transformations on training data only, with pipelines inside cross-validation.
2. [imbalanced-learn: RandomOverSampler](https://imbalanced-learn.org/stable/references/generated/imblearn.over_sampling.RandomOverSampler.html): sampling with replacement; automatic sampling equalizes class counts, so exactly balanced folds require no added rows.
3. [imbalanced-learn: resampling pitfalls](https://imbalanced-learn.org/stable/common_pitfalls.html): resampling before splitting causes leakage and changes the evaluation distribution; use the sampler within a pipeline.
4. [scikit-learn: probability calibration](https://scikit-learn.org/stable/modules/calibration.html): explicit CalibratedClassifierCV wraps the full raw-input SVM pipeline with five inner stratified folds; no held-out data are used for calibration. Model probability is not guaranteed correctness.
5. [Streamlit: AppTest](https://docs.streamlit.io/develop/api-reference/app-testing/st.testing.v1.apptest): simulate the running app, update widgets and explicitly rerun to validate output.
6. [Streamlit: Community Cloud deployment](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/deploy): deploy an accessible repository/branch/entrypoint from the user's Cloud account; a live URL exists only after actual deployment.

Supplementary: [SMOTENC for mixed categorical and numeric predictors](https://imbalanced-learn.org/stable/references/generated/imblearn.over_sampling.SMOTENC.html); [balanced class-weight formula](https://scikit-learn.org/stable/modules/generated/sklearn.utils.class_weight.compute_class_weight.html).
