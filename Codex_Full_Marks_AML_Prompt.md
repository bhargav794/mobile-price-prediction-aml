# Codex master prompt — BITS Pilani Applied Machine Learning assignment

You are Codex acting as a senior applied-ML engineer, data scientist, technical writer, and Streamlit developer. **Implement and verify the complete assignment**, not merely propose an approach. Your objective is to maximize marks honestly and reproducibly across the entire 30-mark rubric. Never fabricate results, screenshots, scores, public deployment URLs, or dataset labels.

## 1. Source of truth and provided files

Read the assignment PDF **first**, then inspect both CSV files on disk, before writing implementation code:

- `S1_26_AML_Assignment.pdf` — authoritative 3-page rubric; BITS Pilani WILP, Applied Machine Learning, `S1-26_DSEZG568`, Mobile Price Prediction; 30 marks and 30% weightage.
- `Mobile_Price_Prediction_train.csv`
- `Mobile_Price_Prediction_test.csv`

Search the current workspace for these names if necessary. In environments where they are mounted under `/mnt/data`, use that location; do not assume `/mnt/data` exists in every Codex environment. Do not download or replace the supplied dataset. Document all differences between the PDF and the actual files.

**Already-verified dataset facts to recheck programmatically:** train has 2,000 rows and 22 columns (including `id` and target `price_range`); test has 1,000 rows and 24 columns. The four target classes `0,1,2,3` each occur exactly 500 times in train. The test is UNLABELED: it has no `price_range`. In addition to the expected predictors and `id`, test has irrelevant `Unnamed: 21`, `Unnamed: 22`, `Unnamed: 23` columns; one even contains a whitespace-only entry. Drop `Unnamed:` fields robustly. The five string-valued predictors are `blue`, `dual_sim`, `mobile_wt`, `touch_screen`, `wifi`: four Yes/No columns and `mobile_wt` with `Low`, `Med`, `High`. `id` is an identifier and must never be a predictive feature. No true missing values were observed in the train CSV, but validate this yourself. `ram` has a strong descriptive correlation with target (~0.917); do not mistake correlation for causality or claim this guarantees performance. Some genuine zeros occur in `sc_w`, `px_height`, and camera fields; investigate before interpreting them as missing or dividing by them.

**Nonnegotiable evaluation distinction:** Since the provided test file lacks ground-truth labels, NEVER calculate accuracy, F1, confusion matrix or classification report on it. Create a stratified, held-out labeled validation/test portion from the provided training CSV and report honest metrics on that split. Separately generate `price_range` predictions for ALL 1,000 unlabeled test examples while preserving their `id` and original row order. Explain this resolution clearly in notebook markdown.

## 2. Deliverables and exact submission rules

Create a polished, self-contained, **fully executed Jupyter notebook** named `<GROUP_NO>.ipynb`, and an HTML export named `<GROUP_NO>.html` with charts, tables, results, and visible executed outputs. Name and student details have not been supplied; expose obvious variables at the top such as `GROUP_NO`, `MEMBER_NAMES`, and `GROUP_DETAILS`, and leave clearly marked fields for genuine user input. Do not invent identities. If the group number remains unknown, use `GROUP_NO.ipynb`/`GROUP_NO.html` as conspicuous temporary filenames and clearly tell the user to replace them before submission.

The PDF insists that **only TWO files be uploaded to Taxila, unzipped**: the `.ipynb` and the `.html` (or PDF) export. Produce these in a clearly labeled `submission/` directory; keep all other project artifacts OUTSIDE `submission/`. Also create supporting local development artifacts as needed: `app.py`, `requirements.txt`, serialized model (`joblib`), schema/metadata files, possibly tests, a local demo guide, and unlabeled predictions CSV. Those are **not extra Taxila uploads**. Ensure the notebook contains executable cells which can regenerate the model, app source, and metadata after re-running it, so the two-file academic submission remains meaningful even without submitting extra files. Do not embed opaque serialized binaries in HTML; include the complete app source in notebook cells and recreate artifacts by running the notebook. If genuine public Streamlit deployment requires credentials or external authorization, do not pretend it is live: implement a working locally tested app, supply precise deployment instructions, and identify the one action the user must take.

Structure the notebook in exactly ordered, clearly titled sections matching the rubric, with a professional title page and concise explanations around outputs:

1. Import Libraries / Dataset — 1 mark
2. Data and Data Models — 1 mark
3. Data Visualization and Exploration — 5 marks
4. Data Pre-processing and Cleaning — 5 marks
5. Model Building — 5 marks
6. Performance Evaluation — 2 marks
7. Model Deployment (Streamlit) — 4 marks
8. Novelty and Innovation — 2 marks
9. Presentation and Viva — 5 marks

Keep the notebook readable and credible, with rationale, measured results and interpretation (not a giant code dump). Ensure every rubric criterion is visibly covered and mapped to a notebook section.

## 3. Detailed notebook requirements

### 1–2. Loading, understanding and sanity checks

Import and use `numpy`, `pandas`, `matplotlib`, scikit-learn (and optionally seaborn, imbalanced-learn, joblib/Streamlit). Load the actual CSVs using relative-path-first robust discovery. Show `train.head()`, explicitly identify `price_range`, shapes, a consolidated table of all feature names/types/unique counts/null counts, and exactly TWO complete sanity-check rows (suitable display options so fields are visible). Show train/test schema mismatch and correct it by dropping only spurious `Unnamed:` fields and excluding `id` from predictor matrices. Show `X_train`, `y_train`, and unlabeled `X_test` shapes and assert matching feature sets/order.

### 3. EDA and evidence-based observations

Produce a concise, visually coordinated EDA dashboard. The PDF says to show attributes in **one frame rather than sequentially**, except for separately justified special-feature analysis. Use sensible subplot grids / combined displays rather than 20 separate huge outputs. Include: class-count chart with proportions and explicit assessment of imbalance; data descriptive statistics; univariate numeric histograms and categorical count plots; bivariate boxplots, scatterplots and/or informative color-by-class plots; correlation heatmap of appropriate numeric variables; and price class vs RAM/battery/pixel characteristics. Clearly label axes/units, include graph captions and interpretations. Rank informative candidate features using training-only evidence and explicitly discuss whether nonlinear patterns are actually observed (do not manufacture claims). Point out why `ram` seems dominant while `battery_power` and resolution also warrant testing, subject to actual results. Include a compact actionable EDA-summary table.

### 4. Leakage-safe preprocessing and feature engineering

Audit missingness, duplicates, invalid values, datatypes, skew, outliers, and domain consistency. Explain which values are true anomalies versus legitimate extreme hardware specifications. DO NOT remove rows, fill in zeros, winsorize, log-transform or drop correlated variables blindly. Any fitted imputation, scaling, clipping or feature selection must happen **inside an sklearn/imblearn Pipeline trained only on the relevant training fold**. Use clear justifications and perhaps a small ablation for meaningful transformations.

Handle Yes/No categorical fields as documented binary/label-style encoding (`No=0`, `Yes=1`) and represent `mobile_wt` with One-Hot Encoding or a deliberately justified ordinal treatment; demonstrate and explain both label/binary encoding and one-hot encoding to address the explicit assignment wording. Use `ColumnTransformer` with `handle_unknown='ignore'`, and `StandardScaler` in SVM/linear-sensitive branches while explaining why trees generally do not require scaling. Support numeric native features and zero-safe derived characteristics. Explore credible engineering candidates such as `pixel_area=px_height*px_width`, `screen_area=sc_h*sc_w`, or resolution/size interactions; test whether they improve CV performance, retain only defensible, reproducible additions. Avoid any notebook-defined lambda/local-class pickle portability issues: prefer sklearn-built-in serializable transformers or generated importable module code shared with Streamlit.

Class balancing: the actual labels are **exactly balanced (500 per class)**. Display and explicitly explain why aggressive balancing is not justified for the real training set. Nonetheless address the rubric by a small, reproducible comparison of no balancing, `class_weight='balanced'`, and a justified oversampling approach such as `RandomOverSampler` or mixed-type-safe `SMOTENC` if installed. Any resampling belongs only WITHIN training CV folds; NEVER resample the validation/holdout/test or apply vanilla SMOTE naively to arbitrary categorical encodings. Since equal class counts may produce no sampling effect, say so plainly. Optionally add a clearly labeled controlled minority-class stress test, using ONLY internal training data, to illustrate balancing effects; keep it separate from final model training and avoid claiming artificial imbalance represents the original data. Display a compact metric comparison with interpretation, even if gains are nil.

Feature importance: show at least two complementary methods where defensible (e.g. mutual information/ANOVA, tree impurity-based importance, held-out permutation importance, plus correlation context). Explain their limitations. Calculate held-out permutation importance on the full fitted pipeline using original input columns and `f1_macro`, so the interpretation uses meaningful raw feature names. Do not label correlations or global importances as causal explanations.

### 5. Modeling and scientifically correct selection

Split the supplied labeled TRAIN CSV once into a stratified 80/20 development/untouched holdout split with `random_state=42`, or justify an alternative. Never use the provided unlabeled test CSV to tune a model. Use stratified cross-validation (e.g. 3 or 5 folds) only on the 80% development portion for model selection and affordable hyperparameter search. Show labeled development/holdout shapes and separate unlabeled test shape; avoid ambiguous use of `X_test` for two different datasets (use `X_holdout` and `X_unlabeled_test`).

At minimum implement and compare **Decision Tree, Random Forest, SVM (RBF), and an ensemble (e.g. soft VotingClassifier)** to satisfy BOTH the PDF objective (DT/RF/SVM/Ensemble) and the section requiring at least two appropriate models. Optionally include `DummyClassifier` and Logistic Regression as sanity baselines. Use proper leakage-safe pipelines for every estimator; sensible, bounded, reproducible parameter searches, avoiding extravagant CPU/runtime. Set seeds, show cross-validation mean/std macro-F1 and accuracy, a tuning summary and model-selection justification. Choose the winning model based on DEVELOPMENT CV `f1_macro` (with accuracy as secondary), NOT opportunistically on the holdout. Once selected, evaluate it on untouched holdout, and optionally show the other models' holdout values for transparent comparison **without changing/tuning the selected model based on them**. Document any gap between best CV model and top holdout score.

### 6. Rich, accurate evaluation

Report **accuracy, macro and weighted precision/recall/F1, individual per-class precision/recall/F1/support for 0–3, confusion matrix (raw counts and normalized), and classification report**, with clearly labeled class names and results computed on labeled holdout ONLY. Provide model-comparison table and understandable per-class error discussion. Compare development CV/training vs holdout metrics to discuss underfitting, overfitting, or satisfactory generalization with evidence rather than asserting it. Consider learning curves or training-vs-validation scores if tractable. Where `predict_proba` is available, report sensible probability/confidence diagnostics without confusing confidence with accuracy. Do not tune/re-test repeatedly on holdout. After honest holdout reporting is locked, **refit the selected configuration on ALL 2,000 labeled rows** for deployment and predictions on ALL 1,000 unlabeled rows. State that holdout metrics were measured before this full-data refit, not on the refitted model. Export a local `test_predictions.csv` with columns `id,price_range` and exactly 1,000 rows, but do not submit this CSV to Taxila.

### 7. Actual working Streamlit deployment

Save the **complete fitted preprocessing + classifier pipeline** using `joblib`, plus a consistent input schema and feature defaults/ranges derived from training data (never manually inconsistent preprocessing between notebook and app). Implement `app.py` with an appealing and usable professional UI, grouped specifications (performance, display, camera, connectivity and hardware). It must:

- Load the saved model once with `st.cache_resource`, validate artifact presence and show friendly errors;
- Accept all required raw predictor values through appropriately typed widgets, include dropdowns for `mobile_wt` and the Yes/No variables, and apply safe bounds/validation;
- On submission, construct a dataframe in EXACT trained schema order, call the persisted pipeline, and display predicted `price_range` (0–3) with a clearly explained relative category label (not invented rupee prices);
- Show per-class probabilities and an intuitive bar chart only if the chosen fitted classifier genuinely supports `predict_proba`; never fake probabilities;
- Provide optional batch CSV inference with validation, allowing `id`, removing known `Unnamed:` garbage columns, verifying required predictors, reporting invalid values and downloadable predictions;
- Provide a meaningful **What-If Simulator**: e.g. adjustable RAM and battery, compare the current prediction/probabilities with a modified scenario, clearly noting the association is predictive, not causal;
- Include accessible labels and restrained styling rather than merely decorative complexity.

Make notebook execution regenerate `app.py`, model files and necessary metadata in the correct locations, and include an explanatory deployment section with the actual command `streamlit run app.py`, a brief demo workflow, and honest evidence of local validation. If feasible, smoke-test app import/startup and use a browser test or Streamlit testing APIs for one prediction. Do NOT claim a live public deployment without one.

### 8. Novelty and innovation, explicitly demonstrated

Earn the 2 innovation marks through a polished, coherent UI AND at least one substantive addition that is genuinely implemented, tested and explained: e.g. what-if comparison, batch scoring, train-time feature-importance insight, error analysis, confidence distributions, or an evidence-backed feature engineering ablation. Novelty must add clear user/technical value, not merely a claim that the project is innovative.

### 9. Presentation and viva preparedness

Add a one-page executive summary/table covering objective, dataset shape/cleaning, final model, CV vs holdout metrics, top drivers, error patterns, limitations, why balancing/scaling were handled as they were, Streamlit instructions, and concise conclusions. Add 8–12 likely viva questions with short correct answers about encoding, missing labels in test, `id`, leakage, stratification, macro-F1, SMOTE, SVM scaling, ensembles, overfitting, persistence and Streamlit. Add a 2–3 minute demo narrative and a short limitations/future-work section. Make the narrative consistent with measured code output and avoid unsupported commercial/pricing claims.

## 4. Quality, runtime and integrity constraints

- Develop against the real supplied files, not a simulated or standard Kaggle variant. Never sneak `id` or the actual target into predictors, nor peek at unlabeled test labels.
- Use reproducible random seeds (`42`), version-aware sklearn APIs, concise comments, readable functions, no deprecated warnings, no error traces, no massive unformatted output, and no hidden cells containing indispensable logic.
- Keep plots consolidated, adequately sized, legible and titled. Prefer `matplotlib`/seaborn so embedded notebook/HTML outputs render offline. Include explanatory Markdown after substantive figures/tables.
- A notebook **without executed outputs will not be evaluated**. Programmatically execute it from a clean kernel using `nbclient` or `jupyter nbconvert --execute`; fix all errors and verify every model metric/chart/important table is visible. Export HTML from the **executed** notebook (prefer an offline/self-contained export), and verify it includes the same output. Do not simply save a notebook containing source code only.
- Add manageable unit/integration smoke tests: expected input schema, class count, train/test alignment, `id` exclusion, spurious-column removal, no fitted transformation outside CV, pipeline persistence/reload prediction equivalence, full 1,000 test predictions, probability rows sum to ~1 when applicable, and Streamlit prediction logic. Fix failures before declaring done.
- Keep parameter tuning moderate, and prefer correctness to artificially inflated accuracy. If optional packages are unavailable, install them if environment permits or implement a simpler sound alternative and document the deviation.
- Provide a `requirements.txt` or equivalent reproducibility details for LOCAL working directory, NOT as an extra Taxila submission file. Make filenames and paths robust across working directories.
- No unverified fabricated scores, charts, model interpretations, UI screenshots or public URLs. If something cannot run in this environment, state exactly what was not verified and produce the maximum functional artifact possible.

## 5. Final acceptance checklist and response format

Before finishing, inspect the assignment PDF one more time and prepare a **rubric-compliance checklist** showing each 30-mark component, exactly where it appears in notebook, and its verification status. Confirm that:

1. The only files in `submission/` are `<GROUP_NO>.ipynb` and `<GROUP_NO>.html` (or the clearly marked temporary placeholders pending group metadata).
2. Both files open, HTML has all real outputs, notebook has successfully executed top-to-bottom, and no ugly errors/needless logs appear.
3. Every 0–3 class has support and per-class metrics shown; confusion matrix/accuracy come from a truly labeled holdout rather than the unlabeled test.
4. Four required classifier families are tested and compared, selected with CV; test predictions contain 1,000 correctly ordered IDs.
5. The saved model loads in the Streamlit app and at least one example inference works; app and training use identical schema/preprocessing.
6. EDA includes consolidated attribute graphs, imbalance assessment, uni-/bivariate plots, correlation and evidence-based observations.
7. Preprocessing, scaling, encoding, balancing, engineered features and importance all have explicit justifications.
8. Innovation and a presentation/viva-ready conclusion are demonstrably present.
9. Any still-unknown group/member metadata and any genuinely required external public-hosting action are clearly flagged—not fabricated.

**Work autonomously until the deliverables and tests are complete.** Do not stop after an outline or partial code; write, execute, inspect, repair and verify. In your final response, report files/paths, model comparison and honestly measured holdout metrics, app run command, compliance checklist, any limitations, and precisely what the student must customize/upload. Optimize for a defensible full-marks submission, not for making exaggerated claims.
