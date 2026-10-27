"""Generate the self-contained academic notebook from readable local sources."""
from pathlib import Path
import textwrap
import nbformat as nbf

ROOT = Path(__file__).resolve().parent
cells = []


def md(text):
    cells.append(nbf.v4.new_markdown_cell(textwrap.dedent(text).strip()))


def code(text):
    cells.append(nbf.v4.new_code_cell(textwrap.dedent(text).strip()))


def source_file(relative):
    content = (ROOT / relative).read_text()
    # Literal, readable source is carried in the notebook, not an external link.
    delimiter = "'''" if "'''" not in content else '"""'
    if delimiter in content:
        raise ValueError(f'Choose a safe literal delimiter for {relative}')
    code(f"# Regenerate {relative}; source is part of this two-file submission.\n"
         f"source = r{delimiter}{content}{delimiter}\n"
         f"destination = ROOT / {relative!r}\n"
         "destination.parent.mkdir(parents=True, exist_ok=True)\n"
         "destination.write_text(source, encoding='utf-8')\n"
         f"print('Recreated {relative}')")


md("""
<div style="background:#122840;color:#fff;padding:32px;border-radius:14px">
<p style="color:#6ee7cf;letter-spacing:2px">BITS PILANI · WORK INTEGRATED LEARNING PROGRAMMES</p>
<h1 style="color:white">Mobile Price Prediction</h1>
<p style="font-size:19px">Technical specifications → four relative price categories</p>
<p>Applied Machine Learning · S1-26_DSEZG568 · Assignment weightage 30%</p>
<p><b>Group: GROUP_NO — REPLACE BEFORE SUBMISSION</b><br>
Members / student IDs: TO BE COMPLETED BY THE GROUP (up to three members)</p>
</div>

**Objective.** Compare multiclass classifiers, explain the evidence, and deliver a working
Streamlit inference application using the exact supplied Mobile Price Prediction CSVs.
The assignment PDF is the authoritative rubric. The nine sections below cover 30 marks.

**Reading guide.** Sections 1–4 establish the data and leakage controls; sections 5–6 contain
the measured experiments and evaluation; sections 7–9 demonstrate deployment, innovation,
and the presentation/viva brief. All reported model scores come from executed cells.

**Submission contract.** Upload only `GROUP_NO.ipynb` and `GROUP_NO.html` to Taxila, unzipped,
after replacing group/member placeholders and exporting again. Supporting CSV predictions,
source, tests, model artifacts and this project's README are local development files.
The two submitted files include readable source that recreates those artifacts. The original
assignment CSVs and the listed Python dependencies remain required to rerun the notebook.
""")

md("""
## 1. Import Libraries / Dataset — 1 mark

Use the supplied files; no substitute dataset is downloaded. Run all cells in order in a
Python environment containing the dependencies shown below. Paths are discovered relative
to the working directory and its parents, so the notebook also works from `submission/`.
Set `AML_DATA_DIR` only if the original CSVs live elsewhere. Source regeneration is deliberate:
the notebook recreates the project modules and app alongside the dataset.
""")
code("""
GROUP_NO = 'GROUP_NO'  # REQUIRED: replace with your genuine group number
MEMBER_NAMES = ['REPLACE: member name(s)']
GROUP_DETAILS = {'student_ids': ['REPLACE: student ID(s)'], 'programme': 'BITS Pilani WILP'}
SEED = 42

import os, sys, json, hashlib, platform, subprocess, time, textwrap, importlib
from pathlib import Path
os.environ.setdefault('OMP_NUM_THREADS', '2')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '2')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn, joblib
get_ipython().run_line_magic('matplotlib', 'inline')
from IPython.display import display, Markdown, HTML
from sklearn.base import clone
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import mutual_info_classif
from sklearn.inspection import permutation_importance
from sklearn.metrics import (accuracy_score, precision_recall_fscore_support,
    classification_report, confusion_matrix, f1_score, log_loss)

def discover_root():
    candidates = ([Path(os.environ['AML_DATA_DIR']).expanduser()] if 'AML_DATA_DIR' in os.environ else [])
    candidates += [Path.cwd(), *Path.cwd().parents, Path('/mnt/data')]
    for candidate in candidates:
        if all((candidate / f'Mobile_Price_Prediction_{split}.csv').is_file() for split in ['train', 'test']):
            return candidate.resolve()
    raise FileNotFoundError('Place both original CSVs alongside the project, or set AML_DATA_DIR.')

ROOT = discover_root()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
for directory in ['artifacts', 'reports', 'tests', 'submission']:
    (ROOT / directory).mkdir(exist_ok=True)
pd.set_option('display.max_columns', None)
pd.set_option('display.max_colwidth', 100)
pd.set_option('display.precision', 4)
sns.set_theme(style='whitegrid', palette=['#137c8b', '#e5a53e', '#6f63a8', '#d76c65'])
plt.rcParams.update({'figure.dpi': 110, 'axes.spines.top': False, 'axes.spines.right': False})
PALETTE = ['#137c8b', '#e5a53e', '#6f63a8', '#d76c65']
display(HTML('<style>.dataframe{font-size:12px} .dataframe th{background:#edf5f8} h2{color:#137c8b}</style>'))
display(pd.DataFrame({'group':[GROUP_NO], 'members':[', '.join(MEMBER_NAMES)], 'details':[str(GROUP_DETAILS)]}))

train = pd.read_csv(ROOT / 'Mobile_Price_Prediction_train.csv')
test_raw = pd.read_csv(ROOT / 'Mobile_Price_Prediction_test.csv')
assert 'price_range' in train and 'price_range' not in test_raw
assert train.shape == (2000, 22) and test_raw.shape == (1000, 24)
print('Target: price_range (four classes 0, 1, 2, 3). Original training records:')
display(train.head())
""")
source_file('requirements.txt')
code("""
from importlib.metadata import version
packages = ['numpy','pandas','matplotlib','seaborn','scikit-learn','imbalanced-learn',
            'joblib','streamlit','nbformat','nbclient','nbconvert','ipykernel','pytest']
versions = {package: version(package) for package in packages}
display(pd.DataFrame(versions.items(), columns=['package', 'executed_version']))
print('Python:', platform.python_version(), '| Seed:', SEED)
hashes = {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
          for name in ['Mobile_Price_Prediction_train.csv','Mobile_Price_Prediction_test.csv']}
_ = (ROOT / 'reports/reproducibility.json').write_text(json.dumps({'python':platform.python_version(),
    'packages':versions, 'input_sha256':hashes, 'seed':SEED}, indent=2))
""")

md("""
## 2. Data and Data Models — 1 mark

The train CSV has **2,000 × 22** fields including identifier and label; the test CSV has
**1,000 × 24** fields and **no label**. Its three blank CSV headers become pandas-generated
`Unnamed: 21`, `Unnamed: 22`, `Unnamed: 23`; one cell contains a single space.
Only these export artifacts are removed. `id` is retained for tracking and excluded from
predictors. The real files contain five string predictors, unlike some numeric versions
of this dataset: four Yes/No variables and `mobile_wt` = Low/Med/High.

**Resolution of rubric ambiguity.** The PDF requests test accuracy but supplies no test
ground truth. Here `X_test` means the supplied **unlabeled** features, while `X_holdout`
means the labeled 20% evaluation partition. No test accuracy is invented. We introduce
the split now so all subsequent plots, rankings, preprocessing choices and CV use only
the 80% development partition. Full-data schema checks and label counts establish the
task; holdout feature/label associations stay untouched until section 6.
""")
source_file('mobile_pipeline.py')
code("""
importlib.invalidate_caches()
from mobile_pipeline import (FEATURES, NUMERIC, BINARY, CATEGORICAL, CLASS_LABELS,
    clean_columns, HardwareFeatures, make_preprocessor, build_schema, validate_inputs, predict_batch)
shape_table = pd.DataFrame({'file':['train','unlabeled test'], 'rows':[len(train),len(test_raw)],
    'columns':[train.shape[1],test_raw.shape[1]], 'target_available':[True,False]})
display(shape_table)
schema_audit = pd.DataFrame({'feature':train.columns, 'dtype':train.dtypes.astype(str).values,
    'unique_values':train.nunique(dropna=False).values, 'null_count':train.isna().sum().values})
display(schema_audit)
print('Exactly TWO complete sanity-check rows (scroll horizontally if needed):')
display(train.iloc[[0,1]].style.set_table_attributes('style="font-size:11px;white-space:nowrap"'))
spurious = test_raw.columns[test_raw.columns.str.match(r'^Unnamed:')].tolist()
display(pd.DataFrame({'test_only_field':spurious,
    'nulls':[int(test_raw[c].isna().sum()) for c in spurious],
    'whitespace_only':[int(test_raw[c].astype('string').str.fullmatch(r'\\s+').fillna(False).sum()) for c in spurious]}))
clean_test = clean_columns(test_raw)
X_train = train.drop(columns=['id','price_range'])  # all labeled predictors, before split
y_train = train['price_range']
X_test = clean_test.drop(columns='id')             # PDF naming; NO labels available
X_unlabeled_test = X_test
assert X_train.columns.tolist() == X_test.columns.tolist() == FEATURES
assert not set(['id','price_range']) & set(FEATURES)
X_dev, X_holdout, y_dev, y_holdout = train_test_split(
    X_train, y_train, test_size=0.20, stratify=y_train, random_state=SEED)
display(pd.DataFrame({'object':['X_train (all labeled)','y_train','X_test / X_unlabeled_test',
    'X_dev','y_dev','X_holdout','y_holdout'], 'shape':[X_train.shape,y_train.shape,X_test.shape,
    X_dev.shape,y_dev.shape,X_holdout.shape,y_holdout.shape]}))
assert len(set(X_dev.index) & set(X_holdout.index)) == 0
split_manifest = {'seed':SEED, 'development_ids':train.loc[X_dev.index,'id'].tolist(),
    'holdout_ids':train.loc[X_holdout.index,'id'].tolist()}
_ = (ROOT / 'reports/split_manifest.json').write_text(json.dumps(split_manifest, indent=2))
""")

md("""
## 3. Data Visualization and Exploration — 5 marks

The following dashboards consolidate related attributes into single frames as requested.
EDA uses the 1,600 development rows; the class-count panel also documents the complete
labeled dataset's balance. Units follow the variable names/conventional dataset meaning:
RAM MB, battery mAh, cameras MP, resolution pixels, screen dimensions cm, depth cm,
clock GHz, storage GB and talk time hours. No supplier data dictionary is provided;
these units are descriptive assumptions, and `mobile_wt` is a categorical weight band.
""")
code("""
fig, axes = plt.subplots(1, 2, figsize=(11, 3.7))
for ax, target, title in [(axes[0],y_train,'All labeled data'),(axes[1],y_dev,'Development data')]:
    counts = target.value_counts().sort_index()
    ax.bar([f'{i} · {CLASS_LABELS[str(i)]}' for i in counts.index], counts.values, color=PALETTE)
    for i, n in enumerate(counts): ax.text(i, n + 6, f'{n} ({n/len(target):.0%})', ha='center')
    ax.set(title=title, ylabel='Phones', xlabel='Relative price class', ylim=(0,counts.max()*1.18))
fig.suptitle('Exactly balanced in the supplied data: each class has 25% support', fontweight='bold')
fig.tight_layout(); plt.show()
display(X_dev.describe().T.assign(skew=X_dev[NUMERIC].skew()).round(3))
display(X_dev[CATEGORICAL].describe().T)
""")
md("""
**Interpretation.** All four labels have 500 rows, of which 400 are in development and 100
in holdout. Class balancing is not indicated by these counts. Macro-F1 still provides a
clear equal-class selection criterion and penalizes uneven precision/recall. Summary
statistics above describe this data, rather than an assumed current phone market.
""")
code("""
units = {'battery_power':'mAh','clock_speed':'GHz','fc':'MP','int_memory':'GB','m_dep':'cm',
         'pc':'MP','px_height':'pixels','px_width':'pixels','ram':'MB','sc_h':'cm','sc_w':'cm','talk_time':'hours'}
fig, axes = plt.subplots(4, 5, figsize=(17, 12))
for ax, name in zip(axes.flat, FEATURES):
    if name in CATEGORICAL or name in ['four_g','three_g','n_cores']:
        order = (['Low','Med','High'] if name == 'mobile_wt' else ['No','Yes']) if name in CATEGORICAL else sorted(X_dev[name].unique())
        counts = X_dev[name].value_counts().reindex(order,fill_value=0)
        ax.bar(counts.index.astype(str), counts.values, color='#137c8b')
    else:
        ax.hist(X_dev[name], bins=20, color='#137c8b', alpha=.85)
    ax.set(title=name, xlabel=units.get(name,'category / count'), ylabel='Phones')
fig.suptitle('All 20 raw predictors in one frame · development data only', fontsize=17, fontweight='bold')
fig.tight_layout(rect=[0,0,1,.97]); plt.show()
""")
md("""
**Interpretation.** The histogram and count grids expose finite hardware ranges and
discrete connectivity/weight levels. Zeros occur in camera, width and resolution fields;
section 4 audits them explicitly. Shape alone is not a reason to transform a predictor:
scaling helps distance-based models, whereas standardization does not make a skewed
distribution Gaussian.
""")
code("""
eda = X_dev.assign(price_range=y_dev)
fig, axes = plt.subplots(2, 3, figsize=(15, 8))
for ax, name in zip(axes[0], ['ram','battery_power','px_height']):
    sns.boxplot(data=eda,x='price_range',y=name,hue='price_range',palette=PALETTE,legend=False,ax=ax)
    ax.set(xlabel='Price class',ylabel=f'{name} ({units[name]})',title=f'{name} across classes')
for ax, (a,b) in zip(axes[1], [('ram','battery_power'),('ram','px_width'),('px_height','px_width')]):
    sns.scatterplot(data=eda,x=a,y=b,hue='price_range',palette=PALETTE,s=15,alpha=.55,ax=ax,legend=(a=='ram' and b=='battery_power'))
    ax.set(xlabel=f'{a} ({units[a]})',ylabel=f'{b} ({units[b]})',title=f'{a} × {b}')
fig.suptitle('Price class vs memory, battery and pixel characteristics',fontweight='bold')
fig.tight_layout(); plt.show()
numeric_corr = eda[NUMERIC + ['price_range']].corr()
fig, ax = plt.subplots(figsize=(13, 10))
sns.heatmap(numeric_corr, cmap='RdBu_r',center=0,vmin=-1,vmax=1,annot=True,fmt='.2f',
            annot_kws={'fontsize':7},square=True,ax=ax,cbar_kws={'shrink':.7})
ax.set_title('Pearson correlation · numeric development predictors and ordinal target',pad=15)
fig.tight_layout(); plt.show()
display(Markdown(f'**Measured association:** development RAM/target Pearson r = **{numeric_corr.loc["ram","price_range"]:.3f}**. '
    'The numeric target represents an ordered class, so this is descriptive association, not causality or an interval-price model. '
    'RAM shifts visibly by class, but overlap shows that a single threshold rule is insufficient. '
    'Battery and pixel resolution are plausible additional predictors. These plots do not establish a specific nonlinear law; '
    'the RBF SVM, trees and area ablation test whether nonlinear flexibility is useful.'))
""")
code("""
# A deterministic representation for training-only descriptive MI; never reused to fit models.
mi_input = X_dev.copy()
for name in BINARY: mi_input[name] = mi_input[name].map({'No':0,'Yes':1})
mi_input['mobile_wt'] = mi_input['mobile_wt'].map({'Low':0,'Med':1,'High':2})
discrete = [name in CATEGORICAL + ['four_g','three_g','n_cores'] for name in FEATURES]
mi_values = mutual_info_classif(mi_input,y_dev,discrete_features=discrete,random_state=SEED)
mi_table = pd.DataFrame({'feature':FEATURES,'mutual_information':mi_values}).sort_values('mutual_information',ascending=False)
fig, ax = plt.subplots(figsize=(10,4))
sns.barplot(data=mi_table.head(10),x='mutual_information',y='feature',color='#137c8b',ax=ax)
ax.set(title='Candidate feature ranking · training-only mutual information',xlabel='Estimated mutual information (nats)',ylabel='Raw predictor')
fig.tight_layout(); plt.show()
display(mi_table.head(8).reset_index(drop=True))
eda_summary = pd.DataFrame([
 ['Class counts','500 each / 400 each in development','No aggressive resampling; compare controls in CV'],
 ['RAM',f'Pearson r={numeric_corr.loc["ram","price_range"]:.3f}; top MI features: '+', '.join(mi_table.head(3).feature),'Retain, scale for SVM/LR'],
 ['Battery / pixels','Class overlap and multivariate variation','Test jointly in classifiers; area ablation'],
 ['Categoricals','Four Yes/No predictors and three weight bands','Binary encoding and one-hot weight bands'],
 ['Nonlinearity','Visual evidence alone is inconclusive','Compare RBF/tree families with linear baseline'],
 ['Zeros / extrema','Recorded finite values; semantics partly undocumented','Audit, retain, avoid unsafe ratios'],
 ],columns=['Finding','Evidence','Action'])
display(eda_summary)
""")
md("""
**Ranking limitations.** Mutual information estimates dependence (including nonlinear
dependence), but finite-sample noise and correlated predictors affect the ranking.
It is a candidate guide, not a causal explanation or a fitted feature-selection filter.
We keep all raw predictors and compare tree importance and full-pipeline permutation
importance after training. No feature is selected using the holdout.
""")

md("""
## 4. Data Pre-processing and Cleaning — 5 marks

**Decisions before fitting.** Keep the original rows, genuine extreme specifications and
correlated resolution dimensions. Do not replace zeros, clip outliers or log-transform
solely because an IQR flag or skew exists. Camera zeros can plausibly indicate absence;
zero screen width and pixel height are physically questionable but their missing-value
semantics are undocumented. Record this limitation and preserve them consistently.
`pixel_area = px_height × px_width` and `screen_area = sc_h × sc_w` are zero-safe
engineering candidates; they are tested together, not automatically retained.

All learned imputation/scaling stays inside each estimator's `Pipeline`/`ColumnTransformer`,
fitted on the corresponding CV training fold. Median and mode imputation are defensive
because these supplied predictors have no missing values. Binary encoding is explicitly
`No=0, Yes=1` (a label-style mapping), while weight bands use one-hot encoding to avoid
an arbitrary numerical distance. `handle_unknown='ignore'` protects weight encoding;
the app performs stricter validation so malformed uploads receive a useful error.

StandardScaler applies to numeric features for Logistic Regression and SVM because
distance and regularization depend on scale. Tree splits do not require standardization.
One-hot/binary values remain 0/1. Min-max normalization, winsorization and logarithms
are not used without an experimentally supported need.
""")
code("""
audit = pd.DataFrame({
 'check':['True missing cells (all train)','Missing raw test predictors','Duplicate complete training rows',
          'Duplicate training predictor rows','Duplicate development predictor rows','Duplicate train IDs',
          'Negative numeric values (development)','Front camera exceeds primary (development)',
          '4G enabled without 3G (development)','Screen width exceeds height (development)'],
 'count':[int(train.isna().sum().sum()),int(X_test.isna().sum().sum()),int(train.duplicated().sum()),
          int(X_train.duplicated().sum()),int(X_dev.duplicated().sum()),int(train.id.duplicated().sum()),
          int((X_dev[NUMERIC]<0).sum().sum()),int((X_dev.fc>X_dev.pc).sum()),
          int(((X_dev.four_g==1)&(X_dev.three_g==0)).sum()),int((X_dev.sc_w>X_dev.sc_h).sum())]})
display(audit)
zero_audit = pd.DataFrame({'field':['sc_w','px_height','fc','pc'],
    'train_zero_count':[int((X_train[c]==0).sum()) for c in ['sc_w','px_height','fc','pc']],
    'development_zero_count':[int((X_dev[c]==0).sum()) for c in ['sc_w','px_height','fc','pc']],
    'test_zero_count':[int((X_test[c]==0).sum()) for c in ['sc_w','px_height','fc','pc']]})
display(zero_audit)
q1,q3 = X_dev[NUMERIC].quantile(.25),X_dev[NUMERIC].quantile(.75)
iqr = q3-q1
outlier_audit = pd.DataFrame({'skew':X_dev[NUMERIC].skew(),
    'IQR_flag_count':((X_dev[NUMERIC] < q1-1.5*iqr) | (X_dev[NUMERIC] > q3+1.5*iqr)).sum(),
    'action':'Retain; no documented invalid value'})
display(outlier_audit)
display(pd.DataFrame({'raw_binary':['No','Yes'],'label_style_code':[0,1]}))
display(pd.DataFrame({'weight_band':['Low','Med','High'], 'weight_Low':[1,0,0],
    'weight_Med':[0,1,0],'weight_High':[0,0,1]}))
display(pd.DataFrame({'operation':['Drop Unnamed columns','Exclude id','Impute median/mode','Encode categories',
    'Standardize numeric','Area products','Balance classes'],
    'fit_boundary':['Deterministic schema rule','Deterministic role rule','CV training fold','CV training fold',
                    'CV training fold (LR/SVM)','Deterministic; choice by development CV','CV training fold only']}))
""")
md("""
**Balancing comparison and engineering acceptance rule.** Section 5 executes all CV
experiments in one reproducible function, with results shown together below. Compare
no balancing, balanced class weights and `RandomOverSampler` inside CV. With five
stratified folds, each training fold has exactly 320 observations per class: balanced
weights equal 1 and oversampling should add zero rows. Oversampling is a transparent
mixed-type-safe control because it duplicates observations, rather than interpolating
fractional categorical codes. Contemporary `SMOTENC` supports continuous/nominal
mixtures, but synthetic data are unjustified for these already balanced classes.

The area ablation holds each model's tuned base hyperparameters fixed. Retain the two
areas only if mean CV macro-F1 improves by more than **0.002** (a predeclared simplicity
rule, not a significance test). All choices use development data. Five-fold means/stds
are descriptive; non-nested tuning can make CV performance optimistic, so the untouched
holdout is the final generalization check.

Method references: [scikit-learn leakage guidance](https://scikit-learn.org/stable/common_pitfalls.html),
[imbalanced-learn leakage guidance](https://imbalanced-learn.org/stable/common_pitfalls.html),
[RandomOverSampler](https://imbalanced-learn.org/stable/references/generated/imblearn.over_sampling.RandomOverSampler.html),
[SMOTENC](https://imbalanced-learn.org/stable/references/generated/imblearn.over_sampling.SMOTENC.html).
Documentation checked on 22 October 2026; package versions are recorded above.
""")

md("""
## 5. Model Building — 5 marks

**Model rationale.** Decision Tree supplies interpretable nonlinear partitions; Random
Forest averages trees to reduce variance; RBF SVM tests smooth nonlinear boundaries;
soft voting combines LR/RF/SVM probabilities. Logistic Regression supplies a strong
linear decision-boundary baseline; the most-frequent Dummy exposes chance-level behavior.
This covers all four model families in the PDF objective, beyond the minimum two.

SVM probabilities use `CalibratedClassifierCV` with sigmoid calibration and five seeded
inner folds (`ensemble=False`). It wraps the **entire raw-input SVM pipeline**, so even
calibration-fold preprocessing is fitted only on its training rows. This uses the current
API rather than scikit-learn 1.9's deprecated `SVC(probability=True)` option. Inner
calibration fits add runtime, but neither calibration nor its parameters use the holdout.

Searches use the same five shuffled stratified development folds (`random_state=42`),
macro-F1 as the primary metric, accuracy only to break exact ties, and two workers.
Base grids have 28 configurations, plus two voting weight choices and two area variants.
Voting member configurations are chosen using development CV, then each member's full
pipeline is refitted inside the voting folds. These repeated choices are not nested CV;
their selection optimism is addressed by the holdout, not by interpreting fold std as a CI.
No code in the experiment function accepts holdout or unlabeled test data.
""")
source_file('model_experiments.py')
code("""
importlib.invalidate_caches()
from model_experiments import run_experiments
experiments = run_experiments(X_dev,y_dev)
cv_table = experiments['cv_table']
selected_name = experiments['selected_name']
selected = experiments['selected_model']
display(cv_table[['model','cv_macro_f1','cv_macro_f1_std','cv_accuracy','cv_accuracy_std',
                  'cv_train_macro_f1','engineered_areas']].round(4))
display(experiments['tuning_table'])
print(f'CV-selected model: {selected_name}. Experiment runtime: {experiments["elapsed_seconds"]:.1f} seconds.')
for key, name in [('cv_table','model_comparison'),('tuning_table','tuning'),
    ('feature_ablation','feature_ablation'),('feature_ablation_folds','feature_ablation_folds'),
    ('balancing_table','balancing')]:
    experiments[key].to_csv(ROOT / f'reports/{name}.csv',index=False)
_ = joblib.dump(experiments['models'],ROOT / 'artifacts/development_models.joblib')
""")
code("""
display(experiments['feature_ablation'].round(4))
display(experiments['balancing_table'][['strategy','cv_macro_f1','cv_macro_f1_std','cv_accuracy',
                                      'mean_added_rows_per_training_fold']].round(4))
retained = experiments['feature_ablation'].query('retained == True')
area_notes = '; '.join(f'{row.model}: {row.feature_variant}' for row in retained.itertuples())
balance = experiments['balancing_table']
balance_gap = float(balance.cv_macro_f1.max()-balance.cv_macro_f1.min())
display(Markdown(f'**Measured engineering decision:** {area_notes}. '
    f'Balancing macro-F1 spread = **{balance_gap:.6f}**; all oversampling fold additions = '
    f'**{balance.mean_added_rows_per_training_fold.max():.0f}** rows. '
    'The balanced original class counts justify retaining the natural data distribution. '
    'A nil gain is a useful negative result, not a reason to manufacture imbalance.'))
fig, ax = plt.subplots(figsize=(10,4.5))
rows = cv_table.sort_values('cv_macro_f1')
ax.barh(rows.model,rows.cv_macro_f1,xerr=rows.cv_macro_f1_std,color='#137c8b',capsize=4)
ax.set(xlim=(0,1.04),xlabel='Development 5-fold macro-F1 (mean ± fold std)',
       title='Selection evidence · holdout has not been evaluated')
fig.tight_layout(); plt.show()
rf = experiments['models']['Random Forest']
transformed_names = rf.named_steps['preprocess'].get_feature_names_out()
tree_importance = pd.DataFrame({'encoded_feature':transformed_names,
    'impurity_importance':rf.named_steps['model'].feature_importances_}).sort_values('impurity_importance',ascending=False)
display(tree_importance.head(10).reset_index(drop=True))
display(Markdown(f'**Selection is now locked:** {selected_name}, with development CV macro-F1 '
    f'**{cv_table.iloc[0].cv_macro_f1:.4f} ± {cv_table.iloc[0].cv_macro_f1_std:.4f}**. '
    'Impurity importance is training-only and can favor continuous/high-cardinality variables or split credit '
    'between correlated features. It complements MI; neither is causal. '
    'The next section evaluates this fixed choice once and will not alter it.'))
""")

md("""
## 6. Performance Evaluation — 2 marks

All metrics in this section use the **400 labeled holdout rows** and the selected pipeline
fitted on **1,600 development rows**. The separate 1,000-row test file has no ground truth.
Macro-F1 is the primary selection metric because every class matters equally; accuracy
is an intuitive secondary measure, and weighted scores and per-class support make the
comparison complete. With equal support, weighted and macro averages coincide here.
""")
code("""
y_pred = selected.predict(X_holdout)
probabilities = selected.predict_proba(X_holdout)
metrics = {'accuracy':accuracy_score(y_holdout,y_pred)}
for average in ['macro','weighted']:
    p,r,f,_ = precision_recall_fscore_support(y_holdout,y_pred,average=average,zero_division=0)
    metrics.update({f'{average}_precision':p,f'{average}_recall':r,f'{average}_f1':f})
display(pd.DataFrame([metrics]).round(4))
labels = [0,1,2,3]
names = [f'{i} · {CLASS_LABELS[str(i)]}' for i in labels]
report = classification_report(y_holdout,y_pred,labels=labels,target_names=names,output_dict=True,zero_division=0)
class_report = pd.DataFrame({name: values for name, values in report.items() if name != 'accuracy'}).T
display(class_report.round(4))
cm = confusion_matrix(y_holdout,y_pred,labels=labels)
fig, axes = plt.subplots(1,2,figsize=(12,4.7))
sns.heatmap(cm,annot=True,fmt='d',cmap='Blues',cbar=False,xticklabels=names,yticklabels=names,ax=axes[0])
sns.heatmap(cm/cm.sum(axis=1,keepdims=True),annot=True,fmt='.1%',cmap='Blues',vmin=0,vmax=1,
            cbar=False,xticklabels=names,yticklabels=names,ax=axes[1])
for ax,title in zip(axes,['Raw confusion counts','Row-normalized recall by true class']):
    ax.set(xlabel='Predicted class',ylabel='True class',title=title)
fig.tight_layout(); plt.show()
# Wilson interval for accuracy, conditional on this fixed holdout; not a CV interval.
n = len(y_holdout); acc = metrics['accuracy']; z = 1.96
denom = 1+z*z/n
center = (acc+z*z/(2*n))/denom
half = z*np.sqrt(acc*(1-acc)/n+z*z/(4*n*n))/denom
accuracy_interval = [float(center-half),float(center+half)]
train_f1 = f1_score(y_dev,selected.predict(X_dev),average='macro')
display(pd.DataFrame({'evaluation':['Development fit (resubstitution)','Development CV mean','Untouched holdout'],
    'macro_F1':[train_f1,cv_table.iloc[0].cv_macro_f1,metrics['macro_f1']]}).round(4))
error_pairs = pd.DataFrame([{'true_class':i,'predicted_class':j,'count':int(cm[i,j])}
    for i in labels for j in labels if i!=j and cm[i,j]>0],
    columns=['true_class','predicted_class','count']).sort_values('count',ascending=False)
display(error_pairs)
errors = np.asarray(y_holdout)!=y_pred
adjacent_errors = int((np.abs(np.asarray(y_holdout)-y_pred)==1).sum())
worst = min(names,key=lambda name:report[name]['f1-score'])
gap = train_f1-metrics['macro_f1']
display(Markdown(f'**Error discussion.** {int(errors.sum())}/{n} holdout phones are misclassified; '
    f'{adjacent_errors} errors are between adjacent classes. The lowest class F1 is '
    f'**{worst}: {report[worst]["f1-score"]:.4f}**. Accuracy Wilson 95% interval is '
    f'**[{accuracy_interval[0]:.3f}, {accuracy_interval[1]:.3f}]**. '
    f'The resubstitution-to-holdout macro-F1 gap is **{gap:.4f}**; compare this with the CV mean and variability above. '
    + ('The modest gap and strong holdout score support satisfactory generalization on this split, without proving performance on new market data.'
       if gap<.05 and metrics['macro_f1']>.85 else
       'The measured gap warrants caution about variance or limited signal; this single holdout cannot establish a universal fit diagnosis.') ))
""")
md("""
**Probability diagnostics.** These are model probabilities, not guaranteed correctness
or monetary estimates. The multiclass Brier score below is the mean **sum** of squared
probability errors across four classes (range 0–2). Log loss penalizes confident mistakes.
Reliability bins compare maximum predicted probability with empirical correctness of
the probability-argmax class. Our SVM uses explicit inner cross-validated calibration;
the calibrated classifier predicts by probability argmax. The generic diagnostic still
counts any `predict`/probability-argmax disagreement for the actually selected pipeline.
[Official calibration guidance](https://scikit-learn.org/stable/modules/calibration.html).
""")
code("""
prob_pred = selected.classes_[probabilities.argmax(axis=1)]
confidence = probabilities.max(axis=1)
prob_correct = (prob_pred==np.asarray(y_holdout)).astype(int)
onehot = np.eye(4)[np.asarray(y_holdout)]
prob_metrics = {'log_loss':log_loss(y_holdout,probabilities,labels=labels),
    'multiclass_Brier_sum':float(np.mean(np.sum((probabilities-onehot)**2,axis=1))),
    'mean_max_probability':float(confidence.mean()),
    'predict_vs_proba_argmax_disagreements':int((y_pred!=prob_pred).sum())}
display(pd.DataFrame([prob_metrics]).round(4))
reliability_data = pd.DataFrame({'confidence':confidence,'correct':prob_correct})
reliability_data['bin'] = pd.cut(confidence,bins=np.linspace(0,1,6),include_lowest=True)
reliability = reliability_data.groupby('bin',observed=True).agg(mean_probability=('confidence','mean'),
    empirical_accuracy=('correct','mean'),count=('correct','size'))
display(reliability.round(4))
fig,axes = plt.subplots(1,2,figsize=(11,3.8))
axes[0].hist([confidence[prob_correct==1],confidence[prob_correct==0]],bins=np.linspace(0,1,11),
             label=['Correct probability argmax','Incorrect probability argmax'],color=['#137c8b','#d76c65'],stacked=True)
axes[0].set(xlabel='Maximum probability',ylabel='Holdout phones',title='Confidence distribution'); axes[0].legend(fontsize=8)
axes[1].plot([0,1],[0,1],'--',color='gray',label='Perfect calibration')
axes[1].plot(reliability.mean_probability,reliability.empirical_accuracy,'o-',color='#137c8b')
axes[1].set(xlim=(0,1),ylim=(0,1.03),xlabel='Mean maximum probability',ylabel='Empirical correctness',title='Coarse reliability (counts in table)')
fig.tight_layout(); plt.show()
""")
md("""
**Interpretation.** Sparse reliability bins are noisy; these diagnostics do not establish
calibration on future data. No recalibration or threshold tuning is performed on this
holdout. The next importance analysis is post-selection explanation only: it does not
feed back into feature choices or a second holdout-based model search.
""")
code("""
permutation = permutation_importance(selected,X_holdout,y_holdout,scoring='f1_macro',
    n_repeats=10,random_state=SEED,n_jobs=2)
permutation_table = pd.DataFrame({'raw_feature':FEATURES,'macro_f1_drop':permutation.importances_mean,
    'repeat_std':permutation.importances_std}).sort_values('macro_f1_drop',ascending=False)
display(permutation_table.reset_index(drop=True).round(4))
fig,ax = plt.subplots(figsize=(10,4.5))
top = permutation_table.head(10).iloc[::-1]
ax.barh(top.raw_feature,top.macro_f1_drop,xerr=top.repeat_std,color='#137c8b',capsize=3)
ax.set(xlabel='Holdout macro-F1 decrease after shuffling (mean ± repeat std)',
       title='Post-selection permutation importance · complete pipeline, raw columns')
fig.tight_layout(); plt.show()
top_drivers = permutation_table.head(3).raw_feature.tolist()
display(Markdown('**Measured top drivers:** '+', '.join(f'`{x}`' for x in top_drivers)+'. '
    'Shuffling passes raw columns through the full pipeline, including any derived products. '
    'Correlated variables can share importance, shuffling may produce unrealistic combinations, '
    'and negative/small drops can reflect sampling noise. Repeat standard deviations are not population confidence intervals. '
    'These are predictive sensitivities, not causal effects.'))
results = {'selected_model':selected_name,'selected_parameters':experiments['selected_params'],
    'selection_metric':'development CV macro-F1','cv_folds':5,'cv_macro_f1':float(cv_table.iloc[0].cv_macro_f1),
    'cv_macro_f1_std':float(cv_table.iloc[0].cv_macro_f1_std),'cv_accuracy':float(cv_table.iloc[0].cv_accuracy),
    'holdout_rows':400,'development_rows':1600,'final_training_rows':2000,
    'class_support':{str(i):int((y_holdout==i).sum()) for i in labels},
    'holdout_metrics':{k:float(v) for k,v in metrics.items()},'accuracy_wilson_95':accuracy_interval,
    'confusion_matrix':cm.tolist(),'probability_diagnostics':prob_metrics,'top_drivers':top_drivers,
    'classification_report':report,'methodology':experiments['methodology']}
(ROOT / 'reports/results.json').write_text(json.dumps(results,indent=2))
permutation_table.to_csv(ROOT / 'reports/permutation_importance.csv',index=False)
class_report.to_csv(ROOT / 'reports/holdout_classification_report.csv')
print('Holdout results locked and saved before full-data refit.')
""")

md("""
## 7. Model Deployment (Streamlit) — 4 marks

After locking the evaluation, clone the selected configuration and fit on all **2,000
labeled rows** for deployment. The recorded holdout scores belong to the preceding
1,600-row development fit; they are not an evaluation of this final refitted artifact.
Persist the complete raw-input pipeline, schema, version information and model card.
Only load trusted local joblib artifacts: pickle-based model files can execute code.

The app source below is complete and executable. It groups hardware inputs, displays
the relative category and genuine class probabilities, validates uploads, allows batch
download, and compares RAM/battery scenarios. Input schema and preprocessing are shared
with training. No rupee prices or nonexistent public deployment URL are implied.
""")
code("""
final_model = clone(selected).fit(X_train,y_train)
model_path = ROOT / 'artifacts/model.joblib'
joblib.dump(final_model,model_path)
reloaded = joblib.load(model_path)
np.testing.assert_array_equal(final_model.predict(X_test),reloaded.predict(X_test))
np.testing.assert_allclose(final_model.predict_proba(X_test),reloaded.predict_proba(X_test))
schema = build_schema(X_train,model_name=selected_name,cv_macro_f1=results['cv_macro_f1'],
    holdout_accuracy=metrics['accuracy'],holdout_macro_f1=metrics['macro_f1'],
    model_versions=versions,metric_training_rows=1600,holdout_rows=400)
(ROOT / 'artifacts/schema.json').write_text(json.dumps(schema,indent=2))
test_predictions = predict_batch(reloaded,test_raw,schema)[['id','price_range']]
assert test_predictions.shape == (1000,2)
assert test_predictions.id.tolist() == test_raw.id.tolist()
test_predictions.to_csv(ROOT / 'test_predictions.csv',index=False)
display(test_predictions.head(10))
display(test_predictions.price_range.value_counts().sort_index().rename('predicted_test_count').to_frame())
print('Saved complete pipeline and schema; generated 1,000 predictions in original ID/row order.')
print('No test accuracy, confusion matrix or classification report is available without test labels.')
""")
source_file('app.py')
source_file('.streamlit/config.toml')
source_file('tests/test_app.py')
source_file('tests/test_project.py')
code("""
test_run = subprocess.run([sys.executable,'-m','pytest','-q','tests','--disable-warnings'],
    cwd=ROOT,text=True,capture_output=True,timeout=180)
print(test_run.stdout.strip())
if test_run.returncode:
    raise RuntimeError(test_run.stdout + test_run.stderr)
verification = {'pytest_returncode':test_run.returncode,'pytest_summary':test_run.stdout.strip(),
    'local_app_tested':True,'public_deployment_verified':False}
_ = (ROOT / 'reports/app_verification.json').write_text(json.dumps(verification,indent=2))
""")
md("""
**Verified locally.** The preceding test output is from this execution, including
Streamlit `AppTest` form submission, a changed what-if scenario, missing-artifact errors,
batch scoring, strict schema validation, model reload equivalence and all 1,000 ordered
test predictions. AppTest executes Streamlit's app logic; it is not a visual browser
screenshot or evidence of public hosting. Automated schema/pipeline checks supplement
the readable fold-local experiment implementation; tests alone cannot prove absence
of every possible methodological leak.

From the project directory, activate the same environment and run:
```bash
python -m streamlit run app.py
# Equivalent once the environment is activated: streamlit run app.py
```
Choose specifications → select **Predict price category** → inspect probability bars →
compare a RAM/battery scenario → upload the supplied test CSV and download predictions.
Class labels mean relative Low/Medium/High/Very high categories (0–3), not monetary values.
Widgets use observed training ranges; the batch validator accepts physically valid
slight extrapolations such as test battery values 500/1999 outside training 501/1998.

**Public deployment remains an external action.** The project is implemented and tested
locally. To publish, create a repository containing `app.py`, `mobile_pipeline.py`,
`requirements.txt`, `.streamlit/config.toml`, `artifacts/model.joblib` and
`artifacts/schema.json`; then sign in to Streamlit Community Cloud, choose that repository
and `app.py`, select a compatible Python version, and deploy. Verify the returned URL
with single and batch predictions before claiming it is live. Public hosting requires
the student's account/repository authorization; none is fabricated here. These hosting
files are **not extra Taxila uploads**.
[Official deployment instructions](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/deploy),
[AppTest reference](https://docs.streamlit.io/develop/api-reference/app-testing/st.testing.v1.apptest).
""")

md("""
## 8. Novelty and Innovation — 2 marks

**Implemented additions.** The app's grouped specification form and model card support
clear decisions; the RAM/battery What-If Simulator shows two actual pipeline evaluations;
batch validation and downloads make the unlabeled file usable; probability diagnostics
reveal uncertainty; the paired feature ablation records whether apparently sensible
engineering helps. These are implemented features with measured evidence, not claims
that a decorative UI improves predictive accuracy.

The demonstration below uses the persisted final pipeline on an actual test-row scenario.
It changes RAM and battery within training ranges, holds other values fixed and compares
probabilities. A class can remain unchanged even when probabilities move. The scenario
shows a model association, **not a causal intervention or a purchasing recommendation**.
""")
code("""
base = X_test.iloc[[0]].copy()
scenario = base.copy()
scenario['ram'] = max(int(X_train.ram.min()), int(base.ram.iloc[0])-1000)
scenario['battery_power'] = min(int(X_train.battery_power.max()),int(base.battery_power.iloc[0])+300)
scenario_inputs = pd.concat([base,scenario],ignore_index=True)
scenario_results = predict_batch(reloaded,scenario_inputs,schema)
scenario_results.insert(0,'scenario',['Original test row 1','RAM −1000 MB / battery +300 mAh (clipped to observed bounds)'])
scenario_results.insert(1,'RAM_MB',scenario_inputs.ram)
scenario_results.insert(2,'battery_mAh',scenario_inputs.battery_power)
display(scenario_results.round(4))
fig,ax = plt.subplots(figsize=(9,3.5))
prob_cols = [f'p_{i}' for i in labels]
scenario_results.set_index('scenario')[prob_cols].T.plot.bar(ax=ax,color=['#137c8b','#e5a53e'],rot=0)
ax.set(ylim=(0,1),xlabel='Relative price class',ylabel='Model probability',title='Actual persisted-model what-if comparison')
ax.legend(['Original','Modified'],fontsize=9)
fig.tight_layout(); plt.show()
display(pd.DataFrame([
 ['What-if simulator','AppTest changes RAM/battery and verifies updated result','Explore sensitivity without retraining'],
 ['Batch inference','All 1,000 supplied rows; ID/order assertions; malformed-input tests','Repeatable scoring and download'],
 ['Feature engineering ablation','Same folds / fixed base parameters; accept only measured improvement','Avoid unsupported complexity'],
 ['Confidence and importance','Held-out probability diagnostics / raw-column permutation','Communicate uncertainty and drivers'],
 ],columns=['Addition','Evidence','Value']))
""")

md("""
## 9. Presentation and Viva — 5 marks

The following compact executive brief is generated from the actual experiment results.
Group/member placeholders must be replaced with genuine details before submitting.
""")
code("""
summary_rows = [
 ('Objective','Classify technical specifications into relative price classes 0–3'),
 ('Data','2,000 labeled ×22; 1,000 unlabeled ×24; 20 raw predictors'),
 ('Cleaning','Drop 3 Unnamed test fields; exclude id; retain audited zeros/extremes; no missing train values'),
 ('Selection',f'{selected_name}; 5-fold stratified development CV; 1,600 / 400 split; seed42'),
 ('CV',f'Macro-F1 {results["cv_macro_f1"]:.4f} ± {results["cv_macro_f1_std"]:.4f}; accuracy {results["cv_accuracy"]:.4f}'),
 ('Holdout',f'Accuracy {metrics["accuracy"]:.4f}; macro-F1 {metrics["macro_f1"]:.4f}; 100 phones per class'),
 ('Drivers',', '.join(top_drivers)+' (permutation, predictive not causal)'),
 ('Errors',f'{int(errors.sum())} errors; {adjacent_errors} adjacent-class errors; lowest F1: {worst}'),
 ('Scaling / encoding','Numeric standardization for LR/SVM; No/Yes=0/1; weight one-hot; fitted within CV'),
 ('Balancing',f'Exactly balanced; CV macro-F1 spread {balance_gap:.6f}; no synthetic data needed'),
 ('Engineering',area_notes),
 ('Deployment','Final pipeline refit on all 2,000; streamlit run app.py; AppTest passed locally'),
 ('Unlabeled inference','1,000 ordered id,price_range rows saved; test accuracy is unknowable'),
 ('Limits','Single supplied dataset/split; no current rupee prices, causal claims or verified public URL'),
 ('Taxila','Only GROUP_NO.ipynb and GROUP_NO.html, unzipped; fill genuine group details first'),
]
display(pd.DataFrame(summary_rows,columns=['Executive brief','Evidence / decision']))
""")
md("""
**Viva questions and short answers**

1. **Why exclude `id`?** It identifies rows, not hardware; learning row order can create spurious shortcuts.
2. **Why not report supplied test accuracy?** It lacks `price_range`. Metrics require ground truth; a stratified labeled holdout supplies it.
3. **Why stratify and keep a holdout?** Stratification preserves class representation; an untouched holdout evaluates a choice made exclusively with development CV.
4. **How is leakage prevented?** Split before EDA decisions; fit encoders, imputers and scalers inside each model/CV fold; resample only fold-training data.
5. **Why both binary and one-hot encoding?** Yes/No has a natural 0/1 meaning; one-hot weight bands avoid imposing arbitrary equal numeric distances.
6. **Why macro-F1?** It averages each class's F1 equally, balancing precision and recall; accuracy alone can hide poor class performance. Here support is balanced.
7. **Why not SMOTE?** Classes already have equal counts. Naive interpolation of categorical codes is invalid; SMOTENC can handle mixed data if an actual imbalance warrants it.
8. **Why scale SVM/LR but not trees?** Distance/regularization depends on feature units; threshold-based tree splits generally do not.
9. **What does the ensemble add?** Soft voting averages probabilities from distinct learners. Diversity can reduce error, but it must beat the baseline in CV; it is not guaranteed to win.
10. **How is overfitting assessed?** Compare resubstitution, CV and untouched holdout scores; use the measured gap and uncertainty, not a claim based only on training accuracy.
11. **What is saved for deployment?** The complete preprocessing/classifier pipeline plus ordered raw-feature schema; a shared importable module avoids notebook-local pickle failures.
12. **Does what-if show causality or confidence equal certainty?** Neither. It explores fitted predictive associations, and probabilities require calibration assessment and representative data.

**Two–three minute demonstration narrative**

- **0:00–0:30 — Task and integrity.** Introduce the four classes, real dataset shapes and missing test labels. Show the split and balanced-class chart.
- **0:30–1:00 — Evidence.** Use the consolidated plots to explain RAM and multivariate predictors; show leakage-safe encoding/scaling and why balancing did not add rows.
- **1:00–1:30 — Models and result.** Point to the CV table, selected model, holdout report and confusion matrices. Explain one common error pair and the train/CV/holdout comparison using the displayed values.
- **1:30–2:15 — Live app.** Run `streamlit run app.py`, submit a valid phone, show genuine probability bars, change RAM/battery and compare scenarios. State that the comparison is predictive, not causal.
- **2:15–2:45 — Reproducibility and limits.** Upload the unlabeled test CSV, download results, show passed tests and the two Taxila files. Explain the full-data refit and the public-hosting/account limitation.

**Limitations and future work.** This is a single supplied dataset with undocumented
sampling provenance and questionable screen/pixel zeros, not a representative survey
of the 2026 phone market. Weight is a coarse category. No real currency prices are
available. Non-nested development tuning is optimistic and one holdout has sampling
uncertainty. Public deployment and visual browser rendering have not been verified by
the notebook's AppTest. Future work should collect documented contemporary labels,
use external/temporal validation, examine zero semantics, and assess probability
calibration inside training CV before deploying for decisions. Do not tune these choices
on this already-used holdout and claim a new unbiased result.
""")
code("""
rubric = pd.DataFrame([
 ['1. Import Libraries / Dataset',1,'§1 imports, real CSV preview, versions/hashes','Executed'],
 ['2. Data and Data Models',1,'§2 shapes/schema, exactly two full sanity rows, split','Executed'],
 ['3. Visualization and Exploration',5,'§3 consolidated grids, balance, statistics, bivariate/heatmap, MI','Executed'],
 ['4. Pre-processing and Cleaning',5,'§4 audits/encoding/scaling rationale; §5 balancing/area comparisons; §6 importance','Executed'],
 ['5. Model Building',5,'§5 DT/RF/RBF-SVM/soft voting + baselines; dev-only CV/tuning','Executed'],
 ['6. Performance Evaluation',2,'§6 labeled holdout accuracy, class reports, confusion matrices, fit analysis','Executed'],
 ['7. Model Deployment',4,'§7 full pipeline persistence, full app source, AppTest and batch checks','Local verified; public hosting pending'],
 ['8. Novelty and Innovation',2,'§8 actual what-if, batch, ablation, probability/importance evidence','Executed / tested'],
 ['9. Presentation and Viva',5,'§9 executive brief, 12 Q&As, timed demo, limitations','Present; genuine identities required'],
 ],columns=['Rubric component','Marks','Location / evidence','Verification'])
assert rubric.Marks.sum()==30
display(rubric)
rubric.to_csv(ROOT / 'reports/rubric_compliance.csv',index=False)
display(Markdown('**Submission check:** replace GROUP_NO/member/student-ID placeholders, rerun and export, '
    'then upload only the notebook and matching HTML (uncompressed). This checklist covers all 30 rubric marks; '
    'the final grade and live viva depend on the examiner and the group’s demonstration.'))
""")

nb = nbf.v4.new_notebook(cells=cells, metadata={
    'kernelspec':{'display_name':'Python 3 (AML environment)','language':'python','name':'python3'},
    'language_info':{'name':'python'},
})
out = ROOT / 'submission/GROUP_NO.ipynb'
out.parent.mkdir(exist_ok=True)
nbf.write(nb,out)
print(f'Created {out}: {len(cells)} cells')
