"""Student-facing explanations and presentation edits for the notebook generator."""
import re
import textwrap

APP_URL = 'https://mobile-price-prediction-aml.streamlit.app/'
GROUP_NUMBER = '2'
MEMBERS = [
    ('YUVARAJ S S', '2025NS04059'),
    ('Jannatul Firdousi', '2025NS04069'),
    ('KOTHAPALLI BHARGAVA AVINASH', '2025NS04077'),
    ('Dyutik Chaudhary Suryadevara', '2025NS04090'),
]
MEMBER_ROWS = '<br>'.join(f'{name} - {student_id}' for name, student_id in MEMBERS)

NARRATIVE = {
0: f'''
<div class="assignment-cover" style="background:#122840;color:#fff;padding:28px;border-radius:12px">
<p style="color:#6ee7cf;letter-spacing:1px">BITS PILANI - WORK INTEGRATED LEARNING PROGRAMMES</p>
<h1 style="color:white">Mobile Price Prediction</h1>
<p>Applied Machine Learning | S1-26_DSEZG568</p>
<p>Group: {GROUP_NUMBER}<br>Members and student IDs:<br>{MEMBER_ROWS}</p>
<p><a style="color:#6ee7cf" href="https://mobile-price-prediction-aml.streamlit.app/">Open the Streamlit application</a><br>
<span style="font-size:12px">https://mobile-price-prediction-aml.streamlit.app/</span></p>
</div>

We use mobile phone specifications to predict one of four price categories:
**0 - Low, 1 - Medium, 2 - High, and 3 - Very high**. We compare Decision Tree,
Random Forest, SVM and a voting ensemble, with Logistic Regression and a Dummy
Classifier as baselines. The selected model is used in the Streamlit application.
''',
1: '''
## 1. Libraries and Dataset

We use NumPy and pandas for data handling, Matplotlib and seaborn for plots, and
scikit-learn for preprocessing, training and evaluation. The two supplied CSVs are
loaded below. The target column is `price_range`.
''',
5: '''
## 2. Data Understanding

The training data contains **2,000 rows and 22 columns**, including `id` and
`price_range`. The test data contains **1,000 rows and 24 columns**. After removing
the identifier and three empty-header test columns, both sets contain the same
**20 predictors**. Four string columns contain Yes/No values; `mobile_wt` contains
Low, Med and High weight categories.

**Assumptions and evaluation setup**

- `id` identifies a row and is excluded from the predictor matrix.
- The supplied test data is used for prediction because it has no target labels.
- We reserve 20% of the labeled data for final evaluation: **1,600 development rows
  and 400 holdout rows**, using stratification and `random_state=42`.
- Exploration and model selection use the development partition. The holdout is
  used after selecting the model with cross-validation.
- The target values are treated as four ordered price categories.

The following tables show the schema, two complete sample records and the split sizes.
''',
8: '''
## 3. Exploratory Data Analysis

We examine class counts, summary statistics, individual feature distributions,
relationships with the target and correlations. Related plots are grouped in one
frame to make comparisons easier.

**Units assumed:** battery capacity in mAh, RAM in MB, storage in GB, clock speed in
GHz, cameras in MP, resolution in pixels, screen dimensions and depth in cm, and
talk time in hours. `mobile_wt` is used as a category.
''',
10: '''
Each class has **500 records**, giving 400 development and 100 holdout records per
class. The labels are exactly balanced. We still use macro-F1 for model selection
because it gives equal importance to each class's precision and recall.
''',
12: '''
The distributions show a wide spread of RAM and battery capacity, discrete
connectivity flags, and three weight bands. Front-camera values and pixel height
are more skewed than RAM or battery capacity. Scaling changes the units of a
feature; it does not remove skewness. Recorded zeros are examined in the next section.
''',
15: '''
**Feature ranking:** mutual information measures how much a feature helps distinguish
the classes, including dependence that a linear correlation may miss. We use it as
an exploratory ranking and retain the original predictors for model comparison.
Later, we compare this ranking with Random Forest importance and permutation importance.
''',
16: '''
## 4. Data Preprocessing

We check missing values, duplicates, negative values, camera consistency, screen
dimensions, skewness and IQR outlier flags before fitting the models.

**Assumptions used**

- Recorded zero values are retained as supplied. Camera zeros can indicate absence;
  zero screen width or pixel height is treated as a recorded value. We avoid ratios
  that would divide by zero.
- Extreme specifications are retained when they fall within the dataset's ranges.
  An IQR flag alone is not used to delete a record.
- Yes/No values use binary encoding: **No = 0, Yes = 1**. Weight categories use
  **one-hot encoding**, so the model does not assume equal numerical gaps between bands.
- Numeric median imputation and categorical mode imputation are included in the
  pipeline. They do not change the supplied complete predictors.

For Logistic Regression and SVM, `StandardScaler` subtracts the training mean and
divides by the training standard deviation. This prevents large-unit features such
as RAM from dominating regularization or distance calculations. Trees choose split
thresholds and therefore use unscaled numeric inputs. The encoder, imputer and
scaler are fitted separately inside each cross-validation training fold.
''',
18: '''
**Class balancing.** We compare no balancing, balanced class weights and random
oversampling within the training folds. Each five-fold training partition has
320 examples per class. Therefore balanced weights are 1 and random oversampling
adds no rows. Random oversampling duplicates existing observations, preserving
categorical values. SMOTENC is an alternative for mixed numeric/categorical data
when classes are imbalanced; synthetic sampling is unnecessary here.

**Feature engineering.** We test two products: `pixel_area = px_height * px_width`
and `screen_area = sc_h * sc_w`. Both are safe when inputs contain zero. For each
comparison we keep the model's tuned hyperparameters fixed and use the same folds.
We retain the extra features only if mean CV macro-F1 improves by more than 0.002.
This threshold favors the simpler feature set when the measured change is small.
''',
19: '''
## 5. Model Training and Comparison

| Classifier | Reason for including it |
|---|---|
| Decision Tree | Learns feature thresholds and nonlinear decision rules |
| Random Forest | Combines trees to reduce the variation of a single tree |
| SVM with RBF kernel | Tests flexible, smooth class boundaries |
| Soft Voting | Combines Logistic Regression, Random Forest and SVM probabilities |
| Logistic Regression | Provides a linear classification baseline |
| Dummy Classifier | Provides a simple reference score by predicting the most frequent class |

We tune the models with **five-fold stratified cross-validation** on the 1,600
development rows. Macro-F1 is the selection metric, with accuracy used to break an
exact tie. Every candidate uses the same folds. The table shows mean scores and
their standard deviation across folds.

SVM probabilities are obtained with sigmoid calibration using five inner folds.
The complete preprocessing and SVM pipeline is fitted within those folds. The
voting model averages its members' class probabilities. Tuning and feature choices
reuse the development folds, so the separate holdout provides the final evaluation.
''',
23: '''
## 6. Performance Evaluation

We evaluate the selected model on the **400 labeled holdout records** after fitting
it on the development data.

- **Accuracy:** fraction of all predictions that are correct.
- **Precision for a class:** correct predictions of that class divided by all predictions of it.
- **Recall for a class:** correct predictions of that class divided by its actual support.
- **F1:** harmonic mean of precision and recall.
- **Macro average:** equal average across classes. **Weighted average:** average weighted by class support.

The confusion matrix uses actual classes as rows and predicted classes as columns.
Row normalization shows the fraction of each actual class assigned to every predicted class.
''',
25: '''
**Class probabilities.** `p_0`, `p_1`, `p_2` and `p_3` are the probabilities assigned
to Low, Medium, High and Very high. They sum to one for each phone. The selected
Logistic Regression model predicts the class with the largest probability.

We also calculate log loss and the multiclass Brier score. Log loss penalizes low
probability assigned to the true class. Here Brier score is the mean sum of squared
differences between the four probabilities and the one-hot target. Lower values
are better. The reliability plot compares average maximum probability with the
observed fraction correct in each bin.
''',
27: '''
Reliability-bin counts help interpret the probability plot: a bin with few records
has a less stable observed accuracy. The following permutation analysis shuffles
one original feature at a time and measures the decrease in holdout macro-F1.
It explains the selected model after evaluation; it is not used to choose new features.
''',
29: '''
## 7. Streamlit Application

**Application:** [Mobile Price Prediction](https://mobile-price-prediction-aml.streamlit.app/)

After evaluation, we refit the selected configuration on all **2,000 labeled
records**. The reported holdout metrics belong to the development-trained model.
The final fitted pipeline combines preprocessing and classification, so the app
uses the same feature order and encoding as training.

The app groups the 20 inputs into performance, display, camera, connectivity, and
hardware/battery settings. It returns the predicted category and four class
probabilities. Batch upload accepts the supplied test data, preserves IDs and row
order, validates predictors and provides downloadable results. The complete app
and preprocessing source are included in the notebook cells.
''',
36: '''
The application checks above cover prediction, probability totals, missing or
invalid inputs, saved-model reloads, batch row order and the two interactive forms.

**Use of the app:** enter a phone's specifications and select **Predict price category**.
The **What-if simulator** compares the current profile with changed RAM and battery
values. **Batch CSV** accepts multiple phones; `price_range` is the predicted class
and `p_0` to `p_3` give the probabilities of the four categories.
''',
37: '''
## 8. Additional Analysis

The what-if comparison runs the fitted model twice: once for the original profile
and once after changing RAM and battery capacity. The other 18 predictors are held
fixed. This makes it easy to inspect how the predicted category and probabilities
change for a selected phone.

Below we use the first supplied test record, reduce RAM by 1,000 MB and increase
battery capacity by 300 mAh, keeping values within the training ranges.
''',
39: '''
## 9. Summary and Conclusions

The following table brings together the dataset preparation, model choice and
evaluation results.
''',
41: '''
Logistic Regression gave the highest development CV macro-F1 among the tested
configurations. The voting ensemble ranked second, followed by SVM, Random Forest
and Decision Tree. This suggests that a linear decision boundary captures much of
the class separation in this dataset. More flexible models did not improve the
measured CV score with the settings tested.

The selected model correctly classified **392 of 400 holdout records**. All eight
errors were between neighboring price classes. Class 2 had the lowest F1 (0.9700),
while every class had recall of at least 0.97. Training macro-F1 was about 0.985,
compared with 0.961 in CV and 0.980 on holdout. These results do not show a large
training-to-holdout gap; the holdout is also a smaller sample than the CV evaluation.

RAM was the strongest predictor in the exploratory ranking and permutation
analysis, followed by battery capacity and pixel height. The area-feature experiment
did not meet our retention threshold, and balancing methods produced the same
scores because the labels were already balanced. We therefore kept the original
20 predictors and the natural class distribution.

The final application uses the selected pipeline fitted on all labeled rows and
supports single-phone prediction, batch prediction and a RAM/battery comparison.
''',
}


def tidy(text):
    return textwrap.dedent(text).strip()


def edit_cells(cells):
    """Apply the same wording changes whenever the notebook is regenerated."""
    assert len(cells) == 43, 'Update the editorial map if the notebook structure changes.'
    for i, text in NARRATIVE.items():
        cells[i].source = tidy(text)
    cells[2].source = cells[2].source.replace(
        "GROUP_NO = 'GROUP_NO'  # REQUIRED: replace with your genuine group number",
        f"GROUP_NO = {GROUP_NUMBER!r}")
    cells[2].source = cells[2].source.replace(
        "['REPLACE: member name(s)']", repr([name for name, _ in MEMBERS])).replace(
        "['REPLACE: student ID(s)']", repr([student_id for _, student_id in MEMBERS]))
    cells[2].source = cells[2].source.replace(
        "display(pd.DataFrame({'group':[GROUP_NO], 'members':[', '.join(MEMBER_NAMES)], 'details':[str(GROUP_DETAILS)]}))", '')
    # Source cells remain executable and contain all implementation details.
    for i in [3, 6, 20, 31, 32, 33, 34]:
        src = cells[i].source
        src = re.sub(r'^# Regenerate .*?\n', '# Implementation used in this notebook.\n', src)
        src = src.replace("destination.write_text(source, encoding='utf-8')",
                          "_ = destination.write_text(source, encoding='utf-8')")
        src = re.sub(r"\nprint\('Recreated [^']+'\)$", '', src)
        cells[i].source = src
    cells[7].source = cells[7].source.replace(
        "print('Exactly TWO complete sanity-check rows (scroll horizontally if needed):')",
        "print('Two sample records:')")
    # Explain measured observations directly alongside the plots.
    start = cells[13].source.index('display(Markdown(')
    cells[13].source = cells[13].source[:start] + tidy('''
    display(Markdown(f'RAM has a development-set Pearson correlation of **{numeric_corr.loc["ram","price_range"]:.3f}** '
        'with the ordered class labels. RAM distributions shift upward across the classes, '
        'while battery capacity and resolution provide additional separation. Overlap in the '
        'plots motivates using several features together. The model comparison tests whether '
        'nonlinear boundaries improve on Logistic Regression.'))
    ''')
    cells[14].source = cells[14].source.replace('not a causal explanation','an exploratory association')
    cells[21].source = cells[21].source.replace("display(experiments['tuning_table'])", tidy('''
    tuning_display = experiments['tuning_table'][['model','grid_candidates','best_parameters']].copy()
    tuning_display['best_parameters'] = tuning_display['best_parameters'].str.replace(
        'model__estimator__model__', '', regex=False).str.replace('model__', '', regex=False)
    display(tuning_display)
    '''))
    cells[22].source = cells[22].source.replace('**Measured engineering decision:**', '**Feature comparison:**')
    cells[22].source = cells[22].source.replace(
        "'The balanced original class counts justify retaining the natural data distribution. '",
        "'Each class has the same support, so the original class distribution is retained. '")
    cells[22].source = cells[22].source.replace(
        "'A nil gain is a useful negative result, not a reason to manufacture imbalance.'", "'The comparison confirms that balancing makes no change here.'")
    cells[22].source = cells[22].source.replace("title='Selection evidence · holdout has not been evaluated'", "title='Model comparison using development cross-validation'")
    cells[22].source = cells[22].source.replace('**Selection is now locked:**', '**Selected model:**')
    cells[22].source = cells[22].source.replace(
        "'between correlated features. It complements MI; neither is causal. '",
        "'between correlated features. It complements the mutual-information ranking. '")
    cells[22].source = cells[22].source.replace(
        "'The next section evaluates this fixed choice once and will not alter it.'",
        "'We evaluate this configuration on the holdout in the next section.'")
    cells[24].source = cells[24].source.replace(
        'The modest gap and strong holdout score support satisfactory generalization on this split, without proving performance on new market data.',
        'The small gap suggests that the selected model fits the development and holdout partitions consistently.')
    cells[24].source = cells[24].source.replace(
        'The measured gap warrants caution about variance or limited signal; this single holdout cannot establish a universal fit diagnosis.',
        'The gap suggests that the model may be sensitive to the partition; the CV results provide an additional comparison.')
    cells[28].source = cells[28].source.replace('**Measured top drivers:**','**Top predictors:**')
    cells[28].source = cells[28].source.replace(
        "'Correlated variables can share importance, shuffling may produce unrealistic combinations, '",
        "'Correlated variables can share importance. '")
    cells[28].source = cells[28].source.replace(
        "'and negative/small drops can reflect sampling noise. Repeat standard deviations are not population confidence intervals. '",
        "'Negative or small drops can reflect sampling variation. Error bars show variation across repeated shuffles. '")
    cells[28].source = cells[28].source.replace("'These are predictive sensitivities, not causal effects.'", "'A larger drop indicates greater reliance on that feature.'")
    cells[28].source = cells[28].source.replace('Holdout results locked and saved before full-data refit.', 'Holdout evaluation complete.')
    cells[30].source = cells[30].source.replace(
        'Saved complete pipeline and schema; generated 1,000 predictions in original ID/row order.',
        'The final pipeline generated 1,000 predictions in the original row order.')
    cells[30].source = cells[30].source.replace(
        "print('No test accuracy, confusion matrix or classification report is available without test labels.')", '')
    cells[35].source = cells[35].source.replace("print(test_run.stdout.strip())", "print('Checks: ' + test_run.stdout.strip().splitlines()[-1])")
    cells[38].source = cells[38].source.replace('RAM −1000 MB / battery +300 mAh (clipped to observed bounds)', 'Modified RAM / battery')
    table_start = cells[38].source.index('display(pd.DataFrame([')
    cells[38].source = cells[38].source[:table_start] + tidy('''
    display(pd.DataFrame([
     ['What-if comparison', 'Changes RAM/battery and holds other predictors fixed'],
     ['Batch prediction', 'Scores uploaded rows and preserves their IDs and order'],
     ['Feature experiment', 'Compares original inputs with pixel and screen areas'],
     ['Probability analysis', 'Shows class probabilities, log loss and Brier score'],
     ], columns=['Feature', 'Implementation']))
    ''')
    cells[40].source = tidy('''
    summary_rows = [
     ('Dataset', '2,000 labeled phones; 1,000 unlabeled test phones; 20 predictors'),
     ('Preparation', 'Remove empty test columns and exclude id; binary and one-hot encoding'),
     ('Evaluation split', '1,600 development / 400 holdout rows; stratified, seed 42'),
     ('Selected classifier', selected_name),
     ('Development CV', f'Macro-F1 {results["cv_macro_f1"]:.4f} +/- {results["cv_macro_f1_std"]:.4f}'),
     ('Holdout', f'Accuracy {metrics["accuracy"]:.4f}; macro-F1 {metrics["macro_f1"]:.4f}'),
     ('Holdout errors', f'{int(errors.sum())} errors, all {adjacent_errors} between adjacent classes'),
     ('Top predictors', ', '.join(top_drivers)),
     ('Feature engineering', 'Original inputs retained after the area-feature comparison'),
     ('Balancing', 'No score change; the supplied classes are already balanced'),
     ('Final model', 'Selected pipeline refitted on all 2,000 labeled rows'),
     ('Application', 'Single prediction, batch upload, class probabilities and what-if comparison'),
    ]
    display(pd.DataFrame(summary_rows, columns=['Summary', 'Result']))
    ''')
    # The former grading checklist is not part of the student report.
    return cells[:42]
