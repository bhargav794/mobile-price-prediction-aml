"""Data, persistence, inference and experimental-integrity acceptance checks."""
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import pytest
from mobile_pipeline import FEATURES, clean_columns, validate_inputs, predict_batch

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope='module')
def bundle():
    train = pd.read_csv(ROOT / 'Mobile_Price_Prediction_train.csv')
    test = pd.read_csv(ROOT / 'Mobile_Price_Prediction_test.csv')
    schema = json.loads((ROOT / 'artifacts/schema.json').read_text())
    model = joblib.load(ROOT / 'artifacts/model.joblib')
    results = json.loads((ROOT / 'reports/results.json').read_text())
    return train, test, schema, model, results


def test_original_data_and_schema(bundle):
    train, test, schema, model, _ = bundle
    assert train.shape == (2000, 22) and test.shape == (1000, 24)
    assert train.price_range.value_counts().sort_index().tolist() == [500] * 4
    assert 'price_range' not in test and 'id' not in FEATURES
    assert schema['feature_order'] == FEATURES
    assert clean_columns(test).drop(columns='id').columns.tolist() == FEATURES
    assert validate_inputs(test, schema).columns.tolist() == FEATURES
    assert np.array_equal(model.classes_, [0, 1, 2, 3])


def test_predictions_and_probabilities(bundle):
    _, test, schema, model, _ = bundle
    predictions = pd.read_csv(ROOT / 'test_predictions.csv')
    assert predictions.shape == (1000, 2)
    assert predictions.columns.tolist() == ['id', 'price_range']
    assert predictions.id.tolist() == test.id.tolist()
    expected = predict_batch(model, test, schema)
    assert predictions.price_range.tolist() == expected.price_range.tolist()
    assert expected.price_range.isin([0, 1, 2, 3]).all()
    assert np.allclose(expected[[f'p_{i}' for i in range(4)]].sum(axis=1), 1)


def test_persistence_equivalence(bundle, tmp_path):
    train, _, schema, model, _ = bundle
    X = train.loc[:9, FEATURES]
    target = tmp_path / 'roundtrip.joblib'
    joblib.dump(model, target)
    recovered = joblib.load(target)
    np.testing.assert_array_equal(model.predict(X), recovered.predict(X))
    np.testing.assert_allclose(model.predict_proba(X), recovered.predict_proba(X))


def test_split_and_selection_integrity(bundle):
    train, _, _, _, results = bundle
    split = json.loads((ROOT / 'reports/split_manifest.json').read_text())
    development = set(split['development_ids'])
    holdout = set(split['holdout_ids'])
    assert len(development) == 1600 and len(holdout) == 400
    assert not development & holdout
    assert development | holdout == set(train.id)
    cv = pd.read_csv(ROOT / 'reports/model_comparison.csv')
    best = cv.sort_values(['cv_macro_f1', 'cv_accuracy'], ascending=False).iloc[0]
    assert results['selected_model'] == best['model']
    assert results['holdout_rows'] == 400 and results['final_training_rows'] == 2000
    assert set(['Decision Tree', 'Random Forest', 'SVM (RBF)', 'Soft Voting']) <= set(cv.model)
    assert all(results['class_support'][str(i)] == 100 for i in range(4))


def test_leakage_safe_pipeline_structure(bundle):
    # Persisted individual pipelines contain fitted transforms inside the estimator;
    # GridSearchCV receives their unfitted clones, not globally preprocessed arrays.
    models = joblib.load(ROOT / 'artifacts/development_models.joblib')
    for name in ['Logistic Regression', 'Decision Tree', 'Random Forest', 'SVM (RBF)']:
        if name == 'SVM (RBF)':
            from sklearn.calibration import CalibratedClassifierCV
            calibrator = models[name].named_steps['model']
            assert isinstance(calibrator, CalibratedClassifierCV)
            assert calibrator.cv.n_splits == 5 and not calibrator.ensemble
            steps = calibrator.calibrated_classifiers_[0].estimator.named_steps
        else:
            steps = models[name].named_steps
        assert 'preprocess' in steps and 'model' in steps
        assert hasattr(steps['preprocess'], 'transformers_')
        assert 'id' not in steps['preprocess'].feature_names_in_
        assert 'price_range' not in steps['preprocess'].feature_names_in_
    results = bundle[-1]
    assert results['cv_folds'] == 5
    assert results['selection_metric'] == 'development CV macro-F1'


def test_malformed_input_rejected(bundle):
    train, _, schema, _, _ = bundle
    frame = train.loc[:1, FEATURES].copy()
    with pytest.raises(ValueError, match='Missing'):
        validate_inputs(frame.drop(columns='ram'), schema)
    with pytest.raises(ValueError, match='finite'):
        validate_inputs(frame.assign(ram=np.nan), schema)
    with pytest.raises(ValueError, match='use one of'):
        validate_inputs(frame.assign(wifi='maybe'), schema)
    with pytest.raises(ValueError, match='Unexpected'):
        validate_inputs(frame.assign(price_range=0), schema)
