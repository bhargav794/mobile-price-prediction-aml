"""Portable preprocessing and inference shared by the notebook and Streamlit."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

SEED = 42
BINARY = ['blue', 'dual_sim', 'touch_screen', 'wifi']
CATEGORICAL = BINARY + ['mobile_wt']
FEATURES = ['battery_power', 'blue', 'clock_speed', 'dual_sim', 'fc', 'four_g',
            'int_memory', 'm_dep', 'mobile_wt', 'n_cores', 'pc', 'px_height',
            'px_width', 'ram', 'sc_h', 'sc_w', 'talk_time', 'three_g',
            'touch_screen', 'wifi']
NUMERIC = [c for c in FEATURES if c not in CATEGORICAL]
FLOATS = ['clock_speed', 'm_dep']
CLASS_LABELS = {'0': 'Low', '1': 'Medium', '2': 'High', '3': 'Very high'}


def clean_columns(frame):
    """Drop only known spreadsheet export garbage, preserve row order."""
    return frame.loc[:, ~frame.columns.astype(str).str.match(r'^Unnamed:')].copy()


class HardwareFeatures(TransformerMixin, BaseEstimator):
    """Deterministic products; no learned state, ratios, or zero replacement."""
    def __init__(self, add_areas=False):
        self.add_areas = add_areas

    def fit(self, X, y=None):
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        self.n_features_in_ = X.shape[1]
        return self

    def transform(self, X):
        out = X.copy()
        if self.add_areas:
            out['pixel_area'] = out['px_height'].astype(float) * out['px_width'].astype(float)
            out['screen_area'] = out['sc_h'].astype(float) * out['sc_w'].astype(float)
        return out

    def get_feature_names_out(self, input_features=None):
        names = list(self.feature_names_in_ if input_features is None else input_features)
        return np.asarray(names + (['pixel_area', 'screen_area'] if self.add_areas else []), dtype=object)


def make_preprocessor(scaled=True, add_areas=False):
    """All fitted steps are cloned and fitted inside each CV training fold."""
    numeric_steps = [('impute', SimpleImputer(strategy='median'))]
    if scaled:
        numeric_steps.append(('scale', StandardScaler()))
    binary = Pipeline([
        ('impute', SimpleImputer(strategy='most_frequent')),
        ('encode', OrdinalEncoder(categories=[['No', 'Yes']] * len(BINARY),
                                  handle_unknown='use_encoded_value', unknown_value=-1)),
    ])
    weight = Pipeline([
        ('impute', SimpleImputer(strategy='most_frequent')),
        ('encode', OneHotEncoder(categories=[['Low', 'Med', 'High']],
                                 handle_unknown='ignore', sparse_output=False)),
    ])
    return ColumnTransformer([
        ('numeric', Pipeline(numeric_steps), NUMERIC + (['pixel_area', 'screen_area'] if add_areas else [])),
        ('binary', binary, BINARY),
        ('weight', weight, ['mobile_wt']),
    ], remainder='drop', verbose_feature_names_out=True)


def build_schema(X, **metadata):
    features = {}
    for name in FEATURES:
        s = X[name]
        if name in CATEGORICAL:
            features[name] = {'dtype': 'category', 'categories': ['Low', 'Med', 'High'] if name == 'mobile_wt' else ['No', 'Yes'],
                              'default': str(s.mode().iloc[0])}
        else:
            cast = float if name in FLOATS else int
            features[name] = {'dtype': 'float' if name in FLOATS else 'int',
                              'min': cast(s.min()), 'max': cast(s.max()), 'default': cast(s.median())}
    return {'feature_order': FEATURES, 'features': features, 'class_labels': CLASS_LABELS,
            'seed': SEED, 'trained_rows': len(X), **metadata}


def validate_inputs(frame, schema):
    """Validate raw user input without silently imputing malformed uploads.

    Observed training ranges are guidance, not physical constraints. In particular,
    supplied test battery values 500 and 1999 lie just outside training extrema.
    """
    if not isinstance(frame, pd.DataFrame) or frame.empty:
        raise ValueError('Provide a non-empty CSV table.')
    if frame.columns.duplicated().any():
        raise ValueError('Duplicate column names are not allowed.')
    cleaned = clean_columns(frame)
    order = schema['feature_order']
    missing = sorted(set(order) - set(cleaned.columns))
    extra = sorted(set(cleaned.columns) - set(order) - {'id'})
    if missing:
        raise ValueError('Missing required predictors: ' + ', '.join(missing))
    if extra:
        raise ValueError('Unexpected columns: ' + ', '.join(extra))
    out = cleaned.loc[:, order].copy()
    positive = {'battery_power', 'clock_speed', 'int_memory', 'm_dep', 'n_cores', 'px_width', 'ram', 'sc_h', 'talk_time'}
    for name in order:
        spec = schema['features'][name]
        if spec['dtype'] == 'category':
            out[name] = out[name].astype('string').str.strip()
            valid = out[name].isin(spec['categories'])
            if not valid.all():
                raise ValueError(f'{name}: use one of {spec["categories"]}; invalid rows {list(out.index[~valid][:5])}.')
            out[name] = out[name].astype(object)
        else:
            values = pd.to_numeric(out[name], errors='coerce')
            if not np.isfinite(values.to_numpy(dtype=float)).all():
                raise ValueError(f'{name}: all values must be finite numbers (no missing values).')
            if (values.abs() > 1e9).any():
                raise ValueError(f'{name}: value is too large for supported inference (maximum 1 billion).')
            if (values < 0).any() or (name in positive and (values <= 0).any()):
                raise ValueError(f'{name}: values must be {"positive" if name in positive else "nonnegative"}.')
            if spec['dtype'] == 'int':
                if (values != np.floor(values)).any():
                    raise ValueError(f'{name}: whole numbers are required.')
                values = values.astype('int64')
            if name in ['three_g', 'four_g'] and not values.isin([0, 1]).all():
                raise ValueError(f'{name}: use only 0 or 1.')
            out[name] = values
    if (out['fc'] > out['pc']).any():
        raise ValueError('Front camera (fc) cannot exceed primary camera (pc) in this dataset schema.')
    if ((out['four_g'] == 1) & (out['three_g'] == 0)).any():
        raise ValueError('A 4G phone must also have 3G enabled in this dataset schema.')
    if (out['sc_w'] > out['sc_h']).any():
        raise ValueError('Screen width (sc_w) cannot exceed screen height (sc_h) in this dataset schema.')
    return out


def predict_batch(model, frame, schema):
    X = validate_inputs(frame, schema)
    result = pd.DataFrame(index=frame.index)
    if 'id' in frame.columns:
        result['id'] = frame['id']
    result['price_range'] = model.predict(X).astype(int)
    if hasattr(model, 'predict_proba'):
        probabilities = model.predict_proba(X)
        for i, label in enumerate(model.classes_):
            result[f'p_{label}'] = probabilities[:, i]
    return result
