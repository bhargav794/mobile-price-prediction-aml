"""Reproducible development-only model selection for Mobile Price Prediction.

This module deliberately has no holdout/test input. Every fitted transformation,
hyperparameter choice, balancing comparison, and feature ablation is confined
to stratified development-set cross-validation.
"""
from __future__ import annotations

import json
import time

import numpy as np
import pandas as pd
from imblearn.over_sampling import RandomOverSampler
from imblearn.pipeline import Pipeline as SamplingPipeline
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from mobile_pipeline import FEATURES, HardwareFeatures, make_preprocessor

SEED = 42
N_SPLITS = 5
N_JOBS = 2
AREA_MIN_GAIN = 0.002
SCORING = {"macro_f1": "f1_macro", "accuracy": "accuracy"}


def make_pipeline(estimator, *, scaled=True, add_areas=False):
    """Build a portable raw-DataFrame -> predictions pipeline."""
    return Pipeline([
        ("features", HardwareFeatures(add_areas=add_areas)),
        ("preprocess", make_preprocessor(scaled=scaled, add_areas=add_areas)),
        ("model", estimator),
    ])


def make_calibrated_svm(*, add_areas=False):
    """Keep preprocessing inside each calibration fold as well as each CV fold.

    SVC's historical probability flag is deprecated in sklearn 1.9. Sigmoid
    calibration uses out-of-fold decision scores from the complete raw-input
    pipeline, followed by a single estimator refit on the available training
    data (ensemble=False). No holdout rows enter calibration.
    """
    raw_svm = make_pipeline(
        SVC(kernel="rbf", random_state=SEED, cache_size=256),
        scaled=True, add_areas=add_areas,
    )
    calibrator = CalibratedClassifierCV(
        estimator=raw_svm, method="sigmoid", ensemble=False, n_jobs=1,
        cv=StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED),
    )
    return Pipeline([("model", calibrator)])


def _refit_by_f1_then_accuracy(results):
    """Use accuracy only to resolve an exact macro-F1 tie; keep order stable."""
    f1 = np.asarray(results["mean_test_macro_f1"])
    accuracy = np.asarray(results["mean_test_accuracy"])
    return int(np.lexsort((np.arange(len(f1)), -accuracy, -f1))[0])


def _search(estimator, param_grid, X, y, splits):
    search = GridSearchCV(
        estimator=estimator,
        param_grid=param_grid,
        scoring=SCORING,
        refit=_refit_by_f1_then_accuracy,
        cv=splits,
        n_jobs=N_JOBS,
        pre_dispatch=N_JOBS,
        return_train_score=True,
        error_score="raise",
    )
    return search.fit(X, y)


def _search_summary(search):
    i = search.best_index_
    result = search.cv_results_
    return {
        "cv_macro_f1": float(result["mean_test_macro_f1"][i]),
        "cv_macro_f1_std": float(result["std_test_macro_f1"][i]),
        "cv_accuracy": float(result["mean_test_accuracy"][i]),
        "cv_accuracy_std": float(result["std_test_accuracy"][i]),
        "cv_train_macro_f1": float(result["mean_train_macro_f1"][i]),
        "cv_train_macro_f1_std": float(result["std_train_macro_f1"][i]),
        "mean_fit_seconds": float(result["mean_fit_time"][i]),
    }


def _score_summary(scores):
    return {
        "cv_macro_f1": float(np.mean(scores["test_macro_f1"])),
        "cv_macro_f1_std": float(np.std(scores["test_macro_f1"])),
        "cv_accuracy": float(np.mean(scores["test_accuracy"])),
        "cv_accuracy_std": float(np.std(scores["test_accuracy"])),
        "cv_train_macro_f1": float(np.mean(scores["train_macro_f1"])),
        "cv_train_macro_f1_std": float(np.std(scores["train_macro_f1"])),
        "mean_fit_seconds": float(np.mean(scores["fit_time"])),
    }


def _cv_scores(estimator, X, y, splits):
    return cross_validate(
        estimator, X, y, cv=splits, scoring=SCORING,
        n_jobs=N_JOBS, pre_dispatch=N_JOBS,
        return_train_score=True, error_score="raise",
    )


def run_experiments(X_dev, y_dev, verbose=False):
    """Fit six families, ablate area features, and compare three balancing rules.

    Parameters are selected by five-fold stratified CV on the development data.
    Models returned in ``models`` are fitted on that development data only.
    ``selected_model`` is therefore ready for a single untouched holdout
    evaluation; the caller must clone/refit it for all-data deployment later.

    The same folds are reused for paired comparisons. Feature areas must improve
    mean CV macro-F1 by more than 0.002 before they are retained. This threshold
    is a practical simplicity rule, not a claim of statistical significance.
    """
    started = time.perf_counter()
    if not isinstance(X_dev, pd.DataFrame):
        raise TypeError("X_dev must be a DataFrame of raw predictors.")
    if list(X_dev.columns) != FEATURES:
        raise ValueError("Development predictors must match FEATURES exactly, with no id or target.")
    y = np.asarray(y_dev)
    if len(y) != len(X_dev) or set(np.unique(y)) != {0, 1, 2, 3}:
        raise ValueError("Development labels must align with X_dev and contain classes 0-3.")
    if min(np.unique(y, return_counts=True)[1]) < N_SPLITS:
        raise ValueError("Each class needs at least five rows for five-fold stratified CV.")
    splitter = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    splits = list(splitter.split(X_dev, y))

    specifications = {
        "Dummy": (
            make_pipeline(DummyClassifier(strategy="most_frequent"), scaled=False),
            {},
        ),
        "Logistic Regression": (
            make_pipeline(LogisticRegression(max_iter=3000, solver="lbfgs", random_state=SEED)),
            {"model__C": [0.1, 1.0, 10.0, 100.0]},
        ),
        "Decision Tree": (
            make_pipeline(DecisionTreeClassifier(random_state=SEED), scaled=False),
            {"model__max_depth": [4, 8, None], "model__min_samples_leaf": [1, 5]},
        ),
        "Random Forest": (
            make_pipeline(RandomForestClassifier(n_estimators=180, random_state=SEED, n_jobs=1), scaled=False),
            {"model__max_depth": [12, None], "model__min_samples_leaf": [1, 3],
             "model__max_features": ["sqrt", 0.7]},
        ),
        "SVM (RBF)": (
            make_calibrated_svm(),
            {"model__estimator__model__C": [1.0, 10.0, 100.0],
             "model__estimator__model__gamma": ["scale", 0.01, 0.1]},
        ),
    }
    models = {}
    summaries = {}
    searches = {}
    tuning_rows = []
    selected_parameters = {}
    for name, (estimator, grid) in specifications.items():
        if verbose:
            print(f"Development CV: {name}")
        search = _search(estimator, grid, X_dev, y, splits)
        models[name] = search.best_estimator_
        summaries[name] = {"model": name, **_search_summary(search), "engineered_areas": False}
        searches[name] = pd.DataFrame(search.cv_results_)
        selected_parameters[name] = dict(search.best_params_)
        tuning_rows.append({
            "model": name,
            "grid_candidates": len(search.cv_results_["params"]),
            "search_cv_fits": len(search.cv_results_["params"]) * N_SPLITS,
            "best_parameters": json.dumps(search.best_params_, sort_keys=True),
        })

    # Products preserve valid zeros and require no ratios or guessed values.
    # Hyperparameters stay fixed across each paired feature comparison.
    ablation_rows = []
    ablation_fold_rows = []
    for name in ["Logistic Regression", "SVM (RBF)"]:
        base_summary = summaries[name].copy()
        base_search_results = searches[name]
        base_best_index = _refit_by_f1_then_accuracy(base_search_results)
        base_f1 = np.array([
            base_search_results.loc[base_best_index, f"split{i}_test_macro_f1"]
            for i in range(N_SPLITS)
        ])
        if name == "SVM (RBF)":
            area_model = clone(models[name]).set_params(
                model__estimator__features__add_areas=True,
                model__estimator__preprocess=make_preprocessor(scaled=True, add_areas=True),
            )
        else:
            area_model = make_pipeline(clone(models[name].named_steps["model"]), scaled=True, add_areas=True)
        area_scores = _cv_scores(area_model, X_dev, y, splits)
        area_summary = _score_summary(area_scores)
        delta = area_summary["cv_macro_f1"] - base_summary["cv_macro_f1"]
        retain_areas = bool(delta > AREA_MIN_GAIN)
        for variant, summary, retained in [
            ("Original features", base_summary, not retain_areas),
            ("Original + pixel_area + screen_area", area_summary, retain_areas),
        ]:
            ablation_rows.append({
                "model": name,
                "feature_variant": variant,
                "cv_macro_f1": summary["cv_macro_f1"],
                "cv_macro_f1_std": summary["cv_macro_f1_std"],
                "cv_accuracy": summary["cv_accuracy"],
                "delta_macro_f1_vs_original": 0.0 if variant == "Original features" else delta,
                "retained": retained,
            })
        for i in range(N_SPLITS):
            ablation_fold_rows.append({
                "model": name, "fold": i + 1,
                "original_macro_f1": float(base_f1[i]),
                "area_macro_f1": float(area_scores["test_macro_f1"][i]),
                "paired_delta": float(area_scores["test_macro_f1"][i] - base_f1[i]),
            })
        if retain_areas:
            models[name] = area_model.fit(X_dev, y)
            summaries[name] = {"model": name, **area_summary, "engineered_areas": True}
        feature_parameter = "model__estimator__features__add_areas" if name == "SVM (RBF)" else "features__add_areas"
        selected_parameters[name][feature_parameter] = retain_areas

    # Each voting member is an entire pipeline, preserving the member's chosen
    # feature configuration and its own fold-fitted transformations.
    voter = Pipeline([("model", VotingClassifier(
        estimators=[
            ("lr", clone(models["Logistic Regression"])),
            ("rf", clone(models["Random Forest"])),
            ("svm", clone(models["SVM (RBF)"])),
        ],
        voting="soft", n_jobs=1,
    ))])
    if verbose:
        print("Development CV: Soft Voting")
    voting_search = _search(voter, {"model__weights": [[1, 1, 1], [2, 1, 2]]}, X_dev, y, splits)
    models["Soft Voting"] = voting_search.best_estimator_
    summaries["Soft Voting"] = {
        "model": "Soft Voting", **_search_summary(voting_search),
        "engineered_areas": any(summaries[n]["engineered_areas"] for n in ["Logistic Regression", "SVM (RBF)"]),
    }
    searches["Soft Voting"] = pd.DataFrame(voting_search.cv_results_)
    selected_parameters["Soft Voting"] = {
        **voting_search.best_params_,
        "lr_parameters": selected_parameters["Logistic Regression"],
        "rf_parameters": selected_parameters["Random Forest"],
        "svm_parameters": selected_parameters["SVM (RBF)"],
    }
    tuning_rows.append({
        "model": "Soft Voting", "grid_candidates": 2, "search_cv_fits": 2 * N_SPLITS,
        "best_parameters": json.dumps(voting_search.best_params_, sort_keys=True),
    })

    # Only training folds are resampled. Random duplication is safe for encoded
    # categorical features; unlike naive SMOTE, it invents no fractional levels.
    no_balance = clone(models["Logistic Regression"])
    weighted = clone(no_balance).set_params(model__class_weight="balanced")
    oversampled = SamplingPipeline([
        ("features", clone(no_balance.named_steps["features"])),
        ("preprocess", clone(no_balance.named_steps["preprocess"])),
        ("sampler", RandomOverSampler(random_state=SEED)),
        ("model", clone(no_balance.named_steps["model"])),
    ])
    fold_counts = [np.unique(y[train_index], return_counts=True)[1] for train_index, _ in splits]
    added_per_fold = [int(len(counts) * counts.max() - counts.sum()) for counts in fold_counts]
    balancing_rows = []
    for strategy, estimator in [
        ("No balancing", no_balance),
        ("Class weight = balanced", weighted),
        ("RandomOverSampler (inside CV)", oversampled),
    ]:
        scores = _cv_scores(estimator, X_dev, y, splits)
        balancing_rows.append({
            "strategy": strategy, "reference_model": "Logistic Regression",
            **_score_summary(scores),
            "mean_added_rows_per_training_fold": float(np.mean(added_per_fold)) if "Sampler" in strategy else 0.0,
        })

    cv_table = pd.DataFrame(summaries.values()).sort_values(
        ["cv_macro_f1", "cv_accuracy", "model"], ascending=[False, False, True],
        kind="stable", ignore_index=True,
    )
    selected_name = str(cv_table.iloc[0]["model"])
    tuning_table = pd.DataFrame(tuning_rows)
    tuning_table["area_ablation_cv_fits"] = tuning_table["model"].isin(["Logistic Regression", "SVM (RBF)"]).astype(int) * N_SPLITS
    tuning_table["retained_area_features"] = tuning_table["model"].map(
        {name: row["engineered_areas"] for name, row in summaries.items()}
    )
    return {
        "models": models,
        "cv_table": cv_table,
        "tuning_table": tuning_table,
        "feature_ablation": pd.DataFrame(ablation_rows),
        "feature_ablation_folds": pd.DataFrame(ablation_fold_rows),
        "balancing_table": pd.DataFrame(balancing_rows),
        "selected_name": selected_name,
        "selected_model": models[selected_name],
        "selected_params": selected_parameters[selected_name],
        "all_selected_params": selected_parameters,
        "search_results": searches,
        "cv_splits": splits,
        "elapsed_seconds": time.perf_counter() - started,
        "methodology": {
            "seed": SEED,
            "folds": N_SPLITS,
            "development_rows": len(X_dev),
            "selection": "Maximum development CV macro-F1; accuracy breaks exact ties.",
            "area_minimum_macro_f1_gain": AREA_MIN_GAIN,
            "area_ablation": "Two zero-safe areas compared jointly; tuned original-feature hyperparameters fixed.",
            "uncertainty": "Fold standard deviations describe variability, not confidence intervals.",
            "selection_bias": "Non-nested CV guides selection; the untouched holdout estimates generalization.",
            "balancing": "Balanced Logistic Regression and RandomOverSampler are training-fold-only controls, not final-model retuning.",
            "svm_probabilities": "Sigmoid CalibratedClassifierCV uses five internal stratified folds with the complete preprocessing/SVC pipeline; ensemble=False refits one estimator on available training rows.",
            "deployment": "Returned models are fitted on development rows only; evaluate once before full-data refit.",
        },
    }
