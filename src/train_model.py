"""
train_model.py
--------------
Crop Yield Prediction Project — CipherSense AI Technical Assessment

This module trains multiple models on the cleaned crop yield dataset,
compares their performance, tunes the best model using RandomizedSearchCV,
and saves the final tuned pipeline to disk.

RandomizedSearchCV is used with refit=True (sklearn default) which means
after finding the best hyperparameters, sklearn automatically retrains
the best model on the full training set before returning it. So
search.best_estimator_ is always a model trained on the complete X_train
with the best parameters found — no manual retraining step needed.

Usage:
    python src/train_model.py --input data/processed/cleaned_yield.csv
                               --output models/trained_model.pkl
"""

import os
import sys
import argparse
import pandas as pd
import numpy as np
import joblib

from sklearn.pipeline        import Pipeline
from sklearn.linear_model    import LinearRegression
from sklearn.ensemble        import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics         import mean_squared_error, mean_absolute_error, r2_score

# Add src/ to path so preprocessing can be imported
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from preprocessing import build_preprocessor, split_data


# ─────────────────────────────────────────────
# MODEL DEFINITIONS
# ─────────────────────────────────────────────

MODELS = {
    'Linear Regression' : LinearRegression(),
    'Random Forest'     : RandomForestRegressor(random_state=42),
    'Gradient Boosting' : GradientBoostingRegressor(random_state=42)
}

# Hyperparameter search space for Random Forest
RF_PARAM_GRID = {
    'model__n_estimators'      : [100, 200, 300, 500],
    'model__max_depth'         : [None, 10, 20, 30],
    'model__min_samples_split' : [2, 5, 10],
    'model__min_samples_leaf'  : [1, 2, 4],
    'model__max_features'      : ['sqrt', 'log2']
}

# Hyperparameter search space for Gradient Boosting
GB_PARAM_GRID = {
    'model__n_estimators'      : [100, 200, 300],
    'model__learning_rate'     : [0.01, 0.05, 0.1, 0.2],
    'model__max_depth'         : [3, 5, 7],
    'model__subsample'         : [0.7, 0.8, 1.0],
    'model__min_samples_split' : [2, 5, 10]
}

PARAM_GRIDS = {
    'Random Forest'     : RF_PARAM_GRID,
    'Gradient Boosting' : GB_PARAM_GRID
}


# ─────────────────────────────────────────────
# HELPER FUNCTIONS
# ─────────────────────────────────────────────

def build_pipeline_for_model(model, verbose=False):
    """
    Assembles a full sklearn Pipeline for a given model by combining
    the preprocessor from preprocessing.py with the model estimator.

    Args:
        model   : Any sklearn-compatible regressor.
        verbose : Whether to print preprocessor summary.

    Returns:
        sklearn.pipeline.Pipeline: Unfitted pipeline.
    """
    preprocessor = build_preprocessor(verbose=verbose)

    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('model',        model)
    ])

    return pipeline


def evaluate_pipeline(pipeline, X_test, y_test):
    """
    Generates predictions from a fitted pipeline and computes
    RMSE, MAE and R² against the test set.

    Args:
        pipeline : Fitted sklearn Pipeline.
        X_test   : Test features.
        y_test   : True test labels.

    Returns:
        dict: {'rmse', 'mae', 'r2', 'y_pred'}
    """
    y_pred = pipeline.predict(X_test)

    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae  = mean_absolute_error(y_test, y_pred)
    r2   = r2_score(y_test, y_pred)

    return {
        'rmse'   : rmse,
        'mae'    : mae,
        'r2'     : r2,
        'y_pred' : y_pred
    }


# ─────────────────────────────────────────────
# SECTION 1 — LOAD DATA
# ─────────────────────────────────────────────

def load_data(input_path, verbose=True):
    """
    Loads the cleaned dataset and performs the train/test split.

    Args:
        input_path (str)  : Path to cleaned_yield.csv.
        verbose    (bool) : Whether to print loading summary.

    Returns:
        X_train, X_test, y_train, y_test
    """
    if not os.path.exists(input_path):
        raise FileNotFoundError(
            f"Cleaned dataset not found at: {input_path}\n"
            f"Run data_cleaning.py first to generate it."
        )

    if verbose:
        print("=" * 55)
        print("LOADING DATA")
        print("=" * 55)
        print(f"Loading from : {input_path}")

    df = pd.read_csv(input_path)

    if verbose:
        print(f"Loaded — shape: {df.shape}")

    X_train, X_test, y_train, y_test = split_data(
        df,
        test_size    = 0.2,
        random_state = 42,
        verbose      = verbose
    )

    return X_train, X_test, y_train, y_test


# ─────────────────────────────────────────────
# SECTION 2 — TRAIN AND COMPARE ALL MODELS
# ─────────────────────────────────────────────

def train_and_compare(X_train, X_test, y_train, y_test, verbose=True):
    """
    Trains all models defined in MODELS, evaluates each on the test
    set and prints a comparison table.

    Args:
        X_train, X_test, y_train, y_test : Output of split_data().
        verbose (bool) : Whether to print training log.

    Returns:
        results   (dict) : Metrics and fitted pipeline per model name.
        best_name (str)  : Name of the best performing model by R².
    """
    results = {}

    if verbose:
        print("\n" + "=" * 55)
        print("TRAINING AND COMPARING ALL MODELS")
        print("=" * 55)

    for name, model in MODELS.items():

        if verbose:
            print(f"\nTraining : {name} ...")

        # Build fresh pipeline for this model
        pipeline = build_pipeline_for_model(model, verbose=False)
        pipeline.fit(X_train, y_train)

        # Evaluate on test set
        metrics = evaluate_pipeline(pipeline, X_test, y_test)

        results[name] = {
            'pipeline' : pipeline,
            'rmse'     : metrics['rmse'],
            'mae'      : metrics['mae'],
            'r2'       : metrics['r2']
        }

        if verbose:
            print(f"  RMSE : {metrics['rmse']:>15,.2f}")
            print(f"  MAE  : {metrics['mae']:>15,.2f}")
            print(f"  R²   : {metrics['r2']:>15.4f}")

    # ── Comparison table ────────────────────────────────────
    if verbose:
        print("\n" + "=" * 55)
        print("MODEL COMPARISON TABLE")
        print("=" * 55)
        print(f"{'Model':<25} {'RMSE':>12} {'MAE':>12} {'R²':>8}")
        print("-" * 55)
        for name, res in results.items():
            print(f"{name:<25} "
                  f"{res['rmse']:>12,.2f} "
                  f"{res['mae']:>12,.2f} "
                  f"{res['r2']:>8.4f}")

    # ── Pick best model by R² ───────────────────────────────
    best_name = max(results, key=lambda k: results[k]['r2'])

    if verbose:
        print(f"\nBest model : {best_name} "
              f"(R² = {results[best_name]['r2']:.4f})")

    return results, best_name


# ─────────────────────────────────────────────
# SECTION 3 — HYPERPARAMETER TUNING
# ─────────────────────────────────────────────

def tune_best_model(best_name, X_train, y_train, verbose=True):
    """
    Runs RandomizedSearchCV on the best model to find better
    hyperparameters than the defaults.

    Uses refit=True (sklearn default) which means after finding
    the best hyperparameters, sklearn automatically retrains the
    model on the full X_train before returning search.best_estimator_.
    No manual retraining step is required.

    RandomizedSearchCV is used over GridSearchCV because the search
    space for tree-based models is large — grid search would try
    every single combination which is computationally expensive.
    Randomized search samples n_iter combinations and reliably finds
    good hyperparameters at a fraction of the cost.

    Settings:
        n_iter = 20  — number of random combinations to try
        cv     = 5   — 5-fold cross validation per combination
        Total fits = 20 x 5 = 100

    Note: If the best model is Linear Regression (no meaningful
    hyperparameters to tune) this step is skipped and a freshly
    fitted default pipeline is returned.

    Args:
        best_name (str) : Name of best model from train_and_compare().
        X_train         : Full training features.
        y_train         : Full training labels.
        verbose (bool)  : Whether to print tuning log.

    Returns:
        tuned_pipeline: Best pipeline found by RandomizedSearchCV,
                        refitted on full X_train automatically.
    """

    if best_name not in PARAM_GRIDS:
        if verbose:
            print(f"\n{best_name} has no hyperparameter grid defined.")
            print("Skipping tuning — returning default fitted pipeline.")

        pipeline = build_pipeline_for_model(
            MODELS[best_name], verbose=False
        )
        pipeline.fit(X_train, y_train)
        return pipeline

    if verbose:
        print("\n" + "=" * 55)
        print(f"HYPERPARAMETER TUNING — {best_name.upper()}")
        print("=" * 55)
        print(f"Method    : RandomizedSearchCV (refit=True)")
        print(f"n_iter    : 20")
        print(f"cv folds  : 5")
        print(f"Total fits: 100")
        print(f"\nNote: refit=True means sklearn automatically retrains")
        print(f"the best model on full X_train after search completes.")
        print(f"\nRunning search — this may take a few minutes ...")

    # Build fresh pipeline for tuning
    pipeline   = build_pipeline_for_model(
        MODELS[best_name], verbose=False
    )
    param_grid = PARAM_GRIDS[best_name]

    search = RandomizedSearchCV(
        estimator           = pipeline,
        param_distributions = param_grid,
        n_iter              = 20,
        cv                  = 5,
        scoring             = 'r2',
        refit               = True,   # retrains on full X_train automatically
        random_state        = 42,
        n_jobs              = -1,
        verbose             = 1
    )

    search.fit(X_train, y_train)

    if verbose:
        print(f"\nBest parameters found:")
        for param, value in search.best_params_.items():
            clean_param = param.replace('model__', '')
            print(f"  {clean_param:<25} : {value}")
        print(f"\nBest cross-validated R² : {search.best_score_:.4f}")
        print(f"\nsearch.best_estimator_ is now refitted on full X_train.")

    # search.best_estimator_ is already refitted on full X_train
    return search.best_estimator_


# ─────────────────────────────────────────────
# SECTION 4 — SAVE BEST MODEL
# ─────────────────────────────────────────────

def save_model(pipeline, output_path, verbose=True):
    """
    Saves the full tuned pipeline to disk using joblib.

    The complete pipeline — preprocessor plus model — is saved,
    not just the model weights. This ensures that evaluate_model.py
    can load it and call .predict() directly on raw unprocessed
    data without needing to reapply scaling or encoding manually.

    Args:
        pipeline    : Fitted tuned sklearn Pipeline to save.
        output_path : Path to save the .pkl file.
        verbose     : Whether to print save confirmation.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    joblib.dump(pipeline, output_path)

    if verbose:
        file_size = os.path.getsize(output_path) / (1024 * 1024)
        print("\n" + "=" * 55)
        print("MODEL SAVED")
        print("=" * 55)
        print(f"Saved to  : {output_path}")
        print(f"File size : {file_size:.2f} MB")


# ─────────────────────────────────────────────
# COMMAND LINE INTERFACE
# ─────────────────────────────────────────────

def _parse_args():
    """
    Parses command line arguments.

    Returns:
        argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description='Train and tune crop yield prediction models.'
    )
    parser.add_argument(
        '--input',
        type    = str,
        default = 'data/processed/cleaned_yield.csv',
        help    = 'Path to cleaned dataset '
                  '(default: data/processed/cleaned_yield.csv)'
    )
    parser.add_argument(
        '--output',
        type    = str,
        default = 'models/trained_model.pkl',
        help    = 'Path to save best model '
                  '(default: models/trained_model.pkl)'
    )
    parser.add_argument(
        '--silent',
        action  = 'store_true',
        help    = 'Suppress training log output'
    )
    return parser.parse_args()


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

if __name__ == '__main__':

    args    = _parse_args()
    verbose = not args.silent

    # ── Section 1 — Load data ───────────────────────────────
    X_train, X_test, y_train, y_test = load_data(
        args.input,
        verbose = verbose
    )

    # ── Section 2 — Train and compare all models ────────────
    results, best_name = train_and_compare(
        X_train, X_test,
        y_train, y_test,
        verbose = verbose
    )

    # ── Section 3 — Tune best model ─────────────────────────
    tuned_pipeline = tune_best_model(
        best_name,
        X_train,
        y_train,
        verbose = verbose
    )

    # ── Evaluate tuned model on test set ────────────────────
    tuned_metrics = evaluate_pipeline(
        tuned_pipeline, X_test, y_test
    )

    if verbose:
        print("\n" + "=" * 55)
        print("TUNED MODEL — FINAL TEST SET PERFORMANCE")
        print("=" * 55)
        print(f"Model : {best_name} (tuned)")
        print(f"RMSE  : {tuned_metrics['rmse']:>15,.2f}")
        print(f"MAE   : {tuned_metrics['mae']:>15,.2f}")
        print(f"R²    : {tuned_metrics['r2']:>15.4f}")

        # Compare tuned vs untuned
        default_r2  = results[best_name]['r2']
        tuned_r2    = tuned_metrics['r2']
        improvement = tuned_r2 - default_r2

        print(f"\nDefault R²      : {default_r2:.4f}")
        print(f"Tuned R²        : {tuned_r2:.4f}")
        print(f"Improvement     : {improvement:+.4f}")

    # ── Section 4 — Save best model ─────────────────────────
    save_model(
        tuned_pipeline,
        args.output,
        verbose = verbose
    )

    if verbose:
        print(f"\nTraining complete.")
        print(f"Run evaluate_model.py to generate final "
              f"evaluation plots and report.")
