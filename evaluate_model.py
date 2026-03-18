"""
evaluate_model.py
-----------------
Crop Yield Prediction Project — CipherSense AI Technical Assessment

This module loads the saved trained pipeline from disk and produces
the official final evaluation report — metrics, visualisations and
feature importance analysis.

This file makes no training decisions. It is the independent final
verdict on the saved model's performance on held-out test data.

Usage:
    python src/evaluate_model.py
        --model  models/trained_model.pkl
        --input  data/processed/cleaned_yield.csv
        --output outputs/
"""

import os
import sys
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import joblib

from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score
)

# Add src/ to path so preprocessing can be imported
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from preprocessing import split_data, NUMERICAL_FEATURES, CATEGORICAL_FEATURES


# ─────────────────────────────────────────────
# SECTION 1 — LOAD MODEL AND DATA
# ─────────────────────────────────────────────

def load_model(model_path, verbose=True):
    """
    Loads the saved pipeline from disk.

    Args:
        model_path (str)  : Path to trained_model.pkl.
        verbose    (bool) : Whether to print loading summary.

    Returns:
        pipeline: Fitted sklearn Pipeline (preprocessor + model).
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Trained model not found at: {model_path}\n"
            f"Run train_model.py first to generate it."
        )

    pipeline = joblib.load(model_path)

    if verbose:
        print("=" * 55)
        print("MODEL LOADED")
        print("=" * 55)
        print(f"Loaded from : {model_path}")
        model_step = pipeline.named_steps['model']
        print(f"Model type  : {type(model_step).__name__}")

        # Print tuned parameters if available
        params = model_step.get_params()
        print(f"\nModel parameters:")
        for param, value in params.items():
            print(f"  {param:<25} : {value}")

    return pipeline


def load_data(input_path, verbose=True):
    """
    Loads the cleaned dataset and recreates the exact same train/test
    split used during training.

    Uses identical random_state=42 and test_size=0.2 to guarantee
    X_test and y_test are the same held-out rows the model never
    trained on.

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
        print("\n" + "=" * 55)
        print("DATA LOADED")
        print("=" * 55)
        print(f"Loaded from : {input_path}")

    df = pd.read_csv(input_path)

    # Recreate exact same split as training
    X_train, X_test, y_train, y_test = split_data(
        df,
        test_size    = 0.2,
        random_state = 42,
        verbose      = verbose
    )

    return X_train, X_test, y_train, y_test


# ─────────────────────────────────────────────
# SECTION 2 — OFFICIAL METRICS REPORT
# ─────────────────────────────────────────────

def compute_metrics(pipeline, X_test, y_test, verbose=True):
    """
    Generates predictions and computes official evaluation metrics.

    Metrics reported:
        RMSE — Root Mean Squared Error. Penalises large errors heavily.
               Sensitive to extreme mispredictions in the upper yield range.
        MAE  — Mean Absolute Error. Treats all errors equally.
               More robust to outlier predictions than RMSE.
        R²   — Coefficient of determination. Proportion of yield variance
               explained by the model. 1.0 = perfect, 0.0 = no better
               than predicting the mean for every row.

    Note on RMSE vs MAE gap:
        A large gap between RMSE and MAE indicates the presence of
        high-error predictions concentrated in the upper yield range.
        This is consistent with the heavily right-skewed target
        distribution identified during EDA — the model predicts
        low-to-moderate yields reliably but struggles with extreme
        high yield values that are underrepresented in training data.

    Args:
        pipeline : Fitted sklearn Pipeline.
        X_test   : Test features.
        y_test   : True test labels.
        verbose  : Whether to print metrics report.

    Returns:
        dict: {'rmse', 'mae', 'r2', 'y_pred'}
    """
    y_pred = pipeline.predict(X_test)

    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae  = mean_absolute_error(y_test, y_pred)
    r2   = r2_score(y_test, y_pred)

    if verbose:
        print("\n" + "=" * 55)
        print("OFFICIAL EVALUATION METRICS")
        print("=" * 55)
        print(f"Test set size : {len(y_test):,} rows")
        print(f"\n{'Metric':<8} {'Value':>15}")
        print("-" * 25)
        print(f"{'RMSE':<8} {rmse:>15,.2f}")
        print(f"{'MAE':<8} {mae:>15,.2f}")
        print(f"{'R²':<8} {r2:>15.4f}")
        print(f"\nInterpretation:")
        print(f"  The model explains {r2 * 100:.1f}% of the variance "
              f"in crop yield.")
        print(f"  On average predictions are off by {mae:,.0f} hg/ha.")
        print(f"  The gap between RMSE ({rmse:,.0f}) and MAE ({mae:,.0f})")
        print(f"  indicates high-error predictions in the upper yield")
        print(f"  range — consistent with the right-skewed target")
        print(f"  distribution identified during EDA.")

    return {
        'rmse'   : rmse,
        'mae'    : mae,
        'r2'     : r2,
        'y_pred' : y_pred
    }


# ─────────────────────────────────────────────
# SECTION 3 — VISUALISATIONS
# ─────────────────────────────────────────────

def plot_actual_vs_predicted(y_test, y_pred, output_dir, verbose=True):
    """
    Scatter plot of actual vs predicted crop yield values.

    Each dot represents one prediction. Dots close to the diagonal
    line indicate accurate predictions. Systematic deviation from
    the diagonal reveals where the model under or over predicts.

    Args:
        y_test     : True test labels.
        y_pred     : Model predictions.
        output_dir : Folder to save the plot.
        verbose    : Whether to print save confirmation.
    """
    fig, ax = plt.subplots(figsize=(10, 8))

    # ── Scatter plot ────────────────────────────────────────
    ax.scatter(
        y_test, y_pred,
        alpha  = 0.3,
        color  = 'steelblue',
        s      = 10,
        label  = 'Predictions'
    )

    # ── Perfect prediction diagonal line ───────────────────
    min_val = min(y_test.min(), y_pred.min())
    max_val = max(y_test.max(), y_pred.max())

    ax.plot(
        [min_val, max_val],
        [min_val, max_val],
        color     = 'red',
        linewidth = 2,
        linestyle = '--',
        label     = 'Perfect prediction line'
    )

    # ── R² annotation ──────────────────────────────────────
    r2 = r2_score(y_test, y_pred)
    ax.annotate(
        f'R² = {r2:.4f}',
        xy         = (0.05, 0.92),
        xycoords   = 'axes fraction',
        fontsize   = 12,
        fontweight = 'bold',
        color      = 'darkred',
        bbox       = dict(
            boxstyle = 'round,pad=0.3',
            facecolor = 'lightyellow',
            edgecolor = 'darkred'
        )
    )

    ax.set_title(
        'Actual vs Predicted Crop Yield',
        fontsize   = 14,
        fontweight = 'bold'
    )
    ax.set_xlabel('Actual Yield (hg/ha)',    fontsize=12)
    ax.set_ylabel('Predicted Yield (hg/ha)', fontsize=12)
    ax.legend(fontsize=10)

    plt.tight_layout()

    save_path = os.path.join(output_dir, 'actual_vs_predicted.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()

    if verbose:
        print(f"  Saved : {save_path}")


def plot_residuals(y_test, y_pred, output_dir, verbose=True):
    """
    Residuals plot — difference between actual and predicted yield
    plotted against predicted values.

    A well-behaved model shows residuals scattered randomly around
    zero with no pattern. A funnel shape (residuals spreading as
    predicted values increase) indicates heteroscedasticity — the
    model is less reliable at higher yield values. This is expected
    given the right-skewed target distribution.

    Args:
        y_test     : True test labels.
        y_pred     : Model predictions.
        output_dir : Folder to save the plot.
        verbose    : Whether to print save confirmation.
    """
    residuals = np.array(y_test) - np.array(y_pred)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # ── Left: Residuals vs Predicted ───────────────────────
    axes[0].scatter(
        y_pred, residuals,
        alpha = 0.3,
        color = 'darkorange',
        s     = 10
    )
    axes[0].axhline(
        0,
        color     = 'red',
        linewidth = 2,
        linestyle = '--',
        label     = 'Zero residual line'
    )
    axes[0].set_title(
        'Residuals vs Predicted Values',
        fontsize=13, fontweight='bold'
    )
    axes[0].set_xlabel('Predicted Yield (hg/ha)', fontsize=11)
    axes[0].set_ylabel('Residual (Actual − Predicted)', fontsize=11)
    axes[0].legend(fontsize=10)

    # ── Right: Residuals distribution ──────────────────────
    axes[1].hist(
        residuals,
        bins      = 50,
        color     = 'steelblue',
        edgecolor = 'white',
        linewidth = 0.5
    )
    axes[1].axvline(
        0,
        color     = 'red',
        linewidth = 2,
        linestyle = '--',
        label     = 'Zero residual'
    )
    axes[1].axvline(
        residuals.mean(),
        color     = 'green',
        linewidth = 2,
        linestyle = '-',
        label     = f'Mean residual: {residuals.mean():,.0f}'
    )
    axes[1].set_title(
        'Residuals Distribution',
        fontsize=13, fontweight='bold'
    )
    axes[1].set_xlabel('Residual (Actual − Predicted)', fontsize=11)
    axes[1].set_ylabel('Frequency', fontsize=11)
    axes[1].legend(fontsize=10)

    plt.suptitle(
        'Residual Analysis',
        fontsize=15, fontweight='bold', y=1.01
    )
    plt.tight_layout()

    save_path = os.path.join(output_dir, 'residuals_plot.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()

    if verbose:
        print(f"  Saved : {save_path}")


def plot_feature_importance(pipeline, output_dir, verbose=True):
    """
    Horizontal bar chart of feature importances from the trained model.

    Feature importances reveal which inputs the model relied on most
    when making predictions. Engineered features appearing in the
    top half of the chart validates the feature engineering work.

    Note: Feature importances are only available for tree-based models
    (Random Forest, Gradient Boosting). This step is skipped for
    Linear Regression which does not have feature_importances_.

    Importance scores are normalised — all bars sum to 1.0.
    A higher score means the model relied on that feature more
    heavily when deciding where to split.

    Args:
        pipeline   : Fitted sklearn Pipeline.
        output_dir : Folder to save the plot.
        verbose    : Whether to print save confirmation.
    """
    model_step = pipeline.named_steps['model']

    # Check if model supports feature importances
    if not hasattr(model_step, 'feature_importances_'):
        if verbose:
            print("  Skipped : Model does not support feature importances.")
            print("  (Feature importances are only available for "
                  "tree-based models.)")
        return

    # ── Get feature names after one-hot encoding ───────────
    preprocessor = pipeline.named_steps['preprocessor']
    cat_encoder  = preprocessor.named_transformers_['cat']
    cat_features = cat_encoder.get_feature_names_out(CATEGORICAL_FEATURES)

    # Full feature name list — numerical first then encoded categorical
    # (matches the order ColumnTransformer outputs them)
    all_feature_names = list(NUMERICAL_FEATURES) + list(cat_features)
    importances       = model_step.feature_importances_

    # ── Build importance dataframe ──────────────────────────
    importance_df = pd.DataFrame({
        'feature'    : all_feature_names,
        'importance' : importances
    }).sort_values('importance', ascending=True)

    # ── Separate engineered vs original features for colouring ─
    engineered = [
        'years_since_1990',
        'rain_temp_interaction',
        'pesticides_per_rainfall',
        'avg_temp_squared'
    ]
    colors = [
        'seagreen' if f in engineered else 'steelblue'
        for f in importance_df['feature']
    ]

    # ── Plot top 20 features for readability ───────────────
    top20 = importance_df.tail(20)
    colors_top20 = colors[-20:]

    fig, ax = plt.subplots(figsize=(12, 10))

    bars = ax.barh(
        top20['feature'],
        top20['importance'],
        color     = colors_top20,
        edgecolor = 'white',
        linewidth = 0.5
    )

    # ── Add value labels ────────────────────────────────────
    for bar in bars:
        width = bar.get_width()
        ax.text(
            width + 0.001,
            bar.get_y() + bar.get_height() / 2,
            f'{width:.4f}',
            va       = 'center',
            fontsize = 8
        )

    # ── Legend for colour coding ────────────────────────────
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='seagreen',  label='Engineered feature'),
        Patch(facecolor='steelblue', label='Original feature')
    ]
    ax.legend(
        handles  = legend_elements,
        fontsize = 10,
        loc      = 'lower right'
    )

    ax.set_title(
        f'Feature Importances — Top 20\n'
        f'({type(model_step).__name__})',
        fontsize   = 14,
        fontweight = 'bold'
    )
    ax.set_xlabel('Importance Score', fontsize=12)
    ax.set_ylabel('Feature',          fontsize=12)

    plt.tight_layout()

    save_path = os.path.join(output_dir, 'feature_importance.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()

    if verbose:
        print(f"  Saved : {save_path}")

        # Print top 10 features
        print(f"\n  Top 10 most important features:")
        top10 = importance_df.tail(10).iloc[::-1]
        for _, row in top10.iterrows():
            tag = ' ← engineered' if row['feature'] in engineered else ''
            print(f"    {row['feature']:<35} : "
                  f"{row['importance']:.4f}{tag}")


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
        description='Evaluate the trained crop yield prediction model.'
    )
    parser.add_argument(
        '--model',
        type    = str,
        default = 'models/trained_model.pkl',
        help    = 'Path to trained model '
                  '(default: models/trained_model.pkl)'
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
        default = 'outputs',
        help    = 'Folder to save evaluation plots '
                  '(default: outputs/)'
    )
    parser.add_argument(
        '--silent',
        action  = 'store_true',
        help    = 'Suppress evaluation log output'
    )
    return parser.parse_args()


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

if __name__ == '__main__':

    args    = _parse_args()
    verbose = not args.silent

    # ── Create output folder ────────────────────────────────
    os.makedirs(args.output, exist_ok=True)

    # ── Section 1 — Load model and data ────────────────────
    pipeline = load_model(args.model, verbose=verbose)

    X_train, X_test, y_train, y_test = load_data(
        args.input,
        verbose = verbose
    )

    # ── Section 2 — Official metrics ───────────────────────
    metrics = compute_metrics(
        pipeline, X_test, y_test,
        verbose = verbose
    )

    # ── Section 3 — Visualisations ─────────────────────────
    if verbose:
        print("\n" + "=" * 55)
        print("GENERATING EVALUATION PLOTS")
        print("=" * 55)

    plot_actual_vs_predicted(
        y_test,
        metrics['y_pred'],
        args.output,
        verbose = verbose
    )

    plot_residuals(
        y_test,
        metrics['y_pred'],
        args.output,
        verbose = verbose
    )

    plot_feature_importance(
        pipeline,
        args.output,
        verbose = verbose
    )

    # ── Final summary ───────────────────────────────────────
    if verbose:
        print("\n" + "=" * 55)
        print("EVALUATION COMPLETE")
        print("=" * 55)
        print(f"RMSE : {metrics['rmse']:,.2f}")
        print(f"MAE  : {metrics['mae']:,.2f}")
        print(f"R²   : {metrics['r2']:.4f}")
        print(f"\nPlots saved to : {args.output}/")
        print(f"  - actual_vs_predicted.png")
        print(f"  - residuals_plot.png")
        print(f"  - feature_importance.png")
