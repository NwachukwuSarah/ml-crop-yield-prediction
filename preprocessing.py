"""
preprocessing.py
----------------
Crop Yield Prediction Project — CipherSense AI Technical Assessment

This module builds the sklearn ColumnTransformer preprocessor for the
crop yield dataset. It handles train/test splitting and column-specific
transformations only. Model selection and pipeline assembly happen in
train_model.py.

The preprocessor is designed to be imported by train_model.py:

    from src.preprocessing import build_preprocessor, split_data

Usage from command line (to verify preprocessing runs without errors):
    python src/preprocessing.py --input data/processed/cleaned_yield.csv
"""

import os
import argparse
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing   import StandardScaler, OneHotEncoder
from sklearn.compose         import ColumnTransformer


# ─────────────────────────────────────────────
# COLUMN DEFINITIONS
# ─────────────────────────────────────────────

# Target variable
TARGET = 'hg/ha_yield'

# Categorical features — will be one-hot encoded
CATEGORICAL_FEATURES = [
    'Area',
    'Item'
]

# Numerical features — will be standardised
# Note: Raw 'Year' column is excluded — replaced by 'years_since_1990'
# during feature engineering in data_cleaning.py
NUMERICAL_FEATURES = [
    'average_rain_fall_mm_per_year',
    'pesticides_tonnes',
    'avg_temp',
    'years_since_1990',
    'rain_temp_interaction',
    'pesticides_per_rainfall',
    'avg_temp_squared'
]

# All features used by the model
ALL_FEATURES = CATEGORICAL_FEATURES + NUMERICAL_FEATURES


# ─────────────────────────────────────────────
# SPLIT FUNCTION
# ─────────────────────────────────────────────

def split_data(df, test_size=0.2, random_state=42, verbose=True):
    """
    Separates features from target and splits into train and test sets.

    The split is performed before any transformation to prevent data
    leakage — the scaler and encoder are fitted on training data only
    and applied to test data without seeing test statistics.

    Args:
        df           (pd.DataFrame) : Cleaned dataset from data_cleaning.py.
        test_size    (float)        : Proportion of data reserved for testing.
                                      Default 0.2 = 80/20 train/test split.
        random_state (int)          : Random seed for reproducibility.
        verbose      (bool)         : Whether to print split summary.

    Returns:
        X_train, X_test, y_train, y_test
    """

    # ── Validate required columns ───────────────────────────
    missing_cols = [c for c in ALL_FEATURES + [TARGET]
                    if c not in df.columns]
    if missing_cols:
        raise ValueError(
            f"The following required columns are missing:\n"
            f"  {missing_cols}\n"
            f"Ensure data_cleaning.py has been run on the raw data."
        )

    # ── Separate features and target ───────────────────────
    X = df[ALL_FEATURES]
    y = df[TARGET]

    # ── Train / test split ─────────────────────────────────
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size    = test_size,
        random_state = random_state
    )

    if verbose:
        print("=" * 55)
        print("TRAIN / TEST SPLIT")
        print("=" * 55)
        print(f"Total samples  : {len(df):,}")
        print(f"Training set   : {len(X_train):,} rows "
              f"({100 * (1 - test_size):.0f}%)")
        print(f"Test set       : {len(X_test):,} rows "
              f"({100 * test_size:.0f}%)")
        print(f"\nFeatures used  : {len(ALL_FEATURES)}")
        print(f"  Numerical    : {NUMERICAL_FEATURES}")
        print(f"  Categorical  : {CATEGORICAL_FEATURES}")
        print(f"\nTarget         : {TARGET}")

    return X_train, X_test, y_train, y_test


# ─────────────────────────────────────────────
# PREPROCESSOR BUILDER
# ─────────────────────────────────────────────

def build_preprocessor(verbose=True):
    """
    Constructs and returns the ColumnTransformer preprocessor.

    The preprocessor applies different transformations to different
    column types simultaneously:

        Numerical features   → StandardScaler
        Categorical features → OneHotEncoder

    This preprocessor is returned unfitted. It is passed into a full
    sklearn Pipeline in train_model.py where it is fitted on training
    data only — preventing data leakage.

    Why StandardScaler for numerical features:
        Numerical features operate on very different scales —
        pesticides_tonnes reaches tens of thousands while avg_temp
        sits between 0 and 30. Scaling brings all numerical features
        onto a comparable scale. Although tree-based models are not
        strictly scale-sensitive, scaling is applied for pipeline
        consistency and to ensure the pipeline works correctly when
        a linear baseline model is also passed through it.

    Why OneHotEncoder for categorical features:
        Area and Item are nominal categories — there is no meaningful
        ordinal relationship between countries or crop types. One-hot
        encoding creates one binary column per category, allowing the
        model to treat each country and crop independently without
        implying any ranking or ordering between them.

        handle_unknown='ignore' ensures that any unseen category
        in test or future data is silently encoded as all zeros
        rather than raising an error.

    Why encoding happens here and not in data_cleaning.py:
        Applying OneHotEncoder inside the Pipeline ensures the encoder
        is fitted on training data only. Encoding before the split
        would expose test set categories to the encoder during fitting
        — a form of data leakage producing overly optimistic results.

    Why sparse_output=False:
        Returns a dense numpy array instead of a sparse matrix.
        This ensures compatibility with downstream pipeline steps
        and avoids type errors during model training.

    Why remainder='drop':
        Any column not explicitly listed in NUMERICAL_FEATURES or
        CATEGORICAL_FEATURES is silently dropped. This acts as a
        safety net against stray columns corrupting model input.

    Args:
        verbose (bool): Whether to print preprocessor summary.

    Returns:
        sklearn.compose.ColumnTransformer: Unfitted preprocessor.
    """

    # ── Numerical transformer ───────────────────────────────
    numerical_transformer = StandardScaler()

    # ── Categorical transformer ─────────────────────────────
    categorical_transformer = OneHotEncoder(
        handle_unknown = 'ignore',
        sparse_output  = False
    )

    # ── ColumnTransformer ───────────────────────────────────
    preprocessor = ColumnTransformer(
        transformers = [
            ('num', numerical_transformer, NUMERICAL_FEATURES),
            ('cat', categorical_transformer, CATEGORICAL_FEATURES)
        ],
        remainder = 'drop'
    )

    if verbose:
        print("=" * 55)
        print("PREPROCESSOR STRUCTURE")
        print("=" * 55)
        print(f"  num → StandardScaler  : {NUMERICAL_FEATURES}")
        print(f"  cat → OneHotEncoder   : {CATEGORICAL_FEATURES}")
        print(f"\nPreprocessor ready — will be fitted in train_model.py")

    return preprocessor


# ─────────────────────────────────────────────
# COMMAND LINE INTERFACE
# ─────────────────────────────────────────────

def _parse_args():
    """
    Parses command line arguments for verifying the preprocessor.

    Returns:
        argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description='Verify the preprocessor runs without errors.'
    )
    parser.add_argument(
        '--input',
        type    = str,
        default = 'data/processed/cleaned_yield.csv',
        help    = 'Path to cleaned dataset '
                  '(default: data/processed/cleaned_yield.csv)'
    )
    parser.add_argument(
        '--test-size',
        type    = float,
        default = 0.2,
        help    = 'Test set proportion (default: 0.2)'
    )
    parser.add_argument(
        '--random-state',
        type    = int,
        default = 42,
        help    = 'Random seed for reproducibility (default: 42)'
    )
    return parser.parse_args()


if __name__ == '__main__':

    from sklearn.pipeline import Pipeline
    from sklearn.ensemble import RandomForestRegressor

    args = _parse_args()

    # ── Load cleaned dataset ────────────────────────────────
    if not os.path.exists(args.input):
        raise FileNotFoundError(
            f"Cleaned dataset not found at: {args.input}\n"
            f"Run data_cleaning.py first to generate it."
        )

    print(f"Loading cleaned dataset from : {args.input}")
    df = pd.read_csv(args.input)
    print(f"Loaded — shape: {df.shape}")

    # ── Split data ──────────────────────────────────────────
    X_train, X_test, y_train, y_test = split_data(
        df,
        test_size    = args.test_size,
        random_state = args.random_state,
        verbose      = True
    )

    # ── Build preprocessor ──────────────────────────────────
    preprocessor = build_preprocessor(verbose=True)

    # ── Assemble temporary pipeline to verify fit ───────────
    # Note: This pipeline is for verification only.
    # The real pipeline is assembled in train_model.py.
    temp_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('model',        RandomForestRegressor(random_state=42))
    ])

    # ── Fit on training data ────────────────────────────────
    print("\n" + "=" * 55)
    print("VERIFICATION — FITTING PIPELINE ON TRAINING DATA")
    print("=" * 55)

    temp_pipeline.fit(X_train, y_train)
    print("Pipeline fitted successfully.")

    # ── Check transformed shape ─────────────────────────────
    preprocessor_step = temp_pipeline.named_steps['preprocessor']
    X_test_transformed = preprocessor_step.transform(X_test)

    cat_encoder  = preprocessor_step.named_transformers_['cat']
    encoded_cats = cat_encoder.get_feature_names_out(CATEGORICAL_FEATURES)

    print(f"\nX_train shape (raw)         : {X_train.shape}")
    print(f"X_test  shape (raw)         : {X_test.shape}")
    print(f"X_test  shape (transformed) : {X_test_transformed.shape}")
    print(f"\nTotal features after encoding : "
          f"{len(NUMERICAL_FEATURES) + len(encoded_cats)}")
    print(f"  Numerical features          : {len(NUMERICAL_FEATURES)}")
    print(f"  One-hot encoded features    : {len(encoded_cats)}")
    print(f"\nSample encoded category names :")
    print(f"  {list(encoded_cats[:5])} ...")

    print("\n" + "=" * 55)
    print("PREPROCESSING VERIFICATION COMPLETE — NO ERRORS")
    print("=" * 55)
