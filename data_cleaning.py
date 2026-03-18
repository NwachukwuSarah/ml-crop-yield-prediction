"""
data_cleaning.py
----------------
Crop Yield Prediction Project — CipherSense AI Technical Assessment

This module contains all data cleaning logic for the crop yield dataset.
It is designed to be reusable — any raw dataset loaded from Kaggle can be
passed through clean_data() and returned in a fully cleaned state ready
for the preprocessing pipeline.

Usage:
    from src.data_cleaning import clean_data
    df_clean = clean_data(df_raw)

Or run directly from the command line:
    python src/data_cleaning.py --input data/raw/yield_df.csv
                                --output data/processed/cleaned_yield.csv
"""

import os
import argparse
import pandas as pd
import numpy as np


# ─────────────────────────────────────────────
# MAIN CLEANING FUNCTION
# ─────────────────────────────────────────────

def clean_data(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """
    Applies all cleaning steps to the raw crop yield dataset.

    Cleaning steps applied (in order):
        1. Drop redundant index column ('Unnamed: 0') if present
        2. Remove duplicate rows
        3. Verify missing values (none expected — logged if found)
        4. Fix inconsistent categorical labels
        5. Correct data types
        6. Feature engineering (derived features)

    Args:
        df (pd.DataFrame): Raw input dataframe loaded from CSV.
        verbose (bool): If True, prints a step-by-step cleaning log.
                        Set to False for silent operation in pipelines.

    Returns:
        pd.DataFrame: Fully cleaned dataframe ready for preprocessing.
    """

    df = df.copy()

    if verbose:
        print("=" * 55)
        print("DATA CLEANING PIPELINE — START")
        print("=" * 55)
        print(f"Input shape  : {df.shape}")

    # ── Step 1: Drop unnamed index column ──────────────────
    df = _drop_unnamed_column(df, verbose)

    # ── Step 2: Remove duplicates ───────────────────────────
    df = _remove_duplicates(df, verbose)

    # ── Step 3: Verify missing values ───────────────────────
    df = _handle_missing_values(df, verbose)

    # ── Step 4: Fix categorical labels ──────────────────────
    df = _fix_categorical_labels(df, verbose)

    # ── Step 5: Fix data types ───────────────────────────────
    df = _fix_data_types(df, verbose)

    # ── Step 6: Feature engineering ─────────────────────────
    df = _engineer_features(df, verbose)

    if verbose:
        print("\n" + "=" * 55)
        print("DATA CLEANING PIPELINE — COMPLETE")
        print("=" * 55)
        print(f"Output shape : {df.shape}")
        print(f"Missing values remaining : {df.isnull().sum().sum()}")
        print(f"Duplicate rows remaining : {df.duplicated().sum()}")
        print("\nFinal columns:")
        for col in df.columns:
            print(f"  - {col}")

    return df


# ─────────────────────────────────────────────
# STEP 1 — DROP UNNAMED INDEX COLUMN
# ─────────────────────────────────────────────

def _drop_unnamed_column(df: pd.DataFrame, verbose: bool) -> pd.DataFrame:
    """
    Drops the 'Unnamed: 0' column if present.

    This column is a leftover artifact from how the original Kaggle CSV
    was saved — it contains a redundant row index that carries no useful
    information for modelling.

    A conditional check is used so this function works safely on both
    raw data (column present) and pre-processed data (column absent).

    Args:
        df (pd.DataFrame): Input dataframe.
        verbose (bool): Whether to print cleaning log.

    Returns:
        pd.DataFrame: Dataframe with 'Unnamed: 0' removed if it existed.
    """
    if verbose:
        print("\n── Step 1: Drop Unnamed Index Column ──────────────")

    if 'Unnamed: 0' in df.columns:
        df = df.drop(columns=['Unnamed: 0'])
        if verbose:
            print("  Applied : 'Unnamed: 0' column dropped.")
    else:
        if verbose:
            print("  Skipped : 'Unnamed: 0' not present.")

    return df


# ─────────────────────────────────────────────
# STEP 2 — REMOVE DUPLICATES
# ─────────────────────────────────────────────

def _remove_duplicates(df: pd.DataFrame, verbose: bool) -> pd.DataFrame:
    """
    Removes duplicate rows, keeping the first occurrence of each.

    Duplicate rows provide no additional information to the model and
    can bias training by overrepresenting certain patterns. All duplicates
    are removed unconditionally — unlike missing values, duplicates have
    no grey area requiring investigation.

    Args:
        df (pd.DataFrame): Input dataframe.
        verbose (bool): Whether to print cleaning log.

    Returns:
        pd.DataFrame: Dataframe with duplicate rows removed.
    """
    if verbose:
        print("\n── Step 2: Remove Duplicate Rows ──────────────────")

    before = len(df)
    df = df.drop_duplicates(keep='first')
    after  = len(df)
    removed = before - after

    if verbose:
        print(f"  Rows before : {before:,}")
        print(f"  Rows after  : {after:,}")
        print(f"  Removed     : {removed:,}")
        print(f"  Duplicates remaining : {df.duplicated().sum()}")

    return df


# ─────────────────────────────────────────────
# STEP 3 — HANDLE MISSING VALUES
# ─────────────────────────────────────────────

def _handle_missing_values(df: pd.DataFrame, verbose: bool) -> pd.DataFrame:
    """
    Checks for missing values and applies imputation if found.

    No missing values were present in the original dataset. This step
    is included as a defensive measure to ensure the cleaning pipeline
    handles future data gracefully — applying median imputation for
    numerical columns and mode imputation for categorical columns.

    Strategy:
        - Numerical columns  : impute with median (robust to outliers)
        - Categorical columns: impute with mode (most frequent value)

    Args:
        df (pd.DataFrame): Input dataframe.
        verbose (bool): Whether to print cleaning log.

    Returns:
        pd.DataFrame: Dataframe with missing values handled.
    """
    if verbose:
        print("\n── Step 3: Handle Missing Values ──────────────────")

    total_missing = df.isnull().sum().sum()

    if total_missing == 0:
        if verbose:
            print("  Confirmed : No missing values found.")
            print("  No imputation required.")
        return df

    # Missing values found — apply imputation
    if verbose:
        print(f"  Warning : {total_missing} missing values found.")
        print("  Applying imputation strategy:")

    numerical_cols  = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()

    # Median imputation for numerical columns
    for col in numerical_cols:
        missing_count = df[col].isnull().sum()
        if missing_count > 0:
            median_val = df[col].median()
            df[col] = df[col].fillna(median_val)
            if verbose:
                print(f"    {col}: {missing_count} values → median ({median_val:.4f})")

    # Mode imputation for categorical columns
    for col in categorical_cols:
        missing_count = df[col].isnull().sum()
        if missing_count > 0:
            mode_val = df[col].mode()[0]
            df[col] = df[col].fillna(mode_val)
            if verbose:
                print(f"    {col}: {missing_count} values → mode ('{mode_val}')")

    if verbose:
        print(f"  Missing values after imputation : {df.isnull().sum().sum()}")

    return df


# ─────────────────────────────────────────────
# STEP 4 — FIX CATEGORICAL LABELS
# ─────────────────────────────────────────────

def _fix_categorical_labels(df: pd.DataFrame, verbose: bool) -> pd.DataFrame:
    """
    Corrects inconsistent categorical labels in the Item (crop type) column.

    Two label issues were identified during EDA:

    Issue 1 — 'Rice, paddy':
        The FAO technical label 'Rice, paddy' refers to rice at the
        post-harvest stage with husk intact. The word 'paddy' describes
        the form of the crop at harvest rather than a different crop
        entirely. Renamed to 'Rice' for consistency with the naming
        convention used across all other crop types in the dataset.

    Issue 2 — 'Plantains and others':
        The 'and others' suffix makes this category ambiguous. Renamed
        to 'Plantains' because the category contains sufficient records
        to contribute meaningful patterns to the model, and discarding
        it entirely would lose more signal than the ambiguity introduces
        noise.

    Args:
        df (pd.DataFrame): Input dataframe.
        verbose (bool): Whether to print cleaning log.

    Returns:
        pd.DataFrame: Dataframe with corrected categorical labels.
    """
    if verbose:
        print("\n── Step 4: Fix Categorical Labels ─────────────────")

    label_fixes = {
        'Rice, paddy'          : 'Rice',
        'Plantains and others' : 'Plantains'
    }

    if 'Item' not in df.columns:
        if verbose:
            print("  Skipped : 'Item' column not found.")
        return df

    for old_label, new_label in label_fixes.items():
        count = (df['Item'] == old_label).sum()
        if count > 0:
            df['Item'] = df['Item'].replace({old_label: new_label})
            if verbose:
                print(f"  Fixed : '{old_label}' → '{new_label}' "
                      f"({count:,} rows updated)")
        else:
            if verbose:
                print(f"  Skipped : '{old_label}' not found in data.")

    # Verification
    if verbose:
        print(f"\n  Verification:")
        for old_label in label_fixes.keys():
            remaining = (df['Item'] == old_label).sum()
            status = "PASS" if remaining == 0 else "FAIL"
            print(f"    [{status}] '{old_label}' remaining : {remaining}")

    return df


# ─────────────────────────────────────────────
# STEP 5 — FIX DATA TYPES
# ─────────────────────────────────────────────

def _fix_data_types(df: pd.DataFrame, verbose: bool) -> pd.DataFrame:
    """
    Ensures all columns have the correct data types for modelling.

    Expected types:
        - Area                          : object  (categorical)
        - Item                          : object  (categorical)
        - Year                          : int64   (numerical)
        - hg/ha_yield                   : float64 (numerical — target)
        - average_rain_fall_mm_per_year : float64 (numerical)
        - pesticides_tonnes             : float64 (numerical)
        - avg_temp                      : float64 (numerical)

    Args:
        df (pd.DataFrame): Input dataframe.
        verbose (bool): Whether to print cleaning log.

    Returns:
        pd.DataFrame: Dataframe with corrected data types.
    """
    if verbose:
        print("\n── Step 5: Fix Data Types ──────────────────────────")

    expected_types = {
        'Area'                          : 'object',
        'Item'                          : 'object',
        'Year'                          : 'int64',
        'hg/ha_yield'                   : 'float64',
        'average_rain_fall_mm_per_year' : 'float64',
        'pesticides_tonnes'             : 'float64',
        'avg_temp'                      : 'float64'
    }

    for col, expected_type in expected_types.items():
        if col not in df.columns:
            if verbose:
                print(f"  Skipped : '{col}' not in dataframe.")
            continue

        actual_type = str(df[col].dtype)
        if actual_type != expected_type:
            try:
                df[col] = df[col].astype(expected_type)
                if verbose:
                    print(f"  Fixed   : {col:<35} "
                          f"{actual_type} → {expected_type}")
            except Exception as e:
                if verbose:
                    print(f"  ERROR   : Could not convert {col} — {e}")
        else:
            if verbose:
                print(f"  OK      : {col:<35} already {expected_type}")

    return df


# ─────────────────────────────────────────────
# STEP 6 — FEATURE ENGINEERING
# ─────────────────────────────────────────────

def _engineer_features(df: pd.DataFrame, verbose: bool) -> pd.DataFrame:
    """
    Creates new derived features from existing columns.

    Four features are engineered based on domain knowledge of
    agricultural patterns and findings from EDA:

    1. years_since_1990:
        Converts raw calendar year into a measure of agricultural
        progress since the start of the dataset. More meaningful
        to the model than a raw year number.

    2. rain_temp_interaction:
        Multiplies rainfall by temperature to capture the combined
        climate effect on yield. Agricultural science shows that
        heat and moisture together drive plant growth — neither
        factor alone captures the full picture.

    3. pesticides_per_rainfall:
        Divides pesticide usage by rainfall to represent effective
        pesticide concentration. Heavy rainfall washes pesticides
        off crops before they take full effect — this ratio captures
        that real-world relationship. A small epsilon (1e-6) is added
        to the denominator to prevent division by zero.

    4. avg_temp_squared:
        Adds a squared temperature term to capture the non-linear
        relationship between temperature and yield. Crop growth
        peaks at an optimal temperature and drops on both sides —
        a pattern linear features cannot express.

    Note:
        Categorical encoding (OneHotEncoder) is intentionally excluded
        from this step. It is applied inside the sklearn Pipeline during
        preprocessing to prevent data leakage — ensuring the encoder
        learns category mappings from training data only and never
        sees test data statistics during fitting.

    Args:
        df (pd.DataFrame): Input dataframe.
        verbose (bool): Whether to print cleaning log.

    Returns:
        pd.DataFrame: Dataframe with four new engineered features added.
    """
    if verbose:
        print("\n── Step 6: Feature Engineering ────────────────────")

    cols_before = df.shape[1]

    # Feature 1 — Years since 1990
    if 'Year' in df.columns:
        df['years_since_1990'] = df['Year'] - 1990
        if verbose:
            print(f"  Created : years_since_1990  "
                  f"(range: {df['years_since_1990'].min()} "
                  f"to {df['years_since_1990'].max()})")

    # Feature 2 — Rainfall × Temperature interaction
    if all(c in df.columns for c in
           ['average_rain_fall_mm_per_year', 'avg_temp']):
        df['rain_temp_interaction'] = (
            df['average_rain_fall_mm_per_year'] * df['avg_temp']
        )
        if verbose:
            print(f"  Created : rain_temp_interaction  "
                  f"(mean: {df['rain_temp_interaction'].mean():.2f})")

    # Feature 3 — Pesticides per rainfall
    if all(c in df.columns for c in
           ['pesticides_tonnes', 'average_rain_fall_mm_per_year']):
        df['pesticides_per_rainfall'] = (
            df['pesticides_tonnes'] /
            (df['average_rain_fall_mm_per_year'] + 1e-6)
        )
        if verbose:
            print(f"  Created : pesticides_per_rainfall  "
                  f"(mean: {df['pesticides_per_rainfall'].mean():.4f})")

    # Feature 4 — Temperature squared
    if 'avg_temp' in df.columns:
        df['avg_temp_squared'] = df['avg_temp'] ** 2
        if verbose:
            print(f"  Created : avg_temp_squared  "
                  f"(mean: {df['avg_temp_squared'].mean():.2f})")

    cols_after = df.shape[1]
    if verbose:
        print(f"\n  Features added : {cols_after - cols_before}")
        print(f"  Total columns  : {cols_after}")

    return df


# ─────────────────────────────────────────────
# COMMAND LINE INTERFACE
# ─────────────────────────────────────────────

def _parse_args():
    """
    Parses command line arguments for running the cleaning script directly.

    Returns:
        argparse.Namespace: Parsed arguments with input and output paths.
    """
    parser = argparse.ArgumentParser(
        description='Clean the raw crop yield dataset.'
    )
    parser.add_argument(
        '--input',
        type=str,
        default='data/raw/yield_df.csv',
        help='Path to raw input CSV file '
             '(default: data/raw/yield_df.csv)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='data/processed/cleaned_yield.csv',
        help='Path to save cleaned output CSV '
             '(default: data/processed/cleaned_yield.csv)'
    )
    parser.add_argument(
        '--silent',
        action='store_true',
        help='Suppress cleaning log output'
    )
    return parser.parse_args()


if __name__ == '__main__':

    args = _parse_args()
    verbose = not args.silent

    # Load raw data
    if verbose:
        print(f"\nLoading data from: {args.input}")

    if not os.path.exists(args.input):
        raise FileNotFoundError(
            f"Input file not found: {args.input}\n"
            f"Please ensure the raw dataset is placed at {args.input}"
        )

    df_raw = pd.read_csv(args.input)

    if verbose:
        print(f"Loaded successfully — shape: {df_raw.shape}")

    # Run cleaning pipeline
    df_cleaned = clean_data(df_raw, verbose=verbose)

    # Save cleaned dataset
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    df_cleaned.to_csv(args.output, index=False)

    if verbose:
        print(f"\nCleaned dataset saved to: {args.output}")
        print(f"Final shape: {df_cleaned.shape}")
