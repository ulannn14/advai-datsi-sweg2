import pandas as pd
import pytest

from ml.scripts.preprocessing import (
    load_dataset,
    inspect_dataset,
    validate_columns,
    validate_dtypes,
    drop_columns,
    compute_composite_score
)

DATASET_PATH = "ml/datasets/S-COMMERCE_GenZ_UniversityStudent_757_DIB.csv"


# ==========================================================
# MODULE 1 - DATASET LOADING
# ==========================================================

def test_load_dataset():
    """SCA-UT-001"""

    df = load_dataset(DATASET_PATH)

    print(f"Loaded dataset successfully.")
    print(f"Dataset shape: {df.shape}")

    assert isinstance(df, pd.DataFrame), "Output is not a pandas DataFrame."
    assert df.shape == (757, 31), f"Expected shape (757, 31), got {df.shape}"


def test_invalid_dataset():
    """SCA-UT-002"""

    with pytest.raises(FileNotFoundError) as exc_info:
        load_dataset("invalid.csv")

    print(f"Expected exception caught: {exc_info.value}")


# ==========================================================
# MODULE 2 - DATASET VALIDATION
# ==========================================================

def test_inspect_dataset():
    """SCA-UT-003"""

    df = load_dataset(DATASET_PATH)

    shape = inspect_dataset(df)

    print(f"inspect_dataset() returned: {shape}")

    assert shape == (757, 31), f"Expected (757, 31), got {shape}"


def test_validate_columns():
    """SCA-UT-004"""

    df = load_dataset(DATASET_PATH)

    columns = validate_columns(df)

    print(f"Number of columns: {len(columns)}")
    print(f"First column: {columns[0]}")
    print(f"Last column: {columns[-1]}")

    assert len(columns) == 31, f"Expected 31 columns, got {len(columns)}"
    assert columns[0] == "Gender", f"Expected first column 'Gender', got '{columns[0]}'"
    assert columns[-1] == "AUB4", f"Expected last column 'AUB4', got '{columns[-1]}'"


def test_validate_dtypes():
    """SCA-UT-005"""

    df = load_dataset(DATASET_PATH)

    dtypes = validate_dtypes(df)

    print("Column data types:")
    print(dtypes)

    assert len(dtypes) == 31, f"Expected 31 data types, got {len(dtypes)}"
    assert all(dtype == "int64" for dtype in dtypes), \
        f"Expected all dtypes to be int64, got {dtypes}"


# ==========================================================
# MODULE 3 - DATA CLEANING
# ==========================================================

def test_remove_job():
    """SCA-UT-006"""

    df = load_dataset(DATASET_PATH)

    cleaned = drop_columns(df, ["Job"])

    print(f"'Job' exists after cleaning? {'Job' in cleaned.columns}")

    assert "Job" not in cleaned.columns, "'Job' column was not removed."


def test_verify_job_removed():
    """SCA-UT-007"""

    df = load_dataset(DATASET_PATH)

    cleaned = drop_columns(df, ["Job"])

    columns = validate_columns(cleaned)

    print("Columns after removing 'Job':")
    print(columns)

    assert "Job" not in columns, "'Job' column still exists."


# ==========================================================
# MODULE 4 - FEATURE ENGINEERING
# ==========================================================

def test_compute_pu():
    """SCA-UT-008"""

    df = pd.DataFrame({
        "PU1": [4],
        "PU2": [5],
        "PU3": [3],
        "PU4": [4]
    })

    result = compute_composite_score(
        df,
        ["PU1", "PU2", "PU3", "PU4"],
        "PU"
    )

    print(f"Computed PU score: {result['PU'].iloc[0]}")

    assert result["PU"].iloc[0] == 4.0, \
        f"Expected PU score 4.0, got {result['PU'].iloc[0]}"