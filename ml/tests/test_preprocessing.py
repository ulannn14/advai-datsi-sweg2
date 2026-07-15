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

    assert isinstance(df, pd.DataFrame)
    assert df.shape == (757, 31)


def test_invalid_dataset():
    """SCA-UT-002"""

    with pytest.raises(FileNotFoundError):
        load_dataset("invalid.csv")


# ==========================================================
# MODULE 2 - DATASET VALIDATION
# ==========================================================

def test_inspect_dataset():
    """SCA-UT-003"""

    df = load_dataset(DATASET_PATH)

    shape = inspect_dataset(df)

    assert shape == (757, 31)


def test_validate_columns():
    """SCA-UT-004"""

    df = load_dataset(DATASET_PATH)

    columns = validate_columns(df)

    assert len(columns) == 31
    assert columns[0] == "Gender"
    assert columns[-1] == "AUB4"


def test_validate_dtypes():
    """SCA-UT-005"""

    df = load_dataset(DATASET_PATH)

    dtypes = validate_dtypes(df)

    assert len(dtypes) == 31
    assert all(dtype == "int64" for dtype in dtypes)


# ==========================================================
# MODULE 3 - DATA CLEANING
# ==========================================================

def test_remove_job():
    """SCA-UT-006"""

    df = load_dataset(DATASET_PATH)

    cleaned = drop_columns(df, ["Job"])

    assert "Job" not in cleaned.columns


def test_verify_job_removed():
    """SCA-UT-007"""

    df = load_dataset(DATASET_PATH)

    cleaned = drop_columns(df, ["Job"])

    columns = validate_columns(cleaned)

    assert "Job" not in columns


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

    assert result["PU"].iloc[0] == 4.0