import pandas as pd
import pytest

from ml.scripts.preprocessing import (
    load_dataset,
    inspect_dataset,
    validate_columns,
    validate_dtypes,
    drop_columns,
    check_missing_values,
    find_duplicates,
    validate_unique_values,
    compute_composite_score
)

DATASET_PATH = "ml/datasets/S-COMMERCE_GenZ_UniversityStudent_757_DIB.csv"

EXPECTED_COLUMNS = [
    "Gender", "Job", "Income", "Area", "Frequently",
    "PU1", "PU2", "PU3", "PU4",
    "PEU1", "PEU2", "PEU3", "PEU4",
    "FSC1", "FSC2", "FSC3",
    "SP1", "SP2", "SP3", "SP4",
    "TP1", "TP2", "TP3",
    "IB1", "IB2", "IB3", "IB4",
    "AUB1", "AUB2", "AUB3", "AUB4"
]

# ==========================================================
# MODULE 1 - DATASET LOADING
# ==========================================================

def test_load_dataset():
    df = load_dataset(DATASET_PATH)

    # Correct object
    assert isinstance(df, pd.DataFrame)

    # Correct dimensions
    assert df.shape == (757, 31)

    # Correct index
    assert list(df.index) == list(range(757))

    # Expected columns
    expected_columns = [
        "Gender", "Job", "Income", "Area", "Frequently",
        "PU1","PU2","PU3","PU4",
        "PEU1","PEU2","PEU3","PEU4",
        "FSC1","FSC2","FSC3",
        "SP1","SP2","SP3","SP4",
        "TP1","TP2","TP3",
        "IB1","IB2","IB3","IB4",
        "AUB1","AUB2","AUB3","AUB4"
    ]

    assert df.columns.tolist() == expected_columns

    # No duplicate column names
    assert df.columns.is_unique

    # Row count
    assert len(df) == 757

    # Column count
    assert len(df.columns) == 31

    # Dataset not empty
    assert not df.empty


def test_invalid_dataset():
    with pytest.raises(FileNotFoundError):
        load_dataset("invalid.csv")


# ==========================================================
# MODULE 2 - DATASET VALIDATION
# ==========================================================

def test_inspect_dataset():
    df = load_dataset(DATASET_PATH)

    shape = inspect_dataset(df)

    assert shape == (757, 31)


def test_validate_columns():
    df = load_dataset(DATASET_PATH)

    columns = validate_columns(df)

    assert columns == EXPECTED_COLUMNS


def test_validate_dtypes():
    df = load_dataset(DATASET_PATH)

    dtypes = validate_dtypes(df)

    assert len(dtypes) == 31
    assert all(dtype == "int64" for dtype in dtypes)


# ==========================================================
# MODULE 3 - DATA CLEANING
# ==========================================================

def test_remove_job():
    df = load_dataset(DATASET_PATH)

    cleaned = drop_columns(df, ["Job"])

    assert "Job" not in cleaned.columns


def test_verify_job_removed():
    df = load_dataset(DATASET_PATH)

    cleaned = drop_columns(df, ["Job"])

    columns = validate_columns(cleaned)

    assert "Job" not in columns


def test_check_missing_values():
    sample = pd.DataFrame({
        "A": [1, None, 3],
        "B": [None, 2, 3]
    })

    missing = check_missing_values(sample)

    assert missing["A"] == 1
    assert missing["B"] == 1


def test_find_duplicates():
    sample = pd.DataFrame({
        "A": [1, 1, 2],
        "B": [3, 3, 4]
    })

    duplicates = find_duplicates(sample)

    assert len(duplicates) == 1


def test_validate_unique_values():
    sample = pd.DataFrame({
        "Gender": [1, 1, 2],
        "Income": [1, 2, 2]
    })

    summary = validate_unique_values(sample)

    assert isinstance(summary, pd.DataFrame)

    assert "Variable" in summary.columns
    assert "Unique Count" in summary.columns
    assert "Unique Values" in summary.columns

    assert summary.loc[0, "Unique Count"] == 2


# ==========================================================
# MODULE 4 - FEATURE ENGINEERING
# ==========================================================

def test_compute_pu():

    sample = pd.DataFrame({
        "PU1": [4],
        "PU2": [5],
        "PU3": [3],
        "PU4": [4]
    })

    result = compute_composite_score(
        sample,
        ["PU1", "PU2", "PU3", "PU4"],
        "PU"
    )

    assert result["PU"].iloc[0] == 4.0


def test_compute_composite_missing_column():

    sample = pd.DataFrame({
        "PU1": [4],
        "PU2": [5],
        "PU3": [3]
    })

    with pytest.raises(ValueError):
        compute_composite_score(
            sample,
            ["PU1", "PU2", "PU3", "PU4"],
            "PU"
        )