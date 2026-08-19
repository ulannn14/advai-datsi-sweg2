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
# SCA-UT-001
# ==========================================================

def test_load_dataset():
    print("=" * 50)
    print("SCA-UT-001")
    print("=" * 50)

    df = load_dataset(DATASET_PATH)

    print("Input:")
    print(DATASET_PATH)

    print("\nOutput:")
    print(type(df))
    print(df.shape)

    assert isinstance(df, pd.DataFrame)
    assert df.shape == (757, 31)


# ==========================================================
# SCA-UT-002
# ==========================================================

def test_invalid_dataset():
    print("\n" + "=" * 50)
    print("SCA-UT-002")
    print("=" * 50)

    print("Input:")
    print("invalid.csv")

    with pytest.raises(FileNotFoundError) as exc:
        load_dataset("invalid.csv")

    print("\nOutput:")
    print(type(exc.value).__name__)


# ==========================================================
# SCA-UT-003
# ==========================================================

def test_inspect_dataset():
    print("\n" + "=" * 50)
    print("SCA-UT-003")
    print("=" * 50)

    df = load_dataset(DATASET_PATH)
    shape = inspect_dataset(df)

    print("Input:")
    print("Loaded dataset DataFrame")

    print("\nOutput:")
    print(shape)

    assert shape == (757, 31)


# ==========================================================
# SCA-UT-004
# ==========================================================

def test_validate_columns():
    print("\n" + "=" * 50)
    print("SCA-UT-004")
    print("=" * 50)

    df = load_dataset(DATASET_PATH)
    columns = validate_columns(df)

    print("Input:")
    print("Loaded dataset DataFrame")

    print("\nOutput:")
    print(f"Number of columns: {len(columns)}")
    print(f"First column: {columns[0]}")
    print(f"Last column: {columns[-1]}")
    print(columns)

    assert len(columns) == 31
    assert columns[0] == "Gender"
    assert columns[-1] == "AUB4"


# ==========================================================
# SCA-UT-005
# ==========================================================

def test_validate_dtypes():
    print("\n" + "=" * 50)
    print("SCA-UT-005")
    print("=" * 50)

    df = load_dataset(DATASET_PATH)
    dtypes = validate_dtypes(df)

    print("Input:")
    print("Loaded dataset DataFrame")

    print("\nOutput:")
    print(f"Number of dtypes: {len(dtypes)}")
    print(dtypes)

    assert len(dtypes) == 31
    assert all(dtype == "int64" for dtype in dtypes)


# ==========================================================
# SCA-UT-006
# ==========================================================

def test_remove_job():
    print("\n" + "=" * 50)
    print("SCA-UT-006")
    print("=" * 50)

    df = load_dataset(DATASET_PATH)
    cleaned_df = drop_columns(df, ["Job"])

    print("Input:")
    print(["Job"])

    print("\nOutput:")
    print("'Job' exists:", "Job" in cleaned_df.columns)
    print(cleaned_df.columns.tolist())

    assert "Job" not in cleaned_df.columns


# ==========================================================
# SCA-UT-007
# ==========================================================

def test_verify_job_removed():
    print("\n" + "=" * 50)
    print("SCA-UT-007")
    print("=" * 50)

    df = load_dataset(DATASET_PATH)
    cleaned_df = drop_columns(df, ["Job"])

    cleaned_columns = validate_columns(cleaned_df)

    print("Input:")
    print("DataFrame after removing Job")

    print("\nOutput:")
    print("'Job' exists:", "Job" in cleaned_columns)
    print(cleaned_columns)

    assert "Job" not in cleaned_columns


# ==========================================================
# SCA-UT-008
# ==========================================================

def test_compute_pu():
    print("\n" + "=" * 50)
    print("SCA-UT-008")
    print("=" * 50)

    sample_df = pd.DataFrame({
        "PU1": [4],
        "PU2": [5],
        "PU3": [3],
        "PU4": [4]
    })

    print("Input:")
    print(sample_df)

    result = compute_composite_score(
        sample_df,
        ["PU1", "PU2", "PU3", "PU4"],
        "PU"
    )

    print("\nOutput:")
    print(result[["PU"]])

    assert result["PU"].iloc[0] == 4.0