import pandas as pd

from ml.scripts.preprocessing import (
    load_dataset,
    inspect_dataset,
    validate_columns,
    validate_dtypes,
    drop_columns,
    compute_composite_score
)

DATASET_PATH = "ml/datasets/S-COMMERCE_GenZ_UniversityStudent_757_DIB.csv"

def test_fail_load_dataset():
    """SCA-DEMO-001"""

    df = load_dataset(DATASET_PATH)

    print(f"Dataset shape: {df.shape}")

    assert df.shape == (700, 30), \
        f"Expected shape (700, 30), got {df.shape}"


def test_fail_inspect_dataset():
    """SCA-DEMO-002"""

    df = load_dataset(DATASET_PATH)

    shape = inspect_dataset(df)

    print(f"inspect_dataset() returned: {shape}")

    assert shape == (700, 30), \
        f"Expected (700, 30), got {shape}"


def test_fail_validate_columns():
    """SCA-DEMO-003"""

    df = load_dataset(DATASET_PATH)

    columns = validate_columns(df)

    print(columns)

    assert len(columns) == 30, \
        f"Expected 30 columns, got {len(columns)}"


def test_fail_validate_dtypes():
    """SCA-DEMO-004"""

    df = load_dataset(DATASET_PATH)

    dtypes = validate_dtypes(df)

    print(dtypes)

    assert all(dtype == "float64" for dtype in dtypes), \
        f"Expected all dtypes to be float64, got {dtypes}"


def test_fail_remove_job():
    """SCA-DEMO-005"""

    df = load_dataset(DATASET_PATH)

    cleaned = drop_columns(df, ["Job"])

    print(cleaned.columns)

    assert "Job" in cleaned.columns, \
        "'Job' column should still exist (intentional failure)."


def test_fail_verify_job_removed():
    """SCA-DEMO-006"""

    df = load_dataset(DATASET_PATH)

    cleaned = drop_columns(df, ["Job"])

    columns = validate_columns(cleaned)

    print(columns)

    assert "Job" in columns, \
        "'Job' column should still exist (intentional failure)."


def test_fail_compute_pu():
    """SCA-DEMO-007"""

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

    print(result)

    assert result["PU"].iloc[0] == 5.0, \
        f"Expected PU score 5.0, got {result['PU'].iloc[0]}"


def test_fail_invalid_dataset():
    """SCA-DEMO-008"""

    # Intentionally do NOT catch the exception
    load_dataset("invalid.csv")