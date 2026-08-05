from unittest import result

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
    compute_composite_score,
)

DATASET_PATH = "ml/datasets/S-COMMERCE_GenZ_UniversityStudent_757_DIB.csv"

EXPECTED_COLUMNS = [
    "Gender",
    "Job",
    "Income",
    "Area",
    "Frequently",
    "PU1", "PU2", "PU3", "PU4",
    "PEU1", "PEU2", "PEU3", "PEU4",
    "FSC1", "FSC2", "FSC3",
    "SP1", "SP2", "SP3", "SP4",
    "TP1", "TP2", "TP3",
    "IB1", "IB2", "IB3", "IB4",
    "AUB1", "AUB2", "AUB3", "AUB4",
]

# ==========================================================
# MODULE 1 - DATASET LOADING
# ==========================================================

def test_load_dataset():
    """SCA-UT-001"""

    # Load the dataset from the specified file path.
    df = load_dataset(DATASET_PATH)

    assert isinstance(df, pd.DataFrame)              # Verify that the dataset was successfully loaded as a pandas DataFrame.
    assert df.shape == (757, 31)                     # Verify that the dataset contains the expected number of rows and columns.
    assert df.columns.tolist() == EXPECTED_COLUMNS   # Verify that all expected variables were loaded in the correct order.


def test_invalid_dataset():
    """SCA-UT-002"""

    with pytest.raises(
        FileNotFoundError,
        match="Dataset 'invalid.csv' does not exist."
    ):
        load_dataset("invalid.csv")                  # Verify that loading a non-existent dataset raises a FileNotFoundError.
                        # Verify that loading a non-existent dataset raises a FileNotFoundError.

# ==========================================================
# MODULE 2 - DATASET VALIDATION
# ==========================================================

def test_inspect_dataset():
    """SCA-UT-003.1"""

    df = load_dataset(DATASET_PATH)                   # Load the dataset for testing.

    rows, columns = inspect_dataset(df)              # Retrieve the dataset dimensions.

    assert rows == 757, f"Expected 757 rows, got {rows}"    # Verify that the function returns the correct number of observations.
    assert columns == len(EXPECTED_COLUMNS), (
        f"Expected {len(EXPECTED_COLUMNS)} columns, got {columns}"
    )   # Verify that the function returns the correct number of variables.


def test_validate_columns():
    """SCA-UT-004"""

    df = load_dataset(DATASET_PATH)                   # Load the dataset for testing.

    columns = validate_columns(df)                   # Retrieve all column names from the dataset.

    assert columns == EXPECTED_COLUMNS               # Verify that all dataset columns are returned in the expected order.


def test_validate_dtypes():
    """SCA-UT-005"""

    df = load_dataset(DATASET_PATH)                   # Load the dataset for testing.

    dtypes = validate_dtypes(df)                     # Retrieve the data type of each dataset column.

    pd.testing.assert_series_equal(
        dtypes,
        df.dtypes
    )   # Verify that the returned data types exactly match the dataset.

    assert dtypes.eq("int64").all(), (
        "All dataset columns should have dtype int64."
    )   # Verify that every dataset variable is stored using the expected integer data type.

def test_check_missing_values():
    """SCA-UT-007.1"""

    df = load_dataset(DATASET_PATH)                   # Load the dataset for testing.

    missing = check_missing_values(df)               # Count the missing values in each dataset column.

    expected = pd.Series(
        [0] * len(EXPECTED_COLUMNS),
        index=EXPECTED_COLUMNS
    )   # Create the expected result where every column contains zero missing values.

    pd.testing.assert_series_equal(
        missing,
        expected
    )   # Verify that all dataset columns contain zero missing values.


# Duplicate records are retained since identical responses may come from different respondents.
def test_find_duplicates():
    """SCA-UT-007.2"""

    df = load_dataset(DATASET_PATH)                   # Load the dataset for testing.

    expected = df[df.duplicated()]                   # Identify duplicate rows using pandas as the expected result.

    result = find_duplicates(df)                     # Execute the wrapper function to find duplicate rows.

    pd.testing.assert_frame_equal(
        result,
        expected
    )   # Verify that the function correctly identifies all duplicate rows.

def test_validate_unique_values():
    """SCA-UT-007.3"""

    df = load_dataset(DATASET_PATH)

    summary = validate_unique_values(df)

    # Verify the returned DataFrame schema
    assert list(summary.columns) == [
        "Variable",
        "Unique Count",
        "Unique Values"
    ]

    # Verify all variables are present and in the correct order
    assert summary["Variable"].tolist() == EXPECTED_COLUMNS

    # Expected number of unique values per variable
    EXPECTED_UNIQUE_COUNTS = [
        3,  # Gender
        1,  # Job
        5,  # Income
        3,  # Area
        4,  # Frequently

        5, 5, 5, 5,          # PU1-PU4
        5, 5, 5, 5,          # PEU1-PEU4
        5, 5, 5,             # FSC1-FSC3
        5, 5, 5, 5,          # SP1-SP4
        5, 5, 5,             # TP1-TP3
        5, 5, 5, 5,          # IB1-IB4
        5, 5, 5, 5           # AUB1-AUB4
    ]

    assert summary["Unique Count"].tolist() == EXPECTED_UNIQUE_COUNTS

    # Expected unique values per variable
    EXPECTED_UNIQUE_VALUES = [
        [1, 2, 3],              # Gender
        [1],                    # Job
        [1, 2, 3, 4, 5],        # Income
        [1, 2, 3],              # Area
        [1, 2, 3, 4],           # Frequently

        [1, 2, 3, 4, 5],        # PU1
        [1, 2, 3, 4, 5],        # PU2
        [1, 2, 3, 4, 5],        # PU3
        [1, 2, 3, 4, 5],        # PU4

        [1, 2, 3, 4, 5],        # PEU1
        [1, 2, 3, 4, 5],        # PEU2
        [1, 2, 3, 4, 5],        # PEU3
        [1, 2, 3, 4, 5],        # PEU4

        [1, 2, 3, 4, 5],        # FSC1
        [1, 2, 3, 4, 5],        # FSC2
        [1, 2, 3, 4, 5],        # FSC3

        [1, 2, 3, 4, 5],        # SP1
        [1, 2, 3, 4, 5],        # SP2
        [1, 2, 3, 4, 5],        # SP3
        [1, 2, 3, 4, 5],        # SP4

        [1, 2, 3, 4, 5],        # TP1
        [1, 2, 3, 4, 5],        # TP2
        [1, 2, 3, 4, 5],        # TP3

        [1, 2, 3, 4, 5],        # IB1
        [1, 2, 3, 4, 5],        # IB2
        [1, 2, 3, 4, 5],        # IB3
        [1, 2, 3, 4, 5],        # IB4

        [1, 2, 3, 4, 5],        # AUB1
        [1, 2, 3, 4, 5],        # AUB2
        [1, 2, 3, 4, 5],        # AUB3
        [1, 2, 3, 4, 5],        # AUB4
    ]

    assert summary["Unique Values"].tolist() == EXPECTED_UNIQUE_VALUES   # Verify that each variable contains only the expected categories or Likert scale values.

# ==========================================================
# MODULE 3 - DATA CLEANING
# ==========================================================

def test_drop_job():
    """SCA-UT-006"""

    df = load_dataset(DATASET_PATH)                   # Load the dataset for testing.

    result = drop_columns(df, ["Job"])               # Remove the Job column from the dataset.

    expected_columns = EXPECTED_COLUMNS.copy()        # Create a copy of the expected dataset schema.
    expected_columns.remove("Job")                   # Remove the Job column from the expected schema.

    assert result.shape == (757, 30)                 # Verify that removing one column reduces the dataset to 30 variables.
    assert result.columns.tolist() == expected_columns   # Verify that the Job column was successfully removed.


def test_drop_peu4():
    """SCA-UT-006.2"""

    df = load_dataset(DATASET_PATH)                   # Load the dataset for testing.

    result = drop_columns(df, ["PEU4"])              # Remove the undocumented PEU4 column from the dataset.

    expected_columns = EXPECTED_COLUMNS.copy()        # Create a copy of the expected dataset schema.
    expected_columns.remove("PEU4")                  # Remove the PEU4 column from the expected schema.

    assert result.shape == (757, 30)                 # Verify that removing one column reduces the dataset to 30 variables.
    assert result.columns.tolist() == expected_columns   # Verify that the PEU4 column was successfully removed.

# ==========================================================
# MODULE 4 - FEATURE ENGINEERING
# ==========================================================

def test_compute_pu():
    """SCA-UT-008.1"""

    df = load_dataset(DATASET_PATH)                       # Load the dataset for testing.

    result = compute_composite_score(
        df,
        ["PU1", "PU2", "PU3", "PU4"],
        "PU"
    )   # Compute the composite score for the Perceived Usefulness (PU) construct.

    expected = EXPECTED_COLUMNS.copy()                    # Create a copy of the expected dataset schema.
    expected.append("PU")                                # Append the new composite score column to the expected schema.

    assert result.columns.tolist() == expected           # Verify that the PU column was added to the dataset.
    assert "PU" in result.columns                        # Verify that the PU composite score column exists.

    assert result.loc[0, "PU"] == 4                      # Verify that the first respondent's PU score is correctly computed.
    assert result.loc[1, "PU"] == 3                      # Verify that the second respondent's PU score is correctly computed.
    assert result.loc[100, "PU"] == 3                    # Verify that respondent 101's PU score is correctly computed.
    assert result.loc[756, "PU"] == 4.5                  # Verify that the last respondent's PU score is correctly computed.

def test_compute_peu():
    """SCA-UT-008.2"""

    df = load_dataset(DATASET_PATH)

    result = compute_composite_score(
        df,
        ["PEU1", "PEU2", "PEU3"],
        "PEU"
    )

    expected = EXPECTED_COLUMNS.copy()
    expected.append("PEU")

    # Verify schema
    assert result.columns.tolist() == expected
    assert "PEU" in result.columns

    # Precomputed values
    assert result.loc[0, "PEU"] == 4
    assert result.loc[1, "PEU"] == 3
    assert result.loc[100, "PEU"] == 3
    assert result.loc[756, "PEU"] == pytest.approx(4.333333333333333)

def test_compute_fsc():
    """SCA-UT-008.3"""

    df = load_dataset(DATASET_PATH)

    result = compute_composite_score(
        df,
        ["FSC1", "FSC2", "FSC3"],
        "FSC"
    )

    expected = EXPECTED_COLUMNS.copy()
    expected.append("FSC")

    # Verify schema
    assert result.columns.tolist() == expected
    assert "FSC" in result.columns

    # Precomputed values
    assert result.loc[0, "FSC"] == 4
    assert result.loc[1, "FSC"] == 3
    assert result.loc[100, "FSC"] == 3
    assert result.loc[756, "FSC"] == pytest.approx(4.333333333333333)


def test_compute_sp():
    """SCA-UT-008.4"""

    df = load_dataset(DATASET_PATH)

    result = compute_composite_score(
        df,
        ["SP1", "SP2", "SP3", "SP4"],
        "SP"
    )

    expected = EXPECTED_COLUMNS.copy()
    expected.append("SP")

    # Verify schema
    assert result.columns.tolist() == expected
    assert "SP" in result.columns

    # Precomputed values
    assert result.loc[0, "SP"] == 4
    assert result.loc[1, "SP"] == 3
    assert result.loc[100, "SP"] == 3
    assert result.loc[756, "SP"] == 4.5


def test_compute_tp():
    """SCA-UT-008.5"""

    df = load_dataset(DATASET_PATH)

    result = compute_composite_score(
        df,
        ["TP1", "TP2", "TP3"],
        "TP"
    )

    expected = EXPECTED_COLUMNS.copy()
    expected.append("TP")

    # Verify schema
    assert result.columns.tolist() == expected
    assert "TP" in result.columns

    # Precomputed values
    assert result.loc[0, "TP"] == 4
    assert result.loc[1, "TP"] == 3
    assert result.loc[100, "TP"] == 3
    assert result.loc[756, "TP"] == 3


def test_compute_ib():
    """SCA-UT-008.6"""

    df = load_dataset(DATASET_PATH)

    result = compute_composite_score(
        df,
        ["IB1", "IB2", "IB3", "IB4"],
        "IB"
    )

    expected = EXPECTED_COLUMNS.copy()
    expected.append("IB")

    # Verify schema
    assert result.columns.tolist() == expected
    assert "IB" in result.columns

    # Precomputed values
    assert result.loc[0, "IB"] == 3
    assert result.loc[1, "IB"] == 3
    assert result.loc[100, "IB"] == 3
    assert result.loc[756, "IB"] == 4


def test_compute_aub():
    """SCA-UT-008.7"""

    df = load_dataset(DATASET_PATH)

    result = compute_composite_score(
        df,
        ["AUB1", "AUB2", "AUB3", "AUB4"],
        "AUB"
    )

    expected = EXPECTED_COLUMNS.copy()
    expected.append("AUB")

    # Verify schema
    assert result.columns.tolist() == expected
    assert "AUB" in result.columns

    # Precomputed values
    assert result.loc[0, "AUB"] == 4
    assert result.loc[1, "AUB"] == 3
    assert result.loc[100, "AUB"] == 3
    assert result.loc[756, "AUB"] == 4

