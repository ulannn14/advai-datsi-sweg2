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
    get_gender_distribution,
    get_income_distribution,
    get_area_distribution,
    get_frequency_distribution,
    get_construct_descriptives,
    get_construct_correlation_matrix,
    get_aub_by_gender_summary,
    run_ttest_aub_gender,
    get_aub_by_area_summary,
    run_anova_aub_area,
    get_aub_by_frequency_summary,
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

# The following test checks:
# - If the CSV file can be loaded successfully.
# - If the returned object is a pandas DataFrame.
#
# This test will fail if:
# - The dataset path is changed.
# - The CSV file is missing.
# - The function no longer returns a DataFrame.from the specified file path.
def test_load_dataset():
    """SCA-UT-001"""

    # Load the dataset from the specified file path.
    df = load_dataset(DATASET_PATH)

    assert isinstance(df, pd.DataFrame)              # Verify that the dataset was successfully loaded as a pandas DataFrame.
    assert df.shape == (757, 31)                     # Verify that the dataset contains the expected number of rows and columns.
    assert df.columns.tolist() == EXPECTED_COLUMNS   # Verify that all expected variables were loaded in the correct order.

# The following test checks:
# - If an invalid dataset path raises a FileNotFoundError.
#
# This test will fail if:
# - The exception is removed.
# - The wrong exception type is raised.
# - The function silently ignores invalid file paths.
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

# The following test checks:
# - If the dataset dimensions are reported correctly.
#
# This test will fail if:
# - Rows are added or removed.
# - Columns are added or removed.
# - The function no longer returns the dataset shape.
def test_inspect_dataset():
    """SCA-UT-003.1"""

    df = load_dataset(DATASET_PATH)                   # Load the dataset for testing.

    rows, columns = inspect_dataset(df)              # Retrieve the dataset dimensions.

    assert rows == 757, f"Expected 757 rows, got {rows}"    # Verify that the function returns the correct number of observations.
    assert columns == len(EXPECTED_COLUMNS), (
        f"Expected {len(EXPECTED_COLUMNS)} columns, got {columns}"
    )   # Verify that the function returns the correct number of variables.

# The following test checks:
# - If all expected columns are present in the dataset.
#
# This test will fail if:
# - A column is renamed.
# - A column is removed.
# - Additional unexpected columns are introduced.
def test_validate_columns():
    """SCA-UT-004"""

    df = load_dataset(DATASET_PATH)                   # Load the dataset for testing.

    columns = validate_columns(df)                   # Retrieve all column names from the dataset.

    assert columns == EXPECTED_COLUMNS               # Verify that all dataset columns are returned in the expected order.

# The following test checks:
# - If all columns retain their expected data types.
#
# This test will fail if:
# - A numeric column becomes a string.
# - A column's data type changes unexpectedly.
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

# The following test checks:
# - If missing values are identified correctly.
#
# This test will fail if:
# - New missing values are introduced.
# - The function no longer counts missing values correctly.
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

# The following test checks:
# - If duplicate rows are identified correctly.
# - If the returned duplicate rows exactly match the original dataset.
# - If the duplicate detection function returns the expected output.
#
# This test will fail if:
# - Duplicate rows are added or removed.
# - The duplicate detection logic is modified.
# - The function returns incorrect rows.
# - The returned DataFrame no longer matches pandas' duplicate detection.
def test_find_duplicates():
    """SCA-UT-007.2"""

    # Load the dataset for testing.
    df = load_dataset(DATASET_PATH)

    # Identify duplicate rows using pandas as the expected result.
    expected = df[df.duplicated()]

    # Execute the wrapper function to find duplicate rows.
    result = find_duplicates(df)

    # Verify that the function returns a pandas DataFrame.
    assert isinstance(result, pd.DataFrame)

    # Verify that the expected number of duplicate rows was identified.
    assert result.shape == (129, 31)

    # Verify that the duplicate rows exactly match pandas' duplicate detection.
    pd.testing.assert_frame_equal(
        result,
        expected
    )
    
# The following test checks:
# - If every variable reports the correct number of unique values.
#
# This test will fail if:
# - Values are modified.
# - Categories are added or removed.
# - The function no longer summarizes unique values correctly.
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

# The following test checks:
# - If the Job column is removed successfully.
#
# This test will fail if:
# - The Job column remains in the dataset.
# - The wrong column is removed.
def test_drop_job():
    """SCA-UT-006"""

    df = load_dataset(DATASET_PATH)                   # Load the dataset for testing.

    result = drop_columns(df, ["Job"])               # Remove the Job column from the dataset.

    expected_columns = EXPECTED_COLUMNS.copy()        # Create a copy of the expected dataset schema.
    expected_columns.remove("Job")                   # Remove the Job column from the expected schema.

    assert result.shape == (757, 30)                 # Verify that removing one column reduces the dataset to 30 variables.
    assert result.columns.tolist() == expected_columns   # Verify that the Job column was successfully removed.

# The following test checks:
# - If duplicate rows are identified correctly.
#
# This test will fail if:
# - Duplicate rows are added.
# - The duplicate detection logic is modified.
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

# The following tests checks:
# - If the features composite scores are computed correctly.
#
# This test will fail if:
# - A questionnaire item is removed.
# - The formula is modified.
# - The wrong aggregation method is used.
# This goes for each of the composite score tests below (PU, PEU, FSC, SP, TP, IB, AUB).

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

    # Verify that no observations were removed
    assert result.shape[0] == df.shape[0]

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

    # Verify that no observations were removed
    assert result.shape[0] == df.shape[0]

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

    # Verify that no observations were removed
    assert result.shape[0] == df.shape[0]

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

    # Verify that no observations were removed
    assert result.shape[0] == df.shape[0]

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

    # Verify that no observations were removed
    assert result.shape[0] == df.shape[0]

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

    # Verify that no observations were removed
    assert result.shape[0] == df.shape[0]

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

    # Verify that no observations were removed
    assert result.shape[0] == df.shape[0]    

    # Precomputed values
    assert result.loc[0, "AUB"] == 4
    assert result.loc[1, "AUB"] == 3
    assert result.loc[100, "AUB"] == 3
    assert result.loc[756, "AUB"] == 4



# ==========================================================
# MODULE 5 - EXPLORATORY DATA ANALYSIS (EDA)
# ==========================================================
@pytest.fixture
def edaDF():
    """Load and preprocess the dataset exactly the way the notebook does."""
    data = load_dataset(DATASET_PATH)
    data = drop_columns(data, ["PEU4"])
    data = drop_columns(data, ["Job"])

    data = compute_composite_score(data, ["PU1", "PU2", "PU3", "PU4"], "PU")
    data = compute_composite_score(data, ["PEU1", "PEU2", "PEU3"], "PEU")
    data = compute_composite_score(data, ["FSC1", "FSC2", "FSC3"], "FSC")
    data = compute_composite_score(data, ["SP1", "SP2", "SP3", "SP4"], "SP")
    data = compute_composite_score(data, ["TP1", "TP2", "TP3"], "TP")
    data = compute_composite_score(data, ["IB1", "IB2", "IB3", "IB4"], "IB")
    data = compute_composite_score(data, ["AUB1", "AUB2", "AUB3", "AUB4"], "AUB")

    return data


# DEMOGRAPHIC DISTRIBUTIONS
# The following tests check:
# - If the demographic distribution matches the original dataset.
# - If all respondents are included in the total count.
#
# This test will fail if:
# - Column values are modified.
# - A row is removed.
# - The column is changed.

def test_get_gender_distribution(edaDF):
    result = get_gender_distribution(edaDF)

    assert result["Male"] == 167
    assert result["Female"] == 588
    assert result["Different"] == 2
    assert result.sum() == 757


def test_get_income_distribution(edaDF):
    result = get_income_distribution(edaDF)

    assert result["< $100"] == 672
    assert result["$100 - $200"] == 68
    assert result["$200 - $300"] == 12
    assert result["$300 - $400"] == 2
    assert result["> $400"] == 3
    assert result.sum() == 757


def test_get_area_distribution(edaDF):
    result = get_area_distribution(edaDF)

    assert result["Urban"] == 461
    assert result["Suburban"] == 74
    assert result["Rural"] == 222
    assert result.sum() == 757


def test_get_frequency_distribution(edaDF):
    result = get_frequency_distribution(edaDF)

    assert result["Daily"] == 725
    assert result["Weekly"] == 14
    assert result["Monthly"] == 7
    assert result["Rarely Used"] == 11
    assert result.sum() == 757


# CONSTRUCT DESCRIPTIVES / CORRELATIONS
# The following test checks:
# - If descriptive statistics are generated for all seven constructs.
# - If the calculated means match the original analysis.
#
# This test will fail if:
# - A construct is removed.
# - Composite scores are modified.
# - Statistical calculations are changed.
def test_get_construct_descriptives_shape_and_range(edaDF):
    result = get_construct_descriptives(edaDF)

    expected_constructs = ["PU", "PEU", "FSC", "SP", "TP", "IB", "AUB"]
    assert result.columns.tolist() == expected_constructs
    assert result.loc["count"].tolist() == [757] * 7

    assert result.loc["mean", "PU"] == pytest.approx(3.77, abs=0.01)
    assert result.loc["mean", "PEU"] == pytest.approx(3.70, abs=0.01)
    assert result.loc["mean", "FSC"] == pytest.approx(3.73, abs=0.01)
    assert result.loc["mean", "SP"] == pytest.approx(3.60, abs=0.01)
    assert result.loc["mean", "TP"] == pytest.approx(3.56, abs=0.01)
    assert result.loc["mean", "IB"] == pytest.approx(3.61, abs=0.01)
    assert result.loc["mean", "AUB"] == pytest.approx(3.68, abs=0.01)

# The following test checks:
# - If the correlation values remain consistent with the original analysis.
# - If the correlation matrix structure remains valid.
#
# This test will fail if:
# - A construct is removed.
# - A composite score changes.
# - The correlation method is changed.
def test_get_construct_correlation_matrix_known_pairs(edaDF):
    """
    Exact values below were pulled directly from
    get_construct_correlation_matrix(edaDF) after fixing the SP composite
    bug (SP now correctly uses SP1-SP4, not SP1-SP3). Tolerance is
    tight (abs=0.001) since these are full-precision, not rounded
    write-up numbers.
    """
    corr = get_construct_correlation_matrix(edaDF)

    assert corr.loc["PU", "PEU"] == pytest.approx(0.788226, abs=0.001)
    assert corr.loc["PU", "FSC"] == pytest.approx(0.804070, abs=0.001)
    assert corr.loc["PU", "SP"] == pytest.approx(0.718057, abs=0.001)
    assert corr.loc["PU", "TP"] == pytest.approx(0.676910, abs=0.001)
    assert corr.loc["PU", "IB"] == pytest.approx(0.655539, abs=0.001)
    assert corr.loc["PU", "AUB"] == pytest.approx(0.771895, abs=0.001)

    assert corr.loc["PEU", "FSC"] == pytest.approx(0.776458, abs=0.001)
    assert corr.loc["PEU", "SP"] == pytest.approx(0.717394, abs=0.001)
    assert corr.loc["PEU", "TP"] == pytest.approx(0.674808, abs=0.001)
    assert corr.loc["PEU", "IB"] == pytest.approx(0.691334, abs=0.001)
    assert corr.loc["PEU", "AUB"] == pytest.approx(0.785858, abs=0.001)

    assert corr.loc["FSC", "SP"] == pytest.approx(0.766774, abs=0.001)
    assert corr.loc["FSC", "TP"] == pytest.approx(0.710337, abs=0.001)
    assert corr.loc["FSC", "IB"] == pytest.approx(0.657147, abs=0.001)
    assert corr.loc["FSC", "AUB"] == pytest.approx(0.744687, abs=0.001)

    assert corr.loc["SP", "TP"] == pytest.approx(0.734756, abs=0.001)
    assert corr.loc["SP", "IB"] == pytest.approx(0.644206, abs=0.001)
    assert corr.loc["SP", "AUB"] == pytest.approx(0.691883, abs=0.001)

    assert corr.loc["TP", "IB"] == pytest.approx(0.663131, abs=0.001)
    assert corr.loc["TP", "AUB"] == pytest.approx(0.662269, abs=0.001)

    assert corr.loc["IB", "AUB"] == pytest.approx(0.683213, abs=0.001)

    # Sanity checks on the matrix shape itself, so a broken pivot/reindex
    # would still get caught even if the specific pairs above happened
    # to still line up.
    for construct in corr.columns:
        assert corr.loc[construct, construct] == pytest.approx(1.0)

# The following test checks:
# - If the SP composite score uses SP1, SP2, SP3, and SP4.
# - If the previously identified regression bug reappears.
#
# This test will fail if:
# - SP4 is removed.
# - Only three SP items are used.
# - The SP calculation is modified.
def test_sp_composite_uses_all_four_items(edaDF):
    """
    Regression test for a bug found in the notebook: SP was originally
    computed from SP1-SP4, but later silently recomputed from only
    SP1-SP3 right before the correlation matrix. This checks SP was
    built from all four items, so that regression can't creep back in
    without a test failing.
    """
    expected_sp = edaDF[["SP1", "SP2", "SP3", "SP4"]].mean(axis=1)
    assert (edaDF["SP"] - expected_sp).abs().max() < 1e-9


# AUB ACROSS DEMOGRAPHIC GROUPS
# The following test checks:
# - If the AUB summary statistics for each gender are computed correctly.
#
# This test will fail if:
# - AUB values are modified.
# - Gender categories are changed.
# - Summary calculations are changed.
def test_get_aub_by_gender_summary(edaDF):
    result = get_aub_by_gender_summary(edaDF)

    assert result.loc["Male", "count"] == 167
    assert result.loc["Female", "count"] == 588

    assert result.loc["Male", "mean"] == pytest.approx(3.66, abs=0.01)
    assert result.loc["Female", "mean"] == pytest.approx(3.69, abs=0.01)

    assert result.loc["Male", "median"] == pytest.approx(3.75)
    assert result.loc["Female", "median"] == pytest.approx(4.00)

    assert result.loc["Male", "std"] == pytest.approx(0.79, abs=0.01)
    assert result.loc["Female", "std"] == pytest.approx(0.67, abs=0.01)

# The following test checks:
# - If the t-test produces the expected t-statistic and p-value.
#
# This test will fail if:
# - AUB values are modified.
# - Gender filtering is changed.
# - The statistical method is modified.
def test_run_ttest_aub_gender(edaDF):
    t_stat, p_value = run_ttest_aub_gender(edaDF)

    assert t_stat == pytest.approx(-0.417, abs=0.01)
    assert p_value == pytest.approx(0.677, abs=0.01)
    assert p_value > 0.05  # not statistically significant, per the report

# The following test checks:
# - If AUB summary statistics for each residential area are computed correctly.
#
# This test will fail if:
# - AUB values are modified.
# - Area categories are changed.
# - Summary calculations are modified.
def test_get_aub_by_area_summary(edaDF):
    result = get_aub_by_area_summary(edaDF)

    assert result.loc["Urban", "count"] == 461
    assert result.loc["Suburban", "count"] == 74
    assert result.loc["Rural", "count"] == 222

    assert result.loc["Urban", "mean"] == pytest.approx(3.71, abs=0.01)
    assert result.loc["Suburban", "mean"] == pytest.approx(3.64, abs=0.01)
    assert result.loc["Rural", "mean"] == pytest.approx(3.62, abs=0.01)

    assert result.loc["Urban", "median"] == pytest.approx(4.00)
    assert result.loc["Suburban", "median"] == pytest.approx(3.75)
    assert result.loc["Rural", "median"] == pytest.approx(3.75)

# The following test checks:
# - If the ANOVA produces the expected F-statistic and p-value.
#
# This test will fail if:
# - AUB values are modified.
# - Area filtering is changed.
# - The statistical method is modified.
def test_run_anova_aub_area(edaDF):
    f_stat, p_value = run_anova_aub_area(edaDF)

    assert f_stat == pytest.approx(1.4681, abs=0.01)
    assert p_value == pytest.approx(0.2310, abs=0.01)
    assert p_value > 0.05  # not statistically significant, per the report

# The following test checks:
# - If AUB summary statistics for each frequency group are computed correctly.
#
# This test will fail if:
# - AUB values are modified.
# - Frequency categories are changed.
# - Summary calculations are modified.
def test_get_aub_by_frequency_summary(edaDF):
    result = get_aub_by_frequency_summary(edaDF)

    assert result.loc["Daily", "count"] == 725
    assert result.loc["Weekly", "count"] == 14
    assert result.loc["Monthly", "count"] == 7
    assert result.loc["Rarely Used", "count"] == 11

