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

DATASET_PATH = (
    "ml/datasets/S-COMMERCE_GenZ_UniversityStudent_757_DIB.csv"
)

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

EXPECTED_UNIQUE_COUNTS = [
    3, 1, 5, 3, 4,
    5, 5, 5, 5,
    5, 5, 5, 5,
    5, 5, 5,
    5, 5, 5, 5,
    5, 5, 5,
    5, 5, 5, 5,
    5, 5, 5, 5,
]

EXPECTED_UNIQUE_VALUES = [
    [1, 2, 3],
    [1],
    [1, 2, 3, 4, 5],
    [1, 2, 3],
    [1, 2, 3, 4],

    [1, 2, 3, 4, 5],
    [1, 2, 3, 4, 5],
    [1, 2, 3, 4, 5],
    [1, 2, 3, 4, 5],

    [1, 2, 3, 4, 5],
    [1, 2, 3, 4, 5],
    [1, 2, 3, 4, 5],
    [1, 2, 3, 4, 5],

    [1, 2, 3, 4, 5],
    [1, 2, 3, 4, 5],
    [1, 2, 3, 4, 5],

    [1, 2, 3, 4, 5],
    [1, 2, 3, 4, 5],
    [1, 2, 3, 4, 5],
    [1, 2, 3, 4, 5],

    [1, 2, 3, 4, 5],
    [1, 2, 3, 4, 5],
    [1, 2, 3, 4, 5],

    [1, 2, 3, 4, 5],
    [1, 2, 3, 4, 5],
    [1, 2, 3, 4, 5],
    [1, 2, 3, 4, 5],

    [1, 2, 3, 4, 5],
    [1, 2, 3, 4, 5],
    [1, 2, 3, 4, 5],
    [1, 2, 3, 4, 5],
]


# The following integration test checks:
# - If the complete preprocessing and EDA pipeline reproduces
#   the exact workflow used in the notebook.
# - If every intermediate output remains identical.
# - If the final statistical results remain unchanged.
#
# This test will fail if:
# - Rows are removed.
# - Columns are removed, renamed, or reordered.
# - Data types change.
# - Missing values are introduced.
# - Duplicate detection changes.
# - PEU4 or Job are not removed.
# - Composite score calculations change.
# - Demographic distributions change.
# - Descriptive statistics change.
# - Correlation coefficients change.
# - The t-test changes.
# - The ANOVA changes.
# - The final notebook results can no longer be reproduced.
def test_preprocessing_and_eda_pipeline():
    """
    SCT-001

    End-to-end integration test for the complete
    preprocessing and EDA workflow.
    """

    # =====================================================
    # DATASET LOADING
    # =====================================================

    df = load_dataset(DATASET_PATH)

    assert isinstance(df, pd.DataFrame)
    assert df.shape == (757, 31)
    assert df.columns.tolist() == EXPECTED_COLUMNS

    rows, columns = inspect_dataset(df)

    assert rows == 757
    assert columns == 31

    assert validate_columns(df) == EXPECTED_COLUMNS

    dtypes = validate_dtypes(df)

    pd.testing.assert_series_equal(
        dtypes,
        df.dtypes
    )

    assert dtypes.eq("int64").all()

    missing = check_missing_values(df)

    expected_missing = pd.Series(
        [0] * len(EXPECTED_COLUMNS),
        index=EXPECTED_COLUMNS
    )

    pd.testing.assert_series_equal(
        missing,
        expected_missing
    )

    duplicates = find_duplicates(df)

    assert duplicates.shape == (129, 31)

    unique_summary = validate_unique_values(df)

    assert list(unique_summary.columns) == [
        "Variable",
        "Unique Count",
        "Unique Values"
    ]

    assert unique_summary["Variable"].tolist() == EXPECTED_COLUMNS
    assert unique_summary["Unique Count"].tolist() == EXPECTED_UNIQUE_COUNTS
    assert unique_summary["Unique Values"].tolist() == EXPECTED_UNIQUE_VALUES

    # =====================================================
    # DATA CLEANING
    # =====================================================

    df = drop_columns(df, ["PEU4"])

    expected_columns = EXPECTED_COLUMNS.copy()
    expected_columns.remove("PEU4")

    assert df.shape == (757, 30)
    assert df.columns.tolist() == expected_columns
    assert "PEU4" not in df.columns

    df = drop_columns(df, ["Job"])

    expected_columns.remove("Job")

    assert df.shape == (757, 29)
    assert df.columns.tolist() == expected_columns
    assert "Job" not in df.columns

    # =====================================================
    # FEATURE ENGINEERING
    # =====================================================

    composites = {
        "PU": ["PU1", "PU2", "PU3", "PU4"],
        "PEU": ["PEU1", "PEU2", "PEU3"],
        "FSC": ["FSC1", "FSC2", "FSC3"],
        "SP": ["SP1", "SP2", "SP3", "SP4"],
        "TP": ["TP1", "TP2", "TP3"],
        "IB": ["IB1", "IB2", "IB3", "IB4"],
        "AUB": ["AUB1", "AUB2", "AUB3", "AUB4"],
    }

    for construct, items in composites.items():
        df = compute_composite_score(
            df,
            items,
            construct
        )

    assert df.shape == (757, 36)

    assert df.loc[0, "PU"] == 4
    assert df.loc[1, "PU"] == 3
    assert df.loc[100, "PU"] == 3
    assert df.loc[756, "PU"] == 4.5

    assert df.loc[0, "PEU"] == 4
    assert df.loc[1, "PEU"] == 3
    assert df.loc[100, "PEU"] == 3
    assert df.loc[756, "PEU"] == pytest.approx(4.333333333333333)

    assert df.loc[0, "FSC"] == 4
    assert df.loc[1, "FSC"] == 3
    assert df.loc[100, "FSC"] == 3
    assert df.loc[756, "FSC"] == pytest.approx(4.333333333333333)

    assert df.loc[0, "SP"] == 4
    assert df.loc[1, "SP"] == 3
    assert df.loc[100, "SP"] == 3
    assert df.loc[756, "SP"] == 4.5

    assert df.loc[0, "TP"] == 4
    assert df.loc[1, "TP"] == 3
    assert df.loc[100, "TP"] == 3
    assert df.loc[756, "TP"] == 3

    assert df.loc[0, "IB"] == 3
    assert df.loc[1, "IB"] == 3
    assert df.loc[100, "IB"] == 3
    assert df.loc[756, "IB"] == 4

    assert df.loc[0, "AUB"] == 4
    assert df.loc[1, "AUB"] == 3
    assert df.loc[100, "AUB"] == 3
    assert df.loc[756, "AUB"] == 4

    # =====================================================
    # DEMOGRAPHIC DISTRIBUTIONS
    # =====================================================

    gender = get_gender_distribution(df)

    assert gender["Male"] == 167
    assert gender["Female"] == 588
    assert gender["Different"] == 2
    assert gender.sum() == 757

    income = get_income_distribution(df)

    assert income["< $100"] == 672
    assert income["$100 - $200"] == 68
    assert income["$200 - $300"] == 12
    assert income["$300 - $400"] == 2
    assert income["> $400"] == 3
    assert income.sum() == 757

    area = get_area_distribution(df)

    assert area["Urban"] == 461
    assert area["Suburban"] == 74
    assert area["Rural"] == 222
    assert area.sum() == 757

    frequency = get_frequency_distribution(df)

    assert frequency["Daily"] == 725
    assert frequency["Weekly"] == 14
    assert frequency["Monthly"] == 7
    assert frequency["Rarely Used"] == 11
    assert frequency.sum() == 757

    # =====================================================
    # CONSTRUCT DESCRIPTIVES
    # =====================================================

    descriptives = get_construct_descriptives(df)

    assert descriptives.columns.tolist() == [
        "PU",
        "PEU",
        "FSC",
        "SP",
        "TP",
        "IB",
        "AUB",
    ]

    assert descriptives.loc["count"].tolist() == [757] * 7

    assert descriptives.loc["mean", "PU"] == pytest.approx(3.77, abs=0.01)
    assert descriptives.loc["mean", "PEU"] == pytest.approx(3.70, abs=0.01)
    assert descriptives.loc["mean", "FSC"] == pytest.approx(3.73, abs=0.01)
    assert descriptives.loc["mean", "SP"] == pytest.approx(3.60, abs=0.01)
    assert descriptives.loc["mean", "TP"] == pytest.approx(3.56, abs=0.01)
    assert descriptives.loc["mean", "IB"] == pytest.approx(3.61, abs=0.01)
    assert descriptives.loc["mean", "AUB"] == pytest.approx(3.68, abs=0.01)

    # =====================================================
    # CORRELATION MATRIX
    # =====================================================

    corr = get_construct_correlation_matrix(df)

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

    for construct in corr.columns:
        assert corr.loc[construct, construct] == pytest.approx(1.0)

    # =====================================================
    # T-TEST
    # =====================================================

    t_stat, p_value = run_ttest_aub_gender(df)

    assert t_stat == pytest.approx(-0.417, abs=0.01)
    assert p_value == pytest.approx(0.677, abs=0.01)
    assert p_value > 0.05

    # =====================================================
    # ANOVA
    # =====================================================

    f_stat, p_value = run_anova_aub_area(df)

    assert f_stat == pytest.approx(1.4681, abs=0.01)
    assert p_value == pytest.approx(0.2310, abs=0.01)
    assert p_value > 0.05

    # =====================================================
    # FINAL OUTPUT
    # =====================================================

    gender_summary = get_aub_by_gender_summary(df)

    assert gender_summary.loc["Male", "count"] == 167
    assert gender_summary.loc["Female", "count"] == 588
    assert gender_summary.loc["Male", "mean"] == pytest.approx(3.66, abs=0.01)
    assert gender_summary.loc["Female", "mean"] == pytest.approx(3.69, abs=0.01)
    assert gender_summary.loc["Male", "median"] == pytest.approx(3.75)
    assert gender_summary.loc["Female", "median"] == pytest.approx(4.00)
    assert gender_summary.loc["Male", "std"] == pytest.approx(0.79, abs=0.01)
    assert gender_summary.loc["Female", "std"] == pytest.approx(0.67, abs=0.01)

    area_summary = get_aub_by_area_summary(df)

    assert area_summary.loc["Urban", "count"] == 461
    assert area_summary.loc["Suburban", "count"] == 74
    assert area_summary.loc["Rural", "count"] == 222

    assert area_summary.loc["Urban", "mean"] == pytest.approx(3.71, abs=0.01)
    assert area_summary.loc["Suburban", "mean"] == pytest.approx(3.64, abs=0.01)
    assert area_summary.loc["Rural", "mean"] == pytest.approx(3.62, abs=0.01)

    assert area_summary.loc["Urban", "median"] == pytest.approx(4.00)
    assert area_summary.loc["Suburban", "median"] == pytest.approx(3.75)
    assert area_summary.loc["Rural", "median"] == pytest.approx(3.75)

    frequency_summary = get_aub_by_frequency_summary(df)

    assert frequency_summary.loc["Daily", "count"] == 725
    assert frequency_summary.loc["Weekly", "count"] == 14
    assert frequency_summary.loc["Monthly", "count"] == 7
    assert frequency_summary.loc["Rarely Used", "count"] == 11