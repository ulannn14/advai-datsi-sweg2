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

# The following integration test checks:
# - If the complete preprocessing and EDA pipeline reproduces the
#   exact workflow used in the notebook.
# - If the dataset remains unchanged throughout every processing stage.
# - If all intermediate outputs match the original analysis.
#
# This test will fail if:
# - The dataset structure changes (rows, columns, or data types).
# - A column is renamed, removed, or added.
# - Missing values are introduced.
# - Duplicate detection changes.
# - The PEU4 or Job columns are not removed correctly.
# - A composite score formula is modified.
# - A questionnaire item used in a composite score is changed or removed.
# - Any respondent is accidentally removed during preprocessing.
# - Demographic distributions no longer match the original dataset.
# - Descriptive statistics are modified.
# - Correlation values change.
# - The t-test implementation is modified.
# - The ANOVA implementation is modified.
# - The final EDA results no longer match the notebook outputs.
#
# In other words, this test verifies that the entire preprocessing and
# EDA workflow still produces the same research results reported in the
# original notebook.
def test_preprocessing_and_eda_pipeline():
    """
    SCT-001

    End-to-end integration test for the complete
    preprocessing and EDA pipeline.
    """

    # =====================================================
    # DATASET LOADING
    # =====================================================

    df = load_dataset(DATASET_PATH)

    assert df.shape == (757, 31)

    assert validate_columns(df) == EXPECTED_COLUMNS

    assert validate_dtypes(df).eq("int64").all()

    rows, columns = inspect_dataset(df)

    assert rows == 757
    assert columns == 31

    assert check_missing_values(df).sum() == 0

    duplicates = find_duplicates(df)

    assert duplicates.shape == (129, 31)

    unique_summary = validate_unique_values(df)

    assert unique_summary.shape == (31, 3)

    # =====================================================
    # DATA CLEANING
    # =====================================================

    df = drop_columns(df, ["PEU4"])

    assert df.shape == (757, 30)

    assert "PEU4" not in df.columns

    df = drop_columns(df, ["Job"])

    assert df.shape == (757, 29)

    assert "Job" not in df.columns

    # =====================================================
    # FEATURE ENGINEERING
    # =====================================================

    df = compute_composite_score(
        df,
        ["PU1", "PU2", "PU3", "PU4"],
        "PU"
    )

    df = compute_composite_score(
        df,
        ["PEU1", "PEU2", "PEU3"],
        "PEU"
    )

    df = compute_composite_score(
        df,
        ["FSC1", "FSC2", "FSC3"],
        "FSC"
    )

    df = compute_composite_score(
        df,
        ["SP1", "SP2", "SP3", "SP4"],
        "SP"
    )

    df = compute_composite_score(
        df,
        ["TP1", "TP2", "TP3"],
        "TP"
    )

    df = compute_composite_score(
        df,
        ["IB1", "IB2", "IB3", "IB4"],
        "IB"
    )

    df = compute_composite_score(
        df,
        ["AUB1", "AUB2", "AUB3", "AUB4"],
        "AUB"
    )

    assert df.shape == (757, 36)

    assert df.loc[0, "PU"] == 4
    assert df.loc[756, "PU"] == 4.5

    assert df.loc[0, "PEU"] == 4
    assert df.loc[756, "PEU"] == pytest.approx(4.333333)

    assert df.loc[0, "FSC"] == 4
    assert df.loc[756, "FSC"] == pytest.approx(4.333333)

    assert df.loc[0, "SP"] == 4
    assert df.loc[756, "SP"] == 4.5

    assert df.loc[0, "TP"] == 4
    assert df.loc[756, "TP"] == 3

    assert df.loc[0, "IB"] == 3
    assert df.loc[756, "IB"] == 4

    assert df.loc[0, "AUB"] == 4
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

    area = get_area_distribution(df)

    assert area["Urban"] == 461
    assert area["Suburban"] == 74
    assert area["Rural"] == 222

    frequency = get_frequency_distribution(df)

    assert frequency["Daily"] == 725
    assert frequency["Weekly"] == 14
    assert frequency["Monthly"] == 7
    assert frequency["Rarely Used"] == 11

    # =====================================================
    # CONSTRUCT DESCRIPTIVES
    # =====================================================

    descriptives = get_construct_descriptives(df)

    assert descriptives.loc["mean", "PU"] == pytest.approx(
        3.77,
        abs=0.01
    )

    assert descriptives.loc["mean", "PEU"] == pytest.approx(
        3.70,
        abs=0.01
    )

    assert descriptives.loc["mean", "AUB"] == pytest.approx(
        3.68,
        abs=0.01
    )

    # =====================================================
    # CORRELATION MATRIX
    # =====================================================

    corr = get_construct_correlation_matrix(df)

    assert corr.loc["PU", "PEU"] == pytest.approx(
        0.788226,
        abs=0.001
    )

    assert corr.loc["PU", "AUB"] == pytest.approx(
        0.771895,
        abs=0.001
    )

    assert corr.loc["IB", "AUB"] == pytest.approx(
        0.683213,
        abs=0.001
    )

    # =====================================================
    # T-TEST
    # =====================================================

    t_stat, p_value = run_ttest_aub_gender(df)

    assert t_stat == pytest.approx(
        -0.417,
        abs=0.01
    )

    assert p_value == pytest.approx(
        0.677,
        abs=0.01
    )

    # =====================================================
    # ANOVA
    # =====================================================

    f_stat, p_value = run_anova_aub_area(df)

    assert f_stat == pytest.approx(
        1.4681,
        abs=0.01
    )

    assert p_value == pytest.approx(
        0.2310,
        abs=0.01
    )

    # =====================================================
    # FINAL OUTPUT
    # =====================================================

    gender_summary = get_aub_by_gender_summary(df)

    assert gender_summary.loc["Male", "mean"] == pytest.approx(
        3.66,
        abs=0.01
    )

    area_summary = get_aub_by_area_summary(df)

    assert area_summary.loc["Urban", "mean"] == pytest.approx(
        3.71,
        abs=0.01
    )

    frequency_summary = get_aub_by_frequency_summary(df)

    assert frequency_summary.loc["Daily", "count"] == 725
