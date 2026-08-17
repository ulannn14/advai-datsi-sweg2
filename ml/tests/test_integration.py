import pandas as pd
import numpy as np
import pytest

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import GridSearchCV


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

from ml.scripts.clustering import (
    select_features,
    standardize_features,
    calculate_vif,
    evaluate_kmeans_clusters,
    perform_kmeans,
    get_cluster_sizes,
    calculate_cluster_profiles,
    perform_shapiro_test,
    perform_kruskal_wallis,
    perform_dunn_test,
)

from ml.scripts.machinelearning import (
    select_features as ml_select_features,
    select_target,
    split_dataset,
    create_kfold,
    standardize_features as ml_standardize_features,
    create_rf_param_grid,
    create_rf_gridsearch,
    fit_rf_gridsearch,
    get_best_rf_params,
    get_best_rf_model,
    predict_rf,
    summarize_rf_results,
    evaluate_regression,
    create_mlp_param_grid,
    create_mlp_gridsearch,
    fit_mlp_gridsearch,
    get_best_mlp_model,
    get_best_mlp_cv_mse,
    predict_mlp,
    compare_models,
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

FEATURES = [
    "PU",
    "PEU",
    "FSC",
    "SP",
    "TP",
    "IB",
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

COMPOSITES = {
    "PU": ["PU1", "PU2", "PU3", "PU4"],
    "PEU": ["PEU1", "PEU2", "PEU3"],
    "FSC": ["FSC1", "FSC2", "FSC3"],
    "SP": ["SP1", "SP2", "SP3", "SP4"],
    "TP": ["TP1", "TP2", "TP3"],
    "IB": ["IB1", "IB2", "IB3", "IB4"],
    "AUB": ["AUB1", "AUB2", "AUB3", "AUB4"],
}

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

    # ================================= DATASET LOADING =================================
    df = load_dataset(DATASET_PATH)

    # Confirms load_dataset() returns the right TYPE, not just "something".
    # If someone accidentally returns a dict, list, or numpy array instead
    # of a DataFrame, this catches it immediately.
    assert isinstance(df, pd.DataFrame)

    # Locks in the exact shape of the RAW dataset (before any cleaning).
    # 757 rows, 31 columns. If a row gets silently dropped during CSV
    # loading (e.g. bad encoding, a stray blank line) or a column gets
    # added/removed upstream, this fails immediately.
    assert df.shape == (757, 31)

    # Checks column NAMES and ORDER against EXPECTED_COLUMNS. This is
    # the check that catches the "PU1 -> zz" style mutation: a renamed
    # column, a reordered column, or a typo in a header would all fail
    # here even though df.shape would still look fine.
    assert df.columns.tolist() == EXPECTED_COLUMNS

    rows, columns = inspect_dataset(df)

    # Redundant with df.shape above on purpose -- this verifies
    # inspect_dataset() itself is wired correctly (unpacks rows/columns
    # in the right order), not just that the raw dataframe is correct.
    assert rows == 757
    assert columns == 31

    # validate_columns() should return the same column list as
    # df.columns.tolist(). This guards against the function silently
    # returning a stale or hardcoded list instead of reading live data.
    assert validate_columns(df) == EXPECTED_COLUMNS

    dtypes = validate_dtypes(df)

    # Confirms validate_dtypes() returns the ACTUAL dtypes of df, not a
    # copy, a guess, or a hardcoded Series. pd.testing.assert_series_equal
    # is used instead of == because Series comparison needs to check
    # index alignment too, not just values.
    pd.testing.assert_series_equal(
        dtypes,
        df.dtypes
    )

    # Every column in this dataset should be int64 (Likert-scale /
    # categorical codes). If someone reads the CSV with a stray decimal
    # point or a text value in one cell, pandas silently upcasts that
    # column to float64 or object -- this assertion catches that.
    assert dtypes.eq("int64").all()

    missing = check_missing_values(df)

    # Builds the expected "0 missing values per column" Series and
    # compares it exactly. This is the test that would fail if a future
    # version of the dataset introduces NaNs (e.g. a respondent skips
    # a question) that the rest of the pipeline isn't built to handle.
    expected_missing = pd.Series(
        [0] * len(EXPECTED_COLUMNS),
        index=EXPECTED_COLUMNS
    )

    pd.testing.assert_series_equal(
        missing,
        expected_missing
    )

    duplicates = find_duplicates(df)

    # 129 duplicate rows is a known, expected property of THIS dataset
    # (Likert-scale survey responses collide often since there are only
    # a handful of possible answer combinations). If this number changes,
    # either the dataset changed or find_duplicates() logic changed,
    # both are worth knowing about.
    assert duplicates.shape == (129, 31)

    unique_summary = validate_unique_values(df)

    # Checks the *shape* of the summary table itself, right column
    # names, in the right order. A broken validate_unique_values() might
    # still run without error but return columns in the wrong order or
    # under different names.
    assert list(unique_summary.columns) == [
        "Variable",
        "Unique Count",
        "Unique Values"
    ]

    # Three separate checks on the CONTENT of the summary:
    # - "Variable" column lists every column, in order
    # - "Unique Count" matches how many distinct values each column has
    #   (e.g. Gender should have exactly 3, Job should have exactly 1)
    # - "Unique Values" matches the exact set of values (e.g. Gender
    #   should only ever contain 1, 2, 3 -- never a stray 4 or a 0)
    # Together these catch bad survey data (out-of-range answers) as
    # well as bugs in how validate_unique_values() computes things.
    assert unique_summary["Variable"].tolist() == EXPECTED_COLUMNS
    assert unique_summary["Unique Count"].tolist() == EXPECTED_UNIQUE_COUNTS
    assert unique_summary["Unique Values"].tolist() == EXPECTED_UNIQUE_VALUES

    # ================================= DATA CLEANING =================================
    df = drop_columns(df, ["PEU4"])

    expected_columns = EXPECTED_COLUMNS.copy()
    expected_columns.remove("PEU4")

    # Three angles on the same fact, checked separately on purpose:
    # 1. Overall shape shrank by exactly 1 column (757, 30)
    # 2. The remaining column order still matches everything else
    #    that was untouched
    # 3. PEU4 specifically is gone, not just "some column" is gone
    # If drop_columns() ever drops the WRONG column by mistake, (1) and
    # (2) would still pass but (3) would fail -- that's why all three
    # checks exist instead of just one.
    assert df.shape == (757, 30)
    assert df.columns.tolist() == expected_columns
    assert "PEU4" not in df.columns

    df = drop_columns(df, ["Job"])

    expected_columns.remove("Job")

    # Same three-angle check, now for Job. Job only had 1 unique value
    # across all 757 respondents (see EXPECTED_UNIQUE_COUNTS above), so
    # it carries zero variance and zero explanatory power -- that's why
    # it's dropped, and this locks that decision in.
    assert df.shape == (757, 29)
    assert df.columns.tolist() == expected_columns
    assert "Job" not in df.columns

    # ================================= FEATURE ENGINEERING =================================

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

    # 29 columns before + 7 new composite columns = 36. This alone
    # wouldn't catch a WRONG composite calculation, which is exactly
    # why the row-level checks below exist too.
    assert df.shape == (757, 36)

    # Each block below checks 4 specific respondents (row 0, row 1, row
    # 100, and the last row 756) for each construct. This is the "test
    # against real dataset values" approach from the professor's
    # feedback -- these are the exact numbers those rows should produce
    # if the mean-of-items formula is applied correctly to the real
    # data. If a composite is accidentally computed from the wrong
    # columns (like the SP1-SP3-instead-of-SP1-SP4 bug found earlier),
    # these values shift and the test catches it.
    assert df.loc[0, "PU"] == 4
    assert df.loc[1, "PU"] == 3
    assert df.loc[100, "PU"] == 3
    assert df.loc[756, "PU"] == 4.5

    assert df.loc[0, "PEU"] == 4
    assert df.loc[1, "PEU"] == 3
    assert df.loc[100, "PEU"] == 3
    # PEU is a 3-item average that doesn't divide evenly, hence
    # pytest.approx() instead of an exact == comparison -- floating
    # point division won't land on a perfectly round number here.
    assert df.loc[756, "PEU"] == pytest.approx(4.333333333333333)

    assert df.loc[0, "FSC"] == 4
    assert df.loc[1, "FSC"] == 3
    assert df.loc[100, "FSC"] == 3
    assert df.loc[756, "FSC"] == pytest.approx(4.333333333333333)

    # SP is the construct that broke earlier when it was silently
    # recomputed from only SP1-SP3 instead of SP1-SP4. This assertion
    # is the regression check for that exact bug -- if it ever creeps
    # back in, df.loc[756, "SP"] stops being 4.5.
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

    # AUB is the eventual prediction target for the ML pipeline (SCT-003),
    # so getting this composite right matters beyond just EDA -- a bug
    # here would silently corrupt the ML model's training labels too.
    assert df.loc[0, "AUB"] == 4
    assert df.loc[1, "AUB"] == 3
    assert df.loc[100, "AUB"] == 3
    assert df.loc[756, "AUB"] == 4

    # ================================= DEMOGRAPHIC DISTRIBUTIONS =================================

    gender = get_gender_distribution(df)

    # Exact respondent counts per category, taken straight from the
    # notebook's own reported output. Checking .sum() == 757 alongside
    # the individual categories confirms no respondents got silently
    # dropped or double-counted by value_counts()/groupby().
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

    # ================================= CONSTRUCT DESCRIPTIVES =================================

    descriptives = get_construct_descriptives(df)

    # Confirms describe() was called on exactly the 7 composite columns,
    # in this order -- not on the raw item-level columns (PU1-PU4 etc.)
    # by mistake, and not with extra/missing constructs.
    assert descriptives.columns.tolist() == [
        "PU",
        "PEU",
        "FSC",
        "SP",
        "TP",
        "IB",
        "AUB",
    ]

    # All 757 respondents contributed to every construct's stats --
    # if any construct's composite score computation produced NaNs
    # (e.g. one questionnaire item had a missing value that wasn't
    # caught earlier), describe()'s count would drop below 757.
    assert descriptives.loc["count"].tolist() == [757] * 7

    # Rounded to 2 decimals with abs=0.01 tolerance because these are
    # the values reported in the write-up (which itself only reports
    # 2 decimal places) -- not because the underlying computation is
    # imprecise.
    assert descriptives.loc["mean", "PU"] == pytest.approx(3.77, abs=0.01)
    assert descriptives.loc["mean", "PEU"] == pytest.approx(3.70, abs=0.01)
    assert descriptives.loc["mean", "FSC"] == pytest.approx(3.73, abs=0.01)
    assert descriptives.loc["mean", "SP"] == pytest.approx(3.60, abs=0.01)
    assert descriptives.loc["mean", "TP"] == pytest.approx(3.56, abs=0.01)
    assert descriptives.loc["mean", "IB"] == pytest.approx(3.61, abs=0.01)
    assert descriptives.loc["mean", "AUB"] == pytest.approx(3.68, abs=0.01)

    # ================================= CORRELATION MATRIX =================================

    corr = get_construct_correlation_matrix(df)

    # Tolerance here is tighter (abs=0.001) than the descriptives above
    # because these values were pulled directly from running
    # get_construct_correlation_matrix() on the real, corrected data
    # (full precision), not eyeballed off a rounded write-up table.
    # Every pair below is checked individually so a bug affecting just
    # ONE construct (like the earlier SP mutation) shows up as isolated
    # failures on the SP-* pairs specifically, not a vague "matrix is
    # wrong somewhere" failure.
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

    # These three (SP-TP, SP-IB, SP-AUB) are exactly the pairs that
    # shifted when the SP composite bug was fixed -- they're the most
    # sensitive part of this whole matrix to a regression.
    assert corr.loc["SP", "TP"] == pytest.approx(0.734756, abs=0.001)
    assert corr.loc["SP", "IB"] == pytest.approx(0.644206, abs=0.001)
    assert corr.loc["SP", "AUB"] == pytest.approx(0.691883, abs=0.001)

    assert corr.loc["TP", "IB"] == pytest.approx(0.663131, abs=0.001)
    assert corr.loc["TP", "AUB"] == pytest.approx(0.662269, abs=0.001)

    assert corr.loc["IB", "AUB"] == pytest.approx(0.683213, abs=0.001)

    # A correlation matrix must always have 1.0 on its diagonal (every
    # construct correlates perfectly with itself). This is a structural
    # sanity check independent of the actual data -- if it ever fails,
    # the bug is in how the matrix itself is built (e.g. wrong axis,
    # wrong method), not in the data.
    for construct in corr.columns:
        assert corr.loc[construct, construct] == pytest.approx(1.0)

    # ================================= T-TEST =================================

    t_stat, p_value = run_ttest_aub_gender(df)

    # Checks BOTH the raw statistic and the resulting conclusion.
    # t_stat/p_value pin down the exact numbers; p_value > 0.05 pins
    # down the INTERPRETATION (no significant gender difference in
    # AUB). If someone changes the test from Welch's to a different
    # variant, the interpretation could stay the same while the raw
    # numbers drift outside tolerance -- checking both catches either
    # kind of change.
    assert t_stat == pytest.approx(-0.417, abs=0.01)
    assert p_value == pytest.approx(0.677, abs=0.01)
    assert p_value > 0.05

    # ================================= ANOVA =================================
    f_stat, p_value = run_anova_aub_area(df)

    # Same reasoning as the t-test above: exact values plus the
    # significance conclusion (no significant difference in AUB across
    # Urban/Suburban/Rural respondents).
    assert f_stat == pytest.approx(1.4681, abs=0.01)
    assert p_value == pytest.approx(0.2310, abs=0.01)
    assert p_value > 0.05

    # ================================= FINAL OUTPUT =================================

    gender_summary = get_aub_by_gender_summary(df)

    # Cross-checks against the earlier gender distribution counts
    # (167 male, 588 female) -- if these counts don't match, it means
    # get_aub_by_gender_summary() is filtering or grouping respondents
    # differently than get_gender_distribution() did, which would be
    # an inconsistency worth catching even if each function looks
    # correct in isolation.
    assert gender_summary.loc["Male", "count"] == 167
    assert gender_summary.loc["Female", "count"] == 588
    assert gender_summary.loc["Male", "mean"] == pytest.approx(3.66, abs=0.01)
    assert gender_summary.loc["Female", "mean"] == pytest.approx(3.69, abs=0.01)
    assert gender_summary.loc["Male", "median"] == pytest.approx(3.75)
    assert gender_summary.loc["Female", "median"] == pytest.approx(4.00)
    assert gender_summary.loc["Male", "std"] == pytest.approx(0.79, abs=0.01)
    assert gender_summary.loc["Female", "std"] == pytest.approx(0.67, abs=0.01)

    area_summary = get_aub_by_area_summary(df)

    # Same cross-check idea as gender: 461 + 74 + 222 must equal 757
    # and must match the earlier area distribution counts exactly.
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

    # Only counts are checked here (not mean/median/std) since the
    # earlier frequency distribution already locked in these exact
    # numbers -- this just confirms get_aub_by_frequency_summary()
    # groups respondents the same way get_frequency_distribution() did.
    assert frequency_summary.loc["Daily", "count"] == 725
    assert frequency_summary.loc["Weekly", "count"] == 14
    assert frequency_summary.loc["Monthly", "count"] == 7
    assert frequency_summary.loc["Rarely Used", "count"] == 11


# The following integration test checks:
# - If the complete clustering workflow reproduces the exact process used
#   in the notebook.
# - If the preprocessing required for clustering produces the same
#   composite construct scores.
# - If the correct clustering features are selected.
# - If feature standardization produces the same standardized values.
# - If the VIF analysis produces the same multicollinearity results.
# - If K-Means model evaluation produces the same inertia and silhouette
#   scores for k = 2 through 10.
# - If the final four-cluster K-Means solution is reproduced.
# - If cluster sizes and cluster profiles remain unchanged.
# - If the statistical inference results across clusters remain unchanged.
# - If the Dunn's post-hoc comparisons remain consistent with the notebook.
#
# This test will fail if:
# - The preprocessing or composite-score calculations change.
# - A clustering feature is removed, renamed, or reordered.
# - Standardization is changed.
# - VIF calculations change.
# - The K-Means configuration or random state changes.
# - The inertia or silhouette results change.
# - Cluster assignments or cluster sizes change.
# - Cluster profile values change.
# - The Shapiro-Wilk, Kruskal-Wallis, or Dunn's test results change.
# - The clustering notebook results can no longer be reproduced.
def test_clustering_pipeline():
    """
    SCT-002

    End-to-end integration test for the complete
    clustering.
    """

    # ================================= DATA PREPROCESSING =================================
    # Recreates the same preprocessing and composite-score calculations
    # performed before clustering in the notebook.
    #
    # These steps are intentionally included in the clustering integration
    # test because the clustering results depend directly on the resulting
    # composite construct values. A change here can propagate through
    # standardization, VIF, K-Means, and all downstream statistical tests.

    # The clustering workflow must begin with the same raw dataset used
    # in the notebook. If the source data changes, every downstream
    # clustering result may also change.
    df = load_dataset(DATASET_PATH)


    # PEU4 and Job are removed using the same preprocessing decisions
    # established in Phase 1. Keeping these operations in the integration
    # test ensures the clustering pipeline receives the same feature space
    # as the notebook.
    df = drop_columns(df, ["Job"])
    df = drop_columns(df, ["PEU4"])

    # The seven composite constructs are recreated using their exact
    # questionnaire items. These values become the inputs for both the
    # clustering features and the AUB variable used during cluster profiling
    # and statistical inference.
    #
    # The SP calculation is especially important because SP previously
    # differed when the wrong questionnaire items were used. Keeping the
    # exact SP1-SP4 calculation here prevents that regression from silently
    # changing the clustering results.
    df = compute_composite_score(df, ["PU1", "PU2", "PU3", "PU4"], "PU")
    df = compute_composite_score(df, ["PEU1", "PEU2", "PEU3"], "PEU")
    df = compute_composite_score(df, ["FSC1", "FSC2", "FSC3"], "FSC")
    df = compute_composite_score(df, ["SP1", "SP2", "SP3", "SP4"], "SP")
    df = compute_composite_score(df, ["TP1", "TP2", "TP3"], "TP")
    df = compute_composite_score(df, ["IB1", "IB2", "IB3", "IB4"], "IB")
    df = compute_composite_score(df, ["AUB1", "AUB2", "AUB3", "AUB4"], "AUB")
    
    # ================================= FEATURE SELECTION =================================

    # select_features() should return only the six behavioral constructs
    # used for clustering, in the exact order used by the notebook.
    #
    # Checking the type, shape, column names, and actual values prevents a
    # feature from being accidentally omitted, added, renamed, reordered,
    # or populated from the wrong source columns.
    result = select_features(df, FEATURES)

    # Confirms the feature-selection wrapper returns a DataFrame rather than
    # an array or another object. The downstream clustering workflow expects
    # the selected features to retain their tabular structure.
    assert isinstance(result, pd.DataFrame)

    # Checks both the feature NAMES and their ORDER. This is important because
    # the standardized matrix preserves this column order, so reordering the
    # constructs would change which standardized values correspond to each
    # feature.
    assert result.shape == (757, 6)
    assert result.columns.tolist() == FEATURES
    assert len(result.columns) == len(FEATURES)

    # Compares the complete selected feature matrix against the source
    # dataframe. This goes beyond checking the schema: it verifies that every
    # respondent's six clustering values are exactly the values supplied to
    # the clustering workflow.
    pd.testing.assert_frame_equal(result, df[FEATURES])

    # ================================= FEATURE STANDARDIZATION =================================
    # Standardizes the six clustering constructs using the same procedure
    # used in the notebook before VIF analysis and K-Means.
    X = select_features(df, FEATURES)
    result = standardize_features( X)

    assert isinstance(result, np.ndarray)
    assert result.shape == (757, 6)

    # No standardized observation should become NaN. A NaN value would
    # indicate that invalid or missing input data entered the clustering
    # workflow and could affect VIF and K-Means.
    assert not np.isnan(result).any()

    # Ensures the standardized matrix contains no infinite values.
    # Infinite values would indicate an invalid transformation and would
    # make the clustering calculations unreliable.
    assert np.isfinite(result).all()

    # StandardScaler should center every clustering feature around zero.
    # This confirms that the standardization actually removed the original
    # feature means as expected.
    assert np.allclose(result.mean(axis=0), 0, atol=1e-10)

    # StandardScaler should scale every feature to unit population standard
    # deviation. This ensures that the constructs are placed on a comparable
    # scale before distance-based K-Means clustering is performed.
    assert np.allclose(result.std(axis=0), 1, atol=1e-10)

    expected = StandardScaler().fit_transform(X)

    # Recreates the expected transformation independently using sklearn.
    # This is stronger than checking only the mean and standard deviation:
    # it verifies that every standardized observation matches the values
    # produced by the StandardScaler implementation used by the notebook.
    np.testing.assert_allclose(result,expected, rtol=1e-10, atol=1e-10)

    # ================================= MULTICOLLINEARITY (VIF) =================================
    # Calculates the Variance Inflation Factor for each clustering construct.
    # VIF is performed on the standardized feature matrix, matching the
    # procedure used in the notebook.
    X_scaled = standardize_features(X)
    result = calculate_vif(X_scaled, FEATURES)

    # The VIF result should contain exactly one row for each clustering
    # construct and the two expected output columns. This catches changes
    # to the structure of the multicollinearity analysis.
    assert isinstance(result, pd.DataFrame)
    assert result.columns.tolist() == [ "Construct", "VIF"]

    assert result.shape == (6, 2)

    # The construct order must match FEATURES so that each VIF value is
    # associated with the correct behavioral construct.
    assert result["Construct"].tolist() == FEATURES

    # VIF values must be numeric because they represent calculated
    # multicollinearity statistics rather than labels or text.
    assert pd.api.types.is_numeric_dtype(result["VIF"])

    # Every VIF value must be finite. Infinite or NaN values would indicate
    # that the VIF calculation is no longer producing a usable result.
    assert np.isfinite(result["VIF"]).all()

    # These are the VIF values established from the notebook output.
    # Comparing the complete vector verifies that the actual VIF result
    # matches the notebook for all six constructs, rather than checking
    # only that the values are valid numbers.
    expected_vif = [3.654939, 3.515319, 3.988501, 3.150629, 2.709242, 2.301402,]

    # Compares the actual VIF values against the established notebook values.
    # A tolerance is used because VIF calculations involve floating-point
    # arithmetic.
    np.testing.assert_allclose(result["VIF"].values, expected_vif, rtol=1e-5, atol=1e-5)

    # ================================= OPTIMAL K SELECTION =================================
    # Evaluates the candidate K-Means solutions used to determine the
    # appropriate number of clusters.
    
    # The notebook evaluates k = 2 through k = 10. Keeping this exact range
    # ensures the integration test covers the same candidate models.
    k_values = range(2, 11)
    inertia, silhouette_scores = (evaluate_kmeans_clusters(X_scaled, k_values))

    assert isinstance(inertia, list)
    assert isinstance(silhouette_scores, list)

    # Nine candidate values of k should produce exactly nine inertia values
    # and nine silhouette scores. A mismatch indicates that one or more
    # candidate models were not evaluated.
    assert len(inertia) == 9
    assert len(silhouette_scores) == 9

    # Ensures every model evaluation produced a valid numerical result.
    # NaN or infinite values would make the model-selection results unusable.
    assert np.isfinite(inertia).all()
    assert np.isfinite(silhouette_scores).all()

    assert all(value > 0 for value in inertia)

    # Silhouette scores must remain within their theoretical range of -1 to 1.
    # This catches invalid or corrupted model-evaluation results.
    assert all(-1 <= value <= 1 for value in silhouette_scores)

    # Inertia should decrease as more clusters are introduced because adding
    # clusters cannot increase the minimum within-cluster sum of squares.
    # This confirms the expected behavior of the evaluated K-Means solutions.
    assert all(inertia[i] > inertia[i + 1] for i in range(len(inertia) - 1))

    # These are the exact inertia values obtained from the notebook for
    # k = 2 through k = 10. They serve as the established baseline for
    # the clustering model-selection stage.
    expected_inertia = [
        2397.3951517810065, 
        1876.6780711816932, 
        1470.371175770248, 
        1289.3260086578698, 
        1194.8825839212625, 
        1138.661553747194, 
        1082.160274964026, 
        1041.3118953265096, 
        995.2666652249214,
    ]

    # These are the corresponding silhouette scores reported by the notebook.
    # Checking all nine values ensures that the complete model-selection
    # stage remains reproducible rather than checking only the selected k.
    expected_silhouette_scores = [
        0.4357624529557227, 
        0.42533701123564177, 
        0.37747256266114426, 
        0.33290308059903284, 
        0.2913130488967038, 
        0.29138177791986325, 
        0.30602460012945765, 
        0.2930900543684983, 
        0.29391001252620486,
    ]

    # Compares the complete actual model-selection results against the
    # established notebook results. This catches changes to preprocessing,
    # standardization, random state, K-Means configuration, or scoring logic
    # that could otherwise alter the selected clustering solution.
    np.testing.assert_allclose(inertia, expected_inertia, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(silhouette_scores, expected_silhouette_scores, rtol=1e-5, atol=1e-5)

    # ================================= CLUSTERING =================================
    # Performs the final four-cluster K-Means model selected from the
    # model-evaluation stage.
    model, clusters = perform_kmeans(X_scaled)

    # Confirms that the wrapper returns the fitted sklearn KMeans model
    # expected by the clustering workflow.
    assert isinstance(model, KMeans)

    # The final notebook solution uses exactly four clusters. If this changes,
    # the cluster assignments, sizes, profiles, and statistical tests will
    # no longer correspond to the established analysis.
    assert model.n_clusters == 4

    # There must be exactly one cluster assignment for every respondent.
    # A different length would indicate that observations were lost or
    # incorrectly assigned.
    assert len(clusters) == 757

    # There must be exactly one cluster assignment for every respondent.
    # A different length would indicate that observations were lost or
    # incorrectly assigned.
    assert len(np.unique(clusters)) == 4
    assert set(np.unique(clusters)) == {0, 1, 2, 3}

    # Confirms that all four expected cluster labels are actually present
    # and that no unexpected cluster IDs were produced.
    assert model.cluster_centers_.shape == (4, 6)

    # Confirms that the K-Means model was actually fitted and contains
    # the calculated within-cluster inertia.
    assert hasattr(model, "inertia_")

    # Confirms that the K-Means model was actually fitted and contains
    # the calculated within-cluster inertia.
    assert np.isfinite(model.cluster_centers_).all()
    assert np.isfinite(clusters).all()

    # ================================= CLUSTER SIZES =================================
    # Calculates the number of respondents assigned to each of the four
    # final clusters.
    result = get_cluster_sizes(clusters)

    assert isinstance(result, pd.Series)

    # Cluster labels must remain ordered as 0, 1, 2, and 3 so that each
    # reported cluster size corresponds to the same cluster used throughout
    # the notebook.
    assert result.index.tolist() == [0, 1, 2, 3]

    # These are the exact cluster sizes established in the notebook.
    # Checking the complete distribution ensures that the final K-Means
    # assignments have not changed even if all four cluster labels are still
    # technically present.
    assert result.tolist() == [97, 258, 20, 382]
    assert result.name == "count"

    # The four cluster sizes must account for all 757 respondents. This catches
    # missing or duplicated assignments that could otherwise be hidden by
    # checking the individual cluster counts alone.
    assert result.sum() == 757

    # Every final cluster must contain at least one respondent. An empty
    # cluster would make the corresponding cluster profile and statistical
    # analysis invalid.
    assert (result > 0).all()

    # The four cluster sizes must account for all 757 respondents. This catches
    # missing or duplicated assignments that could otherwise be hidden by
    # checking the individual cluster counts alone.
    df["Cluster"] = clusters
    result = calculate_cluster_profiles(df, FEATURES)

    assert isinstance(result, pd.DataFrame)

    # Four clusters and seven reported variables (six clustering constructs
    # plus AUB) should produce a 4 x 7 profile table.
    assert result.shape == (4, 7)

    # The profile must contain the six clustering constructs followed by AUB.
    # This ensures the same variables are being used to characterize each
    # cluster as in the notebook.
    assert result.columns.tolist() == (FEATURES + ["AUB"])

    # The profile index must correspond to the four K-Means cluster labels.
    # This keeps each profile associated with the correct cluster.
    assert result.index.tolist() == [0, 1, 2, 3]

    # Every cluster-profile value must be finite. A missing or infinite mean
    # would indicate a problem with the cluster assignments or source data.
    assert np.isfinite(result.values).all()

    # All construct and AUB means should remain within the original
    # questionnaire's 1-to-5 Likert scale. Values outside this range would
    # indicate an invalid aggregation or corrupted source data.
    assert (result[FEATURES] >= 1).all().all()
    assert (result[FEATURES] <= 5).all().all()
    assert (result["AUB"] >= 1).all()
    assert (result["AUB"] <= 5).all()

    # These are the cluster-profile values established from the notebook.
    # Each row represents one cluster and each column represents one
    # behavioral construct or AUB.
    expected_profiles = pd.DataFrame(
        [
            [
                4.6391752577, 4.5841924399, 4.7079037801, 4.5773195876, 4.4707903780, 4.5257731959, 4.4664948454
            ],
            [
                3.2451550388, 3.1550387597, 3.1020671835, 3.0474806202, 3.0348837209, 3.1715116279, 3.1715116279
            ],
            [
                1.6500000000, 1.5666666667, 1.5833333333, 1.7750000000, 1.4666666667, 1.6625000000, 1.7125000000
            ],
            [
                4.0215968586, 3.9554973822, 4.0200698080, 3.8239528796, 3.7870855148, 3.7676701571, 3.9201570681
            ],
        ],
        index=[0, 1, 2, 3],
        columns=FEATURES + ["AUB"]
    )

    # Compares the complete actual cluster-profile matrix against the
    # established notebook results. This verifies that the final cluster
    # composition and the resulting construct means remain unchanged.
    np.testing.assert_allclose(result.values, expected_profiles.values, rtol=1e-5, atol=1e-5)
    
    # ================================= STATISTICAL INFERENCE =================================
    # Reproduces the statistical tests performed on AUB across the four
    # final clusters.
    result = perform_shapiro_test(df)

    assert isinstance(result, list)

    # The Shapiro-Wilk test is performed separately for each cluster.
    # Four clusters should therefore produce four results in cluster-label
    # order.
    assert len(result) == 4
    assert [item[0] for item in result] == [0, 1, 2, 3]

    for cluster, statistic, p_value in result:

        # Confirms that each result contains the expected cluster identifier,
        # Shapiro-Wilk statistic, and p-value types.
        assert isinstance(cluster, (int, np.integer))
        assert isinstance(statistic, (float, np.floating))
        assert isinstance(p_value, (float, np.floating))

        # The Shapiro-Wilk statistic and p-value must both fall within their
        # valid statistical ranges.
        assert 0 <= statistic <= 1
        assert 0 <= p_value <= 1

    # These are the Shapiro-Wilk statistics and p-values obtained from the
    # notebook. Checking them against the actual results verifies that the
    # normality analysis remains reproducible for every cluster.
    expected_statistics = [
        0.8871,
        0.8512,
        0.8787,
        0.8786,
    ]

    expected_p_values = [
        0.0000,
        0.0000,
        0.0168,
        0.0000,
    ]

    # Compares each cluster's actual statistic and p-value against the
    # corresponding notebook result. This prevents one cluster from changing
    # while the other three continue to pass.
    for ((_, statistic, p_value), expected_statistic, expected_p_value) in zip(result, expected_statistics, expected_p_values):
        assert statistic == pytest.approx(expected_statistic, abs=0.0001)
        assert p_value == pytest.approx(expected_p_value, abs=0.0001)

    # Performs the Kruskal-Wallis test on AUB across the four clusters.
    # This is the main non-parametric test used to determine whether AUB
    # differs significantly between the cluster groups.
    H, p = perform_kruskal_wallis(df)

    # These values are the established Kruskal-Wallis results from the
    # notebook. Checking both the statistic and p-value verifies not only
    # the significance conclusion but also the underlying numerical result.
    assert isinstance(H, (float, np.floating))
    assert isinstance(p, (float, np.floating))

    assert H >= 0
    assert 0 <= p <= 1

    assert H == pytest.approx(421.3159298947122, rel=1e-10)
    assert p == pytest.approx(5.341699041631374e-91, rel=1e-10)

    # Confirms the statistical conclusion reported in the notebook:
    # AUB differs significantly across the four clusters.
    assert p < 0.05

    # Performs Dunn's post-hoc test to determine which specific cluster pairs
    # differ in AUB after the significant Kruskal-Wallis result.
    result = perform_dunn_test(df)

    assert isinstance(result, pd.DataFrame)

    # Four clusters should produce a 4 x 4 pairwise comparison matrix.
    # The row and column labels must correspond to the same cluster IDs used
    # throughout the analysis.
    assert result.shape == (4, 4)
    assert result.index.tolist() == [0, 1, 2, 3]
    assert result.columns.tolist() == [0, 1, 2, 3]

    # Every Dunn comparison must produce a valid p-value between 0 and 1.
    # These checks catch malformed or invalid statistical output.
    assert np.isfinite(result.values).all()
    assert (result.values >= 0).all()
    assert (result.values <= 1).all()

    # Every Dunn comparison must produce a valid p-value between 0 and 1.
    # These checks catch malformed or invalid statistical output.
    assert np.allclose(np.diag(result), 1)

    # Pairwise comparisons are symmetric: the comparison between clusters
    # 0 and 1 must be identical to the comparison between clusters 1 and 0.
    # This confirms that the matrix has been constructed consistently.
    np.testing.assert_allclose(result.values, result.values.T)

    # Pairwise comparisons are symmetric: the comparison between clusters
    # 0 and 1 must be identical to the comparison between clusters 1 and 0.
    # This confirms that the matrix has been constructed consistently.
    expected_dunn = pd.DataFrame(
        [
            [1.0000, 0.0000, 0.0000, 0.0000],
            [0.0000, 1.0000, 0.0022, 0.0000],
            [0.0000, 0.0022, 1.0000, 0.0000],
            [0.0000, 0.0000, 0.0000, 1.0000],
        ],
        index=[0, 1, 2, 3],
        columns=[0, 1, 2, 3]
    )

    # Compares the complete actual Dunn matrix against the established
    # notebook matrix. This verifies that every pairwise comparison remains
    # consistent with the original statistical analysis.
    np.testing.assert_allclose(result.round(4).values, expected_dunn.values, rtol=1e-5, atol=1e-5)

    # These cluster pairs were identified as having statistically significant
    # differences in AUB in the notebook. Checking each pair ensures that the
    # reported post-hoc interpretation has not changed.
    assert result.loc[0, 1] < 0.05
    assert result.loc[0, 2] < 0.05
    assert result.loc[0, 3] < 0.05
    assert result.loc[1, 3] < 0.05
    assert result.loc[2, 3] < 0.05


# The following integration test checks:
# - If the machine learning pipeline correctly processes the engineered features.
# - If dataset partitioning and scaling prevent data leakage.
# - If the hyperparameter grid search executes and finds an optimal model.
# - If both Random Forest and MLP models meet the established performance baselines.
#
# This test will fail if:
# - The preprocessing or composite-score calculations change.
# - Target or predictor variables are missing or misaligned.
# - Data leakage occurs during the train-test split or scaling.
# - The parameter grids are altered in a way that breaks GridSearchCV.
# - The models fail to achieve an R² > 0.70 (RF) or > 0.65 (MLP).
def test_machine_learning_pipeline():
    """
    SCT-003

    End-to-end integration test for the machine learning
    predictive modeling pipeline (Random Forest and MLP).
    """

    # ================================= PREPROCESSING =================================
    # Loads the dataset and applies the same preprocessing and composite-score
    # calculations established in Phase 1 to guarantee the machine learning
    # models receive the exact same input as the original analysis.
    df = load_dataset(DATASET_PATH)
    df = drop_columns(df, ["Job", "PEU4"])

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
        df = compute_composite_score(df, items, construct)

    # ================================= FEATURE & TARGET SELECTION =================================
    # Extracts the six behavioral constructs to serve as predictive features
    # and isolates AUB as the prediction target. Checking dimensions ensures
    # no variables were unintentionally dropped.
    X = ml_select_features(df, FEATURES)
    y = select_target(df, "AUB")

    assert isinstance(X, pd.DataFrame)
    assert X.shape == (757, 6)
    assert X.columns.tolist() == FEATURES
    
    assert isinstance(y, pd.Series)
    assert len(y) == 757
    assert y.name == "AUB"

    # ================================= DATASET PARTITIONING & SCALING =================================
    # Partitions the dataset into an 80/20 train/test split. Fixing the
    # random state guarantees reproducible metric evaluation across test runs.
    X_train, X_test, y_train, y_test = split_dataset(
        X, y, test_size=0.20, random_state=1
    )

    assert X_train.shape == (605, 6)
    assert X_test.shape == (152, 6)
    assert len(y_train) == 605
    assert len(y_test) == 152

    # Standardizes the predictors by fitting to the training set only,
    # preventing data leakage. Checking that the scaled training array has
    # a mean of 0 and std of 1 confirms the transformation was successful.
    scaler, X_train_scaled, X_test_scaled = ml_standardize_features(
        X_train, X_test
    )

    assert X_train_scaled.shape == X_train.shape
    assert X_test_scaled.shape == X_test.shape
    assert np.allclose(X_train_scaled.mean(axis=0), 0, atol=1e-7)
    assert np.allclose(X_train_scaled.std(axis=0), 1, atol=1e-7)

    kf = create_kfold(n_splits=10, shuffle=True, random_state=1)
    assert kf.get_n_splits() == 10
    assert kf.shuffle is True

    # ================================= RANDOM FOREST PIPELINE =================================
    # Confirms the Random Forest hyperparameter grid contains the exact 
    # tuning arguments required by the notebook.
    rf_param_grid = create_rf_param_grid()
    assert "n_estimators" in rf_param_grid
    assert "max_depth" in rf_param_grid

    rf_grid = create_rf_gridsearch(rf_param_grid, kf, n_jobs=1)
    assert isinstance(rf_grid, GridSearchCV)
    assert isinstance(rf_grid.estimator, RandomForestRegressor)

    # Validates that the cross-validated grid search executes without error
    # and successfully identifies an optimal set of parameters.
    rf_fitted = fit_rf_gridsearch(rf_grid, X_train, y_train)
    assert hasattr(rf_fitted, "best_estimator_")
    assert hasattr(rf_fitted, "best_params_")

    rf_best_params, rf_best_mse = get_best_rf_params(rf_fitted)
    assert isinstance(rf_best_params, dict)
    assert rf_best_mse >= 0

    rf_summary = summarize_rf_results(rf_fitted)
    assert isinstance(rf_summary, pd.DataFrame)
    assert "rank_test_score" in rf_summary.columns

    rf_model = get_best_rf_model(rf_fitted)
    assert isinstance(rf_model, RandomForestRegressor)

    # Verifies that the best model can successfully process the test set
    # and return an array of predictions equivalent in length to y_test.
    rf_predictions = predict_rf(rf_model, X_test)
    assert isinstance(rf_predictions, np.ndarray)
    assert len(rf_predictions) == len(y_test)

    rf_mae, rf_mse, rf_rmse, rf_r2 = evaluate_regression(y_test, rf_predictions)

    # ================================= MLP PIPELINE =================================
    # Confirms the Multi-Layer Perceptron hyperparameter grid contains 
    # the exact tuning arguments required for neural network optimization.
    mlp_param_grid = create_mlp_param_grid()
    assert "hidden_layer_sizes" in mlp_param_grid
    assert "learning_rate_init" in mlp_param_grid

    mlp_grid = create_mlp_gridsearch(mlp_param_grid, kf, n_jobs=1)
    assert isinstance(mlp_grid, GridSearchCV)
    assert isinstance(mlp_grid.estimator, MLPRegressor)

    # Validates that the scaled data functions properly during MLP grid
    # search and successfully yields an optimized neural network estimator.
    mlp_fitted = fit_mlp_gridsearch(mlp_grid, X_train_scaled, y_train)
    assert hasattr(mlp_fitted, "best_estimator_")
    assert hasattr(mlp_fitted, "best_params_")

    mlp_best_cv_mse = get_best_mlp_cv_mse(mlp_fitted)
    assert mlp_best_cv_mse >= 0

    mlp_model = get_best_mlp_model(mlp_fitted)
    assert isinstance(mlp_model, MLPRegressor)

    mlp_predictions = predict_mlp(mlp_model, X_test_scaled)
    assert isinstance(mlp_predictions, np.ndarray)
    assert len(mlp_predictions) == len(y_test)

    mlp_mae, mlp_mse, mlp_rmse, mlp_r2 = evaluate_regression(y_test, mlp_predictions)

    # ================================= MODEL COMPARISON =================================
    # Extracts the comparative performance metrics for both algorithms.
    # Checking the shape and headers guarantees the comparison table
    # preserves the correct formatting for report generation.
    comparison = compare_models(
        rf_mae, rf_mse, rf_rmse, rf_r2,
        mlp_mae, mlp_mse, mlp_rmse, mlp_r2
    )

    assert comparison.shape == (2, 5)
    assert comparison["Model"].tolist() == ["Random Forest", "MLP"]

    # Verifies the final Random Forest and MLP predictive capabilities against
    # the established performance baselines. If R² dips or RMSE spikes, 
    # the pipeline has suffered a critical regression and fails.
    assert rf_r2 > 0.70
    assert mlp_r2 > 0.65
    assert rf_rmse < 0.45
    assert mlp_rmse < 0.45


# The following system integration test checks:
# - If the complete system can execute from the raw dataset through
#   preprocessing, EDA, clustering, statistical inference, and machine
#   learning without breaking the data flow between stages.
# - If each stage produces the expected output required by the next stage.
# - If the final results remain consistent with the established
#   notebook workflow.
#
# This test will fail if:
# - Dataset loading or validation changes.
# - Data cleaning changes.
# - Composite-score calculations change.
# - EDA results change.
# - Clustering features or standardization change.
# - K-Means results change.
# - Statistical inference results change.
# - Machine-learning inputs or model results change.
# - Any upstream change causes the final system results to differ from
#   the established notebook baseline.
def test_system_pipeline():
    """
    SCT-004

    Full end-to-end system test: load -> validate -> clean ->
    feature engineer -> EDA -> cluster -> statistical inference ->
    machine learning. One `df`, one continuous flow, start to finish.
    """
    
    # ================================= 1. DATASET LOADING =================================
    # Loads the raw dataset and verifies that the system begins with the
    # same data structure used by the individual preprocessing, EDA,
    # clustering, and machine-learning stages.
    df = load_dataset(DATASET_PATH)

    # Confirms that the dataset loader returns the expected pandas DataFrame
    # required by all subsequent pipeline stages.
    assert isinstance(df, pd.DataFrame)

    # Locks in the raw dataset dimensions before any transformation occurs.
    # If respondents or variables are lost at this first stage, every
    # downstream result would be based on different data.
    assert df.shape == (757, 31)

    # Confirms that the raw variables have the expected names and order.
    # This prevents a renamed, missing, or reordered questionnaire item from
    # silently propagating through the entire system.
    assert df.columns.tolist() == EXPECTED_COLUMNS

    # Independently verifies that inspect_dataset() reports the same dimensions
    # as the dataframe itself. This confirms that the validation wrapper is
    # correctly connected to the loaded dataset.
    rows, columns = inspect_dataset(df)
    assert rows == 757
    assert columns == 31

    # Confirms that the validation function reads and returns the actual
    # dataset schema rather than using a stale or hardcoded column list.
    assert validate_columns(df) == EXPECTED_COLUMNS

    # Confirms that the raw dataset retains the expected data types.
    # Unexpected type changes can alter calculations later in the pipeline,
    # especially when computing composite scores.
    dtypes = validate_dtypes(df)
    pd.testing.assert_series_equal(dtypes, df.dtypes)
    assert dtypes.eq("int64").all()

    # Confirms that no missing values have entered the system before
    # preprocessing. This is important because later composite-score,
    # clustering, statistical, and machine-learning calculations depend
    # on complete observations.
    missing = check_missing_values(df)
    expected_missing = pd.Series([0] * len(EXPECTED_COLUMNS), index=EXPECTED_COLUMNS)
    pd.testing.assert_series_equal(missing, expected_missing)

    # Confirms that duplicate detection produces the same result for the
    # source dataset. Duplicate respondents are retained, so this verifies
    # that the system does not silently remove them.
    duplicates = find_duplicates(df)
    assert duplicates.shape == (129, 31)

    # Confirms that the categorical and Likert-scale values remain exactly
    # within the expected categories. This protects the system against
    # malformed survey responses entering downstream analysis.
    unique_summary = validate_unique_values(df)
    assert list(unique_summary.columns) == ["Variable", "Unique Count", "Unique Values"]
    assert unique_summary["Variable"].tolist() == EXPECTED_COLUMNS
    assert unique_summary["Unique Count"].tolist() == EXPECTED_UNIQUE_COUNTS
    assert unique_summary["Unique Values"].tolist() == EXPECTED_UNIQUE_VALUES

    # ================================= 2. DATA CLEANING =================================
    # Applies the same column-removal decisions established during
    # preprocessing before the dataset is passed to feature engineering.
    df = drop_columns(df, ["PEU4"])

    expected_columns = EXPECTED_COLUMNS.copy()
    expected_columns.remove("PEU4")

    # PEU4 must be removed while all 757 observations and the remaining
    # columns are preserved. Checking the shape, column order, and specific
    # column name ensures that the correct variable was removed.
    assert df.shape == (757, 30)
    assert df.columns.tolist() == expected_columns
    assert "PEU4" not in df.columns

    df = drop_columns(df, ["Job"])

    expected_columns.remove("Job")

    # Job must also be removed from the dataset before downstream analysis.
    # The checks confirm that exactly one additional column was removed and
    # that no other variables were unintentionally affected.
    assert df.shape == (757, 29)
    assert df.columns.tolist() == expected_columns
    assert "Job" not in df.columns

    # ================================= 3. FEATURE ENGINEERING =================================
    # Recreates the composite scores used throughout the rest of the system.
    # Because these constructs feed both the EDA and modeling stages, any
    # change in their calculation can propagate to the final results.
    for construct, items in COMPOSITES.items():
        df = compute_composite_score(df, items, construct)

    # Seven composite constructs are added to the cleaned dataset, increasing
    # the dataset from 29 to 36 columns. This confirms that every required
    # construct was created without removing the existing variables.
    assert df.shape == (757, 36)

    # These respondent-level checks compare the actual composite calculations
    # against established values from the notebook. Checking multiple rows,
    # including the first, middle, and last observations, helps detect
    # incorrect item selection or averaging logic.
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

    # ================================= 4. EDA -- DEMOGRAPHIC DISTRIBUTIONS =================================
    # Reproduces the demographic summaries used during exploratory analysis.
    # These checks confirm that the same respondents remain in each demographic
    # category after preprocessing.
    
    # Checks the exact respondent counts for each category. The category
    # counts must collectively account for all 757 respondents so that no
    # observations are silently lost or reassigned.
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

    # ================================= 5. EDA -- CONSTRUCT DESCRIPTIVES =================================
    # Reproduces the descriptive statistics for the seven composite constructs.
    #
    # These checks ensure that the construct-level summaries remain consistent
    # with the corrected composite scores produced earlier in the pipeline.
    descriptives = get_construct_descriptives(df)
    assert descriptives.columns.tolist() == ["PU", "PEU", "FSC", "SP", "TP", "IB", "AUB"]
    assert descriptives.loc["count"].tolist() == [757] * 7

    assert descriptives.loc["mean", "PU"] == pytest.approx(3.77, abs=0.01)
    assert descriptives.loc["mean", "PEU"] == pytest.approx(3.70, abs=0.01)
    assert descriptives.loc["mean", "FSC"] == pytest.approx(3.73, abs=0.01)
    assert descriptives.loc["mean", "SP"] == pytest.approx(3.60, abs=0.01)
    assert descriptives.loc["mean", "TP"] == pytest.approx(3.56, abs=0.01)
    assert descriptives.loc["mean", "IB"] == pytest.approx(3.61, abs=0.01)
    assert descriptives.loc["mean", "AUB"] == pytest.approx(3.68, abs=0.01)

    # ================================= 6. EDA -- CORRELATION MATRIX =================================
    # Reproduces the construct correlation matrix used in the notebook.
    #
    # The correlations are especially useful for detecting changes to one
    # composite construct because a change in that construct can alter every
    # correlation involving it.
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

    # ================================= 7. EDA -- STATISTICAL TESTS =================================
    # Reproduces the inferential EDA results comparing AUB across demographic
    # groups.
    #
    # Both the numerical test results and their statistical conclusions are
    # checked so that a change in the underlying analysis cannot silently
    # alter the reported interpretation.
    t_stat, p_value = run_ttest_aub_gender(df)
    assert t_stat == pytest.approx(-0.417, abs=0.01)
    assert p_value == pytest.approx(0.677, abs=0.01)
    assert p_value > 0.05

    f_stat, p_value = run_anova_aub_area(df)
    assert f_stat == pytest.approx(1.4681, abs=0.01)
    assert p_value == pytest.approx(0.2310, abs=0.01)
    assert p_value > 0.05

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

    # ================================= 8. CLUSTERING -- FEATURE SELECTION + SCALING =================================
    # Passes the already-prepared composite constructs into the clustering
    # stage.
    #
    # These checks confirm that the system uses exactly the same six behavioral
    # constructs and the same standardization procedure established in the
    # dedicated clustering pipeline.
    result = select_features(df, FEATURES)
    assert isinstance(result, pd.DataFrame)
    assert result.shape == (757, 6)
    assert result.columns.tolist() == FEATURES
    pd.testing.assert_frame_equal(result, df[FEATURES])

    X = select_features(df, FEATURES)
    result = standardize_features(X)

    assert isinstance(result, np.ndarray)
    assert result.shape == (757, 6)
    assert not np.isnan(result).any()
    assert np.isfinite(result).all()
    assert np.allclose(result.mean(axis=0), 0, atol=1e-10)
    assert np.allclose(result.std(axis=0), 1, atol=1e-10)

    expected = StandardScaler().fit_transform(X)
    np.testing.assert_allclose(result, expected, rtol=1e-10, atol=1e-10)

    X_scaled = standardize_features(X)

    # ================================= 9. CLUSTERING -- MULTICOLLINEARITY (VIF) =================================
    # Reproduces the VIF analysis within the full system flow.
    #
    # Checking the established VIF values ensures that the same standardized
    # feature relationships are reaching the clustering analysis.
    result = calculate_vif(X_scaled, FEATURES)
    assert isinstance(result, pd.DataFrame)
    assert result.columns.tolist() == ["Construct", "VIF"]
    assert result.shape == (6, 2)
    assert result["Construct"].tolist() == FEATURES
    assert pd.api.types.is_numeric_dtype(result["VIF"])
    assert np.isfinite(result["VIF"]).all()

    expected_vif = [3.654939, 3.515319, 3.988501, 3.150629, 2.709242, 2.301402]
    np.testing.assert_allclose(result["VIF"].values, expected_vif, rtol=1e-5, atol=1e-5)

    # ================================= 10. CLUSTERING -- K SELECTION =================================
    # Reproduces the K-Means model-selection stage within the complete system.
    #
    # Both the model behavior and the established notebook values are checked
    # so that an upstream change cannot silently alter the candidate-cluster
    # evaluation.
    k_values = range(2, 11)
    inertia, silhouette_scores = evaluate_kmeans_clusters(X_scaled, k_values)

    assert isinstance(inertia, list)
    assert isinstance(silhouette_scores, list)
    assert len(inertia) == 9
    assert len(silhouette_scores) == 9
    assert np.isfinite(inertia).all()
    assert np.isfinite(silhouette_scores).all()
    assert all(value > 0 for value in inertia)
    assert all(-1 <= value <= 1 for value in silhouette_scores)
    assert all(inertia[i] > inertia[i + 1] for i in range(len(inertia) - 1))

    expected_inertia = [
        2397.3951517810065, 1876.6780711816932, 1470.371175770248,
        1289.3260086578698, 1194.8825839212625, 1138.661553747194,
        1082.160274964026, 1041.3118953265096, 995.2666652249214,
    ]
    expected_silhouette_scores = [
        0.4357624529557227, 0.42533701123564177, 0.37747256266114426,
        0.33290308059903284, 0.2913130488967038, 0.29138177791986325,
        0.30602460012945765, 0.2930900543684983, 0.29391001252620486,
    ]

    np.testing.assert_allclose(inertia, expected_inertia, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(silhouette_scores, expected_silhouette_scores, rtol=1e-5, atol=1e-5)

    # ================================= 11. CLUSTERING -- FINAL K-MEANS =================================
    # Performs the final four-cluster solution used throughout the remainder
    # of the system.
    #
    # The checks confirm that every respondent receives a valid cluster
    # assignment and that all four expected clusters are present.
    model, clusters = perform_kmeans(X_scaled)

    assert isinstance(model, KMeans)
    assert model.n_clusters == 4
    assert len(clusters) == 757
    assert len(np.unique(clusters)) == 4
    assert set(np.unique(clusters)) == {0, 1, 2, 3}
    assert model.cluster_centers_.shape == (4, 6)
    assert hasattr(model, "inertia_")
    assert np.isfinite(model.cluster_centers_).all()
    assert np.isfinite(clusters).all()

    # Confirms that the final cluster assignments reproduce the established
    # respondent distribution across clusters. The total is also checked to
    # ensure that all 757 observations are accounted for.
    result = get_cluster_sizes(clusters)
    assert isinstance(result, pd.Series)
    assert result.index.tolist() == [0, 1, 2, 3]
    assert result.tolist() == [97, 258, 20, 382]
    assert result.name == "count"
    assert result.sum() == 757
    assert (result > 0).all()

    df["Cluster"] = clusters

    # ================================= 12. CLUSTERING -- CLUSTER PROFILES =================================
    # Summarizes the behavioral constructs and AUB within each final cluster.
    #
    # These values are used to interpret the characteristics of the clusters,
    # so reproducing the notebook profile values confirms that the same
    # respondents were assigned to the same behavioral groups.
    result = calculate_cluster_profiles(df, FEATURES)
    assert isinstance(result, pd.DataFrame)
    assert result.shape == (4, 7)
    assert result.columns.tolist() == (FEATURES + ["AUB"])
    assert result.index.tolist() == [0, 1, 2, 3]
    assert np.isfinite(result.values).all()
    assert (result[FEATURES] >= 1).all().all()
    assert (result[FEATURES] <= 5).all().all()
    assert (result["AUB"] >= 1).all()
    assert (result["AUB"] <= 5).all()

    # The expected profile matrix comes from the established notebook output.
    # Comparing every value ensures that the complete cluster composition,
    # rather than only the cluster sizes, remains unchanged.
    expected_profiles = pd.DataFrame(
        [
            [4.6391752577, 4.5841924399, 4.7079037801, 4.5773195876, 4.4707903780, 4.5257731959, 4.4664948454],
            [3.2451550388, 3.1550387597, 3.1020671835, 3.0474806202, 3.0348837209, 3.1715116279, 3.1715116279],
            [1.6500000000, 1.5666666667, 1.5833333333, 1.7750000000, 1.4666666667, 1.6625000000, 1.7125000000],
            [4.0215968586, 3.9554973822, 4.0200698080, 3.8239528796, 3.7870855148, 3.7676701571, 3.9201570681],
        ],
        index=[0, 1, 2, 3],
        columns=FEATURES + ["AUB"],
    )

    np.testing.assert_allclose(result.values, expected_profiles.values, rtol=1e-5, atol=1e-5)

    # ================================= 13. CLUSTERING -- STATISTICAL INFERENCE =================================
    # Reproduces the statistical analysis performed after the final clusters
    # have been created.
    #
    # Because these tests depend directly on cluster membership and AUB,
    # changes to the clustering stage can propagate into these results.
    # Checking the established statistics therefore verifies the complete
    # downstream analytical chain.

    # The Shapiro-Wilk results are checked for every cluster to confirm that
    # the normality analysis receives the same AUB observations as the notebook.
    result = perform_shapiro_test(df)
    assert isinstance(result, list)
    assert len(result) == 4
    assert [item[0] for item in result] == [0, 1, 2, 3]

    for cluster, statistic, p_value in result:
        assert isinstance(cluster, (int, np.integer))
        assert isinstance(statistic, (float, np.floating))
        assert isinstance(p_value, (float, np.floating))
        assert 0 <= statistic <= 1
        assert 0 <= p_value <= 1

    expected_statistics = [0.8871, 0.8512, 0.8787, 0.8786]
    expected_p_values = [0.0000, 0.0000, 0.0168, 0.0000]

    for (_, statistic, p_value), expected_statistic, expected_p_value in zip(
        result, expected_statistics, expected_p_values
    ):
        assert statistic == pytest.approx(expected_statistic, abs=0.0001)
        assert p_value == pytest.approx(expected_p_value, abs=0.0001)

    # The Kruskal-Wallis statistic and p-value are checked against the notebook
    # to confirm that the overall conclusion about differences in AUB across
    # clusters remains unchanged.
    H, p = perform_kruskal_wallis(df)
    assert isinstance(H, (float, np.floating))
    assert isinstance(p, (float, np.floating))
    assert H >= 0
    assert 0 <= p <= 1
    assert H == pytest.approx(421.3159298947122, rel=1e-10)
    assert p == pytest.approx(5.341699041631374e-91, rel=1e-10)
    assert p < 0.05

    # Dunn's post-hoc results are checked as a complete pairwise matrix.
    # This verifies not only that the overall Kruskal-Wallis conclusion remains
    # significant, but also that the specific cluster-to-cluster differences
    # remain consistent with the notebook.
    result = perform_dunn_test(df)
    assert isinstance(result, pd.DataFrame)
    assert result.shape == (4, 4)
    assert result.index.tolist() == [0, 1, 2, 3]
    assert result.columns.tolist() == [0, 1, 2, 3]
    assert np.isfinite(result.values).all()
    assert (result.values >= 0).all()
    assert (result.values <= 1).all()
    assert np.allclose(np.diag(result), 1)
    np.testing.assert_allclose(result.values, result.values.T)

    expected_dunn = pd.DataFrame(
        [
            [1.0000, 0.0000, 0.0000, 0.0000],
            [0.0000, 1.0000, 0.0022, 0.0000],
            [0.0000, 0.0022, 1.0000, 0.0000],
            [0.0000, 0.0000, 0.0000, 1.0000],
        ],
        index=[0, 1, 2, 3],
        columns=[0, 1, 2, 3],
    )

    np.testing.assert_allclose(result.round(4).values, expected_dunn.values, rtol=1e-5, atol=1e-5)

    assert result.loc[0, 1] < 0.05
    assert result.loc[0, 2] < 0.05
    assert result.loc[0, 3] < 0.05
    assert result.loc[1, 3] < 0.05
    assert result.loc[2, 3] < 0.05

    # ================================= 14. MACHINE LEARNING -- FEATURE & TARGET SELECTION =================================
    # Passes the completed dataset into the machine-learning stage.
    #
    # The six behavioral constructs are used as predictors and AUB is used
    # as the prediction target, matching the modeling workflow established
    # in the notebook.
    X = ml_select_features(df, FEATURES)
    y = select_target(df, "AUB")

    assert isinstance(X, pd.DataFrame)
    assert X.shape == (757, 6)
    assert X.columns.tolist() == FEATURES

    assert isinstance(y, pd.Series)
    assert len(y) == 757
    assert y.name == "AUB"

    # ================================= 15. MACHINE LEARNING -- SPLIT + SCALE =================================
    # Partitions the dataset into training and testing sets using the same
    # test size and random state as the notebook.
    #
    # The fixed dimensions ensure that the same number of observations and
    # predictors reach model training and evaluation.
    X_train, X_test, y_train, y_test = split_dataset(X, y, test_size=0.20, random_state=1)

    assert X_train.shape == (605, 6)
    assert X_test.shape == (152, 6)
    assert len(y_train) == 605
    assert len(y_test) == 152

    # Standardizes the training and testing predictors using the machine-
    # learning preprocessing procedure. The training statistics must be used
    # consistently so that information from the test set does not influence
    # model training.
    scaler, X_train_scaled, X_test_scaled = ml_standardize_features(X_train, X_test)

    assert X_train_scaled.shape == X_train.shape
    assert X_test_scaled.shape == X_test.shape
    assert np.allclose(X_train_scaled.mean(axis=0), 0, atol=1e-7)
    assert np.allclose(X_train_scaled.std(axis=0), 1, atol=1e-7)

    # Creates the same 10-fold cross-validation configuration used during
    # hyperparameter tuning. Keeping the number of folds and shuffle behavior
    # consistent ensures comparable model-selection results.
    kf = create_kfold(n_splits=10, shuffle=True, random_state=1)
    assert kf.get_n_splits() == 10
    assert kf.shuffle is True

    # ================================= 16. MACHINE LEARNING -- RANDOM FOREST =================================
    # Reproduces the complete Random Forest model-selection and evaluation
    # workflow used in the notebook.
    rf_param_grid = create_rf_param_grid()
    assert "n_estimators" in rf_param_grid
    assert "max_depth" in rf_param_grid

    # Confirms that the expected Random Forest hyperparameters are available
    # for grid-search tuning.
    rf_grid = create_rf_gridsearch(rf_param_grid, kf, n_jobs=1)
    assert isinstance(rf_grid, GridSearchCV)
    assert isinstance(rf_grid.estimator, RandomForestRegressor)

    # Confirms that the wrapper constructs the expected sklearn GridSearchCV
    # object using a Random Forest regressor.
    rf_fitted = fit_rf_gridsearch(rf_grid, X_train, y_train)
    assert hasattr(rf_fitted, "best_estimator_")
    assert hasattr(rf_fitted, "best_params_")

    # Confirms that hyperparameter search successfully completed and produced
    # both a best estimator and best parameter configuration.
    rf_best_params, rf_best_mse = get_best_rf_params(rf_fitted)
    assert isinstance(rf_best_params, dict)
    assert rf_best_mse >= 0

    rf_summary = summarize_rf_results(rf_fitted)
    assert isinstance(rf_summary, pd.DataFrame)
    assert "rank_test_score" in rf_summary.columns

    rf_model = get_best_rf_model(rf_fitted)
    assert isinstance(rf_model, RandomForestRegressor)

    # Generates predictions for the held-out test set. The prediction count
    # must match the number of test observations so that regression metrics
    # are calculated on corresponding actual and predicted values.
    rf_predictions = predict_rf(rf_model, X_test)
    assert isinstance(rf_predictions, np.ndarray)
    assert len(rf_predictions) == len(y_test)

    rf_mae, rf_mse, rf_rmse, rf_r2 = evaluate_regression(y_test, rf_predictions)

    # ================================= 17. MACHINE LEARNING -- MLP =================================
    # Reproduces the complete MLP model-selection and evaluation workflow
    # used in the notebook.
    mlp_param_grid = create_mlp_param_grid()
    assert "hidden_layer_sizes" in mlp_param_grid
    assert "learning_rate_init" in mlp_param_grid

    # Confirms that the expected MLP hyperparameters are available for
    # neural-network tuning.
    mlp_grid = create_mlp_gridsearch(mlp_param_grid, kf, n_jobs=1)
    assert isinstance(mlp_grid, GridSearchCV)
    assert isinstance(mlp_grid.estimator, MLPRegressor)

    # Confirms that the MLP grid search successfully completed and produced
    # a fitted best estimator and best parameter configuration.
    mlp_fitted = fit_mlp_gridsearch(mlp_grid, X_train_scaled, y_train)
    assert hasattr(mlp_fitted, "best_estimator_")
    assert hasattr(mlp_fitted, "best_params_")

    mlp_best_cv_mse = get_best_mlp_cv_mse(mlp_fitted)
    assert mlp_best_cv_mse >= 0

    mlp_model = get_best_mlp_model(mlp_fitted)
    assert isinstance(mlp_model, MLPRegressor)

    # Generates MLP predictions for the held-out test observations.
    # The prediction length must match y_test so that the resulting
    # regression metrics are valid.
    mlp_predictions = predict_mlp(mlp_model, X_test_scaled)
    assert isinstance(mlp_predictions, np.ndarray)
    assert len(mlp_predictions) == len(y_test)

    mlp_mae, mlp_mse, mlp_rmse, mlp_r2 = evaluate_regression(y_test, mlp_predictions)

    # ================================= 18. MACHINE LEARNING -- MODEL COMPARISON =================================
    # Compares the final Random Forest and MLP regression results using the
    # same evaluation metrics reported in the notebook.
    comparison = compare_models(
        rf_mae, rf_mse, rf_rmse, rf_r2,
        mlp_mae, mlp_mse, mlp_rmse, mlp_r2,
    )

    # The comparison table should contain exactly two models and the four
    # regression metrics used by the project. This ensures that the final
    # model-comparison output retains the expected structure.
    assert isinstance(comparison, pd.DataFrame)
    assert comparison.shape == (2, 5)
    assert comparison["Model"].tolist() == ["Random Forest", "MLP"]
    assert comparison["MAE"].notna().all()
    assert comparison["MSE"].notna().all()
    assert comparison["RMSE"].notna().all()
    assert comparison["R²"].notna().all()

    # Confirms that both models maintain the established minimum performance
    # baseline. These checks are intentionally based on model performance
    # thresholds rather than exact floating-point values because machine-
    # learning optimization can produce small numerical differences while
    # still satisfying the established performance requirement.
    assert rf_r2 > 0.70
    assert mlp_r2 > 0.65
    assert rf_rmse < 0.45
    assert mlp_rmse < 0.45
    assert rf_mae >= 0
    assert mlp_mae >= 0