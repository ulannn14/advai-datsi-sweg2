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
    get_construct_descriptives,
    get_construct_correlation_matrix,
    get_aub_by_gender_summary,
    run_ttest_aub_gender,
    get_aub_by_area_summary,
    run_anova_aub_area,
    get_aub_by_frequency_summary,
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

def test_clustering_pipeline():
    """
    SCT-002

    End-to-end integration test for the complete
    preprocessing and EDA pipeline.
    """

    # =====================================================
    # DATA PREPROCESSING
    # =====================================================

    df = load_dataset(DATASET_PATH)

    df = drop_columns(
        df,
        ["Job"]
    )

    df = drop_columns(
        df,
        ["PEU4"]
    )

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
    
    # =====================================================
    # CLUSTERING
    # =====================================================

    result = select_features(
        df,
        FEATURES
    )

    assert isinstance(
        result,
        pd.DataFrame
    )

    assert result.shape == (
        757,
        6
    )

    assert result.columns.tolist() == FEATURES

    assert len(result.columns) == len(FEATURES)

    pd.testing.assert_frame_equal(
        result,
        df[FEATURES]
    )

    X = select_features(
        df,
        FEATURES
    )

    result = standardize_features(
        X
    )

    assert isinstance(
        result,
        np.ndarray
    )

    assert result.shape == (
        757,
        6
    )

    assert not np.isnan(
        result
    ).any()

    assert np.isfinite(
        result
    ).all()

    assert np.allclose(
        result.mean(axis=0),
        0,
        atol=1e-10
    )

    assert np.allclose(
        result.std(axis=0),
        1,
        atol=1e-10
    )

    expected = StandardScaler().fit_transform(
        X
    )

    np.testing.assert_allclose(
        result,
        expected,
        rtol=1e-10,
        atol=1e-10
    )

    X_scaled = standardize_features(
        X
    )

    result = calculate_vif(
        X_scaled,
        FEATURES
    )

    assert isinstance(
        result,
        pd.DataFrame
    )

    assert result.columns.tolist() == [
        "Construct",
        "VIF"
    ]

    assert result.shape == (
        6,
        2
    )

    assert result["Construct"].tolist() == FEATURES

    assert pd.api.types.is_numeric_dtype(
        result["VIF"]
    )

    assert np.isfinite(
        result["VIF"]
    ).all()

    expected_vif = [
        3.654939,
        3.515319,
        3.988501,
        3.150629,
        2.709242,
        2.301402,
    ]

    np.testing.assert_allclose(
        result["VIF"].values,
        expected_vif,
        rtol=1e-5,
        atol=1e-5
    )

    k_values = range(
        2,
        11
    )

    inertia, silhouette_scores = (
        evaluate_kmeans_clusters(
            X_scaled,
            k_values
        )
    )

    assert isinstance(
        inertia,
        list
    )

    assert isinstance(
        silhouette_scores,
        list
    )

    assert len(inertia) == 9
    assert len(silhouette_scores) == 9

    assert np.isfinite(
        inertia
    ).all()

    assert np.isfinite(
        silhouette_scores
    ).all()

    assert all(
        value > 0
        for value in inertia
    )

    assert all(
        -1 <= value <= 1
        for value in silhouette_scores
    )

    assert all(
        inertia[i] > inertia[i + 1]
        for i in range(
            len(inertia) - 1
        )
    )

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

    np.testing.assert_allclose(
        inertia,
        expected_inertia,
        rtol=1e-5,
        atol=1e-5
    )

    np.testing.assert_allclose(
        silhouette_scores,
        expected_silhouette_scores,
        rtol=1e-5,
        atol=1e-5
    )

    model, clusters = perform_kmeans(
        X_scaled
    )

    assert isinstance(
        model,
        KMeans
    )

    assert model.n_clusters == 4

    assert len(clusters) == 757

    assert len(
        np.unique(clusters)
    ) == 4

    assert set(
        np.unique(clusters)
    ) == {
        0,
        1,
        2,
        3
    }

    assert model.cluster_centers_.shape == (
        4,
        6
    )

    assert hasattr(
        model,
        "inertia_"
    )

    assert np.isfinite(
        model.cluster_centers_
    ).all()

    assert np.isfinite(
        clusters
    ).all()

    result = get_cluster_sizes(
        clusters
    )

    assert isinstance(
        result,
        pd.Series
    )

    assert result.index.tolist() == [
        0,
        1,
        2,
        3
    ]

    assert result.tolist() == [
        97,
        258,
        20,
        382
    ]

    assert result.name == "count"

    assert result.sum() == 757

    assert (
        result > 0
    ).all()

    df["Cluster"] = clusters

    result = calculate_cluster_profiles(
        df,
        FEATURES
    )

    assert isinstance(
        result,
        pd.DataFrame
    )

    assert result.shape == (
        4,
        7
    )

    assert result.columns.tolist() == (
        FEATURES + ["AUB"]
    )

    assert result.index.tolist() == [
        0,
        1,
        2,
        3
    ]

    assert np.isfinite(
        result.values
    ).all()

    assert (
        result[FEATURES] >= 1
    ).all().all()

    assert (
        result[FEATURES] <= 5
    ).all().all()

    assert (
        result["AUB"] >= 1
    ).all()

    assert (
        result["AUB"] <= 5
    ).all()

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

    np.testing.assert_allclose(
        result.values,
        expected_profiles.values,
        rtol=1e-5,
        atol=1e-5
    )
    
    # =====================================================
    # STATISTICAL INFERENCE
    # =====================================================

    result = perform_shapiro_test(
        df
    )

    assert isinstance(
        result,
        list
    )

    assert len(result) == 4

    assert [
        item[0]
        for item in result
    ] == [
        0,
        1,
        2,
        3
    ]

    for cluster, statistic, p_value in result:

        assert isinstance(
            cluster,
            (int, np.integer)
        )

        assert isinstance(
            statistic,
            (float, np.floating)
        )

        assert isinstance(
            p_value,
            (float, np.floating)
        )

        assert 0 <= statistic <= 1

        assert 0 <= p_value <= 1

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

    for (
        (_, statistic, p_value),
        expected_statistic,
        expected_p_value
    ) in zip(
        result,
        expected_statistics,
        expected_p_values
    ):

        assert statistic == pytest.approx(
            expected_statistic,
            abs=0.0001
        )

        assert p_value == pytest.approx(
            expected_p_value,
            abs=0.0001
        )

    H, p = perform_kruskal_wallis(
        df
    )

    assert isinstance(
        H,
        (float, np.floating)
    )

    assert isinstance(
        p,
        (float, np.floating)
    )

    assert H >= 0
    assert 0 <= p <= 1

    assert H == pytest.approx(
        421.3159298947122,
        rel=1e-10
    )

    assert p == pytest.approx(
        5.341699041631374e-91,
        rel=1e-10
    )

    assert p < 0.05

    result = perform_dunn_test(
        df
    )

    assert isinstance(
        result,
        pd.DataFrame
    )

    assert result.shape == (
        4,
        4
    )

    assert result.index.tolist() == [
        0,
        1,
        2,
        3
    ]

    assert result.columns.tolist() == [
        0,
        1,
        2,
        3
    ]

    assert np.isfinite(
        result.values
    ).all()

    assert (
        result.values >= 0
    ).all()

    assert (
        result.values <= 1
    ).all()

    assert np.allclose(
        np.diag(result),
        1
    )

    np.testing.assert_allclose(
        result.values,
        result.values.T
    )

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

    np.testing.assert_allclose(
        result.round(4).values,
        expected_dunn.values,
        rtol=1e-5,
        atol=1e-5
    )

    assert result.loc[0, 1] < 0.05
    assert result.loc[0, 2] < 0.05
    assert result.loc[0, 3] < 0.05
    assert result.loc[1, 3] < 0.05
    assert result.loc[2, 3] < 0.05

def test_machine_learning_pipeline():
    """
    SCT-003
    Extensive end-to-end integration test for the machine 
    learning predictive modeling pipeline (Random Forest and MLP).
    """

    # =====================================================
    # PREPROCESSING
    # =====================================================
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

    # =====================================================
    # FEATURE & TARGET SELECTION
    # =====================================================
    X = ml_select_features(df, FEATURES)
    y = select_target(df, "AUB")

    assert isinstance(X, pd.DataFrame)
    assert X.shape == (757, 6)
    assert X.columns.tolist() == FEATURES
    
    assert isinstance(y, pd.Series)
    assert len(y) == 757
    assert y.name == "AUB"

    # =====================================================
    # DATASET PARTITIONING & SCALING
    # =====================================================
    X_train, X_test, y_train, y_test = split_dataset(
        X, y, test_size=0.20, random_state=1
    )

    assert X_train.shape == (605, 6)
    assert X_test.shape == (152, 6)
    assert len(y_train) == 605
    assert len(y_test) == 152

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

    # =====================================================
    # RANDOM FOREST
    # =====================================================
    rf_param_grid = create_rf_param_grid()
    assert "n_estimators" in rf_param_grid
    assert "max_depth" in rf_param_grid

    rf_grid = create_rf_gridsearch(rf_param_grid, kf, n_jobs=1)
    assert isinstance(rf_grid, GridSearchCV)
    assert isinstance(rf_grid.estimator, RandomForestRegressor)

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

    rf_predictions = predict_rf(rf_model, X_test)
    assert isinstance(rf_predictions, np.ndarray)
    assert len(rf_predictions) == len(y_test)

    rf_mae, rf_mse, rf_rmse, rf_r2 = evaluate_regression(y_test, rf_predictions)

    # =====================================================
    # MLP
    # =====================================================
    mlp_param_grid = create_mlp_param_grid()
    assert "hidden_layer_sizes" in mlp_param_grid
    assert "learning_rate_init" in mlp_param_grid

    mlp_grid = create_mlp_gridsearch(mlp_param_grid, kf, n_jobs=1)
    assert isinstance(mlp_grid, GridSearchCV)
    assert isinstance(mlp_grid.estimator, MLPRegressor)

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

    # =====================================================
    # MODEL COMPARISON
    # =====================================================
    comparison = compare_models(
        rf_mae, rf_mse, rf_rmse, rf_r2,
        mlp_mae, mlp_mse, mlp_rmse, mlp_r2
    )

    assert isinstance(comparison, pd.DataFrame)
    assert comparison.shape == (2, 5)
    assert comparison["Model"].tolist() == ["Random Forest", "MLP"]
    assert comparison["MAE"].notna().all()
    assert comparison["MSE"].notna().all()
    assert comparison["RMSE"].notna().all()
    assert comparison["R²"].notna().all()

    # Verify models maintain the specific established performance baseline
    assert rf_r2 > 0.70
    assert mlp_r2 > 0.65
    assert rf_rmse < 0.45
    assert mlp_rmse < 0.45
    assert rf_mae >= 0
    assert mlp_mae >= 0

def test_system_pipeline():
    """

    Full end-to-end system test: load -> validate -> clean ->
    feature engineer -> EDA -> cluster -> statistical inference ->
    machine learning. One `df`, one continuous flow, start to finish.
    """
    # 1. DATASET LOADING
    df = load_dataset(DATASET_PATH)

    assert isinstance(df, pd.DataFrame)
    assert df.shape == (757, 31)
    assert df.columns.tolist() == EXPECTED_COLUMNS

    rows, columns = inspect_dataset(df)
    assert rows == 757
    assert columns == 31

    assert validate_columns(df) == EXPECTED_COLUMNS

    dtypes = validate_dtypes(df)
    pd.testing.assert_series_equal(dtypes, df.dtypes)
    assert dtypes.eq("int64").all()

    missing = check_missing_values(df)
    expected_missing = pd.Series([0] * len(EXPECTED_COLUMNS), index=EXPECTED_COLUMNS)
    pd.testing.assert_series_equal(missing, expected_missing)

    duplicates = find_duplicates(df)
    assert duplicates.shape == (129, 31)

    unique_summary = validate_unique_values(df)
    assert list(unique_summary.columns) == ["Variable", "Unique Count", "Unique Values"]
    assert unique_summary["Variable"].tolist() == EXPECTED_COLUMNS
    assert unique_summary["Unique Count"].tolist() == EXPECTED_UNIQUE_COUNTS
    assert unique_summary["Unique Values"].tolist() == EXPECTED_UNIQUE_VALUES

    # 2. DATA CLEANING
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

    # 3. FEATURE ENGINEERING
    for construct, items in COMPOSITES.items():
        df = compute_composite_score(df, items, construct)

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

    # 4. EDA -- DEMOGRAPHIC DISTRIBUTIONS
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

    # 5. EDA -- CONSTRUCT DESCRIPTIVES
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

    # 6. EDA -- CORRELATION MATRIX
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

    # 7. EDA -- T-TEST / ANOVA / GROUP SUMMARIES
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

    # 8. CLUSTERING -- FEATURE SELECTION + SCALING
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

    # 9. CLUSTERING -- MULTICOLLINEARITY (VIF)
    result = calculate_vif(X_scaled, FEATURES)
    assert isinstance(result, pd.DataFrame)
    assert result.columns.tolist() == ["Construct", "VIF"]
    assert result.shape == (6, 2)
    assert result["Construct"].tolist() == FEATURES
    assert pd.api.types.is_numeric_dtype(result["VIF"])
    assert np.isfinite(result["VIF"]).all()

    expected_vif = [3.654939, 3.515319, 3.988501, 3.150629, 2.709242, 2.301402]
    np.testing.assert_allclose(result["VIF"].values, expected_vif, rtol=1e-5, atol=1e-5)

    # 10. CLUSTERING -- K SELECTION
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

    # 11. CLUSTERING -- K-MEANS
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

    result = get_cluster_sizes(clusters)
    assert isinstance(result, pd.Series)
    assert result.index.tolist() == [0, 1, 2, 3]
    assert result.tolist() == [97, 258, 20, 382]
    assert result.name == "count"
    assert result.sum() == 757
    assert (result > 0).all()

    df["Cluster"] = clusters

    # 12. CLUSTERING -- CLUSTER PROFILES
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

    # 13. CLUSTERING -- STATISTICAL INFERENCE ACROSS CLUSTERS
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

    H, p = perform_kruskal_wallis(df)
    assert isinstance(H, (float, np.floating))
    assert isinstance(p, (float, np.floating))
    assert H >= 0
    assert 0 <= p <= 1
    assert H == pytest.approx(421.3159298947122, rel=1e-10)
    assert p == pytest.approx(5.341699041631374e-91, rel=1e-10)
    assert p < 0.05

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

    # 14. MACHINE LEARNING -- FEATURE & TARGET SELECTION
    X = ml_select_features(df, FEATURES)
    y = select_target(df, "AUB")

    assert isinstance(X, pd.DataFrame)
    assert X.shape == (757, 6)
    assert X.columns.tolist() == FEATURES

    assert isinstance(y, pd.Series)
    assert len(y) == 757
    assert y.name == "AUB"

    # 15. MACHINE LEARNING -- SPLIT + SCALE
    X_train, X_test, y_train, y_test = split_dataset(X, y, test_size=0.20, random_state=1)

    assert X_train.shape == (605, 6)
    assert X_test.shape == (152, 6)
    assert len(y_train) == 605
    assert len(y_test) == 152

    scaler, X_train_scaled, X_test_scaled = ml_standardize_features(X_train, X_test)

    assert X_train_scaled.shape == X_train.shape
    assert X_test_scaled.shape == X_test.shape
    assert np.allclose(X_train_scaled.mean(axis=0), 0, atol=1e-7)
    assert np.allclose(X_train_scaled.std(axis=0), 1, atol=1e-7)

    kf = create_kfold(n_splits=10, shuffle=True, random_state=1)
    assert kf.get_n_splits() == 10
    assert kf.shuffle is True

    # 16. MACHINE LEARNING -- RANDOM FOREST
    rf_param_grid = create_rf_param_grid()
    assert "n_estimators" in rf_param_grid
    assert "max_depth" in rf_param_grid

    rf_grid = create_rf_gridsearch(rf_param_grid, kf, n_jobs=1)
    assert isinstance(rf_grid, GridSearchCV)
    assert isinstance(rf_grid.estimator, RandomForestRegressor)

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

    rf_predictions = predict_rf(rf_model, X_test)
    assert isinstance(rf_predictions, np.ndarray)
    assert len(rf_predictions) == len(y_test)

    rf_mae, rf_mse, rf_rmse, rf_r2 = evaluate_regression(y_test, rf_predictions)

    # 17. MACHINE LEARNING -- MLP
    mlp_param_grid = create_mlp_param_grid()
    assert "hidden_layer_sizes" in mlp_param_grid
    assert "learning_rate_init" in mlp_param_grid

    mlp_grid = create_mlp_gridsearch(mlp_param_grid, kf, n_jobs=1)
    assert isinstance(mlp_grid, GridSearchCV)
    assert isinstance(mlp_grid.estimator, MLPRegressor)

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

    # 18. MACHINE LEARNING -- MODEL COMPARISON
    comparison = compare_models(
        rf_mae, rf_mse, rf_rmse, rf_r2,
        mlp_mae, mlp_mse, mlp_rmse, mlp_r2,
    )

    assert isinstance(comparison, pd.DataFrame)
    assert comparison.shape == (2, 5)
    assert comparison["Model"].tolist() == ["Random Forest", "MLP"]
    assert comparison["MAE"].notna().all()
    assert comparison["MSE"].notna().all()
    assert comparison["RMSE"].notna().all()
    assert comparison["R²"].notna().all()

    assert rf_r2 > 0.70
    assert mlp_r2 > 0.65
    assert rf_rmse < 0.45
    assert mlp_rmse < 0.45
    assert rf_mae >= 0
    assert mlp_mae >= 0