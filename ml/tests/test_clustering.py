import numpy as np
import pandas as pd
import pytest

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from ml.scripts.preprocessing import (
    load_dataset,
    drop_columns,
    compute_composite_score,
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


DATASET_PATH = (
    "ml/datasets/"
    "S-COMMERCE_GenZ_UniversityStudent_757_DIB.csv"
)


FEATURES = [
    "PU",
    "PEU",
    "FSC",
    "SP",
    "TP",
    "IB",
]


# ==========================================================
# TEST DATA PREPARATION
# ==========================================================

def prepare_dataset():
    """
    Recreate the Phase 1 preprocessing required before
    executing the Phase 2 clustering pipeline.
    """

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

    return df


# ==========================================================
# MODULE 5 - FEATURE SELECTION
# ==========================================================

def test_select_features():
    """SCA-UT-009"""

    df = prepare_dataset()

    result = select_features(
        df,
        FEATURES
    )

    # Verify return type.
    assert isinstance(
        result,
        pd.DataFrame
    )

    # Verify dimensions.
    assert result.shape == (
        757,
        6
    )

    # Verify selected feature names and order.
    assert result.columns.tolist() == FEATURES

    # Verify only the requested features are returned.
    assert len(result.columns) == len(FEATURES)

    # Verify returned values match the source dataframe.
    pd.testing.assert_frame_equal(
        result,
        df[FEATURES]
    )


# ==========================================================
# MODULE 6 - FEATURE STANDARDIZATION
# ==========================================================

def test_standardize_features():
    """SCA-UT-010"""

    df = prepare_dataset()

    X = select_features(
        df,
        FEATURES
    )

    result = standardize_features(
        X
    )

    # Verify return type.
    assert isinstance(
        result,
        np.ndarray
    )

    # Verify dimensions.
    assert result.shape == (
        757,
        6
    )

    # Verify no missing values were produced.
    assert not np.isnan(
        result
    ).any()

    # Verify no infinite values were produced.
    assert np.isfinite(
        result
    ).all()

    # StandardScaler should produce approximately
    # zero mean for every feature.
    assert np.allclose(
        result.mean(axis=0),
        0,
        atol=1e-10
    )

    # StandardScaler uses population standard deviation.
    assert np.allclose(
        result.std(axis=0),
        1,
        atol=1e-10
    )

    # Verify the wrapper produces the same result
    # as the sklearn StandardScaler used in the wrapper.
    expected = StandardScaler().fit_transform(
        X
    )

    np.testing.assert_allclose(
        result,
        expected,
        rtol=1e-10,
        atol=1e-10
    )


# ==========================================================
# MODULE 7 - MULTICOLLINEARITY ANALYSIS
# ==========================================================

def test_calculate_vif():
    """SCA-UT-011"""

    df = prepare_dataset()

    X = select_features(
        df,
        FEATURES
    )

    X_scaled = standardize_features(
        X
    )

    result = calculate_vif(
        X_scaled,
        FEATURES
    )

    # Verify return type.
    assert isinstance(
        result,
        pd.DataFrame
    )

    # Verify output columns.
    assert result.columns.tolist() == [
        "Construct",
        "VIF"
    ]

    # Verify output dimensions.
    assert result.shape == (
        6,
        2
    )

    # Verify feature names and order.
    assert result["Construct"].tolist() == FEATURES

    # Verify VIF values are numeric.
    assert pd.api.types.is_numeric_dtype(
        result["VIF"]
    )

    # Verify VIF values are finite.
    assert np.isfinite(
        result["VIF"]
    ).all()

    # Verify the VIF results from the notebook.
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


# ==========================================================
# MODULE 8 - OPTIMAL K SELECTION
# ==========================================================

def test_evaluate_kmeans_clusters():
    """SCA-UT-012"""

    df = prepare_dataset()

    X = select_features(
        df,
        FEATURES
    )

    X_scaled = standardize_features(
        X
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

    # Verify return types.
    assert isinstance(
        inertia,
        list
    )

    assert isinstance(
        silhouette_scores,
        list
    )

    # k = 2 through 10 produces nine results.
    assert len(inertia) == 9
    assert len(silhouette_scores) == 9

    # Verify all results are numeric and finite.
    assert np.isfinite(
        inertia
    ).all()

    assert np.isfinite(
        silhouette_scores
    ).all()

    # Inertia must be positive.
    assert all(
        value > 0
        for value in inertia
    )

    # Silhouette scores must be between -1 and 1.
    assert all(
        -1 <= value <= 1
        for value in silhouette_scores
    )

    # Inertia should decrease as the number
    # of clusters increases.
    assert all(
        inertia[i] > inertia[i + 1]
        for i in range(
            len(inertia) - 1
        )
    )

    # Expected results from the original notebook.
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

    # Verify that the wrapper produces the same
    # inertia values as the original notebook.
    np.testing.assert_allclose(
        inertia,
        expected_inertia,
        rtol=1e-5,
        atol=1e-5
    )

    # Verify that the wrapper produces the same
    # silhouette scores as the original notebook.
    np.testing.assert_allclose(
        silhouette_scores,
        expected_silhouette_scores,
        rtol=1e-5,
        atol=1e-5
    )


# ==========================================================
# MODULE 9 - CLUSTERING
# ==========================================================

def test_perform_kmeans():
    """SCA-UT-013"""

    df = prepare_dataset()

    X = select_features(
        df,
        FEATURES
    )

    X_scaled = standardize_features(
        X
    )

    model, clusters = perform_kmeans(
        X_scaled
    )

    # Verify the returned model type.
    assert isinstance(
        model,
        KMeans
    )

    # Verify four clusters are used.
    assert model.n_clusters == 4

    # Verify one cluster label per respondent.
    assert len(clusters) == 757

    # Verify exactly four clusters exist.
    assert len(
        np.unique(clusters)
    ) == 4

    # Verify valid cluster labels.
    assert set(
        np.unique(clusters)
    ) == {
        0,
        1,
        2,
        3
    }

    # Verify cluster centers have the expected dimensions.
    assert model.cluster_centers_.shape == (
        4,
        6
    )

    # Verify the model has been fitted.
    assert hasattr(
        model,
        "inertia_"
    )

    # Verify the model produced finite values.
    assert np.isfinite(
        model.cluster_centers_
    ).all()

    assert np.isfinite(
        clusters
    ).all()


def test_get_cluster_sizes():
    """SCA-UT-014"""

    df = prepare_dataset()

    X = select_features(
        df,
        FEATURES
    )

    X_scaled = standardize_features(
        X
    )

    _, clusters = perform_kmeans(
        X_scaled
    )

    result = get_cluster_sizes(
        clusters
    )

    # Verify return type.
    assert isinstance(
        result,
        pd.Series
    )

    # Verify cluster labels.
    assert result.index.tolist() == [
        0,
        1,
        2,
        3
    ]

    # Verify expected cluster sizes.
    assert result.tolist() == [
        97,
        258,
        20,
        382
    ]

    # Verify Series name.
    assert result.name == "count"

    # Verify all respondents are accounted for.
    assert result.sum() == 757

    # Verify every cluster contains observations.
    assert (
        result > 0
    ).all()


def test_calculate_cluster_profiles():
    """SCA-UT-015"""

    df = prepare_dataset()

    X = select_features(
        df,
        FEATURES
    )

    X_scaled = standardize_features(
        X
    )

    _, clusters = perform_kmeans(
        X_scaled
    )

    df["Cluster"] = clusters

    result = calculate_cluster_profiles(
        df,
        FEATURES
    )

    # Verify return type.
    assert isinstance(
        result,
        pd.DataFrame
    )

    # Verify dimensions.
    assert result.shape == (
        4,
        7
    )

    # Verify expected columns.
    assert result.columns.tolist() == (
        FEATURES + ["AUB"]
    )

    # Verify cluster labels.
    assert result.index.tolist() == [
        0,
        1,
        2,
        3
    ]

    # Verify all profile values are finite.
    assert np.isfinite(
        result.values
    ).all()

    # Verify behavioral constructs remain
    # within the original 1-5 scale.
    assert (
        result[FEATURES] >= 1
    ).all().all()

    assert (
        result[FEATURES] <= 5
    ).all().all()

    # Verify AUB remains within the 1-5 scale.
    assert (
        result["AUB"] >= 1
    ).all()

    assert (
        result["AUB"] <= 5
    ).all()

    # Expected cluster profile values obtained
    # directly from the original notebook.
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

    # Verify that the wrapper output matches
    # the values produced by the original notebook.
    np.testing.assert_allclose(
        result.values,
        expected_profiles.values,
        rtol=1e-5,
        atol=1e-5
    )


# ==========================================================
# MODULE 10 - NORMALITY TESTING
# ==========================================================

def test_perform_shapiro_test():
    """SCA-UT-016"""

    df = prepare_dataset()

    X = select_features(
        df,
        FEATURES
    )

    X_scaled = standardize_features(
        X
    )

    _, clusters = perform_kmeans(
        X_scaled
    )

    df["Cluster"] = clusters

    result = perform_shapiro_test(
        df
    )

    # Verify return type.
    assert isinstance(
        result,
        list
    )

    # Four clusters should produce four results.
    assert len(result) == 4

    # Verify cluster ordering.
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

        # Verify result types.
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

        # Verify valid Shapiro statistic.
        assert 0 <= statistic <= 1

        # Verify valid p-value.
        assert 0 <= p_value <= 1

    # Verify notebook results.
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


# ==========================================================
# MODULE 11 - STATISTICAL INFERENCE
# ==========================================================

def test_perform_kruskal_wallis():
    """SCA-UT-017"""

    df = prepare_dataset()

    X = select_features(
        df,
        FEATURES
    )

    X_scaled = standardize_features(
        X
    )

    _, clusters = perform_kmeans(
        X_scaled
    )

    df["Cluster"] = clusters

    H, p = perform_kruskal_wallis(
        df
    )

    # Verify numeric outputs.
    assert isinstance(
        H,
        (float, np.floating)
    )

    assert isinstance(
        p,
        (float, np.floating)
    )

    # Verify valid statistical ranges.
    assert H >= 0
    assert 0 <= p <= 1

    # Verify notebook results.
    assert H == pytest.approx(
        421.3159298947122,
        rel=1e-10
    )

    assert p == pytest.approx(
        5.341699041631374e-91,
        rel=1e-10
    )

    # Verify statistical significance.
    assert p < 0.05


def test_perform_dunn_test():
    """SCA-UT-018"""

    df = prepare_dataset()

    X = select_features(
        df,
        FEATURES
    )

    X_scaled = standardize_features(
        X
    )

    _, clusters = perform_kmeans(
        X_scaled
    )

    df["Cluster"] = clusters

    result = perform_dunn_test(
        df
    )

    # Verify return type.
    assert isinstance(
        result,
        pd.DataFrame
    )

    # Four clusters produce a 4 x 4 matrix.
    assert result.shape == (
        4,
        4
    )

    # Verify cluster labels.
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

    # Verify all values are finite.
    assert np.isfinite(
        result.values
    ).all()

    # Verify p-values are between 0 and 1.
    assert (
        result.values >= 0
    ).all()

    assert (
        result.values <= 1
    ).all()

    # Self-comparisons should have p = 1.
    assert np.allclose(
        np.diag(result),
        1
    )

    # Dunn's pairwise matrix should be symmetric.
    np.testing.assert_allclose(
        result.values,
        result.values.T
    )

    # Expected Dunn test matrix obtained
    # from the original notebook output.
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

    # Verify that the wrapper output matches
    # the Dunn test results from the original notebook.
    np.testing.assert_allclose(
        result.round(4).values,
        expected_dunn.values,
        rtol=1e-5,
        atol=1e-5
    )

    # Verify significant pairwise differences.
    assert result.loc[0, 1] < 0.05
    assert result.loc[0, 2] < 0.05
    assert result.loc[0, 3] < 0.05
    assert result.loc[1, 3] < 0.05
    assert result.loc[2, 3] < 0.05