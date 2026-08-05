"""
clustering.py

Wrapper functions for feature selection, feature standardization, multicollinearity analysis, optimal k selection, clustering, normality testing, and statistical inference.

Project:
Social Commerce Adoption Among Generation Z University Students in Vietnam
"""

import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from statsmodels.stats.outliers_influence import variance_inflation_factor
from scipy.stats import shapiro, kruskal
from sklearn.metrics import silhouette_score
import scikit_posthocs as sp


# ==========================================================
# MODULE 5 - FEATURE SELECTION
# ==========================================================

def select_features(scommerce_df, features):
    """
    Select the behavioral constructs used for clustering.
    """

    X = scommerce_df[features]

    return X


# ==========================================================
# MODULE 6 - FEATURE STANDARDIZATION
# ==========================================================

def standardize_features(X):
    """
    Standardize the selected clustering features.
    """

    scaler = StandardScaler()

    X_scaled = scaler.fit_transform(X)

    return X_scaled


# ==========================================================
# MODULE 7 - MULTICOLLINEARITY ANALYSIS
# ==========================================================

def calculate_vif(X_scaled, features):
    """
    Calculate Variance Inflation Factor (VIF)
    using the standardized feature values.
    """

    vif_data = pd.DataFrame()
    vif_data["Construct"] = features
    vif_data["VIF"] = [
        variance_inflation_factor(X_scaled, i)
        for i in range(X_scaled.shape[1])
    ]

    return vif_data


# ==========================================================
# MODULE 8 - OPTIMAL K SELECTION
# ==========================================================

def evaluate_kmeans_clusters(X_scaled, k_values):
    """
    Evaluate K-Means models using inertia and
    silhouette scores for the specified k values.
    """

    inertia = []
    silhouette_scores = []

    for k in k_values:
        kmeans = KMeans(
            n_clusters=k,
            random_state=42,
            n_init=10
        )

        labels = kmeans.fit_predict(X_scaled)

        inertia.append(kmeans.inertia_)

        silhouette_scores.append(
            silhouette_score(X_scaled, labels)
        )

    return inertia, silhouette_scores


# ==========================================================
# MODULE 9 - CLUSTERING
# ==========================================================

def perform_kmeans(X_scaled):
    """
    Perform the final K-Means clustering with
    four clusters.
    """

    kmeans = KMeans(
        n_clusters=4,
        random_state=42
    )

    clusters = kmeans.fit_predict(X_scaled)

    return kmeans, clusters


def get_cluster_sizes(clusters):
    """
    Calculate the number of observations in each cluster.
    """

    cluster_sizes = (
        pd.Series(clusters)
        .value_counts()
        .sort_index()
    )

    return cluster_sizes


def calculate_cluster_profiles(scommerce_df, features):
    """
    Calculate the mean values of the selected constructs
    and AUB for each cluster.
    """

    cluster_profile = (
        scommerce_df
        .groupby("Cluster")[features + ["AUB"]]
        .mean()
    )

    return cluster_profile


# ==========================================================
# MODULE 10 - NORMALITY TESTING
# ==========================================================

def perform_shapiro_test(scommerce_df):
    """
    Perform the Shapiro-Wilk normality test for AUB
    within each cluster.
    """

    results = []

    for cluster in sorted(scommerce_df["Cluster"].unique()):

        stat, p = shapiro(
            scommerce_df.loc[
                scommerce_df["Cluster"] == cluster,
                "AUB"
            ]
        )

        results.append(
            (cluster, stat, p)
        )

    return results


# ==========================================================
# MODULE 11 - STATISTICAL INFERENCE
# ==========================================================

def perform_kruskal_wallis(scommerce_df):
    """
    Perform the Kruskal-Wallis H test on AUB
    across the clusters.
    """

    groups = [
        scommerce_df.loc[
            scommerce_df["Cluster"] == cluster,
            "AUB"
        ]
        for cluster in sorted(
            scommerce_df["Cluster"].unique()
        )
    ]

    H, p = kruskal(*groups)

    return H, p


def perform_dunn_test(scommerce_df):
    """
    Perform Dunn's post-hoc test with
    Bonferroni correction.
    """

    dunn = sp.posthoc_dunn(
        scommerce_df,
        val_col="AUB",
        group_col="Cluster",
        p_adjust="bonferroni"
    )

    return dunn