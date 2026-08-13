"""
machinelearning.py

Wrapper functions for feature selection,
dataset partitioning, cross-validation,
and feature scaling.

Project:
Social Commerce Adoption Among Generation Z University Students in Vietnam
"""

import pandas as pd
import numpy as np

from scipy import stats

from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)


# ==========================================================
# MODULE 5 - FEATURE AND TARGET VARIABLE SELECTION
# ==========================================================

def select_features(df, feature_columns):
    """
    Select the predictor variables for machine learning.

    Parameters
    ----------
    df : pandas.DataFrame

    feature_columns : list
        List of predictor variable names.

    Returns
    -------
    pandas.DataFrame
        Selected predictor variables.

    Raises
    ------
    ValueError
        If one or more feature columns do not exist.
    """

    missing_columns = [
        col for col in feature_columns
        if col not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Feature column(s) not found: {missing_columns}"
        )

    return df[feature_columns]


def select_target(df, target_column):
    """
    Select the target variable.

    Parameters
    ----------
    df : pandas.DataFrame

    target_column : str
        Name of the target variable.

    Returns
    -------
    pandas.Series
        Target variable.

    Raises
    ------
    ValueError
        If the target column does not exist.
    """

    if target_column not in df.columns:
        raise ValueError(
            f"Target column '{target_column}' not found."
        )

    return df[target_column]


# ==========================================================
# MODULE 6 - DATASET PARTITIONING
# ==========================================================

def split_dataset(
    X,
    y,
    test_size=0.20,
    random_state=1
):
    """
    Split the dataset into training and testing subsets.

    Parameters
    ----------
    X : pandas.DataFrame

    y : pandas.Series

    test_size : float, default=0.20

    random_state : int, default=1

    Returns
    -------
    tuple
        X_train,
        X_test,
        y_train,
        y_test
    """

    return train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state
    )


def create_kfold(
    n_splits=10,
    shuffle=True,
    random_state=1
):
    """
    Create a K-Fold cross-validation object.

    Parameters
    ----------
    n_splits : int

    shuffle : bool

    random_state : int

    Returns
    -------
    sklearn.model_selection.KFold
    """

    return KFold(
        n_splits=n_splits,
        shuffle=shuffle,
        random_state=random_state
    )


# ==========================================================
# MODULE 7 - FEATURE SCALING
# ==========================================================

def standardize_features(
    X_train,
    X_test
):
    """
    Standardize the predictor variables.

    The scaler is fitted only on the training
    dataset to prevent data leakage.

    Parameters
    ----------
    X_train : pandas.DataFrame

    X_test : pandas.DataFrame

    Returns
    -------
    tuple
        scaler,
        X_train_scaled,
        X_test_scaled
    """

    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(
        X_train
    )

    X_test_scaled = scaler.transform(
        X_test
    )

    return (
        scaler,
        X_train_scaled,
        X_test_scaled
    )

# ==========================================================
# MODULE 8 - RANDOM FOREST
# ==========================================================

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV


def create_rf_param_grid():
    """
    Create the Random Forest hyperparameter search space.

    Returns
    -------
    dict
        Dictionary containing the Random Forest
        hyperparameter search space.
    """

    param_grid = {
        "n_estimators": [100, 200, 300],
        "max_features": [2, 3, 6],
        "max_depth": [None, 10, 20],
        "min_samples_split": [2, 5],
        "min_samples_leaf": [1, 2, 5]
    }

    return param_grid


def create_rf_gridsearch(
    param_grid,
    cv,
    random_state=1,
    scoring="neg_mean_squared_error",
    n_jobs=-1
):
    """
    Initialize a GridSearchCV object for
    Random Forest regression.

    Parameters
    ----------
    param_grid : dict

    cv : cross-validation object

    random_state : int

    scoring : str

    n_jobs : int

    Returns
    -------
    GridSearchCV
    """

    grid_rf = GridSearchCV(
        estimator=RandomForestRegressor(
            random_state=random_state
        ),
        param_grid=param_grid,
        cv=cv,
        scoring=scoring,
        n_jobs=n_jobs
    )

    return grid_rf


def fit_rf_gridsearch(
    grid_rf,
    X_train,
    y_train
):
    """
    Train the GridSearchCV object.

    Parameters
    ----------
    grid_rf : GridSearchCV

    X_train : pandas.DataFrame

    y_train : pandas.Series

    Returns
    -------
    GridSearchCV
        Trained GridSearchCV object.
    """

    grid_rf.fit(
        X_train,
        y_train
    )

    return grid_rf


def get_best_rf_params(grid_rf):
    """
    Retrieve the best Random Forest
    hyperparameters and corresponding
    cross-validation MSE.

    Parameters
    ----------
    grid_rf : GridSearchCV

    Returns
    -------
    tuple
        best_params,
        best_mse
    """

    best_params = grid_rf.best_params_
    best_mse = -grid_rf.best_score_

    return (
        best_params,
        best_mse
    )


def summarize_rf_results(grid_rf):
    """
    Convert GridSearchCV results into
    a summary DataFrame.

    Parameters
    ----------
    grid_rf : GridSearchCV

    Returns
    -------
    pandas.DataFrame
    """

    import pandas as pd

    results = pd.DataFrame(
        grid_rf.cv_results_
    )

    columns = [
        "param_n_estimators",
        "param_max_features",
        "param_max_depth",
        "param_min_samples_split",
        "param_min_samples_leaf",
        "mean_test_score",
        "rank_test_score"
    ]

    return results[
        columns
    ].sort_values(
        "rank_test_score"
    )


def get_best_rf_model(grid_rf):
    """
    Retrieve the best-performing
    Random Forest model.

    Parameters
    ----------
    grid_rf : GridSearchCV

    Returns
    -------
    RandomForestRegressor
    """

    return grid_rf.best_estimator_


def predict_rf(
    model,
    X_test
):
    """
    Generate predictions using the
    optimized Random Forest model.

    Parameters
    ----------
    model : RandomForestRegressor

    X_test : pandas.DataFrame

    Returns
    -------
    numpy.ndarray
    """

    return model.predict(X_test)

# ==========================================================
# MODULE 9 - MLP REGRESSOR
# ==========================================================

from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import GridSearchCV


def create_mlp_param_grid():
    """
    Create the hyperparameter search space for the MLP Regressor.

    Returns
    -------
    dict
        Dictionary containing the hyperparameter grid.
    """

    param_grid = {
        "hidden_layer_sizes": [
            (50,),
            (100,),
            (100, 50)
        ],
        "activation": [
            "relu",
            "tanh"
        ],
        "learning_rate_init": [
            0.001,
            0.01
        ],
        "alpha": [
            0.0001,
            0.001
        ]
    }

    return param_grid


def create_mlp_gridsearch(param_grid, cv, n_jobs=-1):
    """
    Create the GridSearchCV object for the MLP Regressor.

    Parameters
    ----------
    param_grid : dict
        Hyperparameter search space.

    cv : cross-validator
        Cross-validation strategy.

    n_jobs : int, default=-1
        Number of jobs to run in parallel.

    Returns
    -------
    sklearn.model_selection.GridSearchCV
        Configured GridSearchCV object.
    """

    grid = GridSearchCV(
        estimator=MLPRegressor(
            max_iter=1000,
            random_state=1
        ),
        param_grid=param_grid,
        cv=cv,
        scoring="neg_mean_squared_error",
        n_jobs=n_jobs
    )

    return grid


def fit_mlp_gridsearch(grid, X_train_scaled, y_train):
    """
    Train the GridSearchCV object using the training data.

    Parameters
    ----------
    grid : GridSearchCV

    X_train_scaled : numpy.ndarray

    y_train : pandas.Series

    Returns
    -------
    GridSearchCV
        Fitted GridSearchCV object.
    """

    grid.fit(
        X_train_scaled,
        y_train
    )

    return grid


def get_best_mlp_params(grid):
    """
    Retrieve the best hyperparameters found by GridSearchCV.

    Parameters
    ----------
    grid : GridSearchCV

    Returns
    -------
    dict
        Best hyperparameter combination.
    """

    return grid.best_params_


def get_best_mlp_cv_mse(grid):
    """
    Retrieve the best cross-validation Mean Squared Error.

    Parameters
    ----------
    grid : GridSearchCV

    Returns
    -------
    float
        Best cross-validation MSE.
    """

    return -grid.best_score_


def get_best_mlp_model(grid):
    """
    Retrieve the best-performing MLP model.

    Parameters
    ----------
    grid : GridSearchCV

    Returns
    -------
    MLPRegressor
        Best trained MLP model.
    """

    return grid.best_estimator_


def predict_mlp(model, X_test_scaled):
    """
    Generate predictions using the trained MLP model.

    Parameters
    ----------
    model : MLPRegressor

    X_test_scaled : numpy.ndarray

    Returns
    -------
    numpy.ndarray
        Predicted target values.
    """

    return model.predict(X_test_scaled)

# ==========================================================
# MODULE 10 - PERFORMANCE EVALUATION
# ==========================================================

def evaluate_regression(y_true, y_pred):
    """
    Compute common regression evaluation metrics.

    Parameters
    ----------
    y_true : array-like
        Actual target values.

    y_pred : array-like
        Predicted target values.

    Returns
    -------
    tuple
        (mae, mse, rmse, r2)
    """

    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)

    return mae, mse, rmse, r2


def compare_models(
    rf_mae,
    rf_mse,
    rf_rmse,
    rf_r2,
    mlp_mae,
    mlp_mse,
    mlp_rmse,
    mlp_r2
):
    """
    Create a comparison table for Random Forest
    and MLP regression models.

    Parameters
    ----------
    rf_mae, rf_mse, rf_rmse, rf_r2 : float
        Random Forest evaluation metrics.

    mlp_mae, mlp_mse, mlp_rmse, mlp_r2 : float
        MLP evaluation metrics.

    Returns
    -------
    pandas.DataFrame
        Comparison table of both models.
    """

    comparison = pd.DataFrame({
        "Model": [
            "Random Forest",
            "MLP"
        ],
        "MAE": [
            rf_mae,
            mlp_mae
        ],
        "MSE": [
            rf_mse,
            mlp_mse
        ],
        "RMSE": [
            rf_rmse,
            mlp_rmse
        ],
        "R²": [
            rf_r2,
            mlp_r2
        ]
    })

    return comparison


# ==========================================================
# MODULE 8 - EXPLORATORY DATA ANALYSIS
# ==========================================================

def get_construct_descriptives(df):
    """
    Generate descriptive statistics for the seven
    composite constructs.

    Returns
    -------
    pandas.DataFrame
        Descriptive statistics for PU, PEU, FSC, SP,
        TP, IB, and AUB.
    """

    constructs = [
        "PU",
        "PEU",
        "FSC",
        "SP",
        "TP",
        "IB",
        "AUB"
    ]

    return df[constructs].describe()


def get_construct_correlation_matrix(df):
    """
    Generate the Pearson correlation matrix for
    the seven composite constructs.

    Returns
    -------
    pandas.DataFrame
        Pearson correlation matrix.
    """

    constructs = [
        "PU",
        "PEU",
        "FSC",
        "SP",
        "TP",
        "IB",
        "AUB"
    ]

    return df[constructs].corr(method="pearson")


def get_aub_by_gender_summary(df):
    """
    Generate descriptive statistics for AUB
    across gender groups.

    The 'Different' gender category is excluded
    because it contains only two respondents.

    Returns
    -------
    pandas.DataFrame
        AUB descriptive statistics by gender.
    """

    gender_df = df[df["Gender"] != 3]

    summary = (
        gender_df
        .groupby("Gender")["AUB"]
        .agg(["count", "mean", "median", "std"])
    )

    summary.index = summary.index.map({
        1: "Male",
        2: "Female"
    })

    return summary


def run_ttest_aub_gender(df):
    """
    Perform an independent samples t-test comparing
    AUB between male and female respondents.

    Returns
    -------
    tuple
        t-statistic and p-value.
    """

    gender_df = df[df["Gender"] != 3]

    male = gender_df[
        gender_df["Gender"] == 1
    ]["AUB"]

    female = gender_df[
        gender_df["Gender"] == 2
    ]["AUB"]

    t_stat, p_value = stats.ttest_ind(
        male,
        female,
        equal_var=False
    )

    return t_stat, p_value


def get_aub_by_area_summary(df):
    """
    Generate descriptive statistics for AUB
    across residential area groups.

    Returns
    -------
    pandas.DataFrame
        AUB descriptive statistics by area.
    """

    summary = (
        df
        .groupby("Area")["AUB"]
        .agg(["count", "mean", "median", "std"])
    )

    summary.index = summary.index.map({
        1: "Urban",
        2: "Suburban",
        3: "Rural"
    })

    return summary


def run_anova_aub_area(df):
    """
    Perform a one-way ANOVA comparing AUB
    across residential area groups.

    Returns
    -------
    tuple
        F-statistic and p-value.
    """

    urban = df[
        df["Area"] == 1
    ]["AUB"]

    suburban = df[
        df["Area"] == 2
    ]["AUB"]

    rural = df[
        df["Area"] == 3
    ]["AUB"]

    f_stat, p_value = stats.f_oneway(
        urban,
        suburban,
        rural
    )

    return f_stat, p_value


def get_aub_by_frequency_summary(df):
    """
    Generate descriptive statistics for AUB
    across social media usage frequency groups.

    Returns
    -------
    pandas.DataFrame
        AUB descriptive statistics by frequency.
    """

    summary = (
        df
        .groupby("Frequently")["AUB"]
        .agg(["count", "mean", "median", "std"])
    )

    summary.index = summary.index.map({
        1: "Daily",
        2: "Weekly",
        3: "Monthly",
        4: "Rarely"
    })

    return summary