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

from sklearn.model_selection import train_test_split, KFold, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor

# ============================= MODULE 5 - FEATURE AND TARGET VARIABLE SELECTION =============================

# Select the predictor variables for machine learning.
def select_features(df, feature_columns):
    missing_columns = [col for col in feature_columns if col not in df.columns]

    if missing_columns:
        raise ValueError(f"Feature column(s) not found: {missing_columns}")

    return df[feature_columns]

# Select the target variable.
def select_target(df, target_column):
    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found.")

    return df[target_column]


# ============================= MODULE 6 - DATASET PARTITIONING =============================

# Split the dataset into training and testing subsets.
def split_dataset(X, y, test_size=0.20, random_state=1):
    return train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state
    )

# Create a K-Fold cross-validation object.
def create_kfold(n_splits=10, shuffle=True, random_state=1):
    return KFold(
        n_splits=n_splits,
        shuffle=shuffle,
        random_state=random_state
    )


# ============================= MODULE 7 - FEATURE SCALING =============================

# Standardize the predictor variables, fitting only on the training set.
def standardize_features(X_train, X_test):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return scaler, X_train_scaled, X_test_scaled


# ============================= MODULE 8 - RANDOM FOREST =============================

# Create the Random Forest hyperparameter search space.
def create_rf_param_grid():
    param_grid = {
        "n_estimators": [100, 200, 300],
        "max_features": [2, 3, 6],
        "max_depth": [None, 10, 20],
        "min_samples_split": [2, 5],
        "min_samples_leaf": [1, 2, 5]
    }
    return param_grid

# Initialize a GridSearchCV object for Random Forest regression.
def create_rf_gridsearch(param_grid, cv, random_state=1, scoring="neg_mean_squared_error", n_jobs=-1):
    grid_rf = GridSearchCV(
        estimator=RandomForestRegressor(random_state=random_state),
        param_grid=param_grid,
        cv=cv,
        scoring=scoring,
        n_jobs=n_jobs
    )
    return grid_rf

# Train the GridSearchCV object.
def fit_rf_gridsearch(grid_rf, X_train, y_train):
    grid_rf.fit(X_train, y_train)
    return grid_rf

# Retrieve the best Random Forest hyperparameters and cross-validation MSE.
def get_best_rf_params(grid_rf):
    best_params = grid_rf.best_params_
    best_mse = -grid_rf.best_score_
    return best_params, best_mse

# Convert GridSearchCV results into a summary DataFrame.
def summarize_rf_results(grid_rf):
    results = pd.DataFrame(grid_rf.cv_results_)
    columns = [
        "param_n_estimators", "param_max_features", "param_max_depth",
        "param_min_samples_split", "param_min_samples_leaf",
        "mean_test_score", "rank_test_score"
    ]
    return results[columns].sort_values("rank_test_score")

# Retrieve the best-performing Random Forest model.
def get_best_rf_model(grid_rf):
    return grid_rf.best_estimator_

# Generate predictions using the optimized Random Forest model.
def predict_rf(model, X_test):
    return model.predict(X_test)


# ============================= MODULE 9 - MLP REGRESSOR =============================

# Create the hyperparameter search space for the MLP Regressor.
def create_mlp_param_grid():
    param_grid = {
        "hidden_layer_sizes": [(50,), (100,), (100, 50)],
        "activation": ["relu", "tanh"],
        "learning_rate_init": [0.001, 0.01],
        "alpha": [0.0001, 0.001]
    }
    return param_grid

# Create the GridSearchCV object for the MLP Regressor.
def create_mlp_gridsearch(param_grid, cv, n_jobs=-1):
    grid = GridSearchCV(
        estimator=MLPRegressor(max_iter=1000, random_state=1),
        param_grid=param_grid,
        cv=cv,
        scoring="neg_mean_squared_error",
        n_jobs=n_jobs
    )
    return grid

# Train the GridSearchCV object using the training data.
def fit_mlp_gridsearch(grid, X_train_scaled, y_train):
    grid.fit(X_train_scaled, y_train)
    return grid

# Retrieve the best hyperparameters found by GridSearchCV.
def get_best_mlp_params(grid):
    return grid.best_params_

# Retrieve the best cross-validation Mean Squared Error.
def get_best_mlp_cv_mse(grid):
    return -grid.best_score_

# Retrieve the best-performing MLP model.
def get_best_mlp_model(grid):
    return grid.best_estimator_

# Generate predictions using the trained MLP model.
def predict_mlp(model, X_test_scaled):
    return model.predict(X_test_scaled)


# ============================= MODULE 10 - PERFORMANCE EVALUATION =============================

# Compute common regression evaluation metrics.
def evaluate_regression(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    return mae, mse, rmse, r2

# Create a comparison table for Random Forest and MLP regression models.
def compare_models(rf_mae, rf_mse, rf_rmse, rf_r2, mlp_mae, mlp_mse, mlp_rmse, mlp_r2):
    comparison = pd.DataFrame({
        "Model": ["Random Forest", "MLP"],
        "MAE": [rf_mae, mlp_mae],
        "MSE": [rf_mse, mlp_mse],
        "RMSE": [rf_rmse, mlp_rmse],
        "R²": [rf_r2, mlp_r2]
    })
    return comparison