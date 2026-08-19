import numpy as np
import pandas as pd
import pytest

from sklearn.model_selection import GridSearchCV
from sklearn.datasets import make_regression
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor

from ml.scripts.preprocessing import (
    load_dataset,
    drop_columns,
    compute_composite_score,
)

from ml.scripts.machinelearning import (
    select_features,
    select_target,
    split_dataset,
    create_kfold,
    standardize_features,
    create_rf_param_grid,
    create_rf_gridsearch,
    fit_rf_gridsearch,
    get_best_rf_params,
    summarize_rf_results,
    get_best_rf_model,
    predict_rf,
    evaluate_regression,
    create_mlp_param_grid,
    create_mlp_gridsearch,
    fit_mlp_gridsearch,
    get_best_mlp_params,
    get_best_mlp_cv_mse,
    get_best_mlp_model,
    predict_mlp,
    compare_models,
)

DATASET_PATH = "ml/datasets/S-COMMERCE_GenZ_UniversityStudent_757_DIB.csv"


# HELPER FUNCTION
def prepare_dataset():
    """Reproduce the preprocessing performed in the notebook."""
    df = load_dataset(DATASET_PATH)
    df = drop_columns(df, ["Job"])

    constructs = {
        "PU": ["PU1", "PU2", "PU3", "PU4"],
        "PEU": ["PEU1", "PEU2", "PEU3"],
        "FSC": ["FSC1", "FSC2", "FSC3"],
        "SP": ["SP1", "SP2", "SP3", "SP4"],
        "TP": ["TP1", "TP2", "TP3"],
        "IB": ["IB1", "IB2", "IB3", "IB4"],
        "AUB": ["AUB1", "AUB2", "AUB3", "AUB4"]
    }

    for new_col, cols in constructs.items():
        df = compute_composite_score(df, cols, new_col)
    return df


# ============================= MODULE 5 - FEATURE AND TARGET VARIABLE SELECTION =============================

# The following test checks:
# - If predictor features are correctly extracted from the dataset.
#
# This test will fail if:
# - The return type is not a DataFrame.
# - The columns extracted do not match the expected list exactly.
def test_select_features():
    """SCA-UT-009"""
    df = prepare_dataset()
    expected_columns = ["PU", "PEU", "FSC", "SP", "TP", "IB"]
    
    X = select_features(df, expected_columns)

    assert isinstance(X, pd.DataFrame)                 # Verify that the extracted features are returned as a DataFrame.
    assert X.columns.tolist() == expected_columns      # Verify that the column names match exactly.
    assert X.shape == (757, 6)                         # Verify that the shape corresponds to 757 rows and 6 features.

# The following test checks:
# - If a ValueError is raised when requesting a non-existent column.
#
# This test will fail if:
# - The function silently ignores invalid column names.
def test_select_features_invalid_column():
    """SCA-UT-010"""
    df = prepare_dataset()

    with pytest.raises(ValueError):
        select_features(df, ["PU", "INVALID"])         # Verify that an invalid column name triggers a ValueError.


# The following test checks:
# - If the target variable is correctly extracted as a Series.
#
# This test will fail if:
# - The return type is not a Pandas Series.
# - The name of the series does not match the target.
def test_select_target():
    """SCA-UT-011"""
    df = prepare_dataset()
    
    y = select_target(df, "AUB")

    assert isinstance(y, pd.Series)                    # Verify that the target is extracted as a Series.
    assert y.name == "AUB"                             # Verify that the Series is correctly named 'AUB'.
    assert len(y) == 757                               # Verify that the target Series maintains the correct length.

# The following test checks:
# - If a ValueError is raised when requesting a non-existent target column.
#
# This test will fail if:
# - The function fails silently or returns an unexpected error.
def test_select_target_invalid():
    """SCA-UT-012"""
    df = prepare_dataset()

    with pytest.raises(ValueError):
        select_target(df, "INVALID")                   # Verify that an invalid target name triggers a ValueError.


# ============================= MODULE 6 - DATASET PARTITIONING =============================

# The following test checks:
# - If the dataset is correctly partitioned into 80/20 train and test sets.
#
# This test will fail if:
# - The row counts for train/test splits do not match the expected 80/20 ratio.
def test_split_dataset():
    """SCA-UT-013"""
    df = prepare_dataset()
    X = select_features(df, ["PU", "PEU", "FSC", "SP", "TP", "IB"])
    y = select_target(df, "AUB")

    X_train, X_test, y_train, y_test = split_dataset(X, y, test_size=0.20, random_state=1)

    assert X_train.shape == (605, 6)                   # Verify that the training set feature matrix has 605 rows.
    assert X_test.shape == (152, 6)                    # Verify that the test set feature matrix has 152 rows.
    assert len(y_train) == 605                         # Verify that the training target vector has 605 rows.
    assert len(y_test) == 152                          # Verify that the test target vector has 152 rows.


# The following test checks:
# - If the K-Fold object is instantiated with correct parameters.
#
# This test will fail if:
# - The number of splits differs from 10.
# - The shuffle parameter is set to False.
def test_create_kfold():
    """SCA-UT-014"""
    kf = create_kfold()

    assert kf.get_n_splits() == 10                     # Verify that the cross-validator generates 10 folds.
    assert kf.shuffle is True                          # Verify that dataset shuffling is enabled.
    assert kf.random_state == 1                        # Verify that the random state is deterministic.


# ============================= MODULE 7 - FEATURE SCALING =============================

# The following test checks:
# - If the standardization scales values correctly without data leakage.
#
# This test will fail if:
# - The scaled arrays have incorrect dimensions.
# - The mean is not centered at 0 or std is not 1 for the training set.
def test_standardize_features():
    """SCA-UT-015"""
    df = prepare_dataset()
    X = select_features(df, ["PU", "PEU", "FSC", "SP", "TP", "IB"])
    y = select_target(df, "AUB")
    X_train, X_test, _, _ = split_dataset(X, y)

    scaler, X_train_scaled, X_test_scaled = standardize_features(X_train, X_test)

    assert X_train_scaled.shape == X_train.shape       # Verify the scaled training array maintains shape.
    assert X_test_scaled.shape == X_test.shape         # Verify the scaled test array maintains shape.
    np.testing.assert_allclose(X_train_scaled.mean(axis=0), np.zeros(X_train.shape[1]), atol=1e-7)  # Verify training mean is 0.
    np.testing.assert_allclose(X_train_scaled.std(axis=0), np.ones(X_train.shape[1]), atol=1e-7)    # Verify training std is 1.


# ============================= MODULE 8 - RANDOM FOREST =============================

# The following test checks:
# - If the Random Forest hyperparameter grid contains expected parameters.
#
# This test will fail if:
# - Missing expected keys in the parameter dictionary.
def test_create_rf_param_grid():
    """SCA-UT-016"""
    grid = create_rf_param_grid()
    expected_keys = {"n_estimators", "max_features", "max_depth", "min_samples_split", "min_samples_leaf"}

    assert set(grid.keys()) == expected_keys           # Verify all expected hyperparameter keys are present.
    assert grid["n_estimators"] == [100, 200, 300]     # Verify n_estimators choices.
    assert grid["max_features"] == [2, 3, 6]           # Verify max_features choices.
    assert grid["max_depth"] == [None, 10, 20]         # Verify max_depth choices.
    assert grid["min_samples_split"] == [2, 5]         # Verify min_samples_split choices.
    assert grid["min_samples_leaf"] == [1, 2, 5]       # Verify min_samples_leaf choices.


# The following test checks:
# - If the GridSearchCV object for Random Forest is correctly initialized.
#
# This test will fail if:
# - The returned object is not a GridSearchCV instance.
def test_create_rf_gridsearch():
    """SCA-UT-017"""
    param_grid = create_rf_param_grid()
    kf = create_kfold()
    
    grid = create_rf_gridsearch(param_grid, kf)

    assert isinstance(grid, GridSearchCV)              # Verify the returned object is a GridSearchCV instance.
    assert isinstance(grid.estimator, RandomForestRegressor) # Verify the underlying estimator is Random Forest.


# The following test checks:
# - If the GridSearchCV fitting logic works correctly.
#
# This test will fail if:
# - The model does not successfully fit or expose best_estimator_.
def test_fit_rf_gridsearch():
    """SCA-UT-018"""
    df = prepare_dataset()
    X = select_features(df, ["PU", "PEU", "FSC", "SP", "TP", "IB"])
    y = select_target(df, "AUB")
    X_train, X_test, y_train, y_test = split_dataset(X, y, test_size=0.20, random_state=1)

    param_grid = {"n_estimators": [10], "max_features": [2], "max_depth": [5], "min_samples_split": [2], "min_samples_leaf": [1]}
    grid = create_rf_gridsearch(param_grid, create_kfold(), n_jobs=1)
    
    fitted = fit_rf_gridsearch(grid, X_train, y_train)

    assert hasattr(fitted, "best_estimator_")          # Verify the grid search completed and found a best estimator.
    assert hasattr(fitted, "best_params_")             # Verify the grid search retained best parameters.
    assert hasattr(fitted, "best_score_")              # Verify the grid search retained the best score.


# The following test checks:
# - If the optimal Random Forest parameters and MSE can be extracted.
#
# This test will fail if:
# - The returned params are not a dictionary.
# - The returned MSE is negative.
def test_get_best_rf_params():
    """SCA-UT-019"""
    X, y = make_regression(n_samples=80, n_features=6, random_state=1)
    param_grid = {"n_estimators": [10], "max_features": [2], "max_depth": [5], "min_samples_split": [2], "min_samples_leaf": [1]}
    grid = create_rf_gridsearch(param_grid, create_kfold(), n_jobs=1)
    fit_rf_gridsearch(grid, X, y)

    params, mse = get_best_rf_params(grid)

    assert isinstance(params, dict)                    # Verify the optimal parameters are returned as a dictionary.
    assert isinstance(mse, float)                      # Verify the MSE score is a float.
    assert mse >= 0                                    # Verify the MSE is correctly rendered as a non-negative value.


# The following test checks:
# - If the cross-validation summary table is generated correctly.
#
# This test will fail if:
# - Missing expected columns in the summary DataFrame.
def test_summarize_rf_results():
    """SCA-UT-020"""
    X, y = make_regression(n_samples=80, n_features=6, random_state=1)
    param_grid = {"n_estimators": [10], "max_features": [2], "max_depth": [5], "min_samples_split": [2], "min_samples_leaf": [1]}
    grid = create_rf_gridsearch(param_grid, create_kfold(), n_jobs=1)
    fit_rf_gridsearch(grid, X, y)

    summary = summarize_rf_results(grid)
    expected_columns = [
        "param_n_estimators", "param_max_features", "param_max_depth",
        "param_min_samples_split", "param_min_samples_leaf",
        "mean_test_score", "rank_test_score"
    ]

    for col in expected_columns:
        assert col in summary.columns                  # Verify the summary DataFrame contains all specific scoring metrics.


# The following test checks:
# - If the best Random Forest model instance is successfully returned.
#
# This test will fail if:
# - The returned object is not a RandomForestRegressor.
def test_get_best_rf_model():
    """SCA-UT-021"""
    X, y = make_regression(n_samples=80, n_features=6, random_state=1)
    param_grid = {"n_estimators": [10], "max_features": [2], "max_depth": [5], "min_samples_split": [2], "min_samples_leaf": [1]}
    grid = create_rf_gridsearch(param_grid, create_kfold(), n_jobs=1)
    fit_rf_gridsearch(grid, X, y)

    model = get_best_rf_model(grid)

    assert isinstance(model, RandomForestRegressor)    # Verify that the extracted optimal model is a RandomForestRegressor.


# The following test checks:
# - If predictions can be successfully generated from the RF model.
#
# This test will fail if:
# - The predictions are not returned as a numpy array.
# - The length of predictions does not match test data.
def test_predict_rf():
    """SCA-UT-022"""
    df = prepare_dataset()
    X = select_features(df, ["PU", "PEU", "FSC", "SP", "TP", "IB"])
    y = select_target(df, "AUB")
    X_train, X_test, y_train, y_test = split_dataset(X, y, test_size=0.20, random_state=1)
    
    model = RandomForestRegressor(n_estimators=10, random_state=1)
    model.fit(X_train, y_train)
    predictions = predict_rf(model, X_test)

    assert isinstance(predictions, np.ndarray)         # Verify the predictions are output as a NumPy array.
    assert len(predictions) == len(y_test)             # Verify the count of predictions aligns with the testing subset.
    assert np.isfinite(predictions).all()              # Verify that no infinite or NaN predictions were produced.


# ============================= MODULE 9 - MLP REGRESSOR =============================

# The following test checks:
# - If the MLP hyperparameter grid contains expected parameters.
#
# This test will fail if:
# - Missing expected keys in the parameter dictionary.
def test_create_mlp_param_grid():
    """SCA-UT-023"""
    param_grid = create_mlp_param_grid()
    expected_keys = {"hidden_layer_sizes", "activation", "learning_rate_init", "alpha"}

    assert isinstance(param_grid, dict)                # Verify the hyperparameter grid is a dictionary.
    assert set(param_grid.keys()) == expected_keys     # Verify all expected hyperparameter keys for the MLP exist.

    assert param_grid["hidden_layer_sizes"] == [       # Verify the hidden layer configurations are as expected.
            (50,),
            (100,),
            (100, 50)
        ]
    
    assert param_grid["activation"] == [
        "relu",
        "tanh"
    ]

    assert param_grid["learning_rate_init"] == [       # Verify the learning rate configurations are as expected.
        0.001,
        0.01
    ]

    assert param_grid["alpha"] == [                    # Verify the alpha configurations are as expected.
        0.0001,
        0.001
    ]


# The following test checks:
# - If the GridSearchCV object for MLP is correctly initialized.
#
# This test will fail if:
# - The scoring metric is incorrect.
def test_create_mlp_gridsearch():
    """SCA-UT-024"""
    param_grid = create_mlp_param_grid()
    kf = create_kfold()
    
    grid = create_mlp_gridsearch(param_grid, kf)

    assert isinstance(grid, GridSearchCV)              # Verify the returned object is a GridSearchCV instance.
    assert grid.scoring == "neg_mean_squared_error"    # Verify the grid uses negative MSE for optimization scoring.
    assert grid.cv == kf                               # Verify the grid uses the provided K-Fold cross-validator.
    assert grid.n_jobs == -1                           # Verify the grid is set to utilize all available CPU cores for parallel processing.    


# The following test checks:
# - If the MLP GridSearchCV fitting logic works correctly.
#
# This test will fail if:
# - The model does not expose best_params_ or best_estimator_.
def test_fit_mlp_gridsearch():
    """SCA-UT-025"""
    df = prepare_dataset()
    X = select_features(df, ["PU", "PEU", "FSC", "SP", "TP", "IB"])
    y = select_target(df, "AUB")
    X_train, X_test, y_train, y_test = split_dataset(X, y)
    scaler, X_train_scaled, X_test_scaled = standardize_features(X_train, X_test)

    param_grid = create_mlp_param_grid()
    kf = create_kfold()
    grid = create_mlp_gridsearch(param_grid, kf, n_jobs=1)
    
    fitted = fit_mlp_gridsearch(grid, X_train_scaled, y_train)

    assert hasattr(fitted, "best_params_")             # Verify the grid search successfully completed and stored params.
    assert hasattr(fitted, "best_estimator_")          # Verify the grid search stored the optimal estimator.


# The following test checks:
# - If the optimal MLP parameters can be extracted.
#
# This test will fail if:
# - The extracted params dictionary lacks necessary keys.
def test_get_best_mlp_params():
    """SCA-UT-026"""
    df = prepare_dataset()
    X = select_features(df, ["PU", "PEU", "FSC", "SP", "TP", "IB"])
    y = select_target(df, "AUB")
    X_train, X_test, y_train, y_test = split_dataset(X, y)
    scaler, X_train_scaled, X_test_scaled = standardize_features(X_train, X_test)
    grid = create_mlp_gridsearch(create_mlp_param_grid(), create_kfold(), n_jobs=1)
    fitted = fit_mlp_gridsearch(grid, X_train_scaled, y_train)

    params = get_best_mlp_params(fitted)
    expected = {"hidden_layer_sizes", "activation", "learning_rate_init", "alpha"}

    assert isinstance(params, dict)                    # Verify the optimal parameters are returned as a dictionary.
    assert set(params.keys()) == expected              # Verify the dictionary contains all MLP hyperparameter keys.


# The following test checks:
# - If the MLP CV Mean Squared Error is correctly returned.
#
# This test will fail if:
# - The MSE is returned as a negative value.
def test_get_best_mlp_cv_mse():
    """SCA-UT-027"""
    df = prepare_dataset()
    X = select_features(df, ["PU", "PEU", "FSC", "SP", "TP", "IB"])
    y = select_target(df, "AUB")
    X_train, X_test, y_train, y_test = split_dataset(X, y)
    scaler, X_train_scaled, X_test_scaled = standardize_features(X_train, X_test)
    grid = create_mlp_gridsearch(create_mlp_param_grid(), create_kfold(), n_jobs=1)
    fitted = fit_mlp_gridsearch(grid, X_train_scaled, y_train)

    mse = get_best_mlp_cv_mse(fitted)

    assert isinstance(mse, float)                      # Verify the extracted MSE score is a float.
    assert mse >= 0                                    # Verify the MSE is correctly rendered as a non-negative value.


# The following test checks:
# - If the best MLP model instance is successfully returned.
#
# This test will fail if:
# - The returned object is not an MLPRegressor.
def test_get_best_mlp_model():
    """SCA-UT-028"""
    df = prepare_dataset()
    X = select_features(df, ["PU", "PEU", "FSC", "SP", "TP", "IB"])
    y = select_target(df, "AUB")
    X_train, X_test, y_train, y_test = split_dataset(X, y)
    scaler, X_train_scaled, X_test_scaled = standardize_features(X_train, X_test)
    grid = create_mlp_gridsearch(create_mlp_param_grid(), create_kfold(), n_jobs=1)
    fitted = fit_mlp_gridsearch(grid, X_train_scaled, y_train)

    model = get_best_mlp_model(fitted)

    assert isinstance(model, MLPRegressor)             # Verify that the extracted optimal model is an MLPRegressor.


# The following test checks:
# - If predictions can be successfully generated from the MLP model.
#
# This test will fail if:
# - The predictions array length mismatches the test array.
def test_predict_mlp():
    """SCA-UT-029"""
    df = prepare_dataset()
    X = select_features(df, ["PU", "PEU", "FSC", "SP", "TP", "IB"])
    y = select_target(df, "AUB")
    X_train, X_test, y_train, y_test = split_dataset(X, y)
    scaler, X_train_scaled, X_test_scaled = standardize_features(X_train, X_test)
    grid = create_mlp_gridsearch(create_mlp_param_grid(), create_kfold(), n_jobs=1)
    fitted = fit_mlp_gridsearch(grid, X_train_scaled, y_train)
    model = get_best_mlp_model(fitted)

    predictions = predict_mlp(model, X_test_scaled)

    assert isinstance(predictions, np.ndarray)         # Verify the predictions are output as a NumPy array.
    assert len(predictions) == len(y_test)             # Verify the count of predictions aligns with the testing subset.


# ============================= MODULE 10 - PERFORMANCE EVALUATION =============================

# The following test checks:
# - If regression error metrics are correctly calculated.
#
# This test will fail if:
# - Errors are negative, or R² exceeds 1.0.
def test_evaluate_regression():
    """SCA-UT-030"""
    df = prepare_dataset()
    X = select_features(df, ["PU", "PEU", "FSC", "SP", "TP", "IB"])
    y = select_target(df, "AUB")
    X_train, X_test, y_train, y_test = split_dataset(X, y, test_size=0.20, random_state=1)
    
    model = RandomForestRegressor(n_estimators=10, random_state=1)
    model.fit(X_train, y_train)
    predictions = predict_rf(model, X_test)

    mae, mse, rmse, r2 = evaluate_regression(y_test, predictions)

    assert np.isfinite(mae)                            # Verify MAE is finite.
    assert np.isfinite(mse)                            # Verify MSE is finite.
    assert np.isfinite(rmse)                           # Verify RMSE is finite.    
    assert np.isfinite(r2)                             # Verify R² is finite.
    assert mae >= 0                                    # Verify MAE cannot be negative.
    assert mse >= 0                                    # Verify MSE cannot be negative.
    assert rmse >= 0                                   # Verify RMSE cannot be negative.
    assert r2 <= 1                                     # Verify R² score is valid (cannot exceed 1).


# The following test checks:
# - If perfect predictions yield perfect metrics (0 errors, 1.0 R²).
#
# This test will fail if:
# - The function handles zero variance edge-cases poorly.
def test_evaluate_regression_perfect_prediction():
    """SCA-UT-031"""
    y_true = np.array([1, 2, 3, 4, 5])
    mae, mse, rmse, r2 = evaluate_regression(y_true, y_true)

    assert mae == 0                                    # Verify perfect prediction yields 0 MAE.
    assert mse == 0                                    # Verify perfect prediction yields 0 MSE.
    assert rmse == 0                                   # Verify perfect prediction yields 0 RMSE.
    assert r2 == 1                                     # Verify perfect prediction yields 1.0 R².


# The following test checks:
# - If the comparison table between RF and MLP is generated correctly.
#
# This test will fail if:
# - The DataFrame structure or column names are incorrect.
def test_compare_models():
    """SCA-UT-032"""
    df = prepare_dataset()
    X = select_features(df, ["PU", "PEU", "FSC", "SP", "TP", "IB"])
    y = select_target(df, "AUB")
    X_train, X_test, y_train, y_test = split_dataset(X, y, test_size=0.20, random_state=1)

    # Random Forest
    rf_grid = create_rf_gridsearch(create_rf_param_grid(), create_kfold(), n_jobs=1)
    rf_fitted = fit_rf_gridsearch(rf_grid, X_train, y_train)
    rf_model = get_best_rf_model(rf_fitted)
    rf_predictions = predict_rf(rf_model, X_test)
    rf_mae, rf_mse, rf_rmse, rf_r2 = evaluate_regression(y_test, rf_predictions)

    # MLP
    scaler, X_train_scaled, X_test_scaled = standardize_features(X_train, X_test)
    mlp_grid = create_mlp_gridsearch(create_mlp_param_grid(), create_kfold(), n_jobs=1)
    mlp_fitted = fit_mlp_gridsearch(mlp_grid, X_train_scaled, y_train)
    mlp_model = get_best_mlp_model(mlp_fitted)
    mlp_predictions = predict_mlp(mlp_model, X_test_scaled)
    mlp_mae, mlp_mse, mlp_rmse, mlp_r2 = evaluate_regression(y_test, mlp_predictions)

    comparison = compare_models(rf_mae, rf_mse, rf_rmse, rf_r2, mlp_mae, mlp_mse, mlp_rmse, mlp_r2)

    assert isinstance(comparison, pd.DataFrame)        # Verify the comparison is returned as a DataFrame.
    assert comparison.shape == (2, 5)                  # Verify the shape corresponds to 2 models and 5 metric columns.
    assert comparison.columns.tolist() == ["Model", "MAE", "MSE", "RMSE", "R²"] # Verify exact column headers.
    assert comparison["Model"].tolist() == ["Random Forest", "MLP"]             # Verify exact model labels.