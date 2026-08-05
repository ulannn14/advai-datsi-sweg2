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


# ==========================================================
# HELPER FUNCTION
# ==========================================================

def prepare_dataset():
    """
    Reproduce the preprocessing performed in the notebook.
    """

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


# ==========================================================
# MODULE 1 - FEATURE SELECTION
# ==========================================================

def test_select_features():
    """SCA-UT-009"""

    df = prepare_dataset()

    X = select_features(
        df,
        ["PU", "PEU", "FSC", "SP", "TP", "IB"]
    )

    expected_columns = [
        "PU",
        "PEU",
        "FSC",
        "SP",
        "TP",
        "IB"
    ]

    print("Selected feature columns:")
    print(X.columns.tolist())

    assert isinstance(X, pd.DataFrame)

    assert X.columns.tolist() == expected_columns, \
        "Feature columns do not exactly match the expected columns."

    assert X.shape == (757, 6)


def test_select_features_invalid_column():
    """SCA-UT-010"""

    df = prepare_dataset()

    with pytest.raises(ValueError):

        select_features(
            df,
            ["PU", "INVALID"]
        )


# ==========================================================
# MODULE 2 - TARGET VARIABLE
# ==========================================================

def test_select_target():
    """SCA-UT-011"""

    df = prepare_dataset()

    y = select_target(df, "AUB")

    print("Target variable:")
    print(y.name)

    assert isinstance(y, pd.Series)

    assert y.name == "AUB"

    assert len(y) == 757


def test_select_target_invalid():
    """SCA-UT-012"""

    df = prepare_dataset()

    with pytest.raises(ValueError):

        select_target(df, "INVALID")


# ==========================================================
# MODULE 3 - TRAIN TEST SPLIT
# ==========================================================

def test_split_dataset():
    """SCA-UT-013"""

    df = prepare_dataset()

    X = select_features(
        df,
        ["PU", "PEU", "FSC", "SP", "TP", "IB"]
    )

    y = select_target(df, "AUB")

    X_train, X_test, y_train, y_test = split_dataset(
        X,
        y,
        test_size=0.20,
        random_state=1
    )

    print("Training shape:", X_train.shape)
    print("Testing shape:", X_test.shape)

    assert X_train.shape == (605, 6)

    assert X_test.shape == (152, 6)

    assert len(y_train) == 605

    assert len(y_test) == 152


# ==========================================================
# MODULE 4 - KFOLD
# ==========================================================

def test_create_kfold():
    """SCA-UT-014"""

    kf = create_kfold()

    print("Number of folds:", kf.get_n_splits())

    assert kf.get_n_splits() == 10

    assert kf.shuffle is True

    assert kf.random_state == 1


# ==========================================================
# MODULE 5 - STANDARDIZATION
# ==========================================================

def test_standardize_features():
    """SCA-UT-015"""

    df = prepare_dataset()

    X = select_features(
        df,
        ["PU", "PEU", "FSC", "SP", "TP", "IB"]
    )

    y = select_target(df, "AUB")

    X_train, X_test, _, _ = split_dataset(
        X,
        y
    )

    scaler, X_train_scaled, X_test_scaled = standardize_features(
        X_train,
        X_test
    )

    print("Scaled training shape:", X_train_scaled.shape)
    print("Scaled testing shape:", X_test_scaled.shape)

    assert X_train_scaled.shape == X_train.shape

    assert X_test_scaled.shape == X_test.shape

    np.testing.assert_allclose(
        X_train_scaled.mean(axis=0),
        np.zeros(X_train.shape[1]),
        atol=1e-7
    )

    np.testing.assert_allclose(
        X_train_scaled.std(axis=0),
        np.ones(X_train.shape[1]),
        atol=1e-7
    )

# ==========================================================
# MODULE 4 - FEATURE SCALING
# ==========================================================

def test_standardize_features():
    """SCA-UT-012"""

    X_train = pd.DataFrame({
        "A": [1, 2, 3, 4],
        "B": [10, 20, 30, 40]
    })

    X_test = pd.DataFrame({
        "A": [5, 6],
        "B": [50, 60]
    })

    scaler, X_train_scaled, X_test_scaled = standardize_features(
        X_train,
        X_test
    )

    print("Training data standardized successfully.")
    print("Scaled training shape:", X_train_scaled.shape)
    print("Scaled testing shape:", X_test_scaled.shape)

    assert X_train_scaled.shape == (4, 2)
    assert X_test_scaled.shape == (2, 2)

    np.testing.assert_allclose(
        X_train_scaled.mean(axis=0),
        [0, 0],
        atol=1e-7
    )


# ==========================================================
# MODULE 5 - RANDOM FOREST
# ==========================================================

def test_create_rf_param_grid():
    """SCA-UT-013"""

    grid = create_rf_param_grid()

    print(grid)

    expected_keys = {
        "n_estimators",
        "max_features",
        "max_depth",
        "min_samples_split",
        "min_samples_leaf"
    }

    assert set(grid.keys()) == expected_keys

    assert grid["n_estimators"] == [100, 200, 300]
    assert grid["max_features"] == [2, 3, 6]
    assert grid["max_depth"] == [None, 10, 20]
    assert grid["min_samples_split"] == [2, 5]
    assert grid["min_samples_leaf"] == [1, 2, 5]


def test_create_rf_gridsearch():
    """SCA-UT-014"""

    param_grid = create_rf_param_grid()

    kf = create_kfold()

    grid = create_rf_gridsearch(
        param_grid,
        kf
    )

    print(grid)

    assert isinstance(grid, GridSearchCV)
    assert isinstance(grid.estimator, RandomForestRegressor)


def test_fit_rf_gridsearch():
    """SCA-UT-015"""

    X, y = make_regression(
        n_samples=80,
        n_features=6,
        noise=0.1,
        random_state=1
    )

    param_grid = {
        "n_estimators": [10],
        "max_features": [2],
        "max_depth": [5],
        "min_samples_split": [2],
        "min_samples_leaf": [1]
    }

    kf = create_kfold()

    grid = create_rf_gridsearch(
        param_grid,
        kf,
        n_jobs=1
    )

    fitted = fit_rf_gridsearch(
        grid,
        X,
        y
    )

    print("RF GridSearch fitted.")

    assert hasattr(fitted, "best_estimator_")


def test_get_best_rf_params():
    """SCA-UT-016"""

    X, y = make_regression(
        n_samples=80,
        n_features=6,
        random_state=1
    )

    param_grid = {
        "n_estimators": [10],
        "max_features": [2],
        "max_depth": [5],
        "min_samples_split": [2],
        "min_samples_leaf": [1]
    }

    grid = create_rf_gridsearch(
        param_grid,
        create_kfold(),
        n_jobs=1
    )

    fit_rf_gridsearch(
        grid,
        X,
        y
    )

    params, mse = get_best_rf_params(grid)

    print(params)
    print(mse)

    assert isinstance(params, dict)
    assert isinstance(mse, float)
    assert mse >= 0


def test_summarize_rf_results():
    """SCA-UT-017"""

    X, y = make_regression(
        n_samples=80,
        n_features=6,
        random_state=1
    )

    param_grid = {
        "n_estimators": [10],
        "max_features": [2],
        "max_depth": [5],
        "min_samples_split": [2],
        "min_samples_leaf": [1]
    }

    grid = create_rf_gridsearch(
        param_grid,
        create_kfold(),
        n_jobs=1
    )

    fit_rf_gridsearch(
        grid,
        X,
        y
    )

    summary = summarize_rf_results(grid)

    print(summary.head())

    expected_columns = [
        "param_n_estimators",
        "param_max_features",
        "param_max_depth",
        "param_min_samples_split",
        "param_min_samples_leaf",
        "mean_test_score",
        "rank_test_score"
    ]

    for col in expected_columns:
        assert col in summary.columns


def test_get_best_rf_model():
    """SCA-UT-018"""

    X, y = make_regression(
        n_samples=80,
        n_features=6,
        random_state=1
    )

    param_grid = {
        "n_estimators": [10],
        "max_features": [2],
        "max_depth": [5],
        "min_samples_split": [2],
        "min_samples_leaf": [1]
    }

    grid = create_rf_gridsearch(
        param_grid,
        create_kfold(),
        n_jobs=1
    )

    fit_rf_gridsearch(
        grid,
        X,
        y
    )

    model = get_best_rf_model(grid)

    print(model)

    assert isinstance(model, RandomForestRegressor)


def test_predict_rf():
    """SCA-UT-019"""

    X, y = make_regression(
        n_samples=80,
        n_features=6,
        random_state=1
    )

    model = RandomForestRegressor(
        random_state=1
    )

    model.fit(X, y)

    predictions = predict_rf(
        model,
        X
    )

    print(predictions[:5])

    assert len(predictions) == len(X)

# ==========================================================
# MODULE 8 - MLP REGRESSOR
# ==========================================================

def test_create_mlp_param_grid():
    """SCA-UT-020"""

    param_grid = create_mlp_param_grid()

    print("MLP parameter grid:")
    print(param_grid)

    expected_keys = {
        "hidden_layer_sizes",
        "activation",
        "learning_rate_init",
        "alpha"
    }

    assert isinstance(param_grid, dict)
    assert set(param_grid.keys()) == expected_keys

    assert param_grid["hidden_layer_sizes"] == [
        (50,),
        (100,),
        (100, 50)
    ]

    assert param_grid["activation"] == [
        "relu",
        "tanh"
    ]

    assert param_grid["learning_rate_init"] == [
        0.001,
        0.01
    ]

    assert param_grid["alpha"] == [
        0.0001,
        0.001
    ]


def test_create_mlp_gridsearch():
    """SCA-UT-021"""

    param_grid = create_mlp_param_grid()
    kf = create_kfold()

    grid = create_mlp_gridsearch(
        param_grid,
        kf
    )

    print("MLP GridSearchCV successfully created.")

    assert isinstance(grid, GridSearchCV)

    assert grid.scoring == "neg_mean_squared_error"
    assert grid.cv == kf
    assert grid.n_jobs == -1


def test_fit_mlp_gridsearch():
    """SCA-UT-022"""

    df = prepare_dataset()

    feature_columns = [
        "PU",
        "PEU",
        "FSC",
        "SP",
        "TP",
        "IB"
    ]

    X = select_features(df, feature_columns)
    y = select_target(df, "AUB")

    X_train, X_test, y_train, y_test = split_dataset(
        X,
        y
    )

    scaler, X_train_scaled, X_test_scaled = standardize_features(
        X_train,
        X_test
    )

    param_grid = create_mlp_param_grid()
    kf = create_kfold()

    grid = create_mlp_gridsearch(
        param_grid,
        kf,
        n_jobs=1
    )

    fitted = fit_mlp_gridsearch(
        grid,
        X_train_scaled,
        y_train
    )

    print("Best MLP parameters:")
    print(fitted.best_params_)

    assert hasattr(fitted, "best_params_")
    assert hasattr(fitted, "best_estimator_")

# ==========================================================
# MODULE 9 - MLP MODEL
# ==========================================================

def test_get_best_mlp_params():
    """SCA-UT-023"""

    df = prepare_dataset()

    feature_columns = [
        "PU",
        "PEU",
        "FSC",
        "SP",
        "TP",
        "IB"
    ]

    X = select_features(df, feature_columns)
    y = select_target(df, "AUB")

    X_train, X_test, y_train, y_test = split_dataset(
        X,
        y
    )

    scaler, X_train_scaled, X_test_scaled = standardize_features(
        X_train,
        X_test
    )

    grid = create_mlp_gridsearch(
        create_mlp_param_grid(),
        create_kfold(),
        n_jobs=1
    )

    fitted = fit_mlp_gridsearch(
        grid,
        X_train_scaled,
        y_train
    )

    params = get_best_mlp_params(fitted)

    print(params)

    assert isinstance(params, dict)

    expected = {
        "hidden_layer_sizes",
        "activation",
        "learning_rate_init",
        "alpha"
    }

    assert set(params.keys()) == expected


def test_get_best_mlp_cv_mse():
    """SCA-UT-024"""

    df = prepare_dataset()

    feature_columns = [
        "PU",
        "PEU",
        "FSC",
        "SP",
        "TP",
        "IB"
    ]

    X = select_features(df, feature_columns)
    y = select_target(df, "AUB")

    X_train, X_test, y_train, y_test = split_dataset(
        X,
        y
    )

    scaler, X_train_scaled, X_test_scaled = standardize_features(
        X_train,
        X_test
    )

    grid = create_mlp_gridsearch(
        create_mlp_param_grid(),
        create_kfold(),
        n_jobs=1
    )

    fitted = fit_mlp_gridsearch(
        grid,
        X_train_scaled,
        y_train
    )

    mse = get_best_mlp_cv_mse(fitted)

    print("Best CV MSE:", mse)

    assert isinstance(mse, float)
    assert mse >= 0


def test_get_best_mlp_model():
    """SCA-UT-025"""

    df = prepare_dataset()

    feature_columns = [
        "PU",
        "PEU",
        "FSC",
        "SP",
        "TP",
        "IB"
    ]

    X = select_features(df, feature_columns)
    y = select_target(df, "AUB")

    X_train, X_test, y_train, y_test = split_dataset(
        X,
        y
    )

    scaler, X_train_scaled, X_test_scaled = standardize_features(
        X_train,
        X_test
    )

    grid = create_mlp_gridsearch(
        create_mlp_param_grid(),
        create_kfold(),
        n_jobs=1
    )

    fitted = fit_mlp_gridsearch(
        grid,
        X_train_scaled,
        y_train
    )

    model = get_best_mlp_model(fitted)

    print(model)

    assert isinstance(model, MLPRegressor)


def test_predict_mlp():
    """SCA-UT-026"""

    df = prepare_dataset()

    feature_columns = [
        "PU",
        "PEU",
        "FSC",
        "SP",
        "TP",
        "IB"
    ]

    X = select_features(df, feature_columns)
    y = select_target(df, "AUB")

    X_train, X_test, y_train, y_test = split_dataset(
        X,
        y
    )

    scaler, X_train_scaled, X_test_scaled = standardize_features(
        X_train,
        X_test
    )

    grid = create_mlp_gridsearch(
        create_mlp_param_grid(),
        create_kfold(),
        n_jobs=1
    )

    fitted = fit_mlp_gridsearch(
        grid,
        X_train_scaled,
        y_train
    )

    model = get_best_mlp_model(fitted)

    predictions = predict_mlp(
        model,
        X_test_scaled
    )

    print(predictions[:5])

    assert isinstance(predictions, np.ndarray)
    assert len(predictions) == len(y_test)


# ==========================================================
# MODULE 10 - MODEL EVALUATION
# ==========================================================

def test_evaluate_regression():
    """SCA-UT-027"""
    y_true = np.array([1, 2, 3, 4, 5])
    y_pred = np.array([1.1, 2.1, 2.9, 3.8, 5.2])

    mae, mse, rmse, r2 = evaluate_regression(y_true, y_pred)

    assert mae >= 0
    assert mse >= 0
    assert rmse >= 0
    assert r2 <= 1


def test_evaluate_regression_perfect_prediction():
    """SCA-UT-028"""
    y_true = np.array([1, 2, 3, 4, 5])

    mae, mse, rmse, r2 = evaluate_regression(y_true, y_true)

    assert mae == 0
    assert mse == 0
    assert rmse == 0
    assert r2 == 1


# ==========================================================
# MODULE 11 - MODEL COMPARISON
# ==========================================================

def test_compare_models():
    """SCA-UT-029"""
    comparison = compare_models(
        0.20, 0.10, 0.32, 0.80,  # rf_mae, rf_mse, rf_rmse, rf_r2
        0.25, 0.15, 0.39, 0.72   # mlp_mae, mlp_mse, mlp_rmse, mlp_r2
    )

    assert isinstance(comparison, pd.DataFrame)
    assert comparison.shape == (2, 5)

    expected_columns = ["Model", "MAE", "MSE", "RMSE", "R²"]
    assert comparison.columns.tolist() == expected_columns
    assert comparison["Model"].tolist() == ["Random Forest", "MLP"]