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
    get_construct_descriptives,
    get_construct_correlation_matrix,
)

from ml.scripts.machinelearning import (
    select_features,
    select_target,
    split_dataset,
    create_kfold,
    standardize_features,

    get_construct_descriptives,
    get_construct_correlation_matrix,
    get_aub_by_gender_summary,
    run_ttest_aub_gender,
    get_aub_by_area_summary,
    run_anova_aub_area,
    get_aub_by_frequency_summary,

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
    run_ttest_aub_gender, 
    get_aub_by_gender_summary,
    run_anova_aub_area,
    get_aub_by_area_summary,
    get_aub_by_frequency_summary
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
# MODULE 4 - FEATURE SCALING
# ==========================================================

def test_standardize_features():
    """SCA-UT-015"""

    df = prepare_dataset()

    X = select_features(
        df,
        ["PU", "PEU", "FSC", "SP", "TP", "IB"]
    )

    y = select_target(df, "AUB")

    # Split the actual dataset
    X_train, X_test, _, _ = split_dataset(
        X,
        y
    )

    # Standardize the actual training/testing data
    scaler, X_train_scaled, X_test_scaled = standardize_features(
        X_train,
        X_test
    )

    print("\nScaled first 5 training rows:")
    print(X_train_scaled[:5])

    print("\nScaled first 5 testing rows:")
    print(X_test_scaled[:5])

    # Existing tests
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
# MODULE 5 - RANDOM FOREST
# ==========================================================

def test_create_rf_param_grid():
    """SCA-UT-016"""

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
    """SCA-UT-017"""

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
    """SCA-UT-018"""

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

    fitted = fit_rf_gridsearch(
        grid,
        X_train,
        y_train
    )

    print("RF GridSearch fitted.")
    print("Best parameters:", fitted.best_params_)

    assert hasattr(fitted, "best_estimator_")
    assert hasattr(fitted, "best_params_")
    assert hasattr(fitted, "best_score_")


def test_get_best_rf_params():
    """SCA-UT-019"""

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
    """SCA-UT-020"""

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
    """SCA-UT-021"""

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
    """SCA-UT-022"""

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

    model = RandomForestRegressor(
        n_estimators=10,
        random_state=1
    )

    model.fit(
        X_train,
        y_train
    )

    predictions = predict_rf(
        model,
        X_test
    )

    print("First 5 RF predictions:")
    print(predictions[:5])

    assert isinstance(predictions, np.ndarray)
    assert len(predictions) == len(y_test)
    assert np.isfinite(predictions).all()

# ==========================================================
# MODULE 8 - MLP REGRESSOR
# ==========================================================

def test_create_mlp_param_grid():
    """SCA-UT-023"""

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
    """SCA-UT-024"""

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
    """SCA-UT-027"""

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
    """SCA-UT-028"""

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
    """SCA-UT-029"""

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
    """SCA-UT-030"""

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

    model = RandomForestRegressor(
        n_estimators=10,
        random_state=1
    )

    model.fit(
        X_train,
        y_train
    )

    predictions = predict_rf(
        model,
        X_test
    )

    mae, mse, rmse, r2 = evaluate_regression(
        y_test,
        predictions
    )

    print("MAE:", mae)
    print("MSE:", mse)
    print("RMSE:", rmse)
    print("R²:", r2)

    assert np.isfinite(mae)
    assert np.isfinite(mse)
    assert np.isfinite(rmse)
    assert np.isfinite(r2)

    assert mae >= 0
    assert mse >= 0
    assert rmse >= 0
    assert r2 <= 1


def test_evaluate_regression_perfect_prediction():
    """SCA-UT-031"""
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
    """SCA-UT-032"""

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
        y,
        test_size=0.20,
        random_state=1
    )

    # -----------------------------
    # Random Forest
    # -----------------------------

    rf_grid = create_rf_gridsearch(
        create_rf_param_grid(),
        create_kfold(),
        n_jobs=1
    )

    rf_fitted = fit_rf_gridsearch(
        rf_grid,
        X_train,
        y_train
    )

    rf_model = get_best_rf_model(rf_fitted)

    rf_predictions = predict_rf(
        rf_model,
        X_test
    )

    rf_mae, rf_mse, rf_rmse, rf_r2 = evaluate_regression(
        y_test,
        rf_predictions
    )

    # -----------------------------
    # MLP
    # -----------------------------

    scaler, X_train_scaled, X_test_scaled = standardize_features(
        X_train,
        X_test
    )

    mlp_grid = create_mlp_gridsearch(
        create_mlp_param_grid(),
        create_kfold(),
        n_jobs=1
    )

    mlp_fitted = fit_mlp_gridsearch(
        mlp_grid,
        X_train_scaled,
        y_train
    )

    mlp_model = get_best_mlp_model(
        mlp_fitted
    )

    mlp_predictions = predict_mlp(
        mlp_model,
        X_test_scaled
    )

    mlp_mae, mlp_mse, mlp_rmse, mlp_r2 = evaluate_regression(
        y_test,
        mlp_predictions
    )

    # -----------------------------
    # Compare
    # -----------------------------

    comparison = compare_models(
        rf_mae,
        rf_mse,
        rf_rmse,
        rf_r2,
        mlp_mae,
        mlp_mse,
        mlp_rmse,
        mlp_r2
    )

    print(comparison)

    assert isinstance(comparison, pd.DataFrame)
    assert comparison.shape == (2, 5)

    expected_columns = [
        "Model",
        "MAE",
        "MSE",
        "RMSE",
        "R²"
    ]

    assert comparison.columns.tolist() == expected_columns
    assert comparison["Model"].tolist() == ["Random Forest", "MLP"]

# ==========================================================
# EDA UNIT TESTS
# ==========================================================

def test_get_construct_descriptives():
    """SCA-UT-033"""

    df = prepare_dataset()

    result = get_construct_descriptives(df)

    expected_constructs = [
        "PU",
        "PEU",
        "FSC",
        "SP",
        "TP",
        "IB",
        "AUB"
    ]

    assert result.columns.tolist() == expected_constructs

    assert result.loc["count"].tolist() == [757] * 7

    assert result.loc["mean", "PU"] == pytest.approx(
        3.77, abs=0.01
    )

    assert result.loc["mean", "PEU"] == pytest.approx(
        3.70, abs=0.01
    )

    assert result.loc["mean", "FSC"] == pytest.approx(
        3.73, abs=0.01
    )

    assert result.loc["mean", "SP"] == pytest.approx(
        3.60, abs=0.01
    )

    assert result.loc["mean", "TP"] == pytest.approx(
        3.56, abs=0.01
    )

    assert result.loc["mean", "IB"] == pytest.approx(
        3.61, abs=0.01
    )

    assert result.loc["mean", "AUB"] == pytest.approx(
        3.68, abs=0.01
    )

def test_get_construct_correlation_matrix():
    """SCA-UT-034"""

    df = prepare_dataset()

    corr = get_construct_correlation_matrix(df)

    expected_constructs = [
        "PU",
        "PEU",
        "FSC",
        "SP",
        "TP",
        "IB",
        "AUB"
    ]

    assert corr.index.tolist() == expected_constructs
    assert corr.columns.tolist() == expected_constructs

    assert corr.loc["PU", "PEU"] == pytest.approx(
        0.788226, abs=0.001
    )

    assert corr.loc["PU", "FSC"] == pytest.approx(
        0.804070, abs=0.001
    )

    assert corr.loc["PEU", "AUB"] == pytest.approx(
        0.785858, abs=0.001
    )

    assert corr.loc["SP", "IB"] == pytest.approx(
        0.644206, abs=0.001
    )

    assert corr.loc["TP", "AUB"] == pytest.approx(
        0.662269, abs=0.001
    )

    assert corr.loc["IB", "AUB"] == pytest.approx(
        0.683213, abs=0.001
    )

    # Diagonal should always be 1
    for construct in corr.columns:
        assert corr.loc[construct, construct] == pytest.approx(1.0)

def test_get_aub_by_gender_summary():
    """SCA-UT-035"""

    df = prepare_dataset()

    result = get_aub_by_gender_summary(df)

    assert result.loc["Male", "count"] == 167
    assert result.loc["Female", "count"] == 588

    assert result.loc["Male", "mean"] == pytest.approx(
        3.66, abs=0.01
    )

    assert result.loc["Female", "mean"] == pytest.approx(
        3.69, abs=0.01
    )

    assert result.loc["Male", "median"] == pytest.approx(
        3.75
    )

    assert result.loc["Female", "median"] == pytest.approx(
        4.00
    )

    assert result.loc["Male", "std"] == pytest.approx(
        0.79, abs=0.01
    )

    assert result.loc["Female", "std"] == pytest.approx(
        0.67, abs=0.01
    )

def test_run_ttest_aub_gender():
    """SCA-UT-036"""

    df = prepare_dataset()

    t_stat, p_value = run_ttest_aub_gender(df)

    assert t_stat == pytest.approx(
        -0.417, abs=0.01
    )

    assert p_value == pytest.approx(
        0.677, abs=0.01
    )

    assert p_value > 0.05

def test_get_aub_by_area_summary():
    """SCA-UT-037"""

    df = prepare_dataset()

    result = get_aub_by_area_summary(df)

    assert result.loc["Urban", "count"] == 461
    assert result.loc["Suburban", "count"] == 74
    assert result.loc["Rural", "count"] == 222

    assert result.loc["Urban", "mean"] == pytest.approx(
        3.71, abs=0.01
    )

    assert result.loc["Suburban", "mean"] == pytest.approx(
        3.64, abs=0.01
    )

    assert result.loc["Rural", "mean"] == pytest.approx(
        3.62, abs=0.01
    )

    assert result.loc["Urban", "median"] == pytest.approx(
        4.00
    )

    assert result.loc["Suburban", "median"] == pytest.approx(
        3.75
    )

    assert result.loc["Rural", "median"] == pytest.approx(
        3.75
    )

def test_run_anova_aub_area():
    """SCA-UT-038"""

    df = prepare_dataset()

    f_stat, p_value = run_anova_aub_area(df)

    assert f_stat == pytest.approx(
        1.4681, abs=0.01
    )

    assert p_value == pytest.approx(
        0.2310, abs=0.01
    )

    assert p_value > 0.05

def test_get_aub_by_frequency_summary():
    """SCA-UT-039"""

    df = prepare_dataset()

    result = get_aub_by_frequency_summary(df)

    assert result.loc["Daily", "count"] == 725
    assert result.loc["Weekly", "count"] == 14
    assert result.loc["Monthly", "count"] == 7
    assert result.loc["Rarely", "count"] == 11