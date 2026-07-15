"""
preprocessing.py

Wrapper functions for dataset loading, validation,
data cleaning, and feature engineering.

Project:
Social Commerce Adoption Among Generation Z University Students in Vietnam
"""

import pandas as pd


# ==========================================================
# MODULE 1 - DATASET LOADING
# ==========================================================

def load_dataset(file_path):
    """
    Load the dataset from a CSV file.

    Parameters
    ----------
    file_path : str
        Path to the CSV dataset.

    Returns
    -------
    pandas.DataFrame
        Loaded dataset.

    Raises
    ------
    FileNotFoundError
        If the specified dataset does not exist.
    """

    try:
        df = pd.read_csv(file_path)
        return df

    except FileNotFoundError:
        raise FileNotFoundError(
            f"Dataset '{file_path}' does not exist."
        )


# ==========================================================
# MODULE 2 - DATASET VALIDATION
# ==========================================================

def inspect_dataset(df):
    """
    Return the dimensions of the dataset.

    Parameters
    ----------
    df : pandas.DataFrame

    Returns
    -------
    tuple
        (number_of_rows, number_of_columns)
    """

    return df.shape


def validate_columns(df):
    """
    Return all column names.

    Parameters
    ----------
    df : pandas.DataFrame

    Returns
    -------
    list
        List of column names.
    """

    return df.columns.tolist()


def validate_dtypes(df):
    """
    Return the data types of all columns.

    Parameters
    ----------
    df : pandas.DataFrame

    Returns
    -------
    pandas.Series
        Data types of each column.
    """

    return df.dtypes


# ==========================================================
# MODULE 3 - DATA CLEANING
# ==========================================================

def drop_columns(df, columns):
    return df
    """
    Remove one or more columns from the dataset.

    Parameters
    ----------
    df : pandas.DataFrame

    columns : list
        Columns to remove.

    Returns
    -------
    pandas.DataFrame
    
    
    missing_columns = [col for col in columns if col not in df.columns]

    if missing_columns:
        raise ValueError(
            f"Column(s) not found: {missing_columns}"
        )

    return df.drop(columns=columns)
    """


def check_missing_values(df):
    """
    Count missing values for each column.

    Parameters
    ----------
    df : pandas.DataFrame

    Returns
    -------
    pandas.Series
    """

    return df.isnull().sum()


def find_duplicates(df):
    """
    Return duplicate rows.

    Parameters
    ----------
    df : pandas.DataFrame

    Returns
    -------
    pandas.DataFrame
    """

    return df[df.duplicated()]


def validate_unique_values(df):
    """
    Summarize the unique values of every variable.

    Parameters
    ----------
    df : pandas.DataFrame

    Returns
    -------
    pandas.DataFrame
    """

    summary = pd.DataFrame({
        "Variable": df.columns,
        "Unique Count": [df[col].nunique() for col in df.columns],
        "Unique Values": [sorted(df[col].unique().tolist()) for col in df.columns]
    })

    return summary


# ==========================================================
# MODULE 4 - FEATURE ENGINEERING
# ==========================================================

def compute_composite_score(df, columns, new_column):
    """
    Compute the composite score (mean) of questionnaire items.

    Parameters
    ----------
    df : pandas.DataFrame

    columns : list
        Questionnaire item columns.

    new_column : str
        Name of the composite score column.

    Returns
    -------
    pandas.DataFrame
    """

    missing_columns = [col for col in columns if col not in df.columns]

    if missing_columns:
        raise ValueError(
            f"Column(s) not found: {missing_columns}"
        )

    df[new_column] = df[columns].mean(axis=1)

    return df