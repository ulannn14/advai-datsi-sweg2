"""
preprocessing.py

Wrapper functions for dataset loading, validation,
data cleaning, and feature engineering.

Project:
Social Commerce Adoption Among Generation Z University Students in Vietnam
"""
import pandas as pd
from scipy import stats

GENDER_LABELS = {1: "Male", 2: "Female", 3: "Different"}

INCOME_LABELS = {
    1: "< $100",
    2: "$100 - $200",
    3: "$200 - $300",
    4: "$300 - $400",
    5: "> $400",
}

AREA_LABELS = {1: "Urban", 2: "Suburban", 3: "Rural"}

FREQUENCY_LABELS = {1: "Daily", 2: "Weekly", 3: "Monthly", 4: "Rarely Used"}

CONSTRUCTS = ["PU", "PEU", "FSC", "SP", "TP", "IB", "AUB"]

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

    cols = df.columns.tolist()
    df.columns = cols

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
    """

    missing_columns = [col for col in columns if col not in df.columns]

    if missing_columns:
        raise ValueError(
            f"Column(s) not found: {missing_columns}"
        )

    return df.drop(columns=columns)


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

# ==========================================================
# MODULE 5 - EXPLORATORY DATA ANALYSIS (EDA)
# ==========================================================

def get_gender_distribution(df):
    """Counts of respondents per Gender category, labeled."""
    counts = df["Gender"].value_counts().sort_index()
    counts.index = counts.index.map(GENDER_LABELS)
    return counts


def get_income_distribution(df):
    """Counts of respondents per Income category, labeled."""
    counts = df["Income"].value_counts().sort_index()
    counts.index = counts.index.map(INCOME_LABELS)
    return counts


def get_area_distribution(df):
    """Counts of respondents per residential Area, labeled."""
    counts = df["Area"].value_counts().sort_index()
    counts.index = counts.index.map(AREA_LABELS)
    return counts


def get_frequency_distribution(df):
    """Counts of respondents per social media usage Frequency, labeled."""
    counts = df["Frequently"].value_counts().sort_index()
    counts.index = counts.index.map(FREQUENCY_LABELS)
    return counts


# CONSTRUCT DESCRIPTIVES / CORRELATIONS
def get_construct_descriptives(df, constructs=CONSTRUCTS):
    """Descriptive statistics (count/mean/std/etc.) for the composite constructs."""
    return df[constructs].describe()


def get_construct_correlation_matrix(df, constructs=CONSTRUCTS):
    """Pearson correlation matrix between the composite constructs."""
    return df[constructs].corr(method="pearson")


# AUB ACROSS DEMOGRAPHIC GROUPS
def get_aub_by_gender_summary(df, exclude_other=True):
    """AUB count/mean/median/std by gender (Male vs Female only, by default)."""
    data = df[df["Gender"] != 3] if exclude_other else df

    summary = data.groupby("Gender")["AUB"].agg(["count", "mean", "median", "std"])
    summary.index = summary.index.map({1: "Male", 2: "Female"})

    return summary


def run_ttest_aub_gender(df):
    """Welch's t-test comparing AUB between male and female respondents."""
    data = df[df["Gender"] != 3]

    male = data[data["Gender"] == 1]["AUB"]
    female = data[data["Gender"] == 2]["AUB"]

    t_stat, p_value = stats.ttest_ind(male, female, equal_var=False)

    return t_stat, p_value


def get_aub_by_area_summary(df):
    """AUB count/mean/median/std by residential area."""
    summary = df.groupby("Area")["AUB"].agg(["count", "mean", "median", "std"])
    summary.index = summary.index.map(AREA_LABELS)

    return summary


def run_anova_aub_area(df):
    """One-way ANOVA comparing AUB across Urban/Suburban/Rural respondents."""
    urban = df[df["Area"] == 1]["AUB"]
    suburban = df[df["Area"] == 2]["AUB"]
    rural = df[df["Area"] == 3]["AUB"]

    f_stat, p_value = stats.f_oneway(urban, suburban, rural)

    return f_stat, p_value


def get_aub_by_frequency_summary(df):
    """AUB count/mean/median/std by social media usage frequency."""
    summary = df.groupby("Frequently")["AUB"].agg(["count", "mean", "median", "std"])
    summary.index = summary.index.map({1: "Daily", 2: "Weekly", 3: "Monthly", 4: "Rarely Used"})

    return summary


# ==========================================================
# COMPLETE PREPROCESSING PIPELINE FOR 02 AND 03
# ==========================================================
def run_preprocessing(
    file_path="../datasets/S-COMMERCE_GenZ_UniversityStudent_757_DIB.csv"
):
    """
    Execute the complete preprocessing pipeline.

    Parameters
    ----------
    file_path : str
        Path to the dataset.

    Returns
    -------
    pandas.DataFrame
        Fully preprocessed dataset.
    """

    # Load dataset
    df = load_dataset(file_path)

    # Remove unnecessary variables
    df = drop_columns(df, ["PEU4", "Job"])

    # Create composite scores
    composite_scores = {
        "PU": ["PU1", "PU2", "PU3", "PU4"],
        "PEU": ["PEU1", "PEU2", "PEU3"],
        "FSC": ["FSC1", "FSC2", "FSC3"],
        "SP": ["SP1", "SP2", "SP3", "SP4"],
        "TP": ["TP1", "TP2", "TP3"],
        "IB": ["IB1", "IB2", "IB3"],
        "AUB": ["AUB1", "AUB2", "AUB3"],
    }

    for construct, columns in composite_scores.items():
        df = compute_composite_score(
            df,
            columns,
            construct
        )

    return df