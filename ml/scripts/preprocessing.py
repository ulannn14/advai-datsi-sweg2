"""
preprocessing.py

Wrapper functions for dataset loading, validation,
data cleaning, and feature engineering.

Project:
Social Commerce Adoption Among Generation Z University Students in Vietnam
"""
import pandas as pd
from scipy import stats

# Labels and their corresponding values as defined from the study itself
 
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


# ============================= MODULE 1 - DATASET LOADING =============================
"""Load the dataset from a CSV file."""
def load_dataset(file_path):
    try:
        df = pd.read_csv(file_path)
        return df

    except FileNotFoundError:
        raise FileNotFoundError(
            f"Dataset '{file_path}' does not exist."
        )


# ============================= MODULE 2 - DATASET VALIDATION =============================

# Return the dataset's (rows, columns) shape.
def inspect_dataset(df):
    return df.shape

# Return all column names.
def validate_columns(df):
    cols = df.columns.tolist()
    df.columns = cols

    return df.columns.tolist()

# Return the data type of each column.
def validate_dtypes(df):
        return df.dtypes


# ============================= MODULE 3 - DATA CLEANING =============================

# Remove one or more columns from the dataset.
def drop_columns(df, columns):   
    missing_columns = [col for col in columns if col not in df.columns]

    if missing_columns:
        raise ValueError(
            f"Column(s) not found: {missing_columns}"
        )

    return df.drop(columns=columns)

# Count missing values in each column. 
def check_missing_values(df):
    return df.isnull().sum()

# Return duplicate rows.
def find_duplicates(df):
    return df[df.duplicated()]

# Summarize unique values for every column.
def validate_unique_values(df):
    summary = pd.DataFrame({
        "Variable": df.columns,
        "Unique Count": [df[col].nunique() for col in df.columns],
        "Unique Values": [sorted(df[col].unique().tolist()) for col in df.columns]
    })

    return summary


# ============================= MODULE 4 - FEATURE ENGINEERING =============================

# Compute a composite score (row-wise mean) from questionnaire items.
def compute_composite_score(df, columns, new_column):
    missing_columns = [col for col in columns if col not in df.columns]

    if missing_columns:
        raise ValueError(
            f"Column(s) not found: {missing_columns}"
        )

    df[new_column] = df[columns].mean(axis=1)

    return df


# ============================= EXPLORATORY DATA ANALYSIS (EDA) =============================

# Counts of respondents per Gender category, labeled.
def get_gender_distribution(df):
    counts = df["Gender"].value_counts().sort_index()
    counts.index = counts.index.map(GENDER_LABELS)
    return counts


# Counts of respondents per Income category, labeled.
def get_income_distribution(df):
    counts = df["Income"].value_counts().sort_index()
    counts.index = counts.index.map(INCOME_LABELS)
    return counts


# Counts of respondents per residential Area, labeled.
def get_area_distribution(df):
    counts = df["Area"].value_counts().sort_index()
    counts.index = counts.index.map(AREA_LABELS)
    return counts


# Counts of respondents per social media usage Frequency, labeled.
def get_frequency_distribution(df):
    counts = df["Frequently"].value_counts().sort_index()
    counts.index = counts.index.map(FREQUENCY_LABELS)
    return counts


# CONSTRUCT DESCRIPTIVES / CORRELATIONS

# Descriptive stats (count/mean/std/etc.) for the composite constructs.
def get_construct_descriptives(df, constructs=CONSTRUCTS):
    return df[constructs].describe()

# Pearson correlation matrix between the composite constructs.
def get_construct_correlation_matrix(df, constructs=CONSTRUCTS):
    return df[constructs].corr(method="pearson")


# AUB ACROSS DEMOGRAPHIC GROUPS

# AUB count/mean/median/std by gender (Male vs Female only, by default).
def get_aub_by_gender_summary(df, exclude_other=True):
    data = df[df["Gender"] != 3] if exclude_other else df

    summary = data.groupby("Gender")["AUB"].agg(["count", "mean", "median", "std"])
    summary.index = summary.index.map({1: "Male", 2: "Female"})

    return summary

# Welch's t-test comparing AUB between male and female respondents.
def run_ttest_aub_gender(df):
    data = df[df["Gender"] != 3]

    male = data[data["Gender"] == 1]["AUB"]
    female = data[data["Gender"] == 2]["AUB"]

    t_stat, p_value = stats.ttest_ind(male, female, equal_var=False)

    return t_stat, p_value

# AUB count/mean/median/std by residential area.
def get_aub_by_area_summary(df):
    summary = df.groupby("Area")["AUB"].agg(["count", "mean", "median", "std"])
    summary.index = summary.index.map(AREA_LABELS)

    return summary

# One-way ANOVA comparing AUB across Urban/Suburban/Rural respondents.
def run_anova_aub_area(df):
    urban = df[df["Area"] == 1]["AUB"]
    suburban = df[df["Area"] == 2]["AUB"]
    rural = df[df["Area"] == 3]["AUB"]

    f_stat, p_value = stats.f_oneway(urban, suburban, rural)

    return f_stat, p_value


# AUB count/mean/median/std by social media usage frequency.
def get_aub_by_frequency_summary(df):
    summary = df.groupby("Frequently")["AUB"].agg(["count", "mean", "median", "std"])
    summary.index = summary.index.map({1: "Daily", 2: "Weekly", 3: "Monthly", 4: "Rarely Used"})

    return summary


# ============================= COMPLETE PREPROCESSING PIPELINE FOR 02 AND 03 =============================

# Run the full pipeline: load, drop unused columns, compute all composite scores
def run_preprocessing(
    file_path="../datasets/S-COMMERCE_GenZ_UniversityStudent_757_DIB.csv"):    

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
        "IB": ["IB1", "IB2", "IB3", "IB4"],
        "AUB": ["AUB1", "AUB2", "AUB3", "AUB4"],
    }

    for construct, columns in composite_scores.items():
        df = compute_composite_score(
            df,
            columns,
            construct
        )

    return df