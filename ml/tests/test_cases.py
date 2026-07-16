import pandas as pd

from ml.scripts.preprocessing import (
    load_dataset,
    inspect_dataset,
    validate_columns,
    validate_dtypes,
    drop_columns,
    compute_composite_score
)

DATASET_PATH = "ml/datasets/S-COMMERCE_GenZ_UniversityStudent_757_DIB.csv"

print("=" * 50)
print("SCA-UT-001")
print("=" * 50)

df = load_dataset(DATASET_PATH)

print("Input:")
print(DATASET_PATH)

print("\nOutput:")
print(type(df))
print(df.shape)


print("\n" + "=" * 50)
print("SCA-UT-002")
print("=" * 50)

try:
    load_dataset("invalid.csv")
except FileNotFoundError as e:
    print("Input:")
    print("invalid.csv")

    print("\nOutput:")
    print(type(e).__name__)


print("\n" + "=" * 50)
print("SCA-UT-003")
print("=" * 50)

shape = inspect_dataset(df)

print("Input:")
print("Loaded dataset DataFrame")

print("\nOutput:")
print(shape)


print("\n" + "=" * 50)
print("SCA-UT-004")
print("=" * 50)

columns = validate_columns(df)

print("Input:")
print("Loaded dataset DataFrame")

print("\nOutput:")
print(f"Number of columns: {len(columns)}")
print(f"First column: {columns[0]}")
print(f"Last column: {columns[-1]}")
print(columns)


print("\n" + "=" * 50)
print("SCA-UT-005")
print("=" * 50)

dtypes = validate_dtypes(df)

print("Input:")
print("Loaded dataset DataFrame")

print("\nOutput:")
print(f"Number of dtypes: {len(dtypes)}")
print(dtypes)


print("\n" + "=" * 50)
print("SCA-UT-006")
print("=" * 50)

cleaned_df = drop_columns(df, ["Job"])

print("Input:")
print(["Job"])

print("\nOutput:")
print("'Job' exists:", "Job" in cleaned_df.columns)
print(cleaned_df.columns.tolist())


print("\n" + "=" * 50)
print("SCA-UT-007")
print("=" * 50)

cleaned_columns = validate_columns(cleaned_df)

print("Input:")
print("DataFrame after removing Job")

print("\nOutput:")
print("'Job' exists:", "Job" in cleaned_columns)
print(cleaned_columns)


print("\n" + "=" * 50)
print("SCA-UT-008")
print("=" * 50)

sample_df = pd.DataFrame({
    "PU1": [4],
    "PU2": [5],
    "PU3": [3],
    "PU4": [4]
})

print("Input:")
print(sample_df)

result = compute_composite_score(
    sample_df,
    ["PU1", "PU2", "PU3", "PU4"],
    "PU"
)

print("\nOutput:")
print(result[["PU"]])