import pandas as pd

from scripts.preprocessing import load_dataset

def test_load_dataset():

    df = load_dataset("data/SCOMMERCE_GenZ_UniversityStudent_757_DIB.csv")

    assert isinstance(df, pd.DataFrame)

    assert df.shape == (757,31)

def test_columns_exist():

    df = load_dataset("data/SCOMMERCE_GenZ_UniversityStudent_757_DIB.csv")

    expected = [
        "Gender",
        "Job",
        "Income",
        ...
        "AUB4"
    ]

    assert list(df.columns) == expected

from scripts.preprocessing import remove_job

def test_remove_job():

    df = load_dataset("data/SCOMMERCE_GenZ_UniversityStudent_757_DIB.csv")

    df = remove_job(df)

    assert "Job" not in df.columns

from scripts.preprocessing import remove_peu4

def test_remove_peu4():

    df = load_dataset(...)

    df = remove_peu4(df)

    assert "PEU4" not in df.columns

from scripts.preprocessing import compute_composite

def test_compute_pu():

    df = pd.DataFrame({

        "PU1":[4],
        "PU2":[5],
        "PU3":[3],
        "PU4":[4]

    })

    pu = compute_composite(df,["PU1","PU2","PU3","PU4"])

    assert pu.iloc[0] == 4