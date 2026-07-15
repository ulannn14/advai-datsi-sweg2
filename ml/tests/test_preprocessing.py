import pandas as pd

from scripts.preprocessing import load_dataset

def test_load_dataset():

    df = load_dataset("datasets/SCOMMERCE_GenZ_UniversityStudent_757_DIB.csv")

    assert isinstance(df, pd.DataFrame)

    assert df.shape == (757,31)

from scripts.preprocessing import validate_columns

def test_columns_exist():

    df = validate_columns("datasets/SCOMMERCE_GenZ_UniversityStudent_757_DIB.csv")

    expected = [
        "Gender",
        "Job",
        "Income",
        "PEU1",
        "PEU2",
        "...",
        "AUB4"
    ]

    assert list(df.columns) == expected

from scripts.preprocessing import drop_columns

def test_remove_job():

    df = load_dataset("datasets/SCOMMERCE_GenZ_UniversityStudent_757_DIB.csv")

    df = drop_columns(df, ["Job"])

    assert "Job" not in df.columns

from scripts.preprocessing import drop_columns

def test_remove_peu4():

    df = load_dataset("datasets/SCOMMERCE_GenZ_UniversityStudent_757_DIB.csv")

    df = drop_columns(df, ["PEU4"])

    assert "PEU4" not in df.columns

from scripts.preprocessing import compute_composite_score

def test_compute_pu():

    df = pd.DataFrame({

        "PU1":[4],
        "PU2":[5],
        "PU3":[3],
        "PU4":[4]

    })

    pu = compute_composite_score("datasets/SCOMMERCE_GenZ_UniversityStudent_757_DIB.csv",["PU1","PU2","PU3","PU4"], "PU")

    assert pu.iloc[0] == 4