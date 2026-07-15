from unittest import result

import pandas as pd

from ml.scripts.preprocessing import load_dataset

def test_load_dataset():

    df = load_dataset("ml/datasets/S-COMMERCE_GenZ_UniversityStudent_757_DIB.csv")

    assert isinstance(df, pd.DataFrame)

    assert df.shape == (757,31)

from ml.scripts.preprocessing import validate_columns

def test_columns_exist():
    df = load_dataset("ml/datasets/S-COMMERCE_GenZ_UniversityStudent_757_DIB.csv")

    columns = validate_columns(df)

    expected = [
        'Gender',
        'Job',
        'Income',
        'Area',
        'Frequently',
        'PU1',
        'PU2',
        'PU3',
        'PU4',
        'PEU1',
        'PEU2',
        'PEU3',
        'PEU4',
        'FSC1',
        'FSC2',
        'FSC3',
        'SP1',
        'SP2',
        'SP3',
        'SP4',
        'TP1',
        'TP2',
        'TP3',
        'IB1',
        'IB2',
        'IB3',
        'IB4',
        'AUB1',
        'AUB2',
        'AUB3',
        'AUB4'
    ]

    assert columns == expected

from ml.scripts.preprocessing import drop_columns

def test_remove_job():

    df = load_dataset("ml/datasets/S-COMMERCE_GenZ_UniversityStudent_757_DIB.csv")

    df = drop_columns(df, ["Job"])

    assert "Job" not in df.columns

from ml.scripts.preprocessing import drop_columns

def test_remove_peu4():

    df = load_dataset("ml/datasets/S-COMMERCE_GenZ_UniversityStudent_757_DIB.csv")

    df = drop_columns(df, ["PEU4"])

    assert "PEU4" not in df.columns

from ml.scripts.preprocessing import compute_composite_score

def test_compute_pu():

    df = pd.DataFrame({

        "PU1":[4],
        "PU2":[5],
        "PU3":[3],
        "PU4":[4]

    })

    result = compute_composite_score(
    df,
    ["PU1","PU2","PU3","PU4"],
    "PU"
    )

    assert result["PU"].iloc[0] == 4