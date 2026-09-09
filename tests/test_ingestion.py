import os
from unittest.mock import patch

import pandas as pd
import pytest

from src.ingestion.ingestion import load_from_csv, load_titanic, save_raw

TITANIC_COLS = [
    "survived", "pclass", "sex", "age", "sibsp", "parch",
    "fare", "embarked", "class", "who", "adult_male",
    "deck", "embark_town", "alive", "alone",
]


@pytest.fixture
def raw_titanic_df():
    return pd.DataFrame(
        {
            "survived":    [1, 0, 1, 0, 1],
            "pclass":      [1, 3, 1, 3, 2],
            "sex":         ["female", "male", "female", "male", "female"],
            "age":         [29.0, 22.0, None, 35.0, 28.0],
            "sibsp":       [0, 1, 1, 0, 0],
            "parch":       [0, 0, 2, 0, 0],
            "fare":        [211.3, 7.25, 151.5, 8.05, 13.0],
            "embarked":    ["S", "S", "S", "S", "C"],
        }
    )


def test_load_titanic_calls_seaborn(raw_titanic_df):
    with patch("src.ingestion.ingestion.sns.load_dataset", return_value=raw_titanic_df) as mock:
        df = load_titanic()
    mock.assert_called_once_with("titanic")
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 5


def test_save_raw_creates_csv(raw_titanic_df, tmp_path):
    out = str(tmp_path / "raw" / "titanic.csv")
    save_raw(raw_titanic_df, out)
    assert os.path.exists(out)
    reloaded = pd.read_csv(out)
    assert list(reloaded.columns) == list(raw_titanic_df.columns)
    assert len(reloaded) == len(raw_titanic_df)


def test_load_from_csv_round_trip(raw_titanic_df, tmp_path):
    csv_path = tmp_path / "titanic.csv"
    raw_titanic_df.to_csv(csv_path, index=False)
    loaded = load_from_csv(str(csv_path))
    assert len(loaded) == len(raw_titanic_df)
    assert set(loaded.columns) == set(raw_titanic_df.columns)
