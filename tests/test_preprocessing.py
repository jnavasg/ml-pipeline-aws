import pandas as pd
import pytest

from src.preprocessing.preprocessing import clean, encode_categoricals, split


@pytest.fixture
def raw_df():
    return pd.DataFrame(
        {
            "f1": [1.0, 2.0, None, 4.0, 5.0],
            "cat": ["a", "b", "a", "b", "a"],
            "target": [0, 1, 0, 1, 0],
        }
    )


def test_clean_removes_nulls(raw_df):
    cleaned = clean(raw_df)
    assert cleaned.isnull().sum().sum() == 0
    assert len(cleaned) == 4


def test_clean_removes_duplicates():
    df = pd.DataFrame({"a": [1, 1, 2], "b": [3, 3, 4]})
    cleaned = clean(df)
    assert len(cleaned) == 2


def test_encode_categoricals(raw_df):
    df = clean(raw_df)
    encoded = encode_categoricals(df, cat_cols=["cat"])
    assert "cat" not in encoded.columns
    assert any(c.startswith("cat_") for c in encoded.columns)


def test_split_proportions(raw_df):
    df = clean(raw_df)
    X_train, X_val, X_test, y_train, y_val, y_test = split(df, target_col="target")
    total = len(X_train) + len(X_val) + len(X_test)
    assert total == len(df)
