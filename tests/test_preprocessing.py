import pandas as pd
import pytest

from src.preprocessing.preprocessing import clean_titanic, encode_titanic, split

# Fixture con estructura real de Titanic (incluye columnas a eliminar)
@pytest.fixture
def raw_df():
    return pd.DataFrame(
        {
            "survived":    [1, 0, 1, 0, 1, 0, 1, 0, 1, 0,
                            1, 0, 1, 0, 1, 0, 1, 0, 1, 0],
            "pclass":      [1, 3, 1, 3, 2, 3, 1, 2, 3, 1,
                            1, 3, 1, 3, 2, 3, 1, 2, 3, 1],
            "sex":         ["female","male","female","male","female",
                            "male","female","male","female","male",
                            "female","male","female","male","female",
                            "male","female","male","female","male"],
            "age":         [29., 22., None, 35., 28., None, 38., 25., 31., 42.,
                            27., 19., 33., 40., 26., None, 36., 23., 30., 45.],
            "sibsp":       [0]*20,
            "parch":       [0]*20,
            "fare":        [211.3, 7.25, 151.5, 8.05, 13.0, 7.92, 227.5, 26.0,
                            7.75, 52.0, 211.3, 7.25, 151.5, 8.05, 13.0, 7.92,
                            227.5, 26.0, 7.75, 52.0],
            "embarked":    ["S","S","S","S","C","Q","S","C","Q","S",
                            "S","S","S","S","C","Q","S","C",None,"S"],
            # columnas redundantes que se deben eliminar
            "class":       ["First","Third","First","Third","Second",
                            "Third","First","Second","Third","First",
                            "First","Third","First","Third","Second",
                            "Third","First","Second","Third","First"],
            "who":         ["woman","man","woman","man","woman","man",
                            "woman","man","woman","man","woman","man",
                            "woman","man","woman","man","woman","man",
                            "woman","man"],
            "adult_male":  [False,True]*10,
            "deck":        [None]*20,
            "embark_town": ["Southampton"]*20,
            "alive":       ["yes","no"]*10,
            "alone":       [True]*20,
        }
    )


def test_clean_drops_irrelevant_columns(raw_df):
    cleaned = clean_titanic(raw_df)
    for col in ["class", "who", "adult_male", "deck", "embark_town", "alive", "alone"]:
        assert col not in cleaned.columns


def test_clean_imputes_age_nulls(raw_df):
    cleaned = clean_titanic(raw_df)
    assert cleaned["age"].isnull().sum() == 0


def test_clean_imputes_embarked_nulls(raw_df):
    cleaned = clean_titanic(raw_df)
    assert cleaned["embarked"].isnull().sum() == 0


def test_encode_sex_is_binary(raw_df):
    df = clean_titanic(raw_df)
    encoded = encode_titanic(df)
    assert set(encoded["sex"].unique()).issubset({0, 1})


def test_encode_creates_embarked_dummies(raw_df):
    df = clean_titanic(raw_df)
    encoded = encode_titanic(df)
    assert "embarked" not in encoded.columns
    assert "embarked_Q" in encoded.columns
    assert "embarked_S" in encoded.columns


def test_split_preserves_total_rows(raw_df):
    df = clean_titanic(raw_df)
    df = encode_titanic(df)
    X_train, X_test, y_train, y_test = split(df, test_size=0.2)
    assert len(X_train) + len(X_test) == len(df)
    assert len(y_train) + len(y_test) == len(df)


def test_split_feature_columns_exclude_target(raw_df):
    df = clean_titanic(raw_df)
    df = encode_titanic(df)
    X_train, X_test, _, _ = split(df)
    assert "survived" not in X_train.columns
    assert "survived" not in X_test.columns
