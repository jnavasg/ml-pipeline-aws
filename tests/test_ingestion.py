import os
import tempfile

import pandas as pd
import pytest

from src.ingestion.ingestion import load_from_csv, save_raw


@pytest.fixture
def sample_df():
    return pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})


def test_load_from_csv(sample_df, tmp_path):
    csv_path = tmp_path / "sample.csv"
    sample_df.to_csv(csv_path, index=False)
    loaded = load_from_csv(str(csv_path))
    assert len(loaded) == 3
    assert list(loaded.columns) == ["a", "b"]


def test_save_raw(sample_df, tmp_path):
    out = str(tmp_path / "raw" / "out.parquet")
    save_raw(sample_df, out)
    assert os.path.exists(out)
    reloaded = pd.read_parquet(out)
    pd.testing.assert_frame_equal(sample_df, reloaded)
