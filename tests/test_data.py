import pandas as pd
import pytest

from ml.data_pipeline import build


@pytest.mark.parametrize("version", ["v1", "v2"])
def test_no_split_leakage(tmp_path, version):
    build(version, tmp_path)
    splits = [pd.read_csv(tmp_path / f"{name}.csv") for name in ["train", "validation", "test"]]
    for frame in splits:
        assert not frame.isna().any().any()
        assert set(frame.target) == {0, 1, 2}
        # assert set(frame.target) == {1, 2, 3} # test
        assert frame.sample_id.is_unique
    ids = [set(f.sample_id) for f in splits]
    assert not (ids[0] & ids[1] or ids[0] & ids[2] or ids[1] & ids[2])


def test_version_change_preserves_evaluation(tmp_path):
    build("v1", tmp_path / "v1")
    build("v2", tmp_path / "v2")
    for name in ["validation", "test"]:
        assert (tmp_path / "v1" / f"{name}.csv").read_bytes() == (
            tmp_path / "v2" / f"{name}.csv"
        ).read_bytes()
    a = pd.read_csv(tmp_path / "v1/train.csv")
    b = pd.read_csv(tmp_path / "v2/train.csv")
    assert set(a.sample_id) < set(b.sample_id)
