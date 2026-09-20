import string
from pathlib import Path
from zoneinfo import ZoneInfo

import databento as db
import numpy as np
import pandas as pd
import pytest
from pytest import MonkeyPatch

from finm37000 import (
    disk_cache,
    generate_cache_filename,
    get_cache_dir,
    get_cached_dataframe,
)


def test_generate_cache_filename() -> None:
    class FakeClient:
        def __init__(self) -> None:
            self._private_state = 42

    def some_function(
        dataset: db.Dataset,  # noqa: ARG001
        client: FakeClient,  # noqa: ARG001
        contracts: list[str],  # noqa: ARG001
        *,
        start: pd.Timestamp,  # noqa: ARG001
    ) -> pd.DataFrame:
        return pd.DataFrame()

    dataset = db.Dataset.GLBX_MDP3
    client = FakeClient()
    too_many_contracts = [
        "CLZ6",
        "LOZ6 C7000",
        "LOZ6 P7000",
        "LOZ6 C7100",
        "LOZ6 P7100",
    ]
    start_time = pd.Timestamp("2026-09-03 13:14:15.123456789", tzinfo=ZoneInfo("UTC"))

    all_arg_filename = generate_cache_filename(
        some_function,
        (dataset, client, too_many_contracts),
        kwargs={"start": start_time},
        args_to_ignore=[],
    )
    assert all_arg_filename.startswith("some_function")
    assert "GLBX.MDP3" in all_arg_filename
    assert "FakeClient(hash=" in all_arg_filename
    assert "contracts=[CLZ6,LOZ6C7000,LOZ6P7000," in all_arg_filename
    assert "start=2026-09-03T1314_" in all_arg_filename
    assert all_arg_filename.endswith(".parquet")

    some_arg_filename = generate_cache_filename(
        some_function,
        (dataset, client, too_many_contracts),
        kwargs={"start": start_time},
        args_to_ignore=("dataset", "client"),
    )
    assert some_arg_filename.startswith("some_function")
    assert "GLBX.MDP3" not in some_arg_filename
    assert "FakeClient(hash=" not in some_arg_filename
    assert "contracts=[CLZ6,LOZ6C7000,LOZ6P7000," in some_arg_filename
    assert "start=2026-09-03T131415.123456789+0000" in some_arg_filename
    assert some_arg_filename.endswith(".parquet")


def test_get_cache_dir_default(tmp_path: Path, monkeypatch: MonkeyPatch) -> None:
    fake_home = tmp_path / "fake_home"
    repo_root = fake_home / "fake_project"  # No .git yet, defaults relative to home.
    fake_working_dir = repo_root / "src"
    fake_working_dir.mkdir(parents=True)

    monkeypatch.setattr(Path, "home", lambda: fake_home)
    monkeypatch.chdir(fake_working_dir)

    home_cache_dir = fake_home / ".cache" / "finm37000"
    assert not home_cache_dir.exists()
    assert get_cache_dir() == home_cache_dir
    assert home_cache_dir.exists()

    # Now with .git to identify repo root.
    (repo_root / ".git").mkdir(parents=True)
    repo_cache_dir = repo_root / ".cache" / "finm37000"
    assert not repo_cache_dir.exists()
    assert get_cache_dir() == repo_cache_dir
    assert repo_cache_dir.exists()


def test_get_cache_dir_env_var_override(
    tmp_path: Path, monkeypatch: MonkeyPatch
) -> None:
    env_cache = tmp_path / "env_cache"
    expected_cache_dir = env_cache
    assert not expected_cache_dir.exists()
    monkeypatch.setenv("FINM37000_CACHE_DIR", str(env_cache))
    assert get_cache_dir() == expected_cache_dir
    assert expected_cache_dir.exists()


def _make_df(n: int = 10) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "a": range(n),
            "x": np.linspace(0, 1, num=n),
            "name": string.ascii_lowercase[:n],
        }
    )


def _dont_make_df(n: int = 10) -> pd.DataFrame:
    msg = f"This function should not be called {n=}."
    raise RuntimeError(msg)


def test_get_cached_dataframe_writes_cache(tmp_path: Path) -> None:
    cache_file = tmp_path / "test_cache_name.parquet"
    assert not cache_file.exists()
    df = get_cached_dataframe(cache_file, _make_df)
    assert cache_file.exists()
    saved_df = pd.read_parquet(cache_file)
    pd.testing.assert_frame_equal(df, saved_df)


def test_get_cached_dataframe_reads_cache(tmp_path: Path) -> None:
    cache_file = tmp_path / "test_cache_name.parquet"
    generated_df = get_cached_dataframe(cache_file, _make_df)
    read_df = get_cached_dataframe(cache_file, _dont_make_df)
    pd.testing.assert_frame_equal(generated_df, read_df)


def test_get_cached_dataframe_uses_args(tmp_path: Path) -> None:
    cache_file = tmp_path / "test_cache_name"
    rows = 5
    df = get_cached_dataframe(cache_file, _make_df, rows)
    assert len(df) == rows


def test_get_cached_dataframe_uses_kwargs(tmp_path: Path) -> None:
    cache_file = tmp_path / "test_cache_name"
    rows = 7
    df = get_cached_dataframe(cache_file, _make_df, n=rows)
    assert len(df) == rows


def test_disk_cache_decorator(tmp_path: Path, monkeypatch: MonkeyPatch) -> None:
    cache_dir = tmp_path / ".cache"
    monkeypatch.setenv("FINM37000_CACHE_DIR", str(cache_dir))

    call_count = 0

    @disk_cache()
    def df_generator(n: int) -> pd.DataFrame:
        nonlocal call_count
        call_count += 1
        return _make_df(n=n)

    test_sizes = (5, 6, 7)
    for i, test_size in enumerate(test_sizes, start=1):
        initial_df = df_generator(test_size)
        assert len(initial_df) == test_size
        assert call_count == i
        pd.testing.assert_frame_equal(initial_df, df_generator(test_size))
        assert call_count == i

    assert len(list(cache_dir.glob("*.parquet"))) == len(test_sizes)


def test_disk_cache_scope(tmp_path: Path, monkeypatch: MonkeyPatch) -> None:
    cache_dir = tmp_path / ".cache"
    monkeypatch.setenv("FINM37000_CACHE_DIR", str(cache_dir))

    @disk_cache(scope="subdir")
    def df_generator(n: int) -> pd.DataFrame:
        return _make_df(n=n)

    test_sizes = (5, 6, 7)
    for test_size in test_sizes:
        df_generator(test_size)

    assert len(list((cache_dir / "subdir").glob("*.parquet"))) == len(test_sizes)


def test_disk_cache_ignored_args(tmp_path: Path, monkeypatch: MonkeyPatch) -> None:
    cache_dir = tmp_path / ".cache"
    monkeypatch.setenv("FINM37000_CACHE_DIR", str(cache_dir))

    @disk_cache()
    def default_hidden_dataset(n: int, dataset: db.Dataset) -> pd.DataFrame:
        _ignored = dataset
        return _make_df(n=n)

    with pytest.raises(ValueError, match="does not support a bare str"):

        @disk_cache(args_to_ignore="n")
        def bad_hidden_n(n: int, dataset: db.Dataset) -> pd.DataFrame:
            _ignored = dataset
            return _make_df(n=n)

    @disk_cache(args_to_ignore=("n",))
    def hidden_n(n: int, dataset: db.Dataset) -> pd.DataFrame:
        _ignored = dataset
        return _make_df(n=n)

    default_hidden_dataset(7, dataset=db.Dataset.GLBX_MDP3)
    hidden_n(9, dataset=db.Dataset.GLBX_MDP3)

    cached_filenames = [f.name for f in cache_dir.glob("*.parquet")]
    assert "default_hidden_dataset_n=7.parquet" in cached_filenames
    assert "hidden_n_dataset=GLBX.MDP3.parquet" in cached_filenames
