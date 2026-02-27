from __future__ import annotations

from pathlib import Path

from ui.run_inspector import (
    find_config_file,
    find_manifest_file,
    find_seed_log,
    find_summary_file,
    list_files,
    list_seed_dirs,
)


def test_find_config_file(tmp_path: Path) -> None:
    assert find_config_file(tmp_path) is None

    # raw only
    p1 = tmp_path / "config.json"
    p1.touch()
    assert find_config_file(tmp_path) == p1

    # resolved takes precedence
    p2 = tmp_path / "config_resolved.json"
    p2.touch()
    assert find_config_file(tmp_path) == p2


def test_find_manifest_file(tmp_path: Path) -> None:
    assert find_manifest_file(tmp_path) is None

    # meta only
    p1 = tmp_path / "meta.json"
    p1.touch()
    assert find_manifest_file(tmp_path) == p1

    # manifest takes precedence
    p2 = tmp_path / "manifest.json"
    p2.touch()
    assert find_manifest_file(tmp_path) == p2


def test_find_summary_file(tmp_path: Path) -> None:
    assert find_summary_file(tmp_path) is None
    p1 = tmp_path / "summary.csv"
    p1.touch()
    assert find_summary_file(tmp_path) == p1


def test_list_seed_dirs_and_logs(tmp_path: Path) -> None:
    seeds_dir = tmp_path / "seeds"
    assert list_seed_dirs(tmp_path) == []

    seeds_dir.mkdir()
    assert list_seed_dirs(tmp_path) == []

    # create a file (should be ignored)
    (seeds_dir / "not_a_dir").touch()

    # create valid seed dirs
    s2 = seeds_dir / "s00002"
    s2.mkdir()
    s1 = seeds_dir / "s00001"
    s1.mkdir()

    # Should be sorted
    assert list_seed_dirs(tmp_path) == [s1, s2]

    # test find_seed_log
    assert find_seed_log(s1) is None
    log_file = s1 / "run_log.csv"
    log_file.touch()
    assert find_seed_log(s1) == log_file


def test_list_files(tmp_path: Path) -> None:
    assert list_files(tmp_path) == []

    (tmp_path / "file_a.txt").touch()
    d1 = tmp_path / "d1"
    d1.mkdir()
    (d1 / "file_b.txt").touch()

    files = list_files(tmp_path)
    # Should be sorted string relative paths
    assert files == ["d1/file_b.txt", "file_a.txt"]
