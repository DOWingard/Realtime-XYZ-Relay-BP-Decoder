"""Durability of campaign outputs: the files an arm's done.json vouches for are synced first."""
from __future__ import annotations

from pathlib import Path

from rtd import campaign as cp


def test_sync_tree_syncs_every_file_and_directory(tmp_path: Path):
    arm = tmp_path / "relay5"
    (arm / "x").mkdir(parents=True)
    (arm / "z").mkdir()
    for half in ("x", "z"):
        (arm / half / "logical_failure.npy").write_bytes(b"\x93NUMPY")
        (arm / half / "run.json").write_text("{}")
    assert cp.sync_tree(arm) == 4
    assert cp.sync_tree(tmp_path / "relay5" / "x") == 2
