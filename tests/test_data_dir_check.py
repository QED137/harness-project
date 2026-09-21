"""Regression: the container runs as UID 65534, so an owner-only data dir made the
preload fail deep inside the container with a bare PermissionError. The sandbox
now refuses such a config up front with an actionable message. No Docker needed."""

import pytest

from tsagent.sandbox import DockerSandbox, SandboxConfig


def make(tmp_path, dir_mode=0o755, file_mode=0o644, preload="/data/w.parquet"):
    (tmp_path / "w.parquet").write_bytes(b"x")
    (tmp_path / "w.parquet").chmod(file_mode)
    tmp_path.chmod(dir_mode)
    return SandboxConfig(data_dir=tmp_path, preload=preload)


def test_readable_data_dir_accepted(tmp_path):
    DockerSandbox(make(tmp_path))


def test_owner_only_dir_rejected_with_fix_in_message(tmp_path):
    with pytest.raises(ValueError, match="chmod o\\+rx"):
        DockerSandbox(make(tmp_path, dir_mode=0o700))


def test_unreadable_preload_file_rejected(tmp_path):
    with pytest.raises(ValueError, match="chmod o\\+r "):
        DockerSandbox(make(tmp_path, file_mode=0o600))


def test_missing_preload_file_rejected(tmp_path):
    with pytest.raises(ValueError, match="not found"):
        DockerSandbox(make(tmp_path, preload="/data/nope.parquet"))


def test_missing_data_dir_rejected(tmp_path):
    with pytest.raises(ValueError, match="does not exist"):
        DockerSandbox(SandboxConfig(data_dir=tmp_path / "nope"))
