import importlib
import re
import sys
import types
from pathlib import Path

import numpy as np
import pytest
import scipy.io
import torch

import scd

ROOT = Path(__file__).resolve().parents[1]


def test_package_version_matches_project_metadata():
    pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    match = re.search(r'^version = "([^"]+)"$', pyproject, flags=re.MULTILINE)

    assert match is not None
    assert scd.__version__ == match.group(1)


@pytest.mark.parametrize("config_name", ["default", "surface", "intramuscular"])
def test_built_in_configs_fall_back_to_cpu(monkeypatch, config_name):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    config = scd.load_config(config_name)

    assert config.device == "cpu"


def test_load_data_reads_and_transposes_npy(tmp_path):
    path = tmp_path / "data.npy"
    np.save(path, np.arange(6, dtype=np.float32).reshape(2, 3))

    loaded = scd.load_data(path, device="cpu")

    assert loaded.shape == (3, 2)
    assert loaded.device.type == "cpu"


def test_load_data_reads_mat_file(tmp_path):
    path = tmp_path / "data.mat"
    scipy.io.savemat(path, {"emg": np.arange(6).reshape(3, 2)})

    loaded = scd.load_data(path, device="cpu")

    assert loaded.shape == (3, 2)


def test_load_data_falls_back_to_mat73(monkeypatch, tmp_path):
    train_module = importlib.import_module("scd.train")
    path = tmp_path / "v73.mat"
    path.touch()

    def unsupported_by_scipy(_path):
        raise NotImplementedError("MATLAB v7.3 files require an HDF5 reader")

    fake_mat73 = types.ModuleType("mat73")
    fake_mat73.loadmat = lambda _path: {"emg": np.arange(6).reshape(3, 2)}
    monkeypatch.setattr(train_module.sio, "loadmat", unsupported_by_scipy)
    monkeypatch.setitem(sys.modules, "mat73", fake_mat73)

    loaded = scd.load_data(path, device="cpu")

    assert loaded.shape == (3, 2)


def test_load_data_rejects_non_matrix_arrays(tmp_path):
    path = tmp_path / "vector.npy"
    np.save(path, np.arange(3))

    with pytest.raises(ValueError, match="Expected a 2D"):
        scd.load_data(path, device="cpu")
