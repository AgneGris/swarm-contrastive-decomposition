import importlib
import pickle
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


def test_signal_to_array_returns_channels_by_samples_float32():
    tensor = torch.arange(12, dtype=torch.float64).reshape(4, 3)  # (time, channels)

    array = scd.signal_to_array(tensor)

    assert array.shape == (3, 4)
    assert array.dtype == np.float32
    assert array.flags["C_CONTIGUOUS"]
    np.testing.assert_array_equal(array, tensor.numpy().T)


def test_signal_to_array_keeps_channels_by_samples_input():
    array = scd.signal_to_array(np.zeros((3, 4)))

    assert array.shape == (3, 4)


def test_signal_to_array_rejects_non_matrix():
    with pytest.raises(ValueError, match="Expected a 2D"):
        scd.signal_to_array(np.zeros(5))


def test_save_results_stores_signal_under_data(tmp_path):
    path = tmp_path / "nested" / "out.pkl"
    signal = torch.ones(8, 2)  # (time, channels)

    scd.save_results(str(path), {"timestamps": []}, neural_data=signal)

    with open(path, "rb") as f:
        saved = pickle.load(f)
    assert saved["data"].shape == (2, 8)
    assert saved["data"].dtype == np.float32


def test_save_results_without_signal_adds_no_data_key(tmp_path):
    path = tmp_path / "out.pkl"

    scd.save_results(path, {"timestamps": []})

    with open(path, "rb") as f:
        saved = pickle.load(f)
    assert "data" not in saved


def test_save_data_flag_round_trips_through_config():
    assert scd.Config(sampling_frequency=2048).save_data is True
    assert scd.Config(sampling_frequency=2048, save_data=False).save_data is False


def test_preprocessing_snapshot_records_what_preprocess_data_did():
    config = scd.Config(
        sampling_frequency=2048,
        extension_factor=4,
        bad_channels=[3, 7],
        start_time=1,
        end_time=5,
    )
    model = scd.SwarmContrastiveDecomposition()
    model.config = config
    model.initialise_dictionary()

    model._capture_preprocessing_config()

    snapshot = model.decomp["preprocessing_config"]
    assert snapshot["bad_channels"] == [3, 7]
    assert snapshot["start_time"] == 1
    assert snapshot["end_time"] == 5
    assert snapshot["square_sources_spike_det"] is True


def test_preprocessing_snapshot_defaults_bad_channels_to_empty_list():
    model = scd.SwarmContrastiveDecomposition()
    model.config = scd.Config(sampling_frequency=2048, extension_factor=4)
    model.initialise_dictionary()

    model._capture_preprocessing_config()

    assert model.decomp["preprocessing_config"]["bad_channels"] == []


@pytest.mark.parametrize("save_data", [True, False])
def test_train_attaches_signal_as_loaded_only_when_enabled(monkeypatch, tmp_path, save_data):
    train_module = importlib.import_module("scd.train")
    path = tmp_path / "data.npy"
    raw = np.random.default_rng(0).standard_normal((100, 4)).astype(np.float32)
    np.save(path, raw)

    def fake_train_model(neural_data, config):
        return {"timestamps": [], "preprocessing_config": {}}, []

    monkeypatch.setattr(train_module, "train_model", fake_train_model)

    dictionary, _ = scd.train(
        path,
        config_name="default",
        device="cpu",
        bad_channels=[1],
        start_time=0,
        end_time=-1,
        save_data=save_data,
    )

    if save_data:
        # The stored signal is the file contents, not the noise-filled slice
        np.testing.assert_array_equal(dictionary["data"], raw.T)
    else:
        assert "data" not in dictionary


def test_signal_to_array_copies_fortran_ordered_input():
    # scipy returns MATLAB matrices Fortran-ordered; the transposed view is
    # then already C-contiguous and must still be copied, because the loaded
    # tensor is modified in place by preprocess_data.
    fortran = np.asfortranarray(np.zeros((10, 3), dtype=np.float32))
    tensor = torch.from_numpy(fortran)

    array = scd.signal_to_array(tensor)
    tensor[0, 0] = 5.0

    assert not np.shares_memory(array, fortran)
    assert array[0, 0] == 0.0
