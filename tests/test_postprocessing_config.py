"""Tests for the YAML experiment configuration of the post-processing pipeline."""
from pathlib import Path

import pytest

from utilities.config import ExperimentConfig

DEFAULT_YAML = Path(__file__).resolve().parents[1] / "postprocessing" / "config_default.yaml"


def test_defaults_are_valid():
    assert ExperimentConfig().validate() == []


def test_shipped_default_yaml_loads_and_validates():
    config = ExperimentConfig.from_yaml(DEFAULT_YAML)
    assert config.validate() == []
    assert config.tracking.n_segments == 20


def test_yaml_round_trip(tmp_path):
    config = ExperimentConfig(experiment_name="round_trip")
    config.tracking.n_segments = 30
    out = tmp_path / "cfg.yaml"
    config.to_yaml(out)
    assert ExperimentConfig.from_yaml(out).to_dict() == config.to_dict()


def test_invalid_values_are_reported():
    config = ExperimentConfig()
    config.preprocessing.th_img = 300
    config.tracking.scalar_product_threshold = 2.0
    assert len(config.validate()) >= 2


def test_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        ExperimentConfig.from_yaml(tmp_path / "missing.yaml")
