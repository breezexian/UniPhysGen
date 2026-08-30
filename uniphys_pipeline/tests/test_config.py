from __future__ import annotations

import json
from pathlib import Path

import pytest

from uniphys.config import ConfigError, load_config


def test_load_config_derives_dataset_paths(tmp_path: Path) -> None:
    config_file = tmp_path / "config.json"
    config_file.write_text(
        json.dumps(
            {
                "dataset": {
                    "name": "demo",
                    "type_name": "chairs",
                    "input_root": str(tmp_path / "input"),
                    "output_root": str(tmp_path / "output"),
                },
                "pipeline": {"stages": ["render"], "workers": 2},
            }
        ),
        encoding="utf-8",
    )

    config = load_config(config_file)

    assert config.dataset.mesh_root == tmp_path / "input/demo/chairs"
    assert config.dataset.render_root == tmp_path / "output/demo/render_res"
    assert (
        config.dataset.decomposition_root
        == tmp_path / "output/demo/decomposition_res/chairs"
    )
    assert config.dataset.gpt_output_root == tmp_path / "output/demo/gpt_output/chairs"
    assert config.pipeline.stages == ("render",)
    assert config.pipeline.workers == 2
    assert config.pipeline.part_refine_after_merge_use_relative_threshold is False
    assert config.pipeline.post_process_use_relative_threshold is False
    assert config.runtime.gpu == "0"


def test_explicit_null_gpu_disables_default_selection(tmp_path: Path) -> None:
    config_file = tmp_path / "config.json"
    config_file.write_text(
        json.dumps({"runtime": {"gpu": None}}),
        encoding="utf-8",
    )

    assert load_config(config_file).runtime.gpu is None


def test_fingerprint_does_not_contain_api_key(tmp_path: Path, monkeypatch) -> None:
    config_file = tmp_path / "config.json"
    config_file.write_text("{}", encoding="utf-8")
    monkeypatch.setenv("OPENAI_API_KEY", "never-serialize-this-secret")

    fingerprint = load_config(config_file).execution_fingerprint()

    assert len(fingerprint) == 64
    assert "never-serialize" not in fingerprint


def test_string_boolean_is_parsed_explicitly(tmp_path: Path) -> None:
    config_file = tmp_path / "config.json"
    config_file.write_text(
        json.dumps({"pipeline": {"stages": ["render"], "resume": "false"}}),
        encoding="utf-8",
    )

    assert load_config(config_file).pipeline.resume is False


def test_invalid_boolean_is_rejected(tmp_path: Path) -> None:
    config_file = tmp_path / "config.json"
    config_file.write_text(
        json.dumps({"pipeline": {"stages": ["render"], "resume": "maybe"}}),
        encoding="utf-8",
    )

    with pytest.raises(ConfigError, match="pipeline.resume"):
        load_config(config_file)


def test_relative_post_process_threshold_options(tmp_path: Path) -> None:
    config_file = tmp_path / "config.json"
    config_file.write_text(
        json.dumps(
            {
                "pipeline": {
                    "part_refine_after_merge_use_relative_threshold": True,
                    "post_process_use_relative_threshold": "true",
                }
            }
        ),
        encoding="utf-8",
    )

    pipeline = load_config(config_file).pipeline

    assert pipeline.part_refine_after_merge_use_relative_threshold is True
    assert pipeline.post_process_use_relative_threshold is True
