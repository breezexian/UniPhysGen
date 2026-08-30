import json
from pathlib import Path

import pytest

from category.CAT import CATEGORIES
from category.gpt_cat import (
    AnnotationResult,
    CategoryAnnotationError,
    DEFAULT_PROMPT,
    gpt_cate_annotation_main,
    main,
    parse_classification,
)


def test_parse_classification_accepts_marked_json() -> None:
    response = """===BEGIN_JSON===
    {"category": "Furniture", "subcategory": "SeatingFurniture"}
    ===END_JSON==="""

    assert parse_classification(response) == {
        "category": "Furniture",
        "subcategory": "SeatingFurniture",
    }


def test_parse_classification_rejects_unknown_subcategory() -> None:
    response = json.dumps(
        {"category": "Furniture", "subcategory": "InventedFurniture"}
    )

    with pytest.raises(CategoryAnnotationError, match="Unknown subcategory"):
        parse_classification(response)


def test_default_prompt_is_anchored_to_category_directory() -> None:
    assert DEFAULT_PROMPT.is_file()
    assert DEFAULT_PROMPT.name == "prompt.txt"
    assert DEFAULT_PROMPT.parent.name == "category"


def test_existing_valid_annotation_skips_without_api_client(
    tmp_path: Path,
) -> None:
    output = tmp_path / "example.json"
    output.write_text(
        json.dumps(
            {"category": "Furniture", "subcategory": "SeatingFurniture"}
        ),
        encoding="utf-8",
    )

    result = gpt_cate_annotation_main(
        "example",
        "chair",
        "wood chair",
        CATEGORIES,
        tmp_path,
        api_key_env="ENVIRONMENT_VARIABLE_THAT_DOES_NOT_EXIST",
    )

    assert isinstance(result, AnnotationResult)
    assert result.status == "skipped"


def test_main_skips_existing_annotations_without_importing_openai(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    input_dir = tmp_path / "type-a" / "gpt_basic_annotation"
    output_dir = tmp_path / "type-a" / "gpt_category_annotation"
    input_dir.mkdir(parents=True)
    output_dir.mkdir(parents=True)
    (input_dir / "example.json").write_text(
        json.dumps({"category": "chair", "object_name": "wood chair"}),
        encoding="utf-8",
    )
    (output_dir / "example.json").write_text(
        json.dumps(
            {"category": "Furniture", "subcategory": "SeatingFurniture"}
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("TEST_OPENAI_API_KEY", "not-used-for-a-skipped-record")

    exit_code = main(
        [
            "--gpt-root",
            str(tmp_path),
            "--api-key-env",
            "TEST_OPENAI_API_KEY",
            "--workers",
            "1",
        ]
    )

    assert exit_code == 0
    report = json.loads(
        (tmp_path / "category_annotation_report.json").read_text(encoding="utf-8")
    )
    assert report["skipped"] == 1
    assert report["failed"] == 0
