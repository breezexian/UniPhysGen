"""Batch object-category annotation with an OpenAI-compatible API.

Credentials are read exclusively from an environment variable. This script is
also importable for tests and library use without requiring the OpenAI package
until an API request is actually made.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

try:
    from .CAT import CATEGORIES
except ImportError:  # Support direct execution: python category/gpt_cat.py
    from CAT import CATEGORIES


CATEGORY_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CATEGORY_DIR.parent
DEFAULT_PROMPT = CATEGORY_DIR / "prompt.txt"
DEFAULT_GPT_ROOT = PROJECT_ROOT / "outputs/ABO/data/gpt_output"
DEFAULT_API_KEY_ENV = "OPENAI_API_KEY"
JSON_BLOCK = re.compile(r"===BEGIN_JSON===(.*?)===END_JSON===", re.DOTALL)


class CategoryAnnotationError(RuntimeError):
    """Raised when an annotation response or input record is invalid."""


@dataclass(frozen=True)
class AnnotationResult:
    entity: str
    source: str
    output: str
    status: str
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    message: str = ""


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return parsed


def _nonnegative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be 0 or greater")
    return parsed


def _positive_float(value: str) -> float:
    parsed = float(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be greater than 0")
    return parsed


def _read_json_object(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise CategoryAnnotationError(f"Invalid JSON file {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise CategoryAnnotationError(f"Expected a JSON object in {path}.")
    return data


def _write_json_atomic(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(data, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def parse_classification(
    response: str | None,
    categories: Mapping[str, Mapping[str, Any]] = CATEGORIES,
) -> dict[str, str]:
    """Extract and validate a category/subcategory pair from a model response."""

    if not response or not response.strip():
        raise CategoryAnnotationError("Model returned an empty response.")

    match = JSON_BLOCK.search(response)
    payload = match.group(1).strip() if match else response.strip()
    if payload.startswith("```json"):
        payload = payload[len("```json") :].strip()
    elif payload.startswith("```"):
        payload = payload[3:].strip()
    if payload.endswith("```"):
        payload = payload[:-3].strip()

    try:
        data = json.loads(payload)
    except ValueError as exc:
        raise CategoryAnnotationError(
            "Model response does not contain valid classification JSON."
        ) from exc
    if not isinstance(data, dict):
        raise CategoryAnnotationError("Classification response must be a JSON object.")

    category = data.get("category")
    subcategory = data.get("subcategory")
    if not isinstance(category, str) or category not in categories:
        raise CategoryAnnotationError(f"Unknown category: {category!r}.")
    allowed = categories[category].get("subcategories", ())
    if not isinstance(subcategory, str) or subcategory not in allowed:
        raise CategoryAnnotationError(
            f"Unknown subcategory {subcategory!r} for category {category!r}."
        )
    return {"category": category, "subcategory": subcategory}


def _existing_annotation_valid(
    path: Path,
    categories: Mapping[str, Mapping[str, Any]],
) -> bool:
    if not path.is_file():
        return False
    try:
        data = _read_json_object(path)
        parse_classification(json.dumps(data), categories)
    except CategoryAnnotationError:
        return False
    return True


def _save_failed_response(save_dir: Path, entity: str, response: str) -> Path:
    error_path = save_dir / "errors" / f"{entity}.response.txt"
    error_path.parent.mkdir(parents=True, exist_ok=True)
    error_path.write_text(response, encoding="utf-8")
    return error_path


def _create_client(
    *,
    api_key_env: str,
    base_url: str | None,
    timeout: float,
    max_retries: int,
) -> Any:
    api_key = os.environ.get(api_key_env, "").strip()
    if not api_key:
        raise CategoryAnnotationError(
            f"Environment variable '{api_key_env}' is required."
        )

    try:
        from openai import OpenAI
    except ImportError as exc:
        raise CategoryAnnotationError(
            "The 'openai' package is required; install the project runtime dependencies."
        ) from exc

    return OpenAI(
        api_key=api_key,
        base_url=base_url or os.environ.get("OPENAI_BASE_URL") or None,
        timeout=timeout,
        max_retries=max_retries,
    )


def gpt_cate_annotation_main(
    entity: str,
    obj_cate: str,
    obj_name: str,
    predf_cates: Mapping[str, Mapping[str, Any]],
    save_gpt_dir: str | Path,
    use_model: str = "gpt-5",
    *,
    source_file: str | Path | None = None,
    prompt_path: str | Path = DEFAULT_PROMPT,
    api_key_env: str = DEFAULT_API_KEY_ENV,
    base_url: str | None = None,
    timeout: float = 300.0,
    max_retries: int = 2,
    overwrite: bool = False,
) -> AnnotationResult:
    """Classify one object and atomically save its validated JSON annotation."""

    save_dir = Path(save_gpt_dir).expanduser().resolve()
    output_file = save_dir / f"{entity}.json"
    source = str(Path(source_file).resolve()) if source_file else ""
    if not overwrite and _existing_annotation_valid(output_file, predf_cates):
        return AnnotationResult(
            entity=entity,
            source=source,
            output=str(output_file),
            status="skipped",
            message="Existing valid annotation.",
        )

    response_text = ""
    try:
        system_prompt = Path(prompt_path).expanduser().resolve().read_text(
            encoding="utf-8"
        )
        client = _create_client(
            api_key_env=api_key_env,
            base_url=base_url,
            timeout=timeout,
            max_retries=max_retries,
        )
        user_prompt = (
            "Classify the object into the provided categories.\n"
            f'Object Name: "{obj_name}"\n'
            f'Current Category Label: "{obj_cate}"\n\n'
            "Output only the classification JSON between `===BEGIN_JSON===` "
            "and `===END_JSON===`. Use the most specific subcategory in the "
            "provided category system. If nothing fits well, choose the "
            "closest `_Other` subcategory. Do not invent categories."
        )
        response = client.chat.completions.create(
            model=use_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        response_text = response.choices[0].message.content or ""
        classification = parse_classification(response_text, predf_cates)
        _write_json_atomic(output_file, classification)

        usage = getattr(response, "usage", None)
        return AnnotationResult(
            entity=entity,
            source=source,
            output=str(output_file),
            status="succeeded",
            prompt_tokens=getattr(usage, "prompt_tokens", None),
            completion_tokens=getattr(usage, "completion_tokens", None),
        )
    except Exception as exc:
        detail = f"{type(exc).__name__}: {exc}"
        if response_text:
            error_path = _save_failed_response(save_dir, entity, response_text)
            detail = f"{detail} Raw response: {error_path}"
        return AnnotationResult(
            entity=entity,
            source=source,
            output=str(output_file),
            status="failed",
            message=detail,
        )


def _discover_type_directories(
    gpt_root: Path,
    requested: Sequence[str] | None,
    input_subdir: str,
) -> list[tuple[str, Path]]:
    if not gpt_root.is_dir():
        raise CategoryAnnotationError(
            f"GPT output root does not exist or is not a directory: {gpt_root}"
        )

    if requested:
        directories = [(name, gpt_root / name) for name in requested]
    elif (gpt_root / input_subdir).is_dir():
        directories = [(gpt_root.name, gpt_root)]
    else:
        directories = [
            (path.name, path)
            for path in sorted(gpt_root.iterdir())
            if path.is_dir() and (path / input_subdir).is_dir()
        ]

    missing = [
        str(path / input_subdir)
        for _name, path in directories
        if not (path / input_subdir).is_dir()
    ]
    if missing:
        raise CategoryAnnotationError(
            "Missing category input directories: " + ", ".join(missing)
        )
    if not directories:
        raise CategoryAnnotationError(
            f"No '{input_subdir}' directories were found under {gpt_root}."
        )
    return directories


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Classify existing object annotations into the UniPhys taxonomy."
    )
    parser.add_argument(
        "--gpt-root",
        type=Path,
        default=DEFAULT_GPT_ROOT,
        help=f"Root containing type directories (default: {DEFAULT_GPT_ROOT}).",
    )
    parser.add_argument(
        "--type-name",
        action="append",
        help="Only process this type directory; repeat to select multiple types.",
    )
    parser.add_argument(
        "--input-subdir",
        default="gpt_basic_annotation",
        help="Input annotation subdirectory (default: gpt_basic_annotation).",
    )
    parser.add_argument(
        "--output-subdir",
        default="gpt_category_annotation",
        help="Category output subdirectory (default: gpt_category_annotation).",
    )
    parser.add_argument(
        "--prompt",
        type=Path,
        default=DEFAULT_PROMPT,
        help=f"System prompt file (default: {DEFAULT_PROMPT}).",
    )
    parser.add_argument(
        "--model",
        default=os.environ.get("OPENAI_MODEL", "gpt-5"),
        help="OpenAI-compatible model name (default: OPENAI_MODEL or gpt-5).",
    )
    parser.add_argument(
        "--base-url",
        help="API base URL (default: OPENAI_BASE_URL or the OpenAI default).",
    )
    parser.add_argument(
        "--api-key-env",
        default=DEFAULT_API_KEY_ENV,
        help="Environment variable containing the API key (default: OPENAI_API_KEY).",
    )
    parser.add_argument(
        "--workers",
        type=_positive_int,
        default=5,
        help="Number of concurrent API requests (default: 5).",
    )
    parser.add_argument(
        "--timeout",
        type=_positive_float,
        default=300.0,
        help="Per-request timeout in seconds (default: 300).",
    )
    parser.add_argument(
        "--max-retries",
        type=_nonnegative_int,
        default=2,
        help="OpenAI SDK retry count (default: 2).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing valid category annotations.",
    )
    return parser


def _print_result(index: int, total: int, result: AnnotationResult) -> None:
    detail = f" ({result.message})" if result.message else ""
    print(
        f"[{index}/{total}] {result.status.upper()}: {result.entity}{detail}",
        flush=True,
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    gpt_root = args.gpt_root.expanduser().resolve()
    prompt_path = args.prompt.expanduser().resolve()
    if not prompt_path.is_file():
        parser.error(f"Prompt file does not exist: {prompt_path}")
    if not os.environ.get(args.api_key_env, "").strip():
        parser.error(
            f"Environment variable '{args.api_key_env}' is required; "
            "do not place API keys in source code."
        )

    try:
        type_directories = _discover_type_directories(
            gpt_root,
            args.type_name,
            args.input_subdir,
        )
    except CategoryAnnotationError as exc:
        parser.error(str(exc))

    jobs: list[tuple[str, str, str, Path, Path]] = []
    results: list[AnnotationResult] = []
    for _type_name, type_dir in type_directories:
        input_dir = type_dir / args.input_subdir
        output_dir = type_dir / args.output_subdir
        for source_file in sorted(input_dir.glob("*.json")):
            entity = source_file.stem
            output_file = output_dir / f"{entity}.json"
            try:
                record = _read_json_object(source_file)
                obj_cate = record["category"]
                obj_name = record["object_name"]
                if not isinstance(obj_cate, str) or not isinstance(obj_name, str):
                    raise TypeError("category and object_name must be strings")
            except (CategoryAnnotationError, KeyError, TypeError) as exc:
                results.append(
                    AnnotationResult(
                        entity=entity,
                        source=str(source_file),
                        output=str(output_file),
                        status="failed",
                        message=f"Invalid input: {exc}",
                    )
                )
                continue
            jobs.append((entity, obj_cate, obj_name, source_file, output_dir))

    if not jobs and not results:
        parser.error("No input JSON files were found.")

    total = len(jobs) + len(results)
    completed = 0
    for result in results:
        completed += 1
        _print_result(completed, total, result)

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(
                gpt_cate_annotation_main,
                entity,
                obj_cate,
                obj_name,
                CATEGORIES,
                output_dir,
                args.model,
                source_file=source_file,
                prompt_path=prompt_path,
                api_key_env=args.api_key_env,
                base_url=args.base_url,
                timeout=args.timeout,
                max_retries=args.max_retries,
                overwrite=args.overwrite,
            ): entity
            for entity, obj_cate, obj_name, source_file, output_dir in jobs
        }
        for future in as_completed(futures):
            completed += 1
            try:
                result = future.result()
            except Exception as exc:  # Defensive boundary around worker threads.
                entity = futures[future]
                result = AnnotationResult(
                    entity=entity,
                    source="",
                    output="",
                    status="failed",
                    message=f"Worker crashed: {type(exc).__name__}: {exc}",
                )
            results.append(result)
            _print_result(completed, total, result)

    results.sort(key=lambda item: (item.source, item.entity))
    report = {
        "gpt_root": str(gpt_root),
        "model": args.model,
        "total": len(results),
        "succeeded": sum(item.status == "succeeded" for item in results),
        "skipped": sum(item.status == "skipped" for item in results),
        "failed": sum(item.status == "failed" for item in results),
        "results": [asdict(item) for item in results],
    }
    report_path = gpt_root / "category_annotation_report.json"
    _write_json_atomic(report_path, report)
    print(
        "Summary: "
        f"succeeded={report['succeeded']} skipped={report['skipped']} "
        f"failed={report['failed']}"
    )
    print(f"Report: {report_path}")
    return 1 if report["failed"] else 0


if __name__ == "__main__":
    sys.exit(main())
