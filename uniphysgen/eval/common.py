"""Shared JSON input and command-line helpers for evaluation."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
)


EVAL_SCHEMA_VERSION = "uniphysgen.eval.v1"
TASK_ALIASES = {
    "physics": "physics",
    "intrinsic_physics_part": "physics",
    "part": "physics",
    "object_level": "object_level",
    "intrinsic_physics_object": "object_level",
    "object": "object_level",
    "motion": "motion",
    "kinematic_parameters": "motion",
    "kinematic": "motion",
    "group": "group",
    "articulation_structure": "group",
    "structure": "group",
}


@dataclass
class RunningMean:
    total: float = 0.0
    count: int = 0

    def add(self, value: Optional[float]) -> None:
        if value is None or not math.isfinite(value):
            return
        self.total += value
        self.count += 1

    @property
    def mean(self) -> Optional[float]:
        return self.total / self.count if self.count else None


def as_float(value: Any) -> Optional[float]:
    if value is None or isinstance(value, bool):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def numeric_vector(value: Any, length: int) -> Optional[Tuple[float, ...]]:
    if not isinstance(value, (list, tuple)) or len(value) != length:
        return None
    parsed = tuple(as_float(item) for item in value)
    if any(item is None for item in parsed):
        return None
    return tuple(float(item) for item in parsed if item is not None)


def nested(mapping: Any, *keys: str) -> Any:
    current = mapping
    for key in keys:
        if not isinstance(current, Mapping):
            return None
        current = current.get(key)
    return current


def positive_alde(prediction: Any, ground_truth: Any) -> Optional[float]:
    pred = as_float(prediction)
    gt = as_float(ground_truth)
    if pred is None or gt is None or pred <= 0.0 or gt <= 0.0:
        return None
    return abs(math.log(pred) - math.log(gt))


def positive_min_ratio(prediction: Any, ground_truth: Any) -> Optional[float]:
    pred = as_float(prediction)
    gt = as_float(ground_truth)
    if pred is None or gt is None or pred <= 0.0 or gt <= 0.0:
        return None
    return min(pred / gt, gt / pred)


def absolute_error(prediction: Any, ground_truth: Any) -> Optional[float]:
    pred = as_float(prediction)
    gt = as_float(ground_truth)
    if pred is None or gt is None:
        return None
    return abs(pred - gt)


def _json_paths(inputs: Sequence[Path]) -> Iterable[Path]:
    seen: set[Path] = set()
    for input_path in inputs:
        path = input_path.expanduser().resolve(strict=False)
        if path.is_file():
            candidates = [path]
        elif path.is_dir():
            candidates = sorted(
                candidate for candidate in path.rglob("*.json") if candidate.is_file()
            )
        else:
            raise FileNotFoundError(f"evaluation input does not exist: {path}")
        for candidate in candidates:
            if candidate not in seen:
                seen.add(candidate)
                yield candidate


def load_records(
    inputs: Sequence[Path],
) -> Tuple[List[Tuple[Dict[str, Any], Path]], List[str]]:
    """Load individual-record JSON files and consolidated JSON arrays."""

    records: List[Tuple[Dict[str, Any], Path]] = []
    errors: List[str] = []
    for path in _json_paths(inputs):
        try:
            with path.open("r", encoding="utf-8") as stream:
                payload = json.load(stream)
        except Exception as exc:
            errors.append(f"{path}: {exc}")
            continue

        if isinstance(payload, Mapping) and isinstance(payload.get("records"), list):
            payload = payload["records"]
        values = payload if isinstance(payload, list) else [payload]
        for index, value in enumerate(values):
            if isinstance(value, Mapping):
                records.append((dict(value), path))
            else:
                errors.append(f"{path}#{index}: inference record must be a JSON object")
    return records, errors


def canonical_task(value: Any) -> Optional[str]:
    if not isinstance(value, str):
        return None
    return TASK_ALIASES.get(value.strip().lower())


def evaluate_inputs(
    *,
    task_name: str,
    accumulator: Any,
    inputs: Sequence[Path],
    include_extra: bool,
) -> Dict[str, Any]:
    records, load_errors = load_records(inputs)
    selected = 0
    skipped_other_task = 0
    ingest_errors: List[str] = []
    for record, path in records:
        if canonical_task(record.get("task")) != task_name:
            skipped_other_task += 1
            continue
        selected += 1
        try:
            accumulator.ingest(record, record_path=path)
        except Exception as exc:
            ingest_errors.append(
                f"{path} ({record.get('sample_id', '<unknown>')}): {exc}"
            )

    report = accumulator.report(include_extra=include_extra)
    report.update(
        {
            "schema_version": EVAL_SCHEMA_VERSION,
            "task": task_name,
            "inputs": [str(path.expanduser().resolve(strict=False)) for path in inputs],
            "input_summary": {
                "loaded_records": len(records),
                "evaluated_records": selected,
                "skipped_other_task": skipped_other_task,
                "load_error_count": len(load_errors),
                "ingest_error_count": len(ingest_errors),
            },
        }
    )
    if load_errors or ingest_errors:
        report["errors"] = load_errors + ingest_errors
    return report


def _print_report(report: Mapping[str, Any]) -> None:
    print(f"task: {report.get('task')}")
    for name, value in (report.get("paper_metrics") or {}).items():
        rendered = "N/A" if value is None else f"{float(value):.6f}"
        print(f"{name}: {rendered}")
    counts = report.get("counts") or {}
    if counts:
        print("counts: " + ", ".join(f"{key}={value}" for key, value in counts.items()))
    if "extra_metrics" in report:
        print("extra_metrics:")
        for name, value in report["extra_metrics"].items():
            rendered = "N/A" if value is None else f"{float(value):.6f}"
            print(f"  {name}: {rendered}")


def run_task_cli(
    *,
    task_name: str,
    title: str,
    accumulator_factory: Callable[[], Any],
    argv: Optional[Sequence[str]] = None,
) -> None:
    parser = argparse.ArgumentParser(description=title)
    parser.add_argument(
        "inputs",
        nargs="+",
        type=Path,
        help="Inference result JSON file(s) or directories containing JSON files",
    )
    parser.add_argument("--output", type=Path, help="Optional JSON metrics report")
    parser.add_argument(
        "--include-extra",
        action="store_true",
        help="Also report retained diagnostics that are not in the paper's main table",
    )
    args = parser.parse_args(argv)
    try:
        report = evaluate_inputs(
            task_name=task_name,
            accumulator=accumulator_factory(),
            inputs=args.inputs,
            include_extra=args.include_extra,
        )
    except Exception as exc:
        parser.error(str(exc))

    _print_report(report)
    if args.output is not None:
        output = args.output.expanduser().resolve(strict=False)
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("w", encoding="utf-8") as stream:
            json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write("\n")
        print(f"saved: {output}")
