"""Paper-aligned articulation structure set metrics."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Set

try:
    from .common import nested, run_task_cli
except ImportError:  # Support: python eval/articulation_structure.py ...
    from common import nested, run_task_cli


def _part_id(value: Any) -> Optional[str]:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return str(int(value)) if value.is_integer() else None
    if isinstance(value, str):
        normalized = value.strip()
        return normalized or None
    return None


def _member_set(value: Any) -> Optional[Set[str]]:
    if not isinstance(value, (list, tuple)):
        return None
    members: Set[str] = set()
    for raw_member in value:
        member = _part_id(raw_member)
        if member is None:
            return None
        members.add(member)
    return members


@dataclass
class StructureAccumulator:
    records: int = 0
    valid_pairs: int = 0
    missing_pair: int = 0
    iou_sum: float = 0.0
    macro_precision_sum: float = 0.0
    macro_recall_sum: float = 0.0
    macro_f1_sum: float = 0.0
    exact_match_sum: float = 0.0
    true_positive: int = 0
    false_positive: int = 0
    false_negative: int = 0

    def ingest(
        self, record: Mapping[str, Any], record_path: Optional[Path] = None
    ) -> None:
        del record_path
        self.records += 1
        prediction = _member_set(nested(record, "result", "members"))
        ground_truth = _member_set(
            nested(
                record, "source_sample", "kinematic_info", "motion_info", "dependency"
            )
        )
        if prediction is None or ground_truth is None:
            self.missing_pair += 1
            return
        self.valid_pairs += 1

        intersection = prediction & ground_truth
        union = prediction | ground_truth
        self.iou_sum += len(intersection) / len(union) if union else 1.0
        tp = len(intersection)
        fp = len(prediction - ground_truth)
        fn = len(ground_truth - prediction)
        precision = tp / (tp + fp) if tp + fp else (1.0 if not ground_truth else 0.0)
        recall = tp / (tp + fn) if tp + fn else 1.0
        f1 = (
            2.0 * precision * recall / (precision + recall)
            if precision + recall
            else 0.0
        )
        self.macro_precision_sum += precision
        self.macro_recall_sum += recall
        self.macro_f1_sum += f1
        self.exact_match_sum += float(prediction == ground_truth)
        self.true_positive += tp
        self.false_positive += fp
        self.false_negative += fn

    def report(self, *, include_extra: bool = False) -> Dict[str, Any]:
        if self.true_positive + self.false_positive:
            micro_precision = self.true_positive / (
                self.true_positive + self.false_positive
            )
        else:
            micro_precision = 1.0 if self.false_negative == 0 else 0.0
        if self.true_positive + self.false_negative:
            micro_recall = self.true_positive / (
                self.true_positive + self.false_negative
            )
        else:
            micro_recall = 1.0
        if micro_precision + micro_recall:
            micro_f1 = (
                2.0 * micro_precision * micro_recall / (micro_precision + micro_recall)
            )
        else:
            micro_f1 = 0.0
        denominator = self.valid_pairs
        report: Dict[str, Any] = {
            "paper_metrics": {
                "structure_miou": self.iou_sum / denominator if denominator else None,
                "structure_micro_f1": micro_f1 if denominator else None,
            },
            "counts": {
                "records": self.records,
                "valid_pairs": self.valid_pairs,
                "missing_pair": self.missing_pair,
                "true_positive": self.true_positive,
                "false_positive": self.false_positive,
                "false_negative": self.false_negative,
            },
        }
        if include_extra:
            report["extra_metrics"] = {
                "macro_precision": (
                    self.macro_precision_sum / denominator if denominator else None
                ),
                "macro_recall": (
                    self.macro_recall_sum / denominator if denominator else None
                ),
                "macro_f1": self.macro_f1_sum / denominator if denominator else None,
                "micro_precision": micro_precision if denominator else None,
                "micro_recall": micro_recall if denominator else None,
                "exact_match": (
                    self.exact_match_sum / denominator if denominator else None
                ),
            }
        return report


def main(argv: Optional[Sequence[str]] = None) -> None:
    run_task_cli(
        task_name="group",
        title="Evaluate articulation structure from self-contained inference JSON",
        accumulator_factory=StructureAccumulator,
        argv=argv,
    )


if __name__ == "__main__":
    main()
