"""Paper-aligned object-level scale and mass metrics."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

try:
    from .common import (
        RunningMean,
        absolute_error,
        as_float,
        numeric_vector,
        positive_alde,
        positive_min_ratio,
        run_task_cli,
    )
except ImportError:  # Support: python eval/intrinsic_physics_object.py ...
    from common import (
        RunningMean,
        absolute_error,
        as_float,
        numeric_vector,
        positive_alde,
        positive_min_ratio,
        run_task_cli,
    )


def _dimensions(value: Any) -> Optional[Tuple[float, float, float]]:
    parsed = numeric_vector(value, 3)
    if parsed is None or any(item <= 0.0 for item in parsed):
        return None
    # Keep the legacy/table protocol invariant to L/W/H permutation.
    return tuple(sorted(parsed))


@dataclass
class ObjectPhysicsAccumulator:
    records: int = 0
    missing_source_sample: int = 0
    missing_prediction: int = 0
    scale_alde: RunningMean = field(default_factory=RunningMean)
    scale_mnre: RunningMean = field(default_factory=RunningMean)
    mass_alde: RunningMean = field(default_factory=RunningMean)
    mass_mnre: RunningMean = field(default_factory=RunningMean)
    scale_mae: RunningMean = field(default_factory=RunningMean)
    mass_mae: RunningMean = field(default_factory=RunningMean)
    category: RunningMean = field(default_factory=RunningMean)

    def ingest(
        self, record: Mapping[str, Any], record_path: Optional[Path] = None
    ) -> None:
        del record_path
        self.records += 1
        source = record.get("source_sample")
        if not isinstance(source, Mapping):
            self.missing_source_sample += 1
            return
        gt = source.get("object_level")
        prediction = record.get("result")
        if not isinstance(gt, Mapping):
            self.missing_source_sample += 1
            return
        if not isinstance(prediction, Mapping):
            self.missing_prediction += 1
            return

        pred_dimensions = _dimensions(prediction.get("volume"))
        gt_dimensions = _dimensions(gt.get("volume"))
        if pred_dimensions is not None and gt_dimensions is not None:
            aldes = [
                positive_alde(pred, target)
                for pred, target in zip(pred_dimensions, gt_dimensions)
            ]
            ratios = [
                positive_min_ratio(pred, target)
                for pred, target in zip(pred_dimensions, gt_dimensions)
            ]
            maes = [
                absolute_error(pred, target)
                for pred, target in zip(pred_dimensions, gt_dimensions)
            ]
            if all(value is not None for value in aldes):
                self.scale_alde.add(
                    sum(value for value in aldes if value is not None) / 3.0
                )
            if all(value is not None for value in ratios):
                self.scale_mnre.add(
                    sum(value for value in ratios if value is not None) / 3.0
                )
            if all(value is not None for value in maes):
                self.scale_mae.add(
                    sum(value for value in maes if value is not None) / 3.0
                )

        pred_mass = as_float(prediction.get("mass"))
        gt_mass = as_float(gt.get("mass"))
        self.mass_alde.add(positive_alde(pred_mass, gt_mass))
        self.mass_mnre.add(positive_min_ratio(pred_mass, gt_mass))
        self.mass_mae.add(absolute_error(pred_mass, gt_mass))

        pred_category = prediction.get("category")
        gt_category = gt.get("category")
        if isinstance(pred_category, str) and isinstance(gt_category, str):
            self.category.add(
                float(
                    pred_category.strip().casefold() == gt_category.strip().casefold()
                )
            )

    def report(self, *, include_extra: bool = False) -> Dict[str, Any]:
        report: Dict[str, Any] = {
            "paper_metrics": {
                "scale_alde": self.scale_alde.mean,
                "scale_mnre": self.scale_mnre.mean,
                "mass_alde": self.mass_alde.mean,
                "mass_mnre": self.mass_mnre.mean,
            },
            "counts": {
                "records": self.records,
                "missing_source_sample": self.missing_source_sample,
                "missing_prediction": self.missing_prediction,
                "scale": self.scale_alde.count,
                "mass": self.mass_alde.count,
            },
        }
        if include_extra:
            report["extra_metrics"] = {
                "scale_mae": self.scale_mae.mean,
                "mass_mae": self.mass_mae.mean,
                "category_accuracy": self.category.mean,
            }
        return report


def main(argv: Optional[Sequence[str]] = None) -> None:
    run_task_cli(
        task_name="object_level",
        title="Evaluate object-level scale and mass from self-contained inference JSON",
        accumulator_factory=ObjectPhysicsAccumulator,
        argv=argv,
    )


if __name__ == "__main__":
    main()
