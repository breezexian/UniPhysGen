"""Paper-aligned part-level intrinsic physical property metrics."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence

try:
    from .common import RunningMean, absolute_error, nested, positive_alde, run_task_cli
except ImportError:  # Support: python eval/intrinsic_physics_part.py ...
    from common import RunningMean, absolute_error, nested, positive_alde, run_task_cli


def _material_category(value: Any) -> Optional[str]:
    if not isinstance(value, str):
        return None
    category = value.split("/", 1)[0].strip().casefold()
    return category or None


@dataclass
class PartPhysicsAccumulator:
    records: int = 0
    missing_source_sample: int = 0
    missing_prediction: int = 0
    material: RunningMean = field(default_factory=RunningMean)
    density: RunningMean = field(default_factory=RunningMean)
    friction: RunningMean = field(default_factory=RunningMean)
    affordance: RunningMean = field(default_factory=RunningMean)
    young: RunningMean = field(default_factory=RunningMean)
    hardness: RunningMean = field(default_factory=RunningMean)
    poisson: RunningMean = field(default_factory=RunningMean)
    graspable: RunningMean = field(default_factory=RunningMean)

    def ingest(
        self, record: Mapping[str, Any], record_path: Optional[Path] = None
    ) -> None:
        del record_path
        self.records += 1
        source = record.get("source_sample")
        if not isinstance(source, Mapping):
            self.missing_source_sample += 1
            return
        prediction = nested(record, "result", "physical_properties")
        if not isinstance(prediction, Mapping):
            self.missing_prediction += 1
            return

        gt_basic = source.get("basic_info")
        gt_part = source.get("part_level")
        if not isinstance(gt_basic, Mapping):
            self.missing_source_sample += 1
            return

        pred_material = _material_category(prediction.get("material"))
        gt_material = _material_category(gt_basic.get("material"))
        if pred_material is not None and gt_material is not None:
            self.material.add(float(pred_material == gt_material))
        self.density.add(
            positive_alde(prediction.get("density"), gt_basic.get("density"))
        )
        self.friction.add(
            absolute_error(prediction.get("friction"), gt_basic.get("friction"))
        )

        self.young.add(positive_alde(prediction.get("young"), gt_basic.get("young")))
        self.hardness.add(
            positive_alde(prediction.get("hardness"), gt_basic.get("hardness"))
        )
        self.poisson.add(
            absolute_error(prediction.get("poisson"), gt_basic.get("poisson"))
        )
        if isinstance(gt_part, Mapping):
            self.affordance.add(
                absolute_error(prediction.get("affordance"), gt_part.get("affordance"))
            )
            pred_graspable = prediction.get("graspable")
            gt_graspable = gt_part.get("graspable")
            if isinstance(pred_graspable, bool) and isinstance(gt_graspable, bool):
                self.graspable.add(float(pred_graspable == gt_graspable))

    def report(self, *, include_extra: bool = False) -> Dict[str, Any]:
        report: Dict[str, Any] = {
            "paper_metrics": {
                "material_accuracy": self.material.mean,
                "density_alde": self.density.mean,
                "friction_mae": self.friction.mean,
                "affordance_mae": self.affordance.mean,
            },
            "counts": {
                "records": self.records,
                "missing_source_sample": self.missing_source_sample,
                "missing_prediction": self.missing_prediction,
                "material": self.material.count,
                "density": self.density.count,
                "friction": self.friction.count,
                "affordance": self.affordance.count,
            },
        }
        if include_extra:
            report["extra_metrics"] = {
                "young_alde": self.young.mean,
                "hardness_alde": self.hardness.mean,
                "poisson_mae": self.poisson.mean,
                "graspable_accuracy": self.graspable.mean,
            }
        return report


def main(argv: Optional[Sequence[str]] = None) -> None:
    run_task_cli(
        task_name="physics",
        title="Evaluate part-level intrinsic physics from self-contained inference JSON",
        accumulator_factory=PartPhysicsAccumulator,
        argv=argv,
    )


if __name__ == "__main__":
    main()
