"""Paper-aligned kinematic parameter metrics in the saved AABB frame."""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

try:
    from .common import RunningMean, as_float, numeric_vector, run_task_cli
except ImportError:  # Support: python eval/kinematic_parameters.py ...
    from common import RunningMean, as_float, numeric_vector, run_task_cli


Vector3 = Tuple[float, float, float]
Interval = Tuple[float, float]


def axis_vector(value: Any) -> Optional[Vector3]:
    if isinstance(value, Mapping):
        theta = as_float(value.get("theta"))
        phi = as_float(value.get("phi"))
        if theta is None or phi is None:
            return None
        theta_rad = math.radians(theta)
        phi_rad = math.radians(phi)
        value = (
            math.sin(theta_rad) * math.cos(phi_rad),
            math.sin(theta_rad) * math.sin(phi_rad),
            math.cos(theta_rad),
        )
    parsed = numeric_vector(value, 3)
    if parsed is None:
        return None
    norm = math.sqrt(sum(component * component for component in parsed))
    if norm <= 1e-12:
        return None
    return tuple(component / norm for component in parsed)  # type: ignore[return-value]


def axis_angle_degrees(prediction: Any, ground_truth: Any) -> Optional[float]:
    pred = axis_vector(prediction)
    gt = axis_vector(ground_truth)
    if pred is None or gt is None:
        return None
    dot = abs(sum(left * right for left, right in zip(pred, gt)))
    return math.degrees(math.acos(min(1.0, max(0.0, dot))))


def canonical_interval(value: Any) -> Optional[Interval]:
    parsed = numeric_vector(value, 2)
    if parsed is None:
        return None
    absolute = (abs(parsed[0]), abs(parsed[1]))
    return min(absolute), max(absolute)


def interval_iou(prediction: Interval, ground_truth: Interval) -> float:
    intersection = max(
        0.0,
        min(prediction[1], ground_truth[1]) - max(prediction[0], ground_truth[0]),
    )
    union = max(prediction[1], ground_truth[1]) - min(prediction[0], ground_truth[0])
    if union <= 1e-12:
        return 1.0 if abs(prediction[0] - ground_truth[0]) <= 1e-12 else 0.0
    return intersection / union


def point_to_line_distance(
    point: Vector3, line_point: Vector3, line_axis: Any
) -> Optional[float]:
    axis = axis_vector(line_axis)
    if axis is None:
        return None
    delta = tuple(left - right for left, right in zip(point, line_point))
    cross = (
        delta[1] * axis[2] - delta[2] * axis[1],
        delta[2] * axis[0] - delta[0] * axis[2],
        delta[0] * axis[1] - delta[1] * axis[0],
    )
    return math.sqrt(sum(component * component for component in cross))


@dataclass(frozen=True)
class SavedCoordinateFrame:
    center: Vector3
    scale: float
    min_bound: Optional[Vector3]
    shift: Optional[Vector3]

    @classmethod
    def from_record(cls, record: Mapping[str, Any]) -> Optional["SavedCoordinateFrame"]:
        value = record.get("coordinate_frame")
        if not isinstance(value, Mapping):
            return None
        center = numeric_vector(value.get("center"), 3)
        scale = as_float(value.get("scale"))
        min_bound = numeric_vector(value.get("min_bound"), 3)
        shift = numeric_vector(value.get("shift"), 3)
        if center is None or scale is None or scale <= 1e-12:
            return None
        if min_bound is None and shift is None:
            return None
        return cls(
            center=center,  # type: ignore[arg-type]
            scale=scale,
            min_bound=min_bound,  # type: ignore[arg-type]
            shift=shift,  # type: ignore[arg-type]
        )

    def point_to_model(self, point: Any) -> Optional[Vector3]:
        source = numeric_vector(point, 3)
        if source is None:
            return None
        normalized = tuple(
            (value - center) / self.scale for value, center in zip(source, self.center)
        )
        if self.min_bound is not None:
            return tuple(value - bound for value, bound in zip(normalized, self.min_bound))  # type: ignore[return-value]
        assert self.shift is not None
        return tuple(value + shift for value, shift in zip(normalized, self.shift))  # type: ignore[return-value]

    def range_to_model(self, value: Any) -> Optional[Interval]:
        parsed = numeric_vector(value, 2)
        if parsed is None:
            return None
        return parsed[0] / self.scale, parsed[1] / self.scale


def _ground_truth_motion(
    source: Mapping[str, Any]
) -> Optional[Tuple[str, Mapping[str, Any]]]:
    kinematic = source.get("kinematic_info")
    if not isinstance(kinematic, Mapping):
        return None
    motion_info = kinematic.get("motion_info")
    if not isinstance(motion_info, Mapping):
        return None

    candidates = []
    motion_types = kinematic.get("motion_types")
    if isinstance(motion_types, (list, tuple)):
        candidates.extend(str(value).strip() for value in motion_types)
    candidates.extend(str(key).strip() for key in motion_info)
    for motion_type in candidates:
        value = motion_info.get(motion_type)
        if motion_type in {"B", "C"} and isinstance(value, Mapping):
            return motion_type, value
    return None


@dataclass
class KinematicAccumulator:
    axis_threshold_degrees: float = 10.0
    records: int = 0
    missing_source_sample: int = 0
    missing_prediction: int = 0
    missing_coordinate_frame: int = 0
    joint_total: int = 0
    joint_correct: int = 0
    axis_error: RunningMean = field(default_factory=RunningMean)
    axis_accuracy: RunningMean = field(default_factory=RunningMean)
    pivot_error: RunningMean = field(default_factory=RunningMean)
    limit_iou: RunningMean = field(default_factory=RunningMean)

    def ingest(
        self, record: Mapping[str, Any], record_path: Optional[Path] = None
    ) -> None:
        del record_path
        self.records += 1
        source = record.get("source_sample")
        if not isinstance(source, Mapping):
            self.missing_source_sample += 1
            return
        gt_pair = _ground_truth_motion(source)
        if gt_pair is None:
            self.missing_source_sample += 1
            return
        gt_type, gt = gt_pair
        prediction = record.get("result")
        if not isinstance(prediction, Mapping):
            self.missing_prediction += 1
            self.joint_total += 1
            return

        pred_type = prediction.get("motion_type")
        self.joint_total += 1
        if pred_type != gt_type:
            return
        self.joint_correct += 1

        angle = axis_angle_degrees(prediction.get("axis"), gt.get("axis"))
        self.axis_error.add(angle)
        if angle is not None:
            self.axis_accuracy.add(float(angle < self.axis_threshold_degrees))

        frame = SavedCoordinateFrame.from_record(record)
        if frame is None:
            self.missing_coordinate_frame += 1
        elif gt_type == "C":
            pred_pivot = frame.point_to_model(prediction.get("pivot"))
            gt_pivot = frame.point_to_model(gt.get("pos"))
            if pred_pivot is not None and gt_pivot is not None:
                self.pivot_error.add(
                    point_to_line_distance(pred_pivot, gt_pivot, gt.get("axis"))
                )

        pred_range: Any = prediction.get("range")
        gt_range: Any = gt.get("range")
        if gt_type == "B":
            if frame is None:
                pred_range = None
                gt_range = None
            else:
                pred_range = frame.range_to_model(pred_range)
                gt_range = frame.range_to_model(gt_range)
        pred_interval = canonical_interval(pred_range)
        gt_interval = canonical_interval(gt_range)
        if pred_interval is not None and gt_interval is not None:
            self.limit_iou.add(interval_iou(pred_interval, gt_interval))

    def report(self, *, include_extra: bool = False) -> Dict[str, Any]:
        joint_accuracy = (
            self.joint_correct / self.joint_total if self.joint_total else None
        )
        report: Dict[str, Any] = {
            "paper_metrics": {
                "joint_type_accuracy": joint_accuracy,
                "axis_angular_error_degrees": self.axis_error.mean,
                "pivot_distance_error_aabb_0_2": self.pivot_error.mean,
                "limit_miou": self.limit_iou.mean,
            },
            "counts": {
                "records": self.records,
                "missing_source_sample": self.missing_source_sample,
                "missing_prediction": self.missing_prediction,
                "missing_coordinate_frame": self.missing_coordinate_frame,
                "joint_type": self.joint_total,
                "axis": self.axis_error.count,
                "pivot": self.pivot_error.count,
                "limit": self.limit_iou.count,
            },
        }
        if include_extra:
            report["extra_metrics"] = {
                f"axis_accuracy_lt_{self.axis_threshold_degrees:g}_degrees": self.axis_accuracy.mean,
            }
        return report


def main(argv: Optional[Sequence[str]] = None) -> None:
    run_task_cli(
        task_name="motion",
        title="Evaluate kinematic parameters from self-contained inference JSON",
        accumulator_factory=KinematicAccumulator,
        argv=argv,
    )


if __name__ == "__main__":
    main()
