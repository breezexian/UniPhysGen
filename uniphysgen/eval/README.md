# UniPhysGen evaluation

The evaluators consume only JSON written by the four `inference_batch_*.py`
entry points. Ground truth is read exclusively from each record's embedded
`source_sample`; the deprecated `gt` field, external annotation roots, entity
lists, and point-cloud reconstruction paths are not used.

Each successful inference record has this stable shape:

```json
{
  "schema_version": "uniphysgen.inference.v1",
  "task": "motion",
  "source_sample": {"...": "the complete original sample"},
  "coordinate_frame": {
    "name": "source_to_aabb_0_2",
    "center": [0.0, 0.0, 0.0],
    "scale": 1.0,
    "min_bound": [-1.0, -1.0, -1.0]
  },
  "model_result": {"...": "direct normalized model output"},
  "result": {"...": "prediction restored to source/annotation units"},
  "raw_response": "..."
}
```

Run one evaluator on either a consolidated batch JSON file or a directory of
per-sample JSON files:

```bash
python -m eval intrinsic_physics_part PREDICTIONS --output part_metrics.json
python -m eval intrinsic_physics_object PREDICTIONS --output object_metrics.json
python -m eval kinematic_parameters PREDICTIONS --output motion_metrics.json
python -m eval articulation_structure PREDICTIONS --output structure_metrics.json
```

By default, `paper_metrics` contains only the metrics from Table 2. Use
`--include-extra` to expose retained diagnostics. Accuracy, mIoU, F1, and MnRE
are stored in `[0, 1]`; multiply by 100 only when formatting a paper table.

For kinematic evaluation, `result.pivot` and the source GT pivot are both
mapped through the saved transform before computing point-to-GT-axis distance.
Prismatic ranges are divided by the same saved `scale`; revolute ranges remain
in radians. No point cloud is reopened during evaluation.
