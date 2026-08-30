from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("trimesh")

from utils import part_refine_after_merge, post_process


@pytest.mark.parametrize(
    ("use_relative_threshold", "expected"),
    [(False, 1e-2), (True, 0.04)],
)
def test_part_refine_after_merge_threshold(
    tmp_path, monkeypatch, use_relative_threshold, expected
):
    face_map = tmp_path / "face2cls.json"
    face_map.write_text(json.dumps({"0": 0}), encoding="utf-8")
    mesh = SimpleNamespace(
        vertices=np.zeros((3, 3)),
        faces=np.array([[0, 1, 2]]),
        bounding_box=SimpleNamespace(extents=np.array([2.0, 3.0, 4.0])),
    )
    captured = {}

    monkeypatch.setattr(part_refine_after_merge, "read_mesh", lambda _path: mesh)

    def fake_refine(vertices, faces, mask2face, save_dir, eps):
        captured["eps"] = eps
        return [[0]]

    monkeypatch.setattr(
        part_refine_after_merge, "refine_parts_with_connectivity", fake_refine
    )

    result = part_refine_after_merge.part_refine_after_merge_main(
        "mesh.glb",
        face_map,
        tmp_path,
        use_relative_threshold=use_relative_threshold,
    )

    assert result == [[0]]
    assert captured["eps"] == pytest.approx(expected)


@pytest.mark.parametrize(
    ("use_relative_threshold", "expected"),
    [(False, 1e-3), (True, 0.004)],
)
def test_post_process_threshold(
    tmp_path, monkeypatch, use_relative_threshold, expected
):
    mesh = SimpleNamespace(
        vertices=np.empty((0, 3)),
        faces=np.empty((0, 3), dtype=int),
        bounding_box=SimpleNamespace(extents=np.array([2.0, 3.0, 4.0])),
    )
    captured = {}

    monkeypatch.setattr(post_process, "read_mesh", lambda _path: mesh)

    def fake_merge(parts, **kwargs):
        captured["dist_thresh"] = kwargs["dist_thresh"]
        return [], []

    monkeypatch.setattr(post_process, "merge_small_parts", fake_merge)

    result = post_process.post_process_main(
        [],
        "mesh.glb",
        tmp_path,
        use_relative_threshold=use_relative_threshold,
    )

    assert result == []
    assert captured["dist_thresh"] == pytest.approx(expected)
