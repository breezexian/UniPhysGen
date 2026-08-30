from __future__ import annotations

import subprocess
import sys


def test_utils_package_does_not_eagerly_import_optional_modules() -> None:
    command = (
        "import sys, utils; "
        "assert 'utils.physics_basic_annotate' not in sys.modules; "
        "assert 'utils.validate_mujoco_simulation' not in sys.modules"
    )
    subprocess.run([sys.executable, "-c", command], check=True)


def test_legacy_mesh_export_is_resolved_lazily() -> None:
    command = (
        "import sys, utils; "
        "assert 'read_mesh' in dir(utils); "
        "assert 'utils.mesh_handle' not in sys.modules"
    )
    subprocess.run([sys.executable, "-c", command], check=True)
