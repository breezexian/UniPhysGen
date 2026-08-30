"""Object-level intrinsic-physics inference (single sample or JSON batch)."""

from inference_batch_common import run_cli


if __name__ == "__main__":
    run_cli("object_level", program="inference_batch_intrinsic_physics_object.py")
