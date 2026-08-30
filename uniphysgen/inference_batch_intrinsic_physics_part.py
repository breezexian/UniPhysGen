"""Part-level intrinsic-physics inference (single sample or JSON batch)."""

from inference_batch_common import run_cli


if __name__ == "__main__":
    run_cli("physics", program="inference_batch_intrinsic_physics_part.py")
