"""Kinematic-parameter inference (single sample or JSON batch)."""

from inference_batch_common import run_cli


if __name__ == "__main__":
    run_cli("motion", program="inference_batch_kinematic_parameters.py")
