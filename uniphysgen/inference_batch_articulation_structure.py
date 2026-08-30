"""Articulation-structure inference (single sample or JSON batch)."""

from inference_batch_common import run_cli


if __name__ == "__main__":
    run_cli("group", program="inference_batch_articulation_structure.py")
