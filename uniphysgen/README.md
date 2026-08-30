<!-- markdownlint-disable MD013 MD033 MD041 -->

<div align="center">

<h1><a id="top"></a>UniPhysGen</h1>

<p><strong>Unified Physical Grounding for Simulation-Ready 3D Assets</strong></p>

<p>
  <a href="https://arxiv.org/abs/2607.13586">
    <img src="https://img.shields.io/badge/arXiv-2607.13586-b31b1b.svg" alt="arXiv" />
  </a>
  <a href="https://huggingface.co/breezexian/UniPhysGen-1.7B-Physics">
    <img src="https://img.shields.io/badge/Models-Available-FFD21E" alt="Hugging Face models" />
  </a>
  <a href="https://huggingface.co/datasets/spatialverse/UniPhys-40K">
    <img src="https://img.shields.io/badge/Datasets-Available-FFD21E" alt="Hugging Face datasets" />
  </a>
  <img src="https://img.shields.io/badge/Transformers-4.51.0-6C5CE7" alt="Transformers 4.51.0" />
</p>

<p>
  A unified 3D model for grounding articulation semantics and intrinsic physical properties.
</p>

</div>

<p align="center">
  <a href="#model"><b>Model</b></a> |
  <a href="#datasets"><b>Datasets</b></a> |
  <a href="#installation"><b>Installation</b></a> |
  <a href="#inference"><b>Inference</b></a> |
  <a href="#reproducing-uniphys-bench-results"><b>Benchmark</b></a> |
  <a href="#training"><b>Training</b></a> |
  <a href="#license"><b>License</b></a> |
  <a href="#citation"><b>Citation</b></a>
</p>

---

UniPhysGen maps object- and part-level 3D geometry to structured physical
semantics. It uses **Qwen3-1.7B** as the language backbone and **Sonata** as the
point-cloud encoder, and operates directly on heterogeneous part
decompositions without assuming a canonical object hierarchy.

This repository contains the UniPhysGen model implementation, environment
setup, inference and evaluation entry points, training code, and release
documentation for UniPhys-40K and UniPhys-Bench.

![UniPhysGen model overview](assets/uniphysgen.png)

<p align="center"><em>UniPhysGen encodes object- and part-level 3D geometry and produces unified object, structure, kinematics, and physics annotations.</em></p>

## <a id="model"></a>Model

---

Given an object point cloud <i>X</i><sub>o</sub> and, when required, a target-part
point cloud <i>X</i><sub>p</sub>, UniPhysGen predicts four complementary categories
of physical semantics:

| Capability | Internal task | Required geometry | Structured prediction |
| --- | --- | --- | --- |
| Part-level intrinsic physical grounding | `physics` | object + target part | part identity, functional semantics, material, density, friction, affordance, and related physical properties |
| Kinematic parameter grounding | `motion` | object + target part | joint type, axis, pivot, and motion limit |
| Articulation structure grounding | `group` | object + target part | motion-coupled part members |
| Object-level physical grounding | `object_level` | object | object identity, category, dimensions, and mass |

The main model combines three design choices for robust grounding across
heterogeneous assets:

- **Physical semantic alignment** connects local part geometry, global object
  context, functional semantics, and intrinsic physical properties.
- **SO(3)-based augmentation and spherical axis parameterization** reduce
  orientation shortcuts in articulation reasoning.
- **A shared global voxel origin** preserves object-to-part spatial
  correspondence for stable geometric localization.

### Checkpoints

The four task-specific checkpoints for the 1.7B model are available on Hugging
Face. The training-only initialization checkpoint is documented separately in
[Reproducing UniPhysGen Training](#reproducing-uniphysgen-training).

| Checkpoint | Task | Download |
| --- | --- | --- |
| UniPhysGen-1.7B-Physics | Part-level intrinsic physical grounding | 🤗 [Hugging Face](https://huggingface.co/breezexian/UniPhysGen-1.7B-Physics) |
| UniPhysGen-1.7B-Kinematics | Kinematic parameter grounding | 🤗 [Hugging Face](https://huggingface.co/breezexian/UniPhysGen-1.7B-Kinematics) |
| UniPhysGen-1.7B-Structure | Articulation structure grounding | 🤗 [Hugging Face](https://huggingface.co/breezexian/UniPhysGen-1.7B-Structure) |
| UniPhysGen-1.7B-Object | Object-level physical grounding | 🤗 [Hugging Face](https://huggingface.co/breezexian/UniPhysGen-1.7B-Object) |

## <a id="datasets"></a>Datasets

---

| Dataset | Purpose | Scale | Download |
| --- | --- | --- | --- |
| UniPhys-40K | Training unified physical grounding models | 40K (40014) objects, 400K total parts, 370K filtered training parts | 🤗 [Hugging Face](https://huggingface.co/datasets/spatialverse/UniPhys-40K) |
| UniPhys-Bench | Human-verified unified physical grounding evaluation | 1.9K (1927) articulated objects, 16K parts, 5.5K motion-relevant components | 🤗 [Hugging Face](https://huggingface.co/datasets/spatialverse/UniPhys-Bench) |

## <a id="installation"></a>Installation

---

All regular Python dependencies are declared in `pyproject.toml`, including
the recommended `transformers==4.51.0` version.

The model has been tested with:

- Linux
- Python 3.11
- PyTorch 2.4.1
- CUDA 12.4
- Transformers **4.51.0**

> [!IMPORTANT]
> Use `transformers==4.51.0`. This is the tested version for the released
> UniPhysGen implementation and is pinned by the project metadata.

Create the environment:

```bash
conda create -n uniphysgen python=3.11 -y
conda activate uniphysgen
conda install -y -c nvidia/label/cuda-12.4.0 cuda-toolkit
```

Install the tested CUDA 12.4 build of PyTorch:

```bash
python -m pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cu124
```

Install UniPhysGen with the training dependencies:

```bash
python -m pip install -e ".[train]"
```

For an inference-only environment, install the project without the `train`
extra: `python -m pip install -e .`.

Finally, install the CUDA extensions required by the point-cloud encoder:

```bash
bash scripts/install_cuda_extensions.sh
```

The script contains only the non-standard installation steps for FlashAttention,
TorchScatter, and SpConv. Install the CUDA toolkit and PyTorch first, as shown
above. FlashAttention is compiled locally and may take several minutes.

Verify the environment:

```bash
python -c "import torch, transformers, torch_scatter, spconv.pytorch; print('torch:', torch.__version__, 'cuda:', torch.version.cuda, 'transformers:', transformers.__version__)"
```

## <a id="inference"></a>Inference

---

UniPhysGen provides one fixed entry point for each grounding task. Every entry
point supports both single-sample and JSON-batch inference. This section is
intended for running UniPhysGen on custom inputs; see
[Reproducing UniPhys-Bench Results](#reproducing-uniphys-bench-results) for the
complete paper-evaluation workflow.

| Task | Entry point |
| --- | --- |
| Part-level intrinsic physical grounding | `inference_batch_intrinsic_physics_part.py` |
| Kinematic parameter grounding | `inference_batch_kinematic_parameters.py` |
| Articulation structure grounding | `inference_batch_articulation_structure.py` |
| Object-level physical grounding | `inference_batch_intrinsic_physics_object.py` |

### Point-cloud input

The recommended input format is `.npz` with three aligned arrays:

```text
point   float32 [N, 3]   3D coordinates
color   uint8   [N, 3]   RGB values
normal  float32 [N, 3]   surface normals
```

Other point-cloud formats supported by Open3D, such as `.ply` and `.pcd`, can
also be used. For these formats, the current loader uses the available points
and colors and initializes normals to zero.

Articulation-structure grounding additionally reads the following arrays from
the object `.npz` file:

```text
part_names    [K]       candidate part identifiers
part_centers  [K, 3]    candidate part centers in the source coordinate frame
```

Object and target-part point clouds are normalized together inside the
inference pipeline; no manual normalization is required.

### Single-sample inference

#### Part-level intrinsic physical grounding

```bash
CUDA_VISIBLE_DEVICES=0 python inference_batch_intrinsic_physics_part.py \
  --model_path <PATH_TO_PHYSICS_CHECKPOINT> \
  --object_pcd examples/object.npz \
  --part_pcd examples/part.npz \
  --output outputs/intrinsic_physics_part.json
```

#### Kinematic parameter grounding

The released kinematic model uses spherical axis parameterization during
generation. The saved `result.axis` is converted to a unit Cartesian vector.

```bash
CUDA_VISIBLE_DEVICES=0 python inference_batch_kinematic_parameters.py \
  --model_path <PATH_TO_KINEMATICS_CHECKPOINT> \
  --object_pcd examples/object.npz \
  --part_pcd examples/part.npz \
  --spherical_axis \
  --output outputs/kinematic_parameters.json
```

#### Articulation structure grounding

```bash
CUDA_VISIBLE_DEVICES=0 python inference_batch_articulation_structure.py \
  --model_path <PATH_TO_STRUCTURE_CHECKPOINT> \
  --object_pcd examples/object_with_parts.npz \
  --part_pcd examples/part.npz \
  --output outputs/articulation_structure.json
```

#### Object-level physical grounding

```bash
CUDA_VISIBLE_DEVICES=0 python inference_batch_intrinsic_physics_object.py \
  --model_path <PATH_TO_OBJECT_CHECKPOINT> \
  --object_pcd examples/object.npz \
  --output outputs/intrinsic_physics_object.json
```

The default inference configuration uses CUDA and bfloat16. Use `--device` and
`--dtype` to override them when needed:

```bash
python inference_batch_intrinsic_physics_object.py \
  --model_path <PATH_TO_OBJECT_CHECKPOINT> \
  --object_pcd examples/object.npz \
  --device cuda \
  --dtype bf16
```

### Custom batch inference

Batch input is a JSON list. Media paths are resolved relative to the input JSON
file, or relative to `--data_root` when it is provided.

```json
[
  {
    "sample_id": "cabinet-door-0001",
    "object_level": {
      "object_ply": "point_clouds/cabinet.npz"
    },
    "part_level": {
      "part_ply": "point_clouds/door.npz"
    }
  }
]
```

Run a batch:

```bash
CUDA_VISIBLE_DEVICES=0 python inference_batch_kinematic_parameters.py \
  --model_path <PATH_TO_KINEMATICS_CHECKPOINT> \
  --input_json examples/kinematics.json \
  --data_root examples \
  --spherical_axis \
  --output outputs/kinematics.json \
  --output_dir outputs/kinematics_records
```

Useful batch options:

| Option | Description |
| --- | --- |
| `--start N` | Start index, inclusive |
| `--end N` | End index, exclusive; `-1` processes all remaining samples |
| `--output FILE` | Save one consolidated JSON list |
| `--output_dir DIR` | Save one self-contained JSON record per sample |
| `--fail_fast` | Stop on the first failed sample |

The deterministic `--start` / `--end` interval makes it straightforward to
shard a large input list across several workers.

### Output format

Every saved prediction follows the versioned
`uniphysgen.inference.v1` schema:

```json
{
  "schema_version": "uniphysgen.inference.v1",
  "sample_id": "cabinet-door-0001",
  "task": "motion",
  "source_sample": {},
  "inputs": {
    "object_pcd": "...",
    "part_pcd": "..."
  },
  "coordinate_frame": {
    "name": "source_to_aabb_0_2",
    "center": [0.0, 0.0, 0.0],
    "scale": 1.0,
    "min_bound": [-1.0, -1.0, -1.0]
  },
  "model_result": {
    "motion_type": "C",
    "axis": {"theta": 90, "phi": 0},
    "pivot": [1.0, 1.0, 1.0],
    "range": [0.0, 0.25]
  },
  "result": {
    "motion_type": "C",
    "axis": [1.0, 0.0, 0.0],
    "pivot": [0.0, 0.0, 0.0],
    "range": [0.0, 1.5708]
  },
  "raw_response": "..."
}
```

- `model_result` retains the direct model output in the normalized model frame.
- `result` contains the final structured prediction. Kinematic pivots and
  prismatic limits are restored to source units, while revolute limits are
  returned in radians.
- `coordinate_frame` records the reversible mapping between the source and
  normalized point-cloud frames.
- `source_sample` preserves the original batch item for downstream evaluation.

## <a id="reproducing-uniphys-bench-results"></a>Reproducing UniPhys-Bench Results

---

Use this workflow to reproduce the four groups of UniPhysGen results reported
on UniPhys-Bench. Starting from the released benchmark, the full path from raw
assets to paper metrics contains four steps:

```text
UniPhys-Bench
      │
      ▼
generate_npzs.py           Generate model-ready point clouds
      │
      ▼
generate_jsons_for_inference.py
                           Build four task-specific batch manifests
      │
      ▼
UniPhysGen inference       Predict the four categories of physical semantics
      │
      ▼
python -m eval             Compute the paper metrics
```

### 1. Download UniPhys-Bench

Download the released benchmark from Hugging Face:

```bash
hf download \
  spatialverse/UniPhys-Bench \
  --repo-type dataset \
  --local-dir data/UniPhys-Bench
```

The following commands assume this layout:

```text
data/UniPhys-Bench/              Downloaded benchmark
data/UniPhys-Bench-processed/
├── npzs/                        Model-ready object and part point clouds
└── manifests/                   Four task-specific inference JSON files
outputs/UniPhys-Bench/           Model predictions
metrics/UniPhys-Bench/           Final metric reports
```

### 2. Prepare point clouds

Convert every benchmark sample into the `.npz` representation consumed by
UniPhysGen. Within each sample, part meshes are read from `parts/` and must use
the `part_<id>.obj` naming convention:

```text
<sample>/
└── parts/
    ├── part_0.obj
    ├── part_1.obj
    └── ...
```

Run the preprocessing command:

```bash
python pre_process/generate_npzs.py \
  --data_root data/UniPhys-Bench \
  --output_dir data/UniPhys-Bench-processed/npzs
```

The input directory structure is preserved below `--output_dir`. A part named
`part_<id>.obj` is written as `<id>.npz`; the `part_` prefix is not retained in
the generated part ID. Each object or part file contains aligned `point`,
`color`, and `normal` arrays. Object files used by articulation-structure
grounding also contain the numeric IDs in `part_names` and the corresponding
`part_centers`.

### 3. Build task manifests

Generate the four batch manifests from the benchmark annotations and prepared
point clouds:

```bash
python pre_process/generate_jsons_for_inference.py \
  --data_root data/UniPhys-Bench \
  --npz_dir data/UniPhys-Bench-processed/npzs \
  --output_dir data/UniPhys-Bench-processed/manifests
```

No sample-selection file is required: the command processes every entity below
`--data_root`. Part-level intrinsic-physics manifests retain every annotated
part without checking `mass_rate` or `pass_check`. The two articulation
manifests retain parts with non-empty `B` or `C` motion information, and the
object-level manifest contains one sample per entity.

The command writes:

```text
data/UniPhys-Bench-processed/manifests/
├── intrinsic_physics_part.json
├── intrinsic_physics_object.json
├── kinematic_parameters.json
└── articulation_structure.json
```

Each manifest is a JSON list in the batch format described above. It retains
the complete benchmark annotation for every sample; inference embeds that
record unchanged as `source_sample`, which is subsequently used by the
evaluators as ground truth.

### 4. Run benchmark inference

Create the output directory and run the corresponding checkpoint on each
manifest:

```bash
mkdir -p outputs/UniPhys-Bench

CUDA_VISIBLE_DEVICES=0 python inference_batch_intrinsic_physics_part.py \
  --model_path <PATH_TO_PHYSICS_CHECKPOINT> \
  --input_json data/UniPhys-Bench-processed/manifests/intrinsic_physics_part.json \
  --data_root data/UniPhys-Bench-processed/npzs \
  --output outputs/UniPhys-Bench/intrinsic_physics_part.json

CUDA_VISIBLE_DEVICES=0 python inference_batch_intrinsic_physics_object.py \
  --model_path <PATH_TO_OBJECT_CHECKPOINT> \
  --input_json data/UniPhys-Bench-processed/manifests/intrinsic_physics_object.json \
  --data_root data/UniPhys-Bench-processed/npzs \
  --output outputs/UniPhys-Bench/intrinsic_physics_object.json

CUDA_VISIBLE_DEVICES=0 python inference_batch_kinematic_parameters.py \
  --model_path <PATH_TO_KINEMATICS_CHECKPOINT> \
  --input_json data/UniPhys-Bench-processed/manifests/kinematic_parameters.json \
  --data_root data/UniPhys-Bench-processed/npzs \
  --spherical_axis \
  --output outputs/UniPhys-Bench/kinematic_parameters.json

CUDA_VISIBLE_DEVICES=0 python inference_batch_articulation_structure.py \
  --model_path <PATH_TO_STRUCTURE_CHECKPOINT> \
  --input_json data/UniPhys-Bench-processed/manifests/articulation_structure.json \
  --data_root data/UniPhys-Bench-processed/npzs \
  --output outputs/UniPhys-Bench/articulation_structure.json
```

For sharded inference, add `--start` and `--end`, and use `--output_dir` to
write one prediction record per sample. A directory of per-sample records can
be passed directly to the evaluators.

### 5. Compute paper metrics

Run the four evaluators on either the consolidated prediction files shown
above or directories containing per-sample prediction records:

```bash
mkdir -p metrics/UniPhys-Bench

python -m eval intrinsic_physics_part \
  outputs/UniPhys-Bench/intrinsic_physics_part.json \
  --output metrics/UniPhys-Bench/intrinsic_physics_part.json

python -m eval intrinsic_physics_object \
  outputs/UniPhys-Bench/intrinsic_physics_object.json \
  --output metrics/UniPhys-Bench/intrinsic_physics_object.json

python -m eval kinematic_parameters \
  outputs/UniPhys-Bench/kinematic_parameters.json \
  --output metrics/UniPhys-Bench/kinematic_parameters.json

python -m eval articulation_structure \
  outputs/UniPhys-Bench/articulation_structure.json \
  --output metrics/UniPhys-Bench/articulation_structure.json
```

By default, each report exposes the metrics used in the paper's main results.
Add `--include-extra` to retain additional diagnostics. Accuracy, mIoU, F1, and
MnRE values are stored in `[0, 1]`; multiply by 100 only when formatting a
percentage table.

The evaluators read ground truth exclusively from each prediction's embedded
`source_sample`. They do not require a separate annotation directory and do
not reopen point-cloud files during metric computation. See
[`eval/README.md`](eval/README.md) for the prediction schema and metric details.

## <a id="training"></a>Training

---

UniPhysGen is trained in two phases: physical semantic alignment pretraining,
followed by task-specific full-parameter fine-tuning.

```text
Qwen3-1.7B + Sonata
        │
        ▼
Physical Semantic Alignment Pretraining
        │
        ├── Kinematic Parameter Fine-tuning
        ├── Articulation Structure Fine-tuning
        └── Object-Level Physical Grounding Fine-tuning
```

Exact optimization settings are kept in the release configuration files. See
the [paper](https://arxiv.org/abs/2607.13586) for the reported training schedule
and hardware setup.

### Reproducing UniPhysGen Training

The official training workflow starts from UniPhys-40K and the released
UniPhysGen initialization checkpoint:

```text
UniPhys-40K ──> generate_npzs.py ──> generate_jsons_for_training.py
                                              │
UniPhysGen-1.7B-Init ─────────────────────────┘
                                              │
                                              ▼
                            Physical Semantic Alignment
                                              │
                     ┌────────────────────────┼────────────────────────┐
                     ▼                        ▼                        ▼
                 Kinematics                Structure              Object Level
```

#### 1. Download UniPhys-40K

```bash
hf download \
  spatialverse/UniPhys-40K \
  --repo-type dataset \
  --local-dir data/UniPhys-40K
```

#### 2. Download the initialization checkpoint

`UniPhysGen-1.7B-Init` combines the Qwen3-1.7B language backbone with the
Sonata point-cloud encoder. It contains no task-specific grounding training and
is intended only as the starting point for physical semantic alignment.

```bash
hf download \
  breezexian/UniPhysGen-1.7B-Init \
  --local-dir checkpoints/UniPhysGen-1.7B-Init
```

The training launcher loads this checkpoint through Hugging Face
`from_pretrained` by setting:

```yaml
model_name_or_path: checkpoints/UniPhysGen-1.7B-Init
```

#### 3. Prepare point clouds

UniPhys-40K and UniPhys-Bench share the same point-cloud preprocessing entry
point:

```bash
python pre_process/generate_npzs.py \
  --data_root data/UniPhys-40K \
  --output_dir data/UniPhys-40K-processed/npzs
```

#### 4. Build the training datasets

Generate an independent training directory for each task. Every task directory
contains its own `train.json` and `dataset_info.json`:

```bash
python pre_process/generate_jsons_for_training.py \
  --data_root data/UniPhys-40K \
  --npz_dir data/UniPhys-40K-processed/npzs \
  --output_dir data/UniPhys-40K-processed/train
```

Training samples are filtered according to the supervision required by each
task:

| Task directory | Filtering rule |
| --- | --- |
| `physics` | `mass_rate <= 2` and `pass_check == true` |
| `kinematic_parameters` | non-empty `B` or `C` motion information |
| `articulation_structure` | non-empty `B` or `C` motion information |
| `object_level` | `mass_rate <= 2` |

The two motion-related tasks intentionally do not inspect `mass_rate` or
`pass_check`. A part whose simulation validation cleared `motion_info` can
still be used for physical semantic alignment when it passes the physics-data
checks, but it is excluded from motion-related training.

Each generated task directory can be passed directly as `dataset_dir`:

```text
data/UniPhys-40K-processed/
├── npzs/
└── train/
    ├── physics/
    │   ├── dataset_info.json
    │   └── train.json
    ├── kinematic_parameters/
    │   ├── dataset_info.json
    │   └── train.json
    ├── articulation_structure/
    │   ├── dataset_info.json
    │   └── train.json
    └── object_level/
        ├── dataset_info.json
        └── train.json
```

#### 5. Launch the official stages

Before launching, set `dataset_dir`, `media_dir`, and `model_name_or_path` in
the corresponding release config. Set `media_dir` to
`data/UniPhys-40K-processed/npzs`, because all media paths in `train.json` are
relative to that directory. Physical semantic alignment starts from
`UniPhysGen-1.7B-Init`; the three downstream configs start from the resulting
alignment checkpoint.

| Stage | Release config | `dataset_dir` |
| --- | --- | --- |
| Physical semantic alignment | `configs/release/semantic_alignment.yaml` | `data/UniPhys-40K-processed/train/physics` |
| Kinematic parameter grounding | `configs/release/kinematic_parameters.yaml` | `data/UniPhys-40K-processed/train/kinematic_parameters` |
| Articulation structure grounding | `configs/release/articulation_structure.yaml` | `data/UniPhys-40K-processed/train/articulation_structure` |
| Object-level physical grounding | `configs/release/object_level.yaml` | `data/UniPhys-40K-processed/train/object_level` |

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
  python train.py configs/release/semantic_alignment.yaml

CUDA_VISIBLE_DEVICES=0,1,2,3 \
  python train.py configs/release/kinematic_parameters.yaml

CUDA_VISIBLE_DEVICES=0,1,2,3 \
  python train.py configs/release/articulation_structure.yaml

CUDA_VISIBLE_DEVICES=0,1,2,3 \
  python train.py configs/release/object_level.yaml
```

`train.py` automatically uses `torchrun` when multiple GPUs are visible. Keep
the same `output_dir` to resume from the latest checkpoint.

### Training on Custom Data

Custom datasets do not require a separate training pipeline. For each task,
create an independent directory containing `train.json` and
`dataset_info.json`. Convert every sample to the same structure generated for
UniPhys-40K and store model inputs as the `.npz` format described in
[Point-cloud input](#point-cloud-input).

Each sample stores object-level metadata, part-level metadata, intrinsic
physical properties, and kinematic annotations. Point-cloud paths may be
absolute or relative to `media_dir`.

The following example shows the common superset schema. Individual tasks use
only the fields needed for their supervision target.

<details>
<summary><strong>Training sample schema</strong></summary>

```json
{
  "id": "cabinet-door-0001",
  "object_level": {
    "object_ply": "point_clouds/cabinet.npz",
    "object_name": "Cabinet",
    "category": "Furniture/Cabinet",
    "dimension": [80.0, 40.0, 120.0],
    "object_mass": 35.0
  },
  "part_level": {
    "part_ply": "point_clouds/door.npz",
    "part_name": "door",
    "basic_description": "A rigid front panel.",
    "functional_description": "Provides access to the cabinet interior.",
    "movement_description": "Rotates around a side hinge.",
    "grasp_description": "Pulled from the outer edge.",
    "graspable": true,
    "affordance": 2,
    "part_label_to_name": {
      "0": "cabinet body",
      "1": "door"
    }
  },
  "basic_info": {
    "material": "wood",
    "density": 0.70,
    "young": 10.0,
    "hardness": 4.0,
    "poisson": 0.30,
    "friction": 0.45
  },
  "kinematic_info": {
    "motion_types": ["C"],
    "motion_info": {
      "C": {
        "axis": [0.0, 0.0, 1.0],
        "pos": [0.0, 0.0, 0.0],
        "range": [0.0, 1.5708]
      },
      "dependency": [1]
    }
  }
}
```

</details>

<details>
<summary><strong>Example dataset registration</strong></summary>

```json
{
  "my_physics_train": {
    "file_name": "train.json",
    "formatting": "physmeshllm",
    "task_name": "physics",
    "columns": {
      "object_info": "object_level",
      "part_info": "part_level",
      "basic_info": "basic_info",
      "kinematic_info": "kinematic_info"
    }
  }
}
```

For another task, create a sibling directory with its own `train.json` and
`dataset_info.json`, then set `task_name` to `motion`, `group`, or
`object_level` as appropriate.

</details>

Copy the closest release configuration and change only the dataset and path
fields required by the custom data:

```yaml
model_name_or_path: <PATH_TO_INITIALIZATION_OR_ALIGNMENT_CHECKPOINT>
dataset: my_physics_train
dataset_dir: <PATH_TO_TASK_DIRECTORY>
media_dir: <PATH_TO_NPZ_ROOT>
output_dir: outputs/my-uniphysgen-run
```

Then launch it through the same training entry point:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py <PATH_TO_CONFIG.yaml>
```

The release configs remain the source of truth for task-specific
hyperparameters and augmentation settings.

## <a id="license"></a>License

---

The UniPhysGen source code is released under the
[Apache License 2.0](LICENSE.txt). Released UniPhysGen model weights are
distributed under CC BY-NC 4.0; see the `LICENSE` file included with each
checkpoint. UniPhys-40K and UniPhys-Bench contain assets from multiple upstream
sources, so consult the dataset repositories and retained per-asset provenance
for the applicable terms.

## <a id="citation"></a>Citation

---

```bibtex
@article{li2026uniphysgen,
  title   = {UniPhysGen: Unified Physical Grounding for Simulation-Ready 3D Assets},
  author  = {Li, Xian and Wei, Rong and Yang, Lujie and Huang, Haolin and Fang, Junyuan and Tang, Siliang and Xiao, Jun and Tang, Rui and Li, Juncheng},
  journal = {arXiv preprint arXiv:2607.13586},
  year    = {2026}
}
```

#### 🔝 [Back to Top](#top)
