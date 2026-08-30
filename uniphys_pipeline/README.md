# <a id="top"></a>UniPhys Pipeline

---

<p align="left">
  <a href="#overview"><b>Overview</b></a> |
  <a href="#features"><b>Features</b></a> |
  <a href="#requirements"><b>Requirements</b></a> |
  <a href="#installation"><b>Installation</b></a> |
  <a href="#configuration"><b>Configuration</b></a> |
  <a href="#quick-start"><b>Quick Start</b></a> |
  <a href="#custom-meshes"><b>Custom Meshes</b></a> |
  <a href="#resume"><b>Resume</b></a> |
  <a href="#repository-layout"><b>Repository Layout</b></a> |
  <a href="#development"><b>Development</b></a> |
  <a href="#citation"><b>Citation</b></a> |
  <a href="#license"><b>License</b></a>
</p>

UniPhys Pipeline is the automated physical-grounding system introduced in
**UniPhysGen: Unified Physical Grounding for Simulation-Ready 3D Assets**
([paper](https://arxiv.org/abs/2607.13586)). It transforms a raw, heterogeneous
3D mesh into a simulation-ready asset with physically meaningful parts,
intrinsic physical properties, articulation semantics, a MuJoCo model, and
consistency reports.

![Overview of the UniPhys physical-grounding pipeline](assets/uniphys_pipeline_overview.png)

*UniPhys Pipeline. From a heterogeneous raw 3D mesh to a simulation-ready asset
with unified articulation semantics and intrinsic physical properties.*

## <a id="overview"></a>What the pipeline does

---

The implementation follows the four modules described in the paper:

1. **Physically meaningful structural decomposition.** Blender renders
   multi-view RGB and face-ID observations. SAM2 masks are lifted from image
   space to mesh faces to estimate a perceptual part prior. PartField then
   generates geometry-aware candidate decompositions, which are selected by
   Hungarian matching and refined with cross-view merge consensus.
2. **Intrinsic physical property grounding.** Each decomposed part is rendered
   with its appearance preserved and the rest of the object retained as visual
   context. A vision-language model produces schema-constrained part- and
   object-level annotations, including material, density, friction,
   affordance, scale, mass, and structural relations.
3. **Geometry-aware articulation grounding.** Neighbor relations form a motion
   dependency graph. Contact geometry generates feasible joint-axis and pivot
   candidates; rendered candidates and part semantics guide the vision-language
   model to select revolute or prismatic parameters.
4. **Simulation-driven consistency verification.** The pipeline checks
   numerical and material plausibility, intra-part and cross-part consistency,
   object-level mass consistency, and articulated motion in MuJoCo. Failed
   annotations are reported for filtering or refinement.

The paper modules map to the executable stages as follows:

```text
1 render
  └─> 2 decompose
       └─> 3 export_parts
            └─> 4 annotate_basic ───────────────> 11 validate_basic
                 └─> 5 build_kinematic_graph
                      └─> 6 propose_kinematics
                           └─> 7 render_axes
                                ├─> 8 annotate_revolute ─┐
                                └─> 9 annotate_prismatic ┤
                                                         └─> 10 generate_mujoco
                                                              └─> 12 validate_simulation
```

| Paper module | Pipeline stages | Main outputs |
| --- | --- | --- |
| Structural decomposition | `render`, `decompose`, `export_parts` | Multi-view renders, face-to-part labels, part meshes and atlases |
| Physical property grounding | `annotate_basic` | Structured part/object physical annotations and relations |
| Articulation grounding | `build_kinematic_graph`, `propose_kinematics`, `render_axes`, `annotate_revolute`, `annotate_prismatic` | Motion graph, joint types, axes, pivots, and limits |
| Consistency verification | `generate_mujoco`, `validate_basic`, `validate_simulation` | MuJoCo XML and physical/simulation validation reports |

The command-line entry point is [`pipeline.py`](pipeline.py).

## <a id="features"></a>Key engineering features

---

- Typed YAML/JSON configuration with repository-relative path resolution.
- Explicit stage registry and dependency graph.
- Per-entity atomic state in `.pipeline/state.json`.
- Resume, existing-output adoption, forced reruns, and structured failures.
- Deterministic `start`/`end` sharding and process-level entity workers.
- Non-zero exit status when an entity or stage fails.
- Lazy imports, so help, stage listing, and status inspection avoid loading GPU
  dependencies.

## <a id="requirements"></a>System requirements

---

The complete pipeline targets Linux with:

- Conda and Python 3.10;
- an NVIDIA GPU and a compatible CUDA installation;
- **Blender 4.5.x** on `PATH` or configured by absolute path; **4.5.3 is
  recommended** and matches the development environment;
- MuJoCo with an available headless rendering backend such as EGL;
- three separate Conda environments: `uniphys_pipeline`, `sam2`, and
  `partfield`.

The environments are intentionally separate. The official PartField setup pins
PyTorch 2.4/CUDA 12.4, while current SAM2 installation guidance recommends a
newer PyTorch stack. The orchestrator invokes their configured Python
interpreters as subprocesses.

## <a id="installation"></a>Installation

---

### 1. UniPhys Pipeline environment

The commands below assume Conda is already installed:

```bash
conda create -n uniphys_pipeline python=3.10 -y
conda activate uniphys_pipeline
python -m pip install --upgrade pip
python -m pip install -e ".[runtime]"
```

Configure MuJoCo to use EGL for headless rendering:

```bash
export MUJOCO_GL=egl
```

For development, the setup script creates the same Conda environment and
installs both runtime and development dependencies:

```bash
scripts/setup_dev.sh
conda activate uniphys_pipeline
```

Set `UNIPHYS_CONDA_ENV` to use a different environment name.

### 2. Blender 4.5

Install Blender 4.5.x from the
[official Blender downloads](https://www.blender.org/download/). Blender 4.5.3
is recommended. Confirm the executable before continuing:

```bash
blender --version
```

If it is not on `PATH`, set `runtime.blender` in your local configuration to
the absolute Blender executable path.

### 3. SAM2 environment and checkpoint

Install SAM2 from the
[official facebookresearch/sam2 repository](https://github.com/facebookresearch/sam2)
and follow its current `INSTALL.md` for a matching PyTorch/CUDA build:

```bash
git clone https://github.com/facebookresearch/sam2.git
cd sam2
conda create -n sam2 python=3.10 -y
conda activate sam2
# Install the PyTorch/CUDA versions specified by the official SAM2 guide first.
python -m pip install -e .
```

Download the checkpoint yourself from the official repository. The current
UniPhys integration uses the original **SAM2 Hiera Large** configuration, so
select `sam2_hiera_large.pt` from the official README's "SAM 2 checkpoints"
table and set its path in the runtime configuration.

### 4. PartField environment and checkpoint

Install PartField from the
[official nv-tlabs/PartField repository](https://github.com/nv-tlabs/PartField).
Its official environment file provides the reference Python 3.10,
PyTorch 2.4, and CUDA 12.4 setup:

```bash
git clone https://github.com/nv-tlabs/PartField.git
cd PartField
conda env create -f environment.yml
conda activate partfield
```

Download `model_objaverse.ckpt` through the pretrained-model link in the
official PartField README and place it under the upstream repository's
`model/` directory, or another local model directory, then set its path in the
runtime configuration.

### 5. Connect the three environments

Copy the example configuration and point it to the two external interpreters
and user-downloaded checkpoints:

```bash
cp configs/example.yaml configs/local.yaml
```

```yaml
runtime:
  blender: blender
  sam_python: /opt/conda/envs/sam2/bin/python
  partfield_python: /opt/conda/envs/partfield/bin/python
  sam_checkpoint: /path/to/sam2/checkpoints/sam2_hiera_large.pt
  partfield_checkpoint: /path/to/PartField/model/model_objaverse.ckpt
  partfield_config: utils/partfield/configs/final/demo.yaml
  gpu: "0"
```

Conda installations in other locations can be resolved with:

```bash
conda run -n sam2 python -c "import sys; print(sys.executable)"
conda run -n partfield python -c "import sys; print(sys.executable)"
```

`gpu` defaults to `"0"`. Override it in YAML or with `--gpu`, for example
`--gpu 1` or `--gpu 0,1`.

### 6. Vision-model credentials

API keys are read only from the environment variable named by
`gpt.api_key_env`; they are not stored in the pipeline configuration:

```bash
export OPENAI_API_KEY='...'
```

`OPENAI_BASE_URL` may be used when `gpt.base_url` is `null`.

## <a id="configuration"></a>Data layout and configuration

---

Relative paths in configuration files are resolved from the repository root.
The standard layout is derived from:

```yaml
dataset:
  name: ABO
  type_name: "4"
  input_root: samples
  output_root: outputs
```

This produces:

```text
samples/ABO/4/                         # pipeline-ready single-mesh GLBs
outputs/ABO/render_res/4/<entity>/     # RGB, face-ID, and SAM2 observations
outputs/ABO/decomposition_res/4/       # decomposed parts and per-part images
outputs/ABO/gpt_output/4/              # properties, joints, XML, validation
```

Every path can be overridden independently. See
[`configs/example.yaml`](configs/example.yaml) for the complete schema.

## <a id="quick-start"></a>Quick start

---

The repository includes a pipeline-ready example asset at
`samples/ABO/4/B00IIFW2L4.glb`, and the example configuration already selects
it. After completing the installation and filling in the runtime paths in
`configs/local.yaml`, use this bundled asset to verify the complete pipeline
before preparing a custom dataset. No mesh preprocessing is needed for this
quick-start example.

Validate all configured executables, environments, checkpoints, modules, and
credentials before launching a run:

```bash
export MUJOCO_GL=egl
python pipeline.py list-stages
python pipeline.py doctor --config configs/local.yaml
```

Run the configured stage sequence:

```bash
python pipeline.py run --config configs/local.yaml
# equivalent wrapper
scripts/run_pipeline.sh configs/local.yaml
```

Run one stage, a range, or one entity:

```bash
python pipeline.py run --config configs/local.yaml --stages decompose
python pipeline.py run --config configs/local.yaml --from-stage render --to-stage export_parts
python pipeline.py run --config configs/local.yaml --stages 8 9 --with-dependencies
python pipeline.py run --config configs/local.yaml --entity B00IIFW2L4
```

Shard a sorted mesh set and use multiple entity workers:

```bash
python pipeline.py run --config configs/local.yaml --start 0 --end 100 --workers 2
```

Use multiple workers only when GPU memory, Blender/MuJoCo processes, and
temporary storage can support concurrent entities. Preview a plan without
creating state or output files with:

```bash
python pipeline.py run --config configs/local.yaml --dry-run
```

## <a id="custom-meshes"></a>Running on your own meshes

---

The decomposition stages rely on the same stable face ordering in Blender,
Trimesh, and PartField. A custom input must therefore resolve to exactly one
mesh before it enters the pipeline. In particular, preprocess the asset when:

- the source is an OBJ file; or
- the source is a GLB containing multiple mesh geometries or scene instances,
  which is common in Objaverse and other scene-oriented datasets.

The repository provides `pre_process/prepare_meshes.py` for this purpose. Mesh
preprocessing is intentionally separate from pipeline execution: it never runs
as a pipeline stage, it does not modify the raw source, and the pipeline should
only be pointed at the generated GLB directory after preprocessing finishes.

### 1. Preprocess the raw assets

Convert one OBJ or GLB:

```bash
python pre_process/prepare_meshes.py \
  /path/to/raw/chair.obj \
  samples/MyDataset/default
```

This writes `samples/MyDataset/default/chair.glb`. The script loads the complete
scene without mesh processing, bakes scene-node transforms, expands instances,
concatenates all mesh geometry, and verifies that the exported GLB can be
reloaded with the same face count.

Convert every OBJ/GLB in a flat directory, using multiple worker processes:

```bash
python pre_process/prepare_meshes.py \
  /path/to/raw_meshes \
  samples/MyDataset/default \
  --workers 8
```

For datasets laid out as `<entity>/raw_model.obj`, search recursively and name
each output from its entity directory:

```bash
python pre_process/prepare_meshes.py \
  /path/to/raw_dataset \
  samples/MyDataset/default \
  --recursive \
  --name-from parent \
  --workers 8
```

Existing outputs are skipped by default. Use `--overwrite` to replace them,
`--dry-run` to inspect source-to-output mappings, and `--max-faces 0` to disable
the default 500,000-face limit. Each run writes
`preprocess_report.json` in the output directory and returns a non-zero status
if any asset fails. Source and output paths must remain separate.

### 2. Configure the prepared dataset

Copy the working local configuration, then change its dataset section:

```bash
cp configs/local.yaml configs/my_dataset.yaml
```

```yaml
dataset:
  name: MyDataset
  type_name: default
  input_root: samples
  output_root: outputs
  extensions: [".glb"]
```

With this configuration, the pipeline reads the prepared files from
`samples/MyDataset/default/`. Alternatively, set `dataset.mesh_root` to an
absolute prepared-data directory.

### 3. Run the pipeline

Validate the custom configuration and then launch the normal pipeline. The
preprocessing tool is not invoked again:

```bash
python pipeline.py doctor --config configs/my_dataset.yaml
python pipeline.py run --config configs/my_dataset.yaml
```

## <a id="resume"></a>Resume and state

---

Resume is enabled by default. A stage is skipped only when its saved execution
fingerprint matches and its declared outputs still validate. Existing legacy
outputs can be adopted when `adopt_existing_outputs: true`.

```bash
# Force one stage to rerun.
python pipeline.py run --config configs/local.yaml --stages decompose --force decompose

# Inspect state.
python pipeline.py status --config configs/local.yaml
python pipeline.py status --config configs/local.yaml --json
```

Per-entity metadata is stored under:

```text
<decomposition_root>/<entity>/.pipeline/
├── state.json
├── run.lock
└── errors/<stage>.log
```

## <a id="repository-layout"></a>Repository layout

---

```text
pipeline.py                 Stable CLI entry point
uniphys/                    Configuration, orchestration, and stage registry
pre_process/                Standalone raw-mesh to single-GLB preprocessing
assets/                     README figures and project media
configs/                    Reproducible run configurations
scripts/                    Setup, run, and diagnostic helpers
tests/                      CPU-only orchestration tests
utils/                      Geometry, rendering, annotation, and validation code
```

## <a id="development"></a>Development

---

```bash
python -m pytest
python -m ruff check pipeline.py uniphys pre_process tests
python -m mypy pipeline.py uniphys
```

## <a id="citation"></a>Citation

---

If you use UniPhys Pipeline in your research, please cite:

```bibtex
@article{li2026uniphysgen,
  title   = {UniPhysGen: Unified Physical Grounding for Simulation-Ready 3D Assets},
  author  = {Li, Xian and Wei, Rong and Yang, Lujie and Huang, Haolin and Fang, Junyuan and Tang, Siliang and Xiao, Jun and Tang, Rui and Li, Juncheng},
  journal = {arXiv preprint arXiv:2607.13586},
  year    = {2026}
}
```

## <a id="license"></a>License

---

The UniPhys orchestration code is provided under the Apache License 2.0.
Third-party code, checkpoints, and datasets remain governed by their own
licenses.

#### 🔝 [Back to Top](#top)
