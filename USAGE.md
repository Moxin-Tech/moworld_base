# MoWorld Pretraining Guide

Environment setup, initialization checkpoints, data preparation, and expert pretraining. For the short workflow, see the [Quick Start](README.md#installation).

### Installation

Use a Linux Ascend server with a compatible driver, firmware, CANN, and `torch_npu` stack. The bundled MindSpeed-MM package pins **PyTorch 2.7.1** and requires **Python 3.10 or newer**. See the bundled [installation guide](moworld/docs/zh/pytorch/install_guide.md) for platform setup; select the software stack for your hardware before installing the Python packages below.

Clone the repository, including the pinned [MindSpeed](https://gitcode.com/Ascend/MindSpeed) submodule:

```bash
git clone --recursive https://github.com/Moxin-Tech/moworld_base.git
cd moworld_base
export MOWORLD_ROOT="$PWD"
```

If the repository is already cloned, run `git submodule update --init --recursive`. The MindSpeed-MM-based implementation is included in `moworld/` and does not need a separate clone.

Activate your Ascend Python environment. For a new environment, Python 3.11 is an option:

```bash
conda create -n moworld python=3.11 -y
conda activate moworld
```

Install the matching PyTorch 2.7.1 / `torch_npu` packages following the platform guide, then install MindSpeed, Megatron-LM sources, and the bundled package. The Megatron-LM revision below matches the bundled installation script:

```bash
python -m pip install -e ./MindSpeed
git clone --branch core_v0.12.1 https://github.com/NVIDIA/Megatron-LM.git ../Megatron-LM
export PYTHONPATH="$(cd ../Megatron-LM && pwd):${PYTHONPATH:-}"
python -m pip install -e ./moworld
python -m pip check

export MINDSPEED_PATH="$MOWORLD_ROOT/MindSpeed"
cd "$MOWORLD_ROOT/moworld"
```

Run the remaining commands from `moworld_base/moworld` in this environment. Set `ASCEND_ENV` to your CANN environment script if it is installed outside the launchers' standard locations. The NPU attention path uses `torch_npu.npu_fusion_attention`; CUDA FlashAttention installation is not required for that path.

If this environment previously used an editable installation from the old `MindSpeed-MM/` directory, rerun `python -m pip install -e "$MOWORLD_ROOT/moworld"` after the rename. The Python import name remains `mindspeed_mm` and the installed distribution name remains `mindspeed-mm`.

The retained shared runtime uses the `mindspeed_mm` package name. Follow this guide for the MoWorld launchers and configuration files.

### Model Assets

The published [MoWorld base checkpoint](https://huggingface.co/moworld1/moworld_base) contains only `high_noise_model.safetensors` and `config.json` (plus repository metadata). It does not include a low-noise expert, tokenizer, VAE, or text encoder.

Download the checkpoint and set the paths:

```bash
export MOWORLD_WEIGHTS="$MOWORLD_ROOT/models/moworld"
export MODEL_PATH="$MOWORLD_ROOT/models/base"
hf download "moworld1/moworld_base" --local-dir "$MOWORLD_WEIGHTS"
```

The prepared local layout is:

```text
models/
├── moworld/
│   ├── config.json
│   └── high_noise_model.safetensors
├── base/
│   ├── google/umt5-xxl/    # supply separately if required
│   └── high_noise_model/
│       └── diffusion_pytorch_model.safetensors  # link to downloaded weight
└── diffusers/
    ├── vae/              # supply separately for raw-video feature extraction
    └── text_encoder/     # supply separately for raw-video feature extraction
```

Raw-video feature extraction and raw-video pretraining use compatible Diffusers-format VAE and text encoder assets under `models/diffusers/`. Supply these components or update `ae.from_pretrained` and `text_encoder.from_pretrained` in local copies of the JSON configuration. The feature-data workflow uses precomputed encodings; see the selected training configuration for required assets.

JSON paths are resolved from the `moworld/` working directory; the examples use `../models/...` and `../checkpoints/...`. JSON values do not expand shell variables such as `$MOWORLD_ROOT`. Replace example paths with your own paths when necessary.

Training uses **torch DCP** checkpoints. From `moworld/`, create a symbolic link to adapt the published filename to the converter's expected layout, then convert only the high-noise expert:

```bash
mkdir -p "$MODEL_PATH/high_noise_model"
ln -s "$MOWORLD_WEIGHTS/high_noise_model.safetensors" \
  "$MODEL_PATH/high_noise_model/diffusion_pytorch_model.safetensors"
EXPERTS=high_noise_model SOURCE_ROOT="$MODEL_PATH" \
bash examples/moworld/convert_weights.sh
```

The link does not copy the weight file. It must point to the downloaded file; if the destination already exists, check it before proceeding. Keep the downloaded directory in place during conversion. Conversion requires unused output directories and writes `checkpoints/base/mm/high_noise_model` and `checkpoints/base/dcp/high_noise_model`.

This conversion path reads safetensors tensors, not the downloaded `config.json`. Training architecture comes from `pretrain_model_feature_high.json`. Matching filenames does not establish tensor compatibility; the published checkpoint still needs conversion and training validation on the target environment.

Launcher defaults follow this layout: the MindSpeed dependency is at the repository root's `MindSpeed/`, base DCP weights are under `checkpoints/base/dcp/`, training outputs are under `checkpoints/train/`. Override `MINDSPEED_PATH`, `LOAD_PATH`, `SAVE_PATH`, `SOURCE_ROOT`, or `OUTPUT_ROOT` for a different layout.

### Training

#### Prepare feature data

Training examples use precomputed video latents, text embeddings, and camera conditions in `.pth` files. The included manifest is a format example; supply your own feature files and manifest.

Keep local data and configuration under the Git-ignored `examples/moworld/local_data/` directory:

```bash
mkdir -p examples/moworld/local_data
cp examples/moworld/data_feature.json examples/moworld/local_data/train_data.json
```

Edit `local_data/train_data.json` before training:

- Set `dataset_param.basic_parameters.data_path` to your JSONL manifest.
- Set `dataset_param.basic_parameters.data_folder` to the directory containing the feature files.
- Set `dataset_param.tokenizer_config.from_pretrained` to your local tokenizer directory if used by the configuration.

A manifest row uses a feature path relative to `data_folder`:

```json
{"file": "latent1.pth", "sample_num_frames": 125, "sample_size": "125x480x832"}
```

The file must contain compatible features, including the image-conditioning feature and camera conditions required by the selected model. See [CameraControlFeatureDataset](moworld/mindspeed_mm/data/datasets/camera_control_feature_dataset.py) for the accepted fields and shape handling.

For raw-video preprocessing, an optional [feature extraction script](moworld/examples/moworld/feature_extraction.sh) is also included. First configure local copies of `data_process_data.json`, `data_process_model.json`, and `data_process_tools.json` with your raw manifest, encoder weights, and output directory, then pass them through `MM_DATA`, `MM_MODEL`, and `MM_TOOL`. The extractor writes `data.jsonl` and a `features/` directory under its configured output directory.

The example extraction output is `examples/moworld/local_data/feature_extraction/`, which is ignored by Git. To train on this output, set the local training configuration's `data_path` to `./examples/moworld/local_data/feature_extraction/data.jsonl` and `data_folder` to `./examples/moworld/local_data/feature_extraction`. The generated manifest already includes `features/` in each feature path.

#### Start high-noise expert training

The following example uses 16 NPUs with context parallel size 2. Adjust device count, parallelism, batch size, and sequence length for your hardware and workload.

```bash
# High-noise expert
MM_DATA="$PWD/examples/moworld/local_data/train_data.json" \
LOAD_PATH="$MOWORLD_ROOT/checkpoints/base/dcp/high_noise_model" \
SAVE_PATH="$MOWORLD_ROOT/checkpoints/train/high_noise_model" \
NPUS_PER_NODE=16 CP=2 TRAIN_ITERS=10 SAVE_INTERVAL=10 \
bash examples/moworld/pretrain_high.sh
```

The low-noise launcher remains available for users who supply a separate compatible low-noise checkpoint; it is not part of the published checkpoint workflow. This is a short expert-pretraining example, not the full long-horizon curriculum from the paper. Logs are written under `moworld/logs/` and checkpoints under `SAVE_PATH`.

For multiple nodes, run the same launcher on each node with a shared `MASTER_ADDR`/`MASTER_PORT`, the total `NNODES`, and a different `NODE_RANK`. The repository also includes [pretrain_high_dist.sh](moworld/examples/moworld/pretrain_high_dist.sh), an Accelerate launcher with environment-specific settings that must be adapted before use.


This branch provides expert pretraining and data preparation. It does not include MoWorld inference launchers or distillation training. End-to-end training requires validation on the target Ascend environment.
