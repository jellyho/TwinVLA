# TwinVLA: Data-Efficient Bimanual Policy with Twin Single-Arm Vision-Language-Action Models

[![Static Badge](https://img.shields.io/badge/arXiv-2511.05275-red)](https://arxiv.org/abs/2511.05275)
[![Static Badge](https://img.shields.io/badge/🤗-Models-yellow)](https://huggingface.co/collections/jellyho/twinvla)
[![Static Badge](https://img.shields.io/badge/🤗-Datasets-yellow)](https://huggingface.co/collections/jellyho/twinvla-datasets)
[![Static Badge](https://img.shields.io/badge/🌍-Project_Page-blue)](https://jellyho.github.io/TwinVLA/)
[![Static Badge](https://img.shields.io/badge/🤖-Tabletop_Sim-black)](https://github.com/jellyho/Tabletop-Sim)
![Static Badge](https://img.shields.io/badge/Python-3.10-green)



## 📰 News
- [2026/03/01] We release the code of TwinVLA.
- [2026/01/26] [TwinVLA](https://arxiv.org/abs/2511.05275) got accepted to ICLR 2026. 🎉
- [2025/11/10] [TwinVLA](https://arxiv.org/abs/2511.05275) is now on arXiv.

<!-- - [2024/12/17] 🔥 [Scripts](#simulation-benchmark) for evaluating RDT in Maniskill Simulation Benchmark is released!
- [2024/10/23] 🔥 **RDT-170M** (Smaller) model is released, a more VRAM-friendly solution 🚀💻. -->

[**Installation**](#installation) | [**Quick Usage**](#quick-usage) | [**Custom Dataset**](#custom-dataset) | [**RoboTwin**](#robotwin) | [**Tabletop-Sim**](#tabletop-sim) | [**New VLM Backbones**](#new-vlm) | [**Citation**](#citation)



<p align="center">
  <img src="media/twinvla_overview.svg" width="800">
</p>


<a id="installation"></a>
## 💻 Installation

It is recommended to use Anaconda.

```bash
# Cloning TwinVLA
git clone https://github.com/jellyho/TwinVLA.git
cd TwinVLA

# Create Conda env
conda create -n twinvla python=3.10 -y
conda activate twinvla

# For compiling rerun-sdk (LeRobot dependency)
conda install -c conda-forge rust -y

# Install Requirements and TwinVLA
pip install -r requirements.txt

# Additional installation for lerobot
# Install lerobot & downgrade numpy<2.0 (Please ignore dependency conflicts)
pip install "lerobot==0.4.0"
pip install "numpy<2.0.0"
```

<a id="quick-usage"></a>
## 🤖 Quick Usage

```python
from twinvla.model.twinvla import TwinVLA

model = TwinVLA('jellyho/TwinVLA-aloha_handover_box')

actions = model.predict_action(
    unnorm_key='aloha_handover_box', 
    instruction=instruction, 
    image=front_img,
    image_wrist_r=right_wrist_img,
    image_wrist_l=left_wrist_img,
    proprio=proprio,
)

for action in actions:
    robot.excute(action)
```

---

<a id="custom-dataset"></a>
## 🔧 Fine-tune TwinVLA on Custom Dataset

### 1. Fine-tuning (🤗 LeRobot)

We assume you have already uploaded your dataset in [LeRobot](https://github.com/huggingface/lerobot) format.
Action / State dimension should be 20D: 2 × (xyz, 6d rotation, gripper).

#### 1️⃣ Configuration

Change the [lerobot config](https://github.com/jellyho/TwinVLA/blob/main/twinvla/datasets/lerobot/configs.py) file to match your dataset. This config file defines the key to use for training. Refer to `TabletopSimConfig` in [lerobot config](https://github.com/jellyho/TwinVLA/blob/main/twinvla/datasets/lerobot/configs.py) for more details.

#### 2️⃣ Start finetuning

```bash
sh train_twinvla.sh <lerobot-dataset-path> <batch-size> <num-gpu>
```

### 2. Fine-tuning (RLDS)

Since we are based on OpenVLA's RLDS dataset loader, we support fine-tuning with RLDS datasets. We also provide instructions to convert datasets in HDF5 format into the RLDS format. If you already have an RLDS dataset, you can skip to Step 2️⃣.

#### 1️⃣ Prepare your dataset

In this section, we explain how to convert your custom dataset, stored in HDF5 format, into RLDS format.

To convert HDF5 to RLDS, you first need to install the requirements. We recommend installing them in a new environment.

```bash
cd scripts/rlds_gen
pip install -r requirements_rlds.txt
```

Modify `scripts/rlds_gen/rlds_builder_robotwin.py` (or use it as a template) to match the key names and feature specs of your HDF5 file. Note that TwinVLA only requires a 20D EEF pose action space.

```bash
cd scripts/rlds_gen
CUDA_VISIBLE_DEVICES="" python rlds_builder_robotwin.py --task_name $dataset_name
```

#### 2️⃣ Register your dataset

You need to register your RLDS dataset by adding entries to the following files:
- [config](https://github.com/jellyho/TwinVLA/blob/main/twinvla/datasets/rlds/oxe/configs.py)
- [transform function](https://github.com/jellyho/TwinVLA/blob/main/twinvla/datasets/rlds/oxe/transforms.py)
- [control frequency](https://github.com/jellyho/TwinVLA/blob/main/twinvla/datasets/rlds/oxe/hzs.py)

#### 3️⃣ Start finetuning

Run `train_twinvla.sh`. Make sure you specify the arguments correctly, including changing `--output_dir` for checkpoints.

```bash
sh train_twinvla.sh <task-name> <batch-size> <num-gpu>
```

---

<a id="robotwin"></a>
## 🤖 Fine-tune & Evaluate TwinVLA on RoboTwin

### 1. Fine-tuning (RLDS)

#### 1️⃣ Download RoboTwin dataset

Use the code below to download the RoboTwin data converted to RLDS.

> Note that this data is simply the [RoboTwin](https://github.com/robotwin-Platform/RoboTwin) dataset converted to RLDS format to use 20D EEF pose action space.

```bash
huggingface-cli download jellyho/robotwin2_rlds --repo-type dataset --local-dir ./robotwin2_rlds
```

#### 2️⃣ Finetune TwinVLA on RoboTwin

After downloading the dataset, replace the value of the `--data_root_dir` argument in `train_twinvla.sh` with the path to the downloaded `robotwin2_rlds` directory:

```bash
...
--data_root_dir "path/to/robotwin2_rlds" \
...
```

Then, you can start training by running:

```bash
sh train_twinvla.sh <task-name> <batch-size> <num-gpu>
# e.g. sh train_twinvla.sh robotwin_open_laptop 4 2
```

### 2. Evaluation

You can evaluate TwinVLA using the RoboTwin codebase with just a few simple setup steps.

#### 1️⃣ Clone RoboTwin

```bash
git clone https://github.com/RoboTwin-Platform/RoboTwin.git --recursive
cd RoboTwin
```

#### 2️⃣ Set up the environment

Create and activate a new Conda environment following the [RoboTwin installation guide](https://robotwin-platform.github.io/doc/usage/robotwin-install.html#1-install):

After installing RoboTwin's dependencies, return to the **TwinVLA** project directory and install its dependencies into the same Conda environment:

```bash
pip install -r requirements.txt && pip install -e .
```

#### 3️⃣ Place TwinVLA inside `RoboTwin/policy`

Copy the `./TwinVLA_robotwin` folder from this project into RoboTwin's `./policy` directory, and rename it to `TwinVLA`:

```bash
mv TwinVLA_robotwin ../RoboTwin/policy/TwinVLA
```

#### 4️⃣ Run evaluation

Move into the `RoboTwin/policy/TwinVLA` directory and run evaluation using:

```bash
bash eval.sh <ckpt-path> <task-name> <task-config> <ckpt-setting> <seed> <gpu-id>
```

For example, to evaluate a model fine-tuned on `demo_clean` for the `open_laptop` task, but run evaluation in the `demo_randomized` setup (with `seed=42` and `gpu_id=0`):

```bash
bash eval.sh /path/to/ckpt open_laptop demo_randomized demo_clean 42 0
```

For more details, refer to the [RoboTwin policy deploy documentation](https://robotwin-platform.github.io/doc/usage/deploy-your-policy.html).

---

<a id="tabletop-sim"></a>
## 🕹️ Fine-tune & Evaluate TwinVLA on Tabletop-Sim

### 1. Fine-tuning (RLDS)

#### 1️⃣ Download dataset

You can download the Tabletop-Sim dataset by running:

```bash
huggingface-cli download jellyho/tabletop-simulation-rlds --repo-type dataset --local-dir ./tabletop-simulation-rlds  # Total size: 56GB
```

#### 2️⃣ Finetune TwinVLA on Tabletop-Sim

After downloading the dataset, replace the value of the `--data_root_dir` argument in `train_twinvla.sh` with the path to the downloaded `tabletop-simulation-rlds` directory:

```bash
...
--data_root_dir "path/to/tabletop-simulation-rlds" \
--data_mix "$data_mix" \
...
```

Then, you can start training by running:

```bash
sh train_twinvla.sh <task-name> <batch-size> <num-gpu>
# e.g. sh train_twinvla.sh aloha_handover_box 4 2
```

### 2. Fine-tuning (🤗 LeRobot)

You can directly load the LeRobot dataset from Hugging Face by entering the dataset path.

```bash
sh train_twinvla.sh <lerobot-dataset-path> <batch-size> <num-gpu>
# e.g. sh train_twinvla.sh jellyho/aloha_handover_box 8 2
```

### 3. Evaluation

After fine-tuning TwinVLA on the Tabletop-Sim dataset, you can rollout the model in Tabletop-Sim.

#### 1️⃣ Install Tabletop-Sim

To run the simulation rollout, first download and install Tabletop-Sim:

```bash
git clone https://github.com/jellyho/Tabletop-Sim.git --recursive

cd Tabletop-Sim
pip install -r requirements.txt
pip install 'numpy<2'
```

#### 2️⃣ Evaluate TwinVLA

After installing the simulator, you can run the evaluation by executing:

```bash
sh tabletop_run.sh /path/to/checkpoint <task-name>  # e.g. aloha_handover_box
```

You can also try using our fine-tuned model ([Models](https://huggingface.co/collections/jellyho/twinvla)) on Tabletop-Sim:
```bash
sh tabletop_run.sh jellyho/aloha_dish_drainer aloha_dish_drainer
```


<a id="new-vlm"></a>
## 💡 Try new VLM backbones!

Our version of TwinVLA is built on a modular architecture that makes it easy to experiment with different VLM backbones. You can quickly integrate new models using our automated template generator.

### 1. Generate SingleVLA Template

Since TwinVLA builds upon SingleVLA, start by creating a template for your desired backbone. Run the following command:

```bash
python3 singlevla_gen.py --model_type <YourModelName>
```

Example:
```bash
python3 singlevla_gen.py --model_type InternVL3_1B
```

This will generate a new Python file at `twinvla/model/singlevlas/<your_model_name>.py` (e.g., `internvl3_1b.py`).

### 2. Implement the Template

Open the generated file. It contains a class definition for your new VLA model (e.g., `InternVL3_1BVLA`). You need to:

1.  **Direct Imports**: Update the `TODO` section to import your specific model's configuration and class (e.g., from `transformers` or a local file).
2.  **Set Pretrained Path**: Fill in the `pretrained_path` in the config class with the Hugging Face model ID.
3.  **Implement Methods**: Fill in the method stubs that raise `NotImplementedError`. The generated docstrings provide detailed instructions for each method:
    *   `init_processor_tokenizer`: Initialize tokenizers/processors.
    *   `text_backbone` & `vision_backbone`: Return the underlying model components.
    *   `process_image` & `image_embeds`: Define how images are preprocessed and encoded.
    *   `image_seq_len`, `image_start/end_token`: Specify token details.

**Tip**: You can refer to existing implementations like `twinvla/model/singlevlas/eagle2_1b.py` or `smolvlm2.py` for guidance.

### 3. Register the Model

To ensure your new model is recognized by the training script, add an import statement to `twinvla/model/singlevlas/__init__.py`:

```python
from .<your_model_name_lower> import *
```

### 4. Train with the New Backbone

You can now train your new VLA model by specifying its name in the `--model_type` argument:

```bash
# Example for SingleVLA training
accelerate launch scripts/train.py \
    --model_type <YourModelName>VLA \
    --output_dir checkpoints/my_new_backbone \
    ...
```

To extend this to a dual-arm **TwinVLA**, create a corresponding configuration in `twinvla/model/twinvlas/` following the existing patterns (e.g., `eagle2_1b.py` in that directory) and implement any necessary attention mechanism overrides if using a non-standard architecture.

## Acknowledgments

This repository leverages code from the following open-source projects:
- [OpenVLA](https://github.com/openvla/openvla) for the RLDS data loading codebase.
- [DiT](https://github.com/facebookresearch/DiT) for the DiT policy head.
- [MobileVLM-V2](https://github.com/Meituan-Waimai/MobileVLM) for the training pipeline.


## Citation

If you find this work useful, please consider citing:
```bibtex
@inproceedings{
im2026twinvla,
title={Twin{VLA}: Data-Efficient Bimanual Manipulation with Twin Single-Arm Vision-Language-Action Models},
author={Hokyun Im and Euijin Jeong and Andrey Kolobov and Jianlong Fu and Youngwoon Lee},
booktitle={The Fourteenth International Conference on Learning Representations},
year={2026},
url={https://openreview.net/forum?id=jG9W6nAwVz}
}
```