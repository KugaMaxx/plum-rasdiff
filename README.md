# Sensor-based Smoke Reconstruction via Retrieval-Augmented Generative Model

**Retrieval-Augmented Smoke Diffusion (RAS-Diff)** is a generative model designed
 for reconstructin dynamic smoke distribution in complex urban fire scenarios by 
 only using several sensor data readings. It leverages a Stable Diffusion model 
 trained on simulated smoke fields to generate possible smoke dispersion patterns, 
 with a Retrieval-Augmented Generation (RAG) module used for eliminating hallucination
 issues.

<span id="animation"></span>
![animation](https://raw.githubusercontent.com/KugaMaxx/plum-rasdiff/main/assets/images/demonstration.webp "animation")

## Quick Start

### Preparation

Clone the repository:

```bash
git clone https://github.com/KugaMaxx/plum-rasdiff.git
```

Install the required packages:

```bash
pip install -r requirements.txt
```

### Running the pipeline

You can run the following script to perform inference using a pretrained RAS-Diff model:

```bash
export MODEL_NAME="KugaMaxx/ras-diff"
export VALIDATION_CASE="KugaMaxx/smokepv-control/validation/corridor_s00_c04_h1618"
export DATABASE_NAME="KugaMaxx/smokepv-control"

python3 run_pipeline_controlnet.py \
  --pretrained_model_name_or_path $MODEL_NAME \
  --validation_case $VALIDATION_CASE \
  --database_name_or_path $DATABASE_NAME \
  --top_k 3 \
  --index_dim 1024
```

**NOTE:** Since our RAS-Diff follows the [🤗Diffusers](https://huggingface.co/docs/diffusers/index)
 library, the pretrained model and dataset will be automatically downloaded from 
 the [🤗Hugging Face](https://huggingface.co/) when you run the script above. Or you 
 can download from [here](https://huggingface.co/KugaMaxx) 
 and specify the local path.

## Custom Model Training

### Train your own RAS-Diff

You can run the following script to train your own RAS-Diff model, which will be
 saved to the `--output_dir`.

```bash
export MODEL_NAME="stable-diffusion-v1-5/stable-diffusion-v1-5"
export DATABASE_NAME="KugaMaxx/smokepv-control"
export OUTPUT_DIR=<path_to_save>

accelerate launch python3 train_controlnet.py \
  --pretrained_model_name_or_path $MODEL_NAME \
  --dataset_name_or_path $DATABASE_NAME \
  --output_dir $OUTPUT_DIR \
  --train_batch_size 12 \
  --num_train_epochs 3 \
  --learning_rate 1e-5 \
  --validation_steps 1000 \
  --validation_ids 1500 5500 8500
```

**NOTE:** The following hyperparameters can significantly reduce GPU VRAM.
- `--mixed_precision` supports `fp16` or `bf16` mode to activate mixed precision training.
- `--gradient_checkpointing` and `--gradient_accumulation_steps` enable gradient checkpointing and gradient accumulation.
- `--use_8bit_adam` use 8-bit Adam optimizer to reduce memory usage.
- `--enable_xformers_memory_efficient_attention` activate xFormers optimized sparse attention implementation.

### Prepare training data

To train your own RAS-Diff model, you first need to prepare the training data
 in the [🤗Datasets](https://huggingface.co/docs/datasets/index) format, which
 should follow the below structure:

```markdown
data_files:
- split: train
  path: "train/*/*.parquet"
- split: validation
  path: "validation/*/*.parquet"
dataset_info:
  features:
  - name: case
    dtype: string
  - name: image
    dtype: image
  - name: conditioning_image
    dtype: image
  - name: text
    dtype: string
  - name: min_value
    dtype: float32
  - name: max_value
    dtype: float32
```

**NOTE:** The description of each field is listed below.
- `case`: The unique identifier for each smoke simulation case.
- `image`: The ground truth smoke distribution image.
- `conditioning_image`: The hybrid image constructed from the top-K retrieved smoke
 fields based on the sensor readings.
- `text`: The sensor readings in a comma-separated string format.
- `min_value` and `max_value`: The minimum and maximum values of the smoke density
 in the corresponding smoke distribution, used for normalization.

## BibTeX

If you find RAS-Diff inspiring or helpful for your own research, please consider
 citing our paper:

```bibtex
@article{ding2025rasdiff,
  title = {Sensor-based Smoke Reconstruction via Retrieval-Augmented Generative Model},
}
```

## Acknowledgement

We would like to thank [Weikang XIE](mailto:wei-kang.xie@connect.polyu.hk) for 
 his valuable insights and support in this project.
