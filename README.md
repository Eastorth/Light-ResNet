# Light ResNet for PRPD Classification

This project trains a lightweight ResNet-style classifier for partial discharge pattern recognition (PRPD). It includes synthetic data generation, training, evaluation, and inference utilities. No public dataset is required; a synthetic dataset is generated automatically.

## Setup

```bash
pip install -r requirements.txt
```

## Generate Synthetic Dataset

Creates `datasets/synthetic_prpd/` with four PRPD pattern classes.

```bash
python src/synthetic_prpd.py
```

## Train

```bash
python src/train.py \
  --data_root datasets/synthetic_prpd \
  --out_dir runs/synth_run \
  --epochs 30 \
  --batch_size 32 \
  --input_channels 1
```

## Evaluate

```bash
python src/eval.py --data_root datasets/synthetic_prpd --ckpt runs/synth_run/best_model.pt --input_channels 1
```

## Inference

```bash
python src/infer.py --ckpt runs/synth_run/best_model.pt --image_path <path_to_png> --input_channels 1
```
