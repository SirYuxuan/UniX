# Training

## Environment

Refer to [Installation](INSTALLATION.md) for environment setup.

## Data Preparation

The data preparation process for UniX is similar to the original project. To obtain the example dataset, run:

```bash
wget -O bagel_example.zip \
  https://lf3-static.bytednsdoc.com/obj/eden-cn/nuhojubrps/bagel_example.zip
unzip bagel_example.zip -d ./
```

The `DATASET_INFO` in `data/dataset_info.py` is configured with default paths. If creating your own dataset, maintain consistency with the example dataset processing logic.

## Training Tasks

> **Note**: If your hardware supports P2P (e.g., NVLink between GPUs), comment out `export NCCL_P2P_DISABLE=1` in the scripts.

### Medical Understanding Finetuning

Fine-tuning for medical image understanding tasks.

1. In `data/configs/dataset.yaml`, set the `weight` under `vlm_sft` to `1`, and ensure `weight` is set to `0` for all other entries.
2. Run:

```bash
bash train/scripts/medical_und_ft.sh
```

### Medical Generation Pretraining

Pretraining for medical image generation tasks.

1. In `weights/rad-dino/preprocessor_config.json`, modify `height` and `width` in `crop_size` to `224`, and set `shortest_edge` in `size` to `224`.
2. In `data/configs/dataset.yaml`, set the `weight` under `t2i_pretrain` to `1`, and ensure `weight` is set to `0` for all other entries.
3. Run:

```bash
export RAD_DINO_PATH="weights/rad-dino"
bash train/scripts/medical_gen_pt.sh
```

### Medical Generation Finetuning

Fine-tuning for medical image generation tasks.

1. In `data/configs/dataset.yaml`, set the `weight` under `t2i_finetune` to `1`, and ensure `weight` is set to `0` for all other entries.
2. Run:

```bash
bash train/scripts/medical_gen_ft.sh
```

## Dataset Configuration

The `data/configs/dataset.yaml` file controls which training task is active by adjusting weights:

| Task | Dataset Key | Image Size |
|------|-------------|------------|
| Understanding SFT | `vlm_sft` | 384x384 |
| Generation Pretrain | `t2i_pretrain` | 256x256 |
| Generation Finetune | `t2i_finetune` | 512x512 |
