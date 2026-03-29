export NCCL_P2P_DISABLE=1

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun \
  --nnodes=1 \
  --node_rank=0 \
  --nproc_per_node=8 \
  --master_addr=127.0.0.1 \
  --master_port=12346 \
  pretrain_unified_navit.py \
  --results_dir ./results \
  --checkpoint_dir ./results/checkpoints \
  --dataset_config_file data/configs/dataset.yaml \
  --model_path weights/UniX \
  --llm_path weights/Janus-Pro-1B \
  --vit_path weights/siglip-large-patch16-384 \
  --vae_path weights/vae/f16d16/kl-f16d16.ckpt \
  --layer_module Qwen2MoTDecoderLayer \
  --auto_resume True \
  --resume_model_only False \
  --max_latent_size 16 \
  --latent_patch_size 1 \
  --log_every 1 \
  --save_every 640 \
  --lr 1e-4 \
  --num_workers 1 \
  --num_replicate 1 \
  --num_shard 8 \
  --micro_batch_size 32 \
  --global_batch_size 256 \
  --max_tokens_per_batch 45000 \
  --max_seq_len 10000 \
  --warmup_steps 80 \
  --visual_gen False \
  --visual_und True 