# Copyright 2025 Bytedance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0

import functools
import os
import yaml
from copy import deepcopy
from dataclasses import dataclass, field
import time

import torch
import torch.distributed as dist
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    CheckpointImpl,
    apply_activation_checkpointing,
    checkpoint_wrapper,
)
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from transformers import HfArgumentParser, set_seed
from transformers.optimization import (
    get_constant_schedule_with_warmup,
    get_cosine_with_min_lr_schedule_with_warmup,
)

from data.dataset_base import DataConfig, PackedDataset, collate_wrapper
from data.data_utils import add_special_tokens
from modeling.autoencoder import load_ae
from modeling.unix import (
    UniXConfig, UniX, Qwen2Config, Qwen2ForCausalLM, SiglipVisionConfig, SiglipVisionModel
)
from modeling.qwen2 import Qwen2Tokenizer
from modeling.unix_vlm_manager import UnixVLMComponentManager
from train.train_utils import create_logger, get_latest_ckpt
from train.fsdp_utils import (
    FSDPCheckpoint, FSDPConfig, grad_checkpoint_check_fn, fsdp_wrapper, 
    fsdp_ema_setup, fsdp_ema_update,
)


# UnixVLMComponentManager has been moved to modeling/unix_vlm_manager.py


@dataclass
class ModelArguments:
    model_path: str = field(
        default=None,
        metadata={"help": "Path of the pretrained UniX model directory containing model.safetensors."}
    )
    config_path: str = field(
        default=None,
        metadata={"help": "Explicit path to the unified config.json file (defaults to inference/config.json)."}
    )
    llm_path: str = field(
        default=None,
        metadata={"help": "Path to Janus language model weights. If None, read from config.json."}
    )
    llm_qk_norm: bool = field(
        default=True,
        metadata={"help": "Enable QK LayerNorm (qk_norm) inside the attention blocks."}
    )
    tie_word_embeddings: bool = field(
        default=False,
        metadata={"help": "Share input and output word embeddings (tied embeddings)."}
    )
    layer_module: str = field(
        default=None,
        metadata={"help": "Python class name of the decoder layer to instantiate. If None, read from config.json."}
    )
    vae_path: str = field(
        default=None,
        metadata={"help": "Path to the pretrained VAE checkpoint. If None, read from config.json."}
    )
    vit_path: str = field(
        default=None,
        metadata={"help": "Path to Janus SigLIP Vision Transformer. If None, read from config.json."}
    )
    max_latent_size: int = field(
        default=None,
        metadata={"help": "Maximum latent grid size. If None, read from config.json."}
    )
    latent_patch_size: int = field(
        default=None,
        metadata={"help": "Spatial size covered by each latent patch. If None, read from config.json."}
    )
    vit_patch_size: int = field(
        default=None,
        metadata={"help": "Patch size for the ViT encoder. If None, read from config.json."}
    )
    vit_max_num_patch_per_side: int = field(
        default=None,
        metadata={"help": "Maximum number of ViT patches along one image side. If None, read from config.json."}
    )
    connector_act: str = field(
        default=None,
        metadata={"help": "Activation function for connector MLP. If None, read from config.json."}
    )
    interpolate_pos: bool = field(
        default=None,
        metadata={"help": "Interpolate positional embeddings. If None, read from config.json."}
    )
    vit_select_layer: int = field(
        default=-1,
        metadata={"help": "Which hidden layer of the ViT to take as the visual feature (negative = from the end)."}
    )
    vit_rope: bool = field(
        default=False,
        metadata={"help": "Replace ViT positional encodings with RoPE."}
    )

    text_cond_dropout_prob: float = field(
        default=0.1,
        metadata={"help": "Probability of dropping text embeddings during training."}
    )
    vae_cond_dropout_prob: float = field(
        default=0.3,
        metadata={"help": "Probability of dropping VAE latent inputs during training."}
    )
    vit_cond_dropout_prob: float = field(
        default=0.3,
        metadata={"help": "Probability of dropping ViT visual features during training."}
    )


@dataclass
class DataArguments:
    dataset_config_file: str = field(
        default="data/configs/example.yaml",
        metadata={"help": "YAML file specifying dataset groups, weights, and preprocessing rules."}
    )
    prefetch_factor: int = field(
        default=2,
        metadata={"help": "How many batches each DataLoader worker pre-loads in advance."}
    )
    num_workers: int = field(
        default=4,
        metadata={"help": "Number of background workers for the PyTorch DataLoader."}
    )
    max_seq_len: int = field(
        default=16384,
        metadata={"help": "Maximum tokens allowed in one raw sample; longer samples are skipped."}
    )
    max_tokens_per_batch: int = field(
        default=36864,
        metadata={"help": "Hard limit on tokens in a packed batch; flush if adding a sample would exceed it."}
    )
    prefer_buffer_before: int = field(
        default=16384,
        metadata={"help": "While batch length is below this, pop from the overflow buffer before new sampling."}
    )
    max_buffer_size: int = field(
        default=50,
        metadata={"help": "Maximum number of oversized samples kept in the overflow buffer."}
    )
    data_seed: int = field(
        default=42,
        metadata={"help": "Seed used when shuffling / sampling data shards to ensure reproducibility."}
    )


@dataclass
class TrainingArguments:
    # --- modality switches ---
    visual_gen: bool = field(
        default=True,
        metadata={"help": "Train image generation branch."}
    )
    visual_und: bool = field(
        default=True,
        metadata={"help": "Train image understanding branch."}
    )

    # --- bookkeeping & logging ---
    results_dir: str = field(
        default="./results",
        metadata={"help": "Root directory for logs."}
    )
    checkpoint_dir: str = field(
        default="./results/checkpoints",
        metadata={"help": "Root directory for model checkpoints."}
    )

    # --- reproducibility & resume ---
    global_seed: int = field(
        default=4396,
        metadata={"help": "Base random seed; actual seed is offset by rank for DDP."}
    )
    auto_resume: bool = field(
        default=False,
        metadata={"help": "Automatically pick up the latest checkpoint found in checkpoint_dir."}
    )
    resume_from: str = field(
        default=None,
        metadata={"help": "Explicit checkpoint path to resume from (overrides auto_resume)." }
    )
    resume_model_only: bool = field(
        default=False,
        metadata={"help": "Load only model weights, ignoring optimizer/scheduler states."}
    )
    finetune_from_ema: bool = field(
        default=False,
        metadata={"help": "When resume_model_only=True, load the EMA (exponential moving average) weights instead of raw weights."}
    )
    finetune_from_hf: bool = field(
        default=False,
        metadata={"help": "Whether finetune from HugginFace model."}
    )

    # --- reporting frequency ---
    log_every: int = field(
        default=10,
        metadata={"help": "Print / log every N training steps."}
    )
    save_every: int = field(
        default=2000,
        metadata={"help": "Save a checkpoint every N training steps."}
    )
    total_steps: int = field(
        default=5_000_000,
        metadata={"help": "Total number of optimizer steps to train for."}
    )

    # --- optimization & scheduler ---
    warmup_steps: int = field(
        default=2000,
        metadata={"help": "Linear warm-up steps before applying the main LR schedule."}
    )
    lr_scheduler: str = field(
        default="constant",
        metadata={"help": "Type of LR schedule: 'constant' or 'cosine'."}
    )
    lr: float = field(
        default=1e-4,
        metadata={"help": "Peak learning rate after warm-up."}
    )
    min_lr: float = field(
        default=1e-7,
        metadata={"help": "Minimum learning rate for cosine schedule (ignored for constant)."}
    )
    beta1: float = field(
        default=0.9,
        metadata={"help": "AdamW β₁ coefficient."}
    )
    beta2: float = field(
        default=0.95,
        metadata={"help": "AdamW β₂ coefficient."}
    )
    eps: float = field(
        default=1e-15,
        metadata={"help": "AdamW ε for numerical stability."}
    )
    ema: float = field(
        default=0.9999,
        metadata={"help": "Decay rate for the exponential moving average of model weights."}
    )
    max_grad_norm: int = field(
        default=1.0,
        metadata={"help": "Gradient clipping threshold (L2 norm)."}
    )
    timestep_shift: float = field(
        default=1.0,
        metadata={"help": "Shift applied to diffusion timestep indices (for latent prediction)."}
    )
    mse_weight: float = field(
        default=1.0,
        metadata={"help": "Scaling factor for the image-reconstruction MSE loss term."}
    )
    ce_weight: float = field(
        default=1.0,
        metadata={"help": "Scaling factor for the language cross-entropy loss term."}
    )
    ce_loss_reweighting: bool = field(
        default=False,
        metadata={"help": "Reweight CE loss by token importance (provided via ce_loss_weights)."}
    )
    expected_num_tokens: int = field(
        default=32768,
        metadata={"help": "Soft target token count; yield the batch once it reaches or exceeds this size."}
    )
    micro_batch_size: int = field(
        default=None,
        metadata={"help": "Number of samples per GPU per step (local batch size). If None, use expected_num_tokens instead."}
    )
    global_batch_size: int = field(
        default=256,
        metadata={"help": "Total number of samples across all GPUs for gradient accumulation (global batch size)."}
    )

    # --- distributed training / FSDP ---
    num_replicate: int = field(
        default=1,
        metadata={"help": "Number of model replicas per GPU rank for tensor parallelism."}
    )
    num_shard: int = field(
        default=4,
        metadata={"help": "Number of parameter shards when using FSDP HYBRID_SHARD."}
    )
    sharding_strategy: str = field(
        default="HYBRID_SHARD",
        metadata={"help": "FSDP sharding strategy: FULL_SHARD, SHARD_GRAD_OP, HYBRID_SHARD, etc."}
    )
    backward_prefetch: str = field(
        default="BACKWARD_PRE",
        metadata={"help": "FSDP backward prefetch strategy (BACKWARD_PRE or NO_PREFETCH)."}
    )
    cpu_offload: bool = field(
        default=False,
        metadata={"help": "Enable FSDP parameter offload to CPU."}
    )

    # --- module freezing ---
    freeze_llm: bool = field(
        default=False,
        metadata={"help": "Keep language-model weights fixed (no gradient updates)."}
    )
    freeze_vit: bool = field(
        default=False,
        metadata={"help": "Keep ViT weights fixed during training."}
    )
    freeze_vae: bool = field(
        default=True,
        metadata={"help": "Keep VAE weights fixed; only predict latents, don't fine-tune encoder/decoder."}
    )
    freeze_und: bool = field(
        default=False,
        metadata={"help": "Freeze the visual understanding connector layers."}
    )
    copy_init_moe: bool = field(
        default=True,
        metadata={"help": "Duplicate initial MoE experts so each has identical initialisation."}
    )
    use_flex: bool = field(
        default=False,
        metadata={"help": "Enable FLEX (flash-ext friendly) packing algorithm for sequence data."}
    )

    # --- REPA configuration ---
    use_repa: bool = field(
        default=False,
        metadata={"help": "Enable REPA (Representation Alignment) for improved training efficiency."}
    )
    repa_enc_type: str = field(
        default="dinov2-vit-b",
        metadata={"help": "REPA encoder type (e.g., dinov2, raddino)."}
    )
    repa_proj_coeff: float = field(
        default=0.5,
        metadata={"help": "REPA projection loss coefficient."}
    )
    repa_projector_dim: int = field(
        default=2048,
        metadata={"help": "REPA projector hidden dimension."}
    )
    repa_encoder_depth: int = field(
        default=8,
        metadata={"help": "REPA encoder depth for feature extraction."}
    )


def main():
    assert torch.cuda.is_available()
    dist.init_process_group("nccl")
    device = dist.get_rank() % torch.cuda.device_count()
    torch.cuda.set_device(device)
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    
    # Setup logging:
    if dist.get_rank() == 0:
        os.makedirs(training_args.results_dir, exist_ok=True)
        os.makedirs(training_args.checkpoint_dir, exist_ok=True)
        logger = create_logger(training_args.results_dir, dist.get_rank())
        # Setup TensorBoard logging
        tensorboard_log_dir = os.path.join(training_args.results_dir, "tensorboard_logs")
        os.makedirs(tensorboard_log_dir, exist_ok=True)
        writer = SummaryWriter(log_dir=tensorboard_log_dir)
        logger.info(f"TensorBoard logs will be saved to: {tensorboard_log_dir}")
    else:
        logger = create_logger(None, dist.get_rank())
        writer = None
    dist.barrier()
    logger.info(f'Training arguments {training_args}')
    logger.info(f'Model arguments {model_args}')
    logger.info(f'Data arguments {data_args}')

    # prepare auto resume logic:
    if training_args.auto_resume:
        resume_from = get_latest_ckpt(training_args.checkpoint_dir)
        if resume_from is None:
            resume_from = training_args.resume_from
            resume_model_only = training_args.resume_model_only
            if resume_model_only:
                finetune_from_ema = training_args.finetune_from_ema
            else:
                finetune_from_ema = False
        else:
            # 当auto_resume找到检查点时，仍然尊重用户设置的resume_model_only
            resume_model_only = training_args.resume_model_only
            if resume_model_only:
                finetune_from_ema = training_args.finetune_from_ema
            else:
                finetune_from_ema = False
    else:
        resume_from = training_args.resume_from
        resume_model_only = training_args.resume_model_only
        if resume_model_only:
            finetune_from_ema = training_args.finetune_from_ema
        else:
            finetune_from_ema = False

    # Set seed:
    seed = training_args.global_seed * dist.get_world_size() + dist.get_rank()
    set_seed(seed)

    # Setup model configuration:
    config_file = model_args.config_path if model_args.config_path else os.path.join(os.path.dirname(os.path.abspath(__file__)), "train", "config.json")
    logger.info(f"Loading unified configuration from {config_file}...")
    
    # Load unified config
    config = UniXConfig.from_json_file(config_file)
    
    # Override config with command line arguments if provided
    if model_args.llm_path: config.llm_path = model_args.llm_path
    if model_args.vit_path: config.vit_path = model_args.vit_path
    if model_args.vae_path: config.vae_path = model_args.vae_path
    if model_args.max_latent_size: config.max_latent_size = model_args.max_latent_size
    if model_args.latent_patch_size: config.latent_patch_size = model_args.latent_patch_size
    if model_args.vit_patch_size: config.vit_config.patch_size = model_args.vit_patch_size
    if model_args.vit_max_num_patch_per_side: config.vit_max_num_patch_per_side = model_args.vit_max_num_patch_per_side
    if model_args.connector_act: config.connector_act = model_args.connector_act
    if model_args.interpolate_pos is not None: config.interpolate_pos = model_args.interpolate_pos
    if model_args.layer_module: config.llm_config.layer_module = model_args.layer_module
    
    # Sync visual_gen/visual_und from training_args
    config.visual_gen = training_args.visual_gen
    config.visual_und = training_args.visual_und
    config.timestep_shift = training_args.timestep_shift

    # Update model_args for UnixVLMComponentManager
    if not model_args.llm_path: model_args.llm_path = getattr(config, 'llm_path', None)
    if not model_args.vit_path: model_args.vit_path = getattr(config, 'vit_path', None)
    if not model_args.vae_path: model_args.vae_path = getattr(config, 'vae_path', None)
    if not model_args.layer_module: model_args.layer_module = config.llm_config.layer_module

    # Load components using parameters from unified config
    llm_config = config.llm_config

    # Initialize UnixVLMComponentManager
    unix_vlm_manager = UnixVLMComponentManager(
        llm_path=model_args.llm_path,
        vit_path=model_args.vit_path,
        rank=dist.get_rank(),
        logger=logger
    )
    
    language_model = unix_vlm_manager.load_language_model(llm_config, training_args)
    
    # Load vision model if path is provided and exists
    vit_model = None
    if model_args.vit_path and os.path.exists(model_args.vit_path):
        logger.info("Loading Janus vision transformer...")
        vit_model = unix_vlm_manager.load_vision_model()
        
        # Log vit_model loading success
        logger.info(f"Vision model loaded successfully, type: {type(vit_model)}")
        if vit_model is None:
            logger.warning("Vision model is None!")
                

    # Load VAE model if path is provided and exists
    vae_model = None
    vae_config = config.vae_config
    if (model_args.vae_path and os.path.exists(model_args.vae_path)):
        logger.info("Loading VAE model...")
        # Note: load_ae returns (model, config)
        vae_model, _ = load_ae(local_path=model_args.vae_path)
        if vae_model is not None:
            vae_model = vae_model.to(dtype=torch.bfloat16)

    logger.info("Creating UniX model with Janus components...")
    model = UniX(
        language_model, 
        vit_model,
        config
    )

    # Load Janus aligner if vit_model is loaded
    if vit_model is not None:
        model = unix_vlm_manager.replace_model_connector(model)

    # REPA is now integrated into the model itself
    if training_args.use_repa and training_args.visual_gen:
        logger.info("REPA is integrated into the language model")
        logger.info(f"REPA config: enc_type={training_args.repa_enc_type}, "
                   f"proj_coeff={training_args.repa_proj_coeff}, "
                   f"projector_dim={training_args.repa_projector_dim}, "
                   f"encoder_depth={training_args.repa_encoder_depth}")
    else:
        logger.info(f"REPA not enabled: use_repa={training_args.use_repa}, visual_gen={training_args.visual_gen}")

    # Setup tokenizer for model using Janus approach:
    logger.info("Loading tokenizer using Janus VLChatProcessor...")
    tokenizer = unix_vlm_manager.load_tokenizer()
    
    # Get Janus token IDs (pre-defined in Janus tokenizer)
    new_token_ids = unix_vlm_manager.get_new_token_ids()
    
    # Verify token IDs are valid
    unix_vlm_manager.verify_token_ids()
    
    # Set tokenizer for model analysis
    model.set_tokenizer(tokenizer)

    # Ensure the entire model is in bf16
    logger.info("Ensuring all model parameters are in bf16...")
    model = model.to(dtype=torch.bfloat16)

    # maybe freeze something:
    if vae_model is not None:
        if training_args.freeze_vae or not training_args.visual_gen:
            logger.info("Freezing VAE model parameters...")
            for param in vae_model.parameters():
                param.requires_grad = False
    
    # Freeze generation layers in UniX if not training generation
    if not training_args.visual_gen and hasattr(model, 'vae2llm'):
        logger.info("Freezing UniX generation layers (visual_gen=False)...")
        model.vae2llm.requires_grad_(False)
        model.llm2vae.requires_grad_(False)
        model.time_embedder.requires_grad_(False)
        model.latent_pos_embed.requires_grad_(False)
    
    # Auto-freeze language model non-MoE parameters when visual_und=False
    if not training_args.visual_und:
        logger.info("Auto-freezing language model non-MoE parameters (visual_und=False)...")
        model.language_model.eval()
        frozen_count = 0
        total_count = 0
        
        for name, param in model.language_model.named_parameters():
            total_count += 1
            # Only freeze parameters that don't contain 'moe_gen' or 'repa' in their name
            if 'moe_gen' not in name and 'repa' not in name:
                param.requires_grad = False
                frozen_count += 1
        
        logger.info(f"Frozen {frozen_count}/{total_count} language model parameters (non-MoE, non-REPA)")
        logger.info("MoE and REPA parameters remain trainable for generation tasks")
    
    # Manual freeze if explicitly requested
    if training_args.freeze_llm:
        logger.info("Freezing all language model parameters (manual override)...")
        model.language_model.eval()
        for param in model.language_model.parameters():
            param.requires_grad = False
    
    if vit_model is not None:
        if training_args.freeze_vit or not training_args.visual_und:
            logger.info("Freezing ViT model parameters...")
            model.vit_model.eval()
            for param in model.vit_model.parameters():
                param.requires_grad = False
        
        if training_args.freeze_und or not training_args.visual_und:
            logger.info("Freezing visual understanding connector layers...")
            model.connector.requires_grad_(False)
            model.vit_pos_embed.requires_grad_(False)

    # Setup FSDP and load pretrained model:
    logger.info("Setting up FSDP...")
    
    # Check if we have mixed requires_grad parameters
    has_frozen_params = False
    has_trainable_params = False
    
    for param in model.parameters():
        if param.requires_grad:
            has_trainable_params = True
        else:
            has_frozen_params = True
        if has_frozen_params and has_trainable_params:
            break
    
    # If we have mixed requires_grad, we need to set use_orig_params=True
    if has_frozen_params and has_trainable_params:
        logger.info("Detected mixed requires_grad parameters, setting use_orig_params=True in FSDP")
        use_orig_params = True
    else:
        use_orig_params = False
    
    fsdp_config = FSDPConfig(
        sharding_strategy=training_args.sharding_strategy,
        backward_prefetch=training_args.backward_prefetch,
        cpu_offload=training_args.cpu_offload,
        num_replicate=training_args.num_replicate,
        num_shard=training_args.num_shard,
        use_orig_params=use_orig_params,
    )
    ema_model = deepcopy(model)
    model, ema_model = FSDPCheckpoint.try_load_ckpt(
        resume_from, logger, model, ema_model, resume_from_ema=finetune_from_ema
    )
    
    # Save model weights information before FSDP wrapping
    unix_vlm_manager.save_model_weights_info(model, training_args, vae_model, vae_config)
    
    ema_model = fsdp_ema_setup(ema_model, fsdp_config)
    fsdp_model = fsdp_wrapper(model, fsdp_config)
    
    apply_activation_checkpointing(
        fsdp_model, 
        checkpoint_wrapper_fn=functools.partial(
            checkpoint_wrapper, checkpoint_impl=CheckpointImpl.NO_REENTRANT
        ), 
        check_fn=grad_checkpoint_check_fn
    )

    # Setup optimizer and scheduler
    optimizer = torch.optim.AdamW(
        fsdp_model.parameters(), 
        lr=training_args.lr, 
        betas=(training_args.beta1, training_args.beta2), 
        eps=training_args.eps, 
        weight_decay=0
    )
    if training_args.lr_scheduler == 'cosine':
        scheduler = get_cosine_with_min_lr_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=training_args.warmup_steps,
            num_training_steps=training_args.total_steps,
            min_lr=training_args.min_lr,
        )
    elif training_args.lr_scheduler == 'constant':
        scheduler = get_constant_schedule_with_warmup(
            optimizer=optimizer, num_warmup_steps=training_args.warmup_steps
        )
    else:
        raise ValueError

    # maybe resume optimizer, scheduler, and train_steps
    if resume_model_only:
        train_step = 0
        data_status = None
    else:
        optimizer, scheduler, train_step, data_status = FSDPCheckpoint.try_load_train_state(
            resume_from, optimizer, scheduler, fsdp_config, 
        )

    # Setup packed dataloader
    logger.info("Setting up data loader...")
    with open(data_args.dataset_config_file, "r") as stream:
        dataset_meta = yaml.safe_load(stream)
    
    # 当 visual_gen=False 时，过滤掉生成相关的数据集配置
    if not training_args.visual_gen:
        filtered_datasets = {}
        generation_datasets = ['t2i_pretrain', 't2i_finetune']
        
        for name, ds_config in dataset_meta.items():
            if name not in generation_datasets:
                filtered_datasets[name] = ds_config
            else:
                if dist.get_rank() == 0:
                    logger.info(f"Filtering out generation dataset: {name}")
        
        dataset_meta = filtered_datasets
        if dist.get_rank() == 0:
            logger.info(f"Filtered out generation datasets, remaining: {list(filtered_datasets.keys())}")
            logger.info(f"Original dataset count: {len(dataset_meta) + len(generation_datasets)}")
            logger.info(f"Filtered dataset count: {len(filtered_datasets)}")
    
    dataset_config = DataConfig(grouped_datasets=dataset_meta)
    
    # 当 visual_und=False 时，需要过滤掉需要图像理解的数据集
    if not training_args.visual_und:
        filtered_datasets = {}
        # 需要图像理解或图像编辑的数据集（都需要图像输入）
        understanding_datasets = ['vlm_sft']
        
        for name, ds_config in dataset_meta.items():
            if name not in understanding_datasets:
                filtered_datasets[name] = ds_config
            else:
                if dist.get_rank() == 0:
                    logger.info(f"Filtering out understanding/editing dataset: {name} (visual_und=False)")
        
        dataset_meta = filtered_datasets
        if dist.get_rank() == 0:
            logger.info(f"Filtered out understanding/editing datasets, remaining: {list(filtered_datasets.keys())}")
            logger.info(f"Original dataset count: {len(dataset_meta) + len(understanding_datasets)}")
            logger.info(f"Filtered dataset count: {len(filtered_datasets)}")
        
        # 重新创建配置
        dataset_config = DataConfig(grouped_datasets=dataset_meta)
    
    # 设置vit相关配置
    if training_args.visual_und:
        dataset_config.vit_patch_size = model_args.vit_patch_size if model_args.vit_patch_size is not None else config.vit_config.patch_size
        dataset_config.max_num_patch_per_side = model_args.vit_max_num_patch_per_side if model_args.vit_max_num_patch_per_side is not None else config.vit_max_num_patch_per_side
    
    # 设置vae相关配置
    if training_args.visual_gen:
        # Use config values as fallback for model_args
        latent_patch_size = model_args.latent_patch_size if model_args.latent_patch_size is not None else config.latent_patch_size
        max_latent_size = model_args.max_latent_size if model_args.max_latent_size is not None else config.max_latent_size
        
        vae_image_downsample = latent_patch_size * vae_config.downsample
        dataset_config.vae_image_downsample = vae_image_downsample
        dataset_config.max_latent_size = max_latent_size
        dataset_config.text_cond_dropout_prob = model_args.text_cond_dropout_prob
        dataset_config.vae_cond_dropout_prob = model_args.vae_cond_dropout_prob
        dataset_config.vit_cond_dropout_prob = model_args.vit_cond_dropout_prob

    train_dataset = PackedDataset(
        dataset_config,
        tokenizer=tokenizer,
        special_tokens=new_token_ids,
        local_rank=dist.get_rank(),
        world_size=dist.get_world_size(),
        num_workers=data_args.num_workers,
        expected_num_tokens=training_args.expected_num_tokens,
        micro_batch_size=training_args.micro_batch_size,
        max_seq_len=data_args.max_seq_len,
        max_tokens_per_batch=data_args.max_tokens_per_batch,
        max_buffer_size=data_args.max_buffer_size,
        prefer_buffer_before=data_args.prefer_buffer_before,
        interpolate_pos=model_args.interpolate_pos,
        use_flex=training_args.use_flex,
        data_status=data_status,
    )
    train_dataset.set_epoch(data_args.data_seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=1, # batch size is 1 packed dataset
        num_workers=data_args.num_workers,
        pin_memory=True,
        collate_fn=collate_wrapper(),
        drop_last=True,
        prefetch_factor=data_args.prefetch_factor,
    )
    
    # Prepare models for training:
    # Move VAE model to GPU (already in bf16 from loading)
    if training_args.visual_gen:
        logger.info("Moving VAE model to GPU...")
        vae_model = vae_model.to(device=device).eval()
    fsdp_model.train()
    ema_model.eval()

    # Calculate gradient accumulation steps
    if training_args.micro_batch_size is not None:
        local_batch_size = training_args.micro_batch_size
        global_batch_size = training_args.global_batch_size
        world_size = dist.get_world_size()
        gradient_accumulation_steps = global_batch_size // (local_batch_size * world_size)
        if gradient_accumulation_steps < 1:
            gradient_accumulation_steps = 1
        logger.info(f"Gradient accumulation: local_batch_size={local_batch_size}, global_batch_size={global_batch_size}, world_size={world_size}, accumulation_steps={gradient_accumulation_steps}")
    else:
        gradient_accumulation_steps = 1
        logger.info("No gradient accumulation (using token-based batching)")
    
    # Always use gradient accumulation display when gradient_accumulation_steps > 1
    use_gradient_accumulation_display = gradient_accumulation_steps > 1
    logger.info(f"Using gradient accumulation display: {use_gradient_accumulation_display} (gradient_accumulation_steps={gradient_accumulation_steps})")

    # train loop
    start_time = time.time()
    logger.info(f"Training for {training_args.total_steps} steps, starting at {train_step}...")
    
    # Log training configuration and enabled loss types
    enabled_losses = []
    if training_args.visual_und:
        enabled_losses.append("CE (Cross-Entropy)")
    if training_args.visual_gen:
        enabled_losses.append("MSE (Mean Squared Error)")
    if training_args.use_repa and training_args.visual_gen:
        enabled_losses.append("REPA (Representation Alignment)")
    
    logger.info(f"Training configuration:")
    logger.info(f"  - Visual Understanding: {training_args.visual_und}")
    logger.info(f"  - Visual Generation: {training_args.visual_gen}")
    logger.info(f"  - REPA: {training_args.use_repa}")
    logger.info(f"  - Enabled losses: {', '.join(enabled_losses) if enabled_losses else 'None'}")
    logger.info(f"  - Loss weights: CE={training_args.ce_weight}, MSE={training_args.mse_weight}, REPA={training_args.repa_proj_coeff}")
    
    # Initialize task statistics tracking
    cumulative_task_counts = {}
    step_task_history = []
    
    # Initialize gradient accumulation
    accumulated_loss = 0.0
    accumulation_count = 0
    accumulated_samples = 0  # Track samples in current accumulation
    accumulated_task_counts = {}  # Track task counts in current accumulation
    
    # Initialize actual training step counter (only increments after gradient updates)
    # Start from 0 for display purposes
    actual_training_step = 0
    
    for curr_step, data in enumerate(train_loader, start=train_step):

        data = data.cuda(device).to_dict()
        data_indexes = data.pop('batch_data_indexes', None)
        ce_loss_weights = data.pop('ce_loss_weights', None)

        with torch.amp.autocast("cuda", enabled=True, dtype=torch.bfloat16):

            if training_args.visual_gen:
                with torch.no_grad():
                    # 训练时使用随机编码，与latent-diffusion原仓库保持一致
                    data['padded_latent'] = vae_model.encode(data.pop('padded_images'), sample_posterior=True)
                if training_args.use_repa and 'original_images' in data:
                    # Add REPA parameters to data
                    data['return_intermediate_features'] = True
                    data['intermediate_depth'] = training_args.repa_encoder_depth
            else:
                # 如果不需要图像生成，确保移除 padded_images
                if 'padded_images' in data:
                    data.pop('padded_images')

            loss_dict = fsdp_model(**data)


        loss = 0
        ce = loss_dict["ce"]
        if ce is not None:
            total_ce_tokens = torch.tensor(len(data['ce_loss_indexes']), device=device)
            dist.all_reduce(total_ce_tokens, op=dist.ReduceOp.SUM)
            if training_args.ce_loss_reweighting:
                ce = ce * ce_loss_weights
                total_ce_loss_weights = ce_loss_weights.sum()
                dist.all_reduce(total_ce_loss_weights, op=dist.ReduceOp.SUM)
                ce = ce.sum() * dist.get_world_size() / total_ce_loss_weights
            else:
                ce = ce.sum() * dist.get_world_size() / total_ce_tokens
            loss_dict["ce"] = ce.detach()
            loss = loss + ce * training_args.ce_weight
        else:
            assert not training_args.visual_und
            loss_dict["ce"] = torch.tensor(0, device=device)
            total_ce_tokens = torch.tensor(0, device=device)

        if training_args.visual_gen:
            mse = loss_dict["mse"]
            total_mse_tokens = torch.tensor(len(data['mse_loss_indexes']), device=device)
            dist.all_reduce(total_mse_tokens, op=dist.ReduceOp.SUM)
            mse = mse.mean(dim=-1).sum() * dist.get_world_size() / total_mse_tokens
            loss_dict["mse"] = mse.detach()
            loss = loss + mse * training_args.mse_weight
        else:
            assert not training_args.visual_gen
            loss_dict["mse"] = torch.tensor(0, device=device)
            total_mse_tokens = torch.tensor(0, device=device)

        # Compute REPA loss if enabled
        if training_args.use_repa:
            if 'repa_loss' not in loss_dict or loss_dict['repa_loss'] is None:
                raise ValueError(f"REPA loss not found in loss_dict. Please check if REPA encoder is properly configured.")
            repa_loss = loss_dict['repa_loss']
            loss = loss + repa_loss * training_args.repa_proj_coeff

        # Scale loss for gradient accumulation
        scaled_loss = loss / gradient_accumulation_steps
        accumulated_loss += loss.item()  # Keep original loss for logging
        
        # Accumulate samples for current batch
        current_batch_samples = len(data['sample_lens'])
        accumulated_samples += current_batch_samples
        
        # Accumulate task counts for current batch
        for item in data_indexes:
            dataset_name = item['dataset_name']
            if dataset_name not in accumulated_task_counts:
                accumulated_task_counts[dataset_name] = 0
            accumulated_task_counts[dataset_name] += 1
        
        # Zero gradients at the start of each accumulation cycle
        if accumulation_count == 0:
            optimizer.zero_grad()
        
        scaled_loss.backward()
        accumulation_count += 1

        # Only update parameters after accumulating gradients
        if accumulation_count % gradient_accumulation_steps == 0:
            total_norm = fsdp_model.clip_grad_norm_(training_args.max_grad_norm)
            optimizer.step()
            scheduler.step()
            fsdp_ema_update(ema_model, fsdp_model, decay=training_args.ema)
            
            # Increment actual training step only after parameter update
            actual_training_step += 1
            
            # Save current accumulated stats for display before resetting
            current_accumulated_samples = accumulated_samples
            current_accumulated_task_counts = accumulated_task_counts.copy()
            
            # Reset accumulation
            accumulated_loss = 0.0
            accumulation_count = 0
            accumulated_samples = 0
            accumulated_task_counts = {}

        # Track task statistics
        current_step_task_counts = {}
        for item in data_indexes:
            dataset_name = item['dataset_name']
            if dataset_name not in current_step_task_counts:
                current_step_task_counts[dataset_name] = 0
            current_step_task_counts[dataset_name] += 1
        
        # Update cumulative statistics
        for dataset_name, count in current_step_task_counts.items():
            if dataset_name not in cumulative_task_counts:
                cumulative_task_counts[dataset_name] = 0
            cumulative_task_counts[dataset_name] += count
        
        # Keep track of recent history (last 100 steps)
        if use_gradient_accumulation_display:
            # For token-based batching, only track after gradient accumulation
            if accumulation_count % gradient_accumulation_steps == 0:
                step_task_history.append(current_step_task_counts.copy())
                if len(step_task_history) > 100:
                    step_task_history.pop(0)
        else:
            # For sample-based batching, track every step
            step_task_history.append(current_step_task_counts.copy())
            if len(step_task_history) > 100:
                step_task_history.pop(0)

        # Log every log_every steps, including step 0
        if use_gradient_accumulation_display:
            # For token-based batching, log only after gradient accumulation
            # Only log when gradient accumulation is complete
            should_log = (actual_training_step % training_args.log_every == 0 and accumulation_count % gradient_accumulation_steps == 0)
        else:
            # For sample-based batching, log every log_every steps
            should_log = (curr_step % training_args.log_every == 0 or curr_step == 0)
        
        if should_log:
            # Measure training speed and log loss first
            torch.cuda.synchronize()
            end_time = time.time()
            # Calculate steps per second based on actual steps processed
            steps_per_sec = training_args.log_every / (end_time - start_time)
            if use_gradient_accumulation_display:
                # For gradient accumulation, show actual training step (0-based)
                display_step = actual_training_step
                message = f"(training_step={display_step:07d}) "
            else:
                message = f"(step={curr_step:07d}) "
            
            # Log losses to both console and TensorBoard based on training configuration
            # Only log losses that are actually computed based on current settings
            
            # CE Loss (Cross-Entropy) - computed when visual_und=True
            if training_args.visual_und and "ce" in loss_dict:
                avg_loss = torch.tensor(loss_dict["ce"].item(), device=device)
                dist.all_reduce(avg_loss, op=dist.ReduceOp.SUM)
                avg_loss = avg_loss.item() / dist.get_world_size()
                message += f"Train Loss CE: {avg_loss:.4f}, "
                
                # Log to TensorBoard (only on rank 0)
                if writer is not None:
                    writer.add_scalar("Loss/CE", avg_loss, display_step if use_gradient_accumulation_display else curr_step)
            
            # MSE Loss (Mean Squared Error) - computed when visual_gen=True
            if training_args.visual_gen and "mse" in loss_dict:
                avg_loss = torch.tensor(loss_dict["mse"].item(), device=device)
                dist.all_reduce(avg_loss, op=dist.ReduceOp.SUM)
                avg_loss = avg_loss.item() / dist.get_world_size()
                message += f"Train Loss MSE: {avg_loss:.4f}, "
                
                # Log to TensorBoard (only on rank 0)
                if writer is not None:
                    writer.add_scalar("Loss/MSE", avg_loss, display_step if use_gradient_accumulation_display else curr_step)
            
            # REPA Loss - computed when use_repa=True and visual_gen=True
            if training_args.use_repa and training_args.visual_gen and "repa_loss" in loss_dict and loss_dict["repa_loss"] is not None:
                avg_loss = torch.tensor(loss_dict["repa_loss"].item(), device=device)
                dist.all_reduce(avg_loss, op=dist.ReduceOp.SUM)
                avg_loss = avg_loss.item() / dist.get_world_size()
                message += f"Train Loss REPA: {avg_loss:.4f}, "
                
                # Log to TensorBoard (only on rank 0)
                if writer is not None:
                    writer.add_scalar("Loss/REPA", avg_loss, display_step if use_gradient_accumulation_display else curr_step)
            
            # Log total loss
            total_loss = loss.item()
            total_loss_tensor = torch.tensor(total_loss, device=device)
            dist.all_reduce(total_loss_tensor, op=dist.ReduceOp.SUM)
            total_loss_avg = total_loss_tensor.item() / dist.get_world_size()
            message += f"Train Loss Total: {total_loss_avg:.4f}, "
            
            # Log total loss to TensorBoard
            if writer is not None:
                writer.add_scalar("Loss/Total", total_loss_avg, display_step if use_gradient_accumulation_display else curr_step)
            
            # Log training speed to TensorBoard
            if writer is not None:
                writer.add_scalar("Training/Steps_Per_Sec", steps_per_sec, display_step if use_gradient_accumulation_display else curr_step)
            
            message += f"Train Steps/Sec: {steps_per_sec:.2f}, "
            logger.info(message)


            start_time = time.time()

        if data_status is None:
            data_status = {}
        for item in data_indexes:
            if item['dataset_name'] not in data_status.keys():
                data_status[item['dataset_name']] = {}
            data_status[item['dataset_name']][item['worker_id']] = item['data_indexes']

        if use_gradient_accumulation_display:
            should_save = actual_training_step > 0 and actual_training_step % training_args.save_every == 0 and accumulation_count % gradient_accumulation_steps == 0
        else:
            should_save = curr_step > 0 and curr_step % training_args.save_every == 0
        
        if should_save:
            if dist.get_rank() == 0:
                gather_list = [None] * dist.get_world_size()
            else:
                gather_list = None
            dist.gather_object(data_status, gather_list, dst=0)

            FSDPCheckpoint.fsdp_save_ckpt(
                ckpt_dir=training_args.checkpoint_dir, 
                train_steps=actual_training_step if use_gradient_accumulation_display else curr_step, 
                model=fsdp_model, 
                ema_model=ema_model, 
                optimizer=optimizer, 
                scheduler=scheduler, 
                logger=logger,
                fsdp_config=fsdp_config,
                data_status=gather_list
            )

    logger.info("Done!")
    
    # Close TensorBoard writer
    if writer is not None:
        writer.close()
        logger.info("TensorBoard writer closed.")
    
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
