# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
This example demonstrates how to use the AutoBridge to perform a round-trip
conversion between a Hugging Face model and a Megatron-LM model on multiple GPUs.

The process is as follows:
1. An AutoBridge is initialized from a pretrained Hugging Face model
    (e.g., "meta-llama/Llama-3.2-1B"). This downloads the model from the Hub and loads it.
2. The bridge's `to_megatron_provider` method is called to get a Megatron-LM compatible model provider.
3. The model provider is configured for multi-GPU execution.
4. The model provider is used to instantiate the Megatron-LM model.
5. The weights of the converted Megatron-LM model are verified against the original
    Hugging Face model.
6. The `save_hf_pretrained` method is used to save the Megatron-LM
    model back into the Hugging Face format. A new directory, named after the
    model, will be created for the converted model files. By default, this
    directory is created in the current working directory, but a different
    parent directory can be specified via the `--output-dir` argument.
7. Optionally, the `save_megatron_model` method can be used to save the model
    in Megatron's native checkpoint format by specifying the `--megatron-save-path` argument.

Usage:
    uv run python examples/conversion/hf_megatron_roundtrip_multi_gpu.py --hf-model-id meta-llama/Llama-3.2-1B

    uv run python examples/conversion/hf_megatron_roundtrip_multi_gpu.py --hf-model-id meta-llama/Llama-3.2-1B \
       --megatron-save-path ./megatron_checkpoint

    uv run python -m torch.distributed.run --nproc_per_node=8 examples/conversion/hf_megatron_roundtrip_multi_gpu.py \
      --hf-model-id Qwen/Qwen3-30B-A3B --tp 1 --pp 8
"""

import argparse
import os
import sys

import torch
from rich.console import Console
from rich.table import Table

from megatron.bridge import AutoBridge
from megatron.bridge.models.decorators import torchrun_main
from megatron.bridge.models.hf_pretrained.utils import is_safe_repo


import os
from typing import List, Optional, Union

import torch
from typing_extensions import TypedDict, Unpack

from megatron.bridge import AutoBridge
from megatron.bridge.models import GPTModelProvider
from megatron.bridge.recipes.utils.dataset_utils import get_blend_fields_from_data_paths
from megatron.bridge.recipes.utils.optimizer_utils import distributed_fused_adam_with_cosine_annealing
from megatron.bridge.recipes.utils.tokenizer_utils import DEFAULT_NULL_TOKENIZER_VOCAB_SIZE
from megatron.bridge.training.comm_overlap import CommOverlapConfig
from megatron.bridge.training.config import (
    CheckpointConfig,
    ConfigContainer,
    DistributedDataParallelConfig,
    GPTDatasetConfig,
    LoggerConfig,
    RNGConfig,
    TokenizerConfig,
    TrainingConfig,
)
from megatron.bridge.training.flex_dispatcher_backend import apply_flex_dispatcher_backend
from megatron.bridge.training.mixed_precision import MixedPrecisionConfig


def set_deepseek_v3_pipeline_model_parallel_layout(
    model_cfg: GPTModelProvider, layout: Optional[Union[str, List[List[str]]]] = None
) -> None:
    """Set the DeepSeek-V3 pipeline model parallel layout."""
    mtp_layers = getattr(model_cfg, "mtp_num_layers", 1) or 0
    last_layer = ["mtp"] * mtp_layers + ["loss"]
    pp_size = model_cfg.pipeline_model_parallel_size or 1
    vp_size = model_cfg.virtual_pipeline_model_parallel_size or 1
    layout_map = {
        (1, 1): None,
        (4, 1): [["embedding"] + ["decoder"] * 16, ["decoder"] * 16, ["decoder"] * 16, ["decoder"] * 13 + last_layer],
        (8, 1): [["embedding"] + ["decoder"] * 8] + [["decoder"] * 8] * 6 + [["decoder"] * 5 + last_layer],
        (4, 2): [["embedding"] + ["decoder"] * 8] + [["decoder"] * 8] * 6 + [["decoder"] * 5 + last_layer],
        (16, 1): [["embedding"] + ["decoder"] * 4] + [["decoder"] * 4] * 14 + [["decoder"] + last_layer],
        (8, 2): [["embedding"] + ["decoder"] * 4] + [["decoder"] * 4] * 14 + [["decoder"] + last_layer],
        (4, 4): [["embedding"] + ["decoder"] * 4] + [["decoder"] * 4] * 14 + [["decoder"] + last_layer],
    }
    if layout is not None:
        model_cfg.pipeline_model_parallel_layout = layout
    elif (pp_size, vp_size) in layout_map:
        model_cfg.pipeline_model_parallel_layout = layout_map[(pp_size, vp_size)]


class DeepSeekV3CommonKwargs(TypedDict, total=False):
    """Typed options accepted by DeepSeek V3 recipe helper functions."""

    # Core identifiers
    hf_path: str
    dir: Optional[str]
    name: str
    # Dataset configuration
    data_paths: Optional[List[str]]
    data_args_path: Optional[str]
    train_data_path: Optional[List[str]]
    valid_data_path: Optional[List[str]]
    test_data_path: Optional[List[str]]
    per_split_data_args_path: Optional[str]
    mock: bool
    # Model configuration
    tensor_model_parallel_size: int
    pipeline_model_parallel_size: int
    pipeline_dtype: Optional[torch.dtype]
    virtual_pipeline_model_parallel_size: Optional[int]
    context_parallel_size: int
    expert_model_parallel_size: int
    sequence_parallel: bool
    use_megatron_fsdp: bool
    check_for_nan_in_grad: bool
    # Recompute configuration
    recompute_granularity: Optional[str]
    recompute_modules: Optional[List[str]]
    recompute_method: Optional[str]
    recompute_num_layers: Optional[int]
    # MTP support
    mtp_num_layers: Optional[int]
    mtp_loss_scaling_factor: Optional[float]
    # Training hyperparameters
    train_iters: int
    global_batch_size: int
    micro_batch_size: int
    seq_length: int
    lr: float
    min_lr: float
    lr_warmup_iters: int
    lr_decay_iters: Optional[int]
    eval_interval: int
    save_interval: int
    use_null_tokenizer: bool
    # Precision / overlap configs
    precision_config: Optional[Union[MixedPrecisionConfig, str]]
    comm_overlap_config: Optional[CommOverlapConfig]
    moe_flex_dispatcher_backend: str
    apply_rope_fusion: bool
    layout: Optional[Union[str, List[List[str]]]]


def deepseek_v3_pretrain_config(**user_kwargs: Unpack[DeepSeekV3CommonKwargs]) -> ConfigContainer:
    """Return a pre-training config for DeepSeek-V3.

    See `_deepseek_v3_common` for the full list of parameters.
    """
    recommended_kwargs: DeepSeekV3CommonKwargs = {
        "hf_path": "deepseek-ai/DeepSeek-V3",
        "tensor_model_parallel_size": 2,
        "pipeline_model_parallel_size": 16,
        "expert_model_parallel_size": 64,
        "pipeline_dtype": torch.bfloat16,
        # Old recipe-compatible defaults passed via wrapper
        "recompute_granularity": "selective",
        "precision_config": MixedPrecisionConfig(
            bf16=True,
            params_dtype=torch.bfloat16,
            pipeline_dtype=torch.bfloat16,
            autocast_enabled=False,
            grad_reduce_in_fp32=False,
        ),
    }
    combined_kwargs: DeepSeekV3CommonKwargs = {**recommended_kwargs, **user_kwargs}
    return _deepseek_v3_common(**combined_kwargs)


def _deepseek_v3_common(
    hf_path: str,
    dir: Optional[str] = None,
    name: str = "default",
    # Dataset configuration
    data_paths: Optional[List[str]] = None,
    data_args_path: Optional[str] = None,
    train_data_path: Optional[List[str]] = None,
    valid_data_path: Optional[List[str]] = None,
    test_data_path: Optional[List[str]] = None,
    per_split_data_args_path: Optional[str] = None,
    mock: bool = False,
    # Model configuration
    tensor_model_parallel_size: int = 2,
    pipeline_model_parallel_size: int = 16,
    pipeline_dtype: Optional[torch.dtype] = torch.bfloat16,
    virtual_pipeline_model_parallel_size: Optional[int] = None,
    context_parallel_size: int = 1,
    expert_model_parallel_size: int = 64,
    sequence_parallel: bool = True,
    use_megatron_fsdp: bool = False,
    check_for_nan_in_grad: bool = True,
    # Recompute configuration
    recompute_granularity: Optional[str] = "selective",
    recompute_modules: Optional[List[str]] = None,
    recompute_method: Optional[str] = None,
    recompute_num_layers: Optional[int] = None,
    # MTP support
    mtp_num_layers: Optional[int] = 1,
    mtp_loss_scaling_factor: Optional[float] = 0.1,
    # Training hyperparameters
    train_iters: int = 1_000_000,
    global_batch_size: int = 4096,
    micro_batch_size: int = 1,
    seq_length: int = 4096,
    lr: float = 3e-4,
    min_lr: float = 3e-5,
    lr_warmup_iters: int = 2000,
    lr_decay_iters: Optional[int] = None,
    eval_interval: int = 2000,
    save_interval: int = 2000,
    use_null_tokenizer: bool = True,
    # Precision recipe
    precision_config: Optional[Union[MixedPrecisionConfig, str]] = None,
    comm_overlap_config: Optional[CommOverlapConfig] = None,
    moe_flex_dispatcher_backend: str = None,
    apply_rope_fusion: bool = False,
    layout: Optional[Union[str, List[List[str]]]] = None,
) -> ConfigContainer:
    """
    Create a pre-training configuration for DeepSeek-V3 models using a given HuggingFace path.
    """
    base_output_dir = dir if dir is not None else os.path.join(os.getcwd(), "nemo_experiments")
    run_output_dir = os.path.join(base_output_dir, name)
    checkpoint_dir = os.path.join(run_output_dir, "checkpoints")
    tensorboard_dir = os.path.join(run_output_dir, "tb_logs")

    blend, blend_per_split, split = get_blend_fields_from_data_paths(
        data_paths, data_args_path, train_data_path, valid_data_path, test_data_path, per_split_data_args_path, mock
    )

    bridge = AutoBridge.from_hf_pretrained(hf_path)
    model_cfg = bridge.to_megatron_provider(load_weights=True)
    model_cfg.tensor_model_parallel_size = tensor_model_parallel_size
    model_cfg.pipeline_model_parallel_size = pipeline_model_parallel_size
    model_cfg.pipeline_dtype = pipeline_dtype
    model_cfg.virtual_pipeline_model_parallel_size = virtual_pipeline_model_parallel_size
    model_cfg.context_parallel_size = context_parallel_size
    model_cfg.expert_model_parallel_size = expert_model_parallel_size
    model_cfg.sequence_parallel = sequence_parallel
    model_cfg.seq_length = seq_length

    model_cfg.expert_tensor_parallel_size = 1
    # MTP configuration (allow None to disable by setting to 0)
    model_cfg.mtp_num_layers = 0 if mtp_num_layers is None else mtp_num_layers
    model_cfg.mtp_loss_scaling_factor = mtp_loss_scaling_factor
    model_cfg.init_method_std = 0.006
    model_cfg.rotary_base = 10000.0
    model_cfg.rotary_scaling_factor = 40
    model_cfg.rotary_base = float(model_cfg.rotary_base)
    model_cfg.rotary_scaling_factor = int(model_cfg.rotary_scaling_factor)

    model_cfg.recompute_granularity = recompute_granularity
    model_cfg.recompute_modules = recompute_modules
    model_cfg.recompute_method = recompute_method
    model_cfg.recompute_num_layers = recompute_num_layers

    set_deepseek_v3_pipeline_model_parallel_layout(model_cfg, layout)

    # Pipeline split for asymmetric stages are specified with map_pp_vp_to_layout below
    model_cfg.account_for_embedding_in_pipeline_split = False
    model_cfg.account_for_loss_in_pipeline_split = False
    model_cfg.num_layers_in_first_pipeline_stage = None
    model_cfg.num_layers_in_last_pipeline_stage = None

    # Performance optimization knobs
    model_cfg.moe_permute_fusion = True
    apply_flex_dispatcher_backend(model_cfg, moe_flex_dispatcher_backend)

    opt_config, scheduler = distributed_fused_adam_with_cosine_annealing(
        lr_warmup_iters=lr_warmup_iters,
        lr_decay_iters=lr_decay_iters,
        adam_beta1=0.9,
        adam_beta2=0.95,
        adam_eps=1e-5,
        weight_decay=0.1,
        max_lr=lr,
        min_lr=min_lr,
    )
    opt_config.use_precision_aware_optimizer = True
    opt_config.main_params_dtype = torch.float32
    opt_config.main_grads_dtype = torch.bfloat16
    opt_config.exp_avg_dtype = torch.bfloat16
    opt_config.exp_avg_sq_dtype = torch.bfloat16

    if precision_config is None:
        precision_config = MixedPrecisionConfig(
            bf16=True,
            params_dtype=torch.bfloat16,
            pipeline_dtype=torch.bfloat16,
            autocast_enabled=False,
            grad_reduce_in_fp32=False,
        )

    cfg = ConfigContainer(
        model=model_cfg,
        train=TrainingConfig(
            train_iters=train_iters,
            eval_interval=eval_interval,
            eval_iters=32,
            global_batch_size=global_batch_size,
            micro_batch_size=micro_batch_size,
            manual_gc=True,
            manual_gc_interval=5,
            manual_gc_eval=5,
        ),
        optimizer=opt_config,
        scheduler=scheduler,
        ddp=DistributedDataParallelConfig(
            check_for_nan_in_grad=check_for_nan_in_grad,
            grad_reduce_in_fp32=False,  # V3 recipe sets this to False
            overlap_grad_reduce=True,
            overlap_param_gather=True,
            average_in_collective=True,
            use_distributed_optimizer=True,
            use_megatron_fsdp=use_megatron_fsdp,  # need use_distributed_optimizer=True
        ),
        dataset=GPTDatasetConfig(
            random_seed=1234,
            reset_attention_mask=False,
            reset_position_ids=False,
            eod_mask_loss=False,
            seq_length=seq_length,
            num_dataset_builder_threads=1,
            blend=blend,
            blend_per_split=blend_per_split,
            split=split,
            data_sharding=True,
            dataloader_type="single",
            num_workers=8,
            skip_getting_attention_mask_from_dataset=True,
        ),
        logger=LoggerConfig(
            log_interval=10,
            tensorboard_dir=tensorboard_dir,
            log_timers_to_tensorboard=True,
        ),
        tokenizer=TokenizerConfig(
            tokenizer_type="NullTokenizer" if use_null_tokenizer else "HuggingFaceTokenizer",
            tokenizer_model=hf_path if not use_null_tokenizer else None,
            vocab_size=DEFAULT_NULL_TOKENIZER_VOCAB_SIZE if use_null_tokenizer else None,
        ),
        checkpoint=CheckpointConfig(
            save_interval=save_interval,
            save=checkpoint_dir,
            load=checkpoint_dir,
            ckpt_format="torch_dist",
            fully_parallel_save=True,
            async_save=False,
        ),
        rng=RNGConfig(seed=1234),
        comm_overlap=comm_overlap_config,
        mixed_precision=precision_config,
    )
    if apply_rope_fusion:
        cfg.dist.enable_megatron_core_experimental = True  # mla rope fusion is experimental

    # Ensure comm_overlap exists with old default tp_comm_overlap=False when not provided
    if cfg.comm_overlap is None:
        cfg.comm_overlap = CommOverlapConfig(tp_comm_overlap=False)

    return cfg, bridge


HF_MODEL_ID = "meta-llama/Llama-3.2-1B"
console = Console()


@torchrun_main
def main(
    hf_model_id: str = HF_MODEL_ID,
    output_dir: str = None,
    tp: int = 1,
    pp: int = 1,
    vp: int | None = None,
    ep: int = 1,
    etp: int = 1,
    megatron_save_path: str | None = None,
    megatron_load_path: str | None = None,
    trust_remote_code: bool | None = None,
    strict: bool = False,
) -> None:
    """Perform round-trip conversion between HuggingFace and Megatron-LM models on multiple GPUs."""
    if os.environ.get("WORLD_SIZE") is None:
        console.print("This script must be launched with torchrun. Please run:")
        console.print(f"torchrun --nproc_per_node <gpus> {sys.argv[0]}")
        sys.exit(1)

    model_name = hf_model_id.split("/")[-1]
    if output_dir:
        save_path = os.path.join(output_dir, model_name)
    else:
        save_path = model_name

    # bridge = AutoBridge.from_hf_pretrained(
    #     hf_model_id,
    #     trust_remote_code=is_safe_repo(
    #         trust_remote_code=trust_remote_code,
    #         hf_path=hf_model_id,
    #     ),
    #     torch_dtype=torch.bfloat16,
    # )

    if megatron_load_path:
        pass
    #     model_provider = bridge.to_megatron_provider(load_weights=False)
    #     model_provider.tensor_model_parallel_size = tp
    #     model_provider.pipeline_model_parallel_size = pp
    #     model_provider.virtual_pipeline_model_parallel_size = vp
    #     model_provider.pipeline_dtype = torch.bfloat16
    #     model_provider.params_dtype = torch.bfloat16
    #     model_provider.expert_model_parallel_size = ep
    #     model_provider.expert_tensor_parallel_size = etp

    #     # Once all overrides are set, finalize the model provider to ensure the post initialization logic is run
    #     model_provider.finalize()
    #     model_provider.initialize_model_parallel(seed=0)
    #     megatron_model = bridge.load_megatron_model(
    #         megatron_load_path,
    #         mp_overrides={
    #             "tensor_model_parallel_size": tp,
    #             "pipeline_model_parallel_size": pp,
    #             "expert_model_parallel_size": ep,
    #             "expert_tensor_parallel_size": etp,
    #             "pipeline_dtype": torch.bfloat16,
    #             "params_dtype": torch.bfloat16,
    #         },
    #         wrap_with_ddp=False,
    #     )
    #     megatron_model = [m.cuda() for m in megatron_model]

    else:
        config, bridge = deepseek_v3_pretrain_config()
        model_provider = config.model
        # model_provider = bridge.to_megatron_provider(load_weights=True)
        model_provider.tensor_model_parallel_size = tp
        model_provider.pipeline_model_parallel_size = pp
        model_provider.virtual_pipeline_model_parallel_size = vp
        model_provider.pipeline_dtype = torch.bfloat16
        model_provider.params_dtype = torch.bfloat16
        model_provider.expert_model_parallel_size = ep
        model_provider.sequence_parallel = (tp > 1)
        # model_provider.expert_tensor_parallel_size = etp
        # model_provider.pipeline_model_parallel_layout = [['embedding', 'decoder', 'decoder', 'decoder', 'decoder'],
        #                                                  ['decoder', 'decoder', 'decoder', 'decoder'],
        #                                                  ['decoder', 'decoder', 'decoder', 'decoder'],
        #                                                  ['decoder', 'decoder', 'decoder', 'decoder'],
        #                                                  ['decoder', 'decoder', 'decoder', 'decoder'],
        #                                                  ['decoder', 'decoder', 'decoder', 'decoder'],
        #                                                  ['decoder', 'decoder', 'decoder', 'decoder'],
        #                                                  ['decoder', 'decoder', 'decoder', 'decoder'],
        #                                                  ['decoder', 'decoder', 'decoder', 'decoder'],
        #                                                  ['decoder', 'decoder', 'decoder', 'decoder'],
        #                                                  ['decoder', 'decoder', 'decoder', 'decoder'],
        #                                                  ['decoder', 'decoder', 'decoder', 'decoder'],
        #                                                  ['decoder', 'decoder', 'decoder', 'decoder'],
        #                                                  ['decoder', 'decoder', 'decoder', 'decoder'],
        #                                                  ['decoder', 'decoder', 'decoder', 'decoder'],
        #                                                  ['decoder', 'mtp', 'loss']]
        # Once all overrides are set, finalize the model provider to ensure the post initialization logic is run
        model_provider.finalize()
        model_provider.initialize_model_parallel(seed=0)
        megatron_model = model_provider.provide_distributed_model(wrap_with_ddp=False)

    # Now we can check for rank
    is_rank_0 = torch.distributed.get_rank() == 0

    # Formatting
    if is_rank_0:
        table = Table(title="Hugging Face Weights Verification")
        table.add_column("Weight Name", style="cyan")
        table.add_column("Shape")
        table.add_column("DType")
        table.add_column("Device")
        table.add_column("Matches Original", justify="center")

    if is_rank_0:
        console.print(f"[yellow]Tensor parallel size: {model_provider.tensor_model_parallel_size}[/yellow]")
        console.print(f"[yellow]Pipeline parallel size: {model_provider.pipeline_model_parallel_size}[/yellow]")
        console.print(f"[yellow]Expert parallel size: {model_provider.expert_model_parallel_size}[/yellow]")
        console.print(f"[yellow]Expert tensor parallel size: {model_provider.expert_tensor_parallel_size}[/yellow]")

    # for name, param in bridge.export_hf_weights(megatron_model, show_progress=False):
    #     if is_rank_0:
    #         original_param = bridge.hf_pretrained.state[name]
    #         match = torch.allclose(
    #             param, original_param.to(dtype=param.dtype, device=param.device), atol=1e-1
    #         )  # Increased tolerance for bfloat16
    #         print(f"Params match for {name}: {match}")
    #         table.add_row(
    #             name,
    #             str(tuple(param.shape)),
    #             str(param.dtype).replace("torch.", ""),
    #             str(param.device),
    #             "✅" if match else "❌",
    #         )

    if is_rank_0:
        console.print(table)
        console.print(f"Saving HF-ckpt in {save_path}...")

    # bridge.save_hf_pretrained(megatron_model, save_path, strict=strict)

    # Save in Megatron format if path is provided
    if megatron_save_path:
        if is_rank_0:
            console.print(f"Saving Megatron checkpoint in {megatron_save_path}...")
        bridge.save_megatron_model(megatron_model, megatron_save_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert between HuggingFace and Megatron-LM model formats on multiple GPUs"
    )
    parser.add_argument("--hf-model-id", type=str, default=HF_MODEL_ID, help="HuggingFace model ID to convert")
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="The directory where the converted model directory will be created. Defaults to the current working directory.",
    )
    parser.add_argument("--tp", type=int, default=1, help="Tensor parallelism size")
    parser.add_argument("--pp", type=int, default=1, help="Pipeline parallelism size")
    parser.add_argument("--vp", type=int, default=None, help="Virtual pipeline parallelism size")
    parser.add_argument("--ep", type=int, default=1, help="Expert parallelism size")
    parser.add_argument("--etp", type=int, default=1, help="Expert tensor parallelism size")

    parser.add_argument(
        "--megatron-save-path",
        type=str,
        default=None,
        help="Path to save the model in Megatron checkpoint format. If not provided, model will not be saved in Megatron format.",
    )
    parser.add_argument(
        "--megatron-load-path",
        type=str,
        default=None,
        help="Path to load the model in Megatron checkpoint format. If provided, model will not start from HF checkpoint.",
    )
    parser.add_argument("--trust-remote-code", action="store_true", help="if trust_remote_code")
    parser.add_argument("--not-strict", action="store_true", help="Perform loose validation during weight export")
    args = parser.parse_args()
    main(
        args.hf_model_id,
        args.output_dir,
        args.tp,
        args.pp,
        args.vp,
        args.ep,
        args.etp,
        args.megatron_save_path,
        args.megatron_load_path,
        args.trust_remote_code,
    )

    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()
