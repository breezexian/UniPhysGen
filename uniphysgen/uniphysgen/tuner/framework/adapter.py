# Copyright 2025 the LlamaFactory team.
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

from typing import TYPE_CHECKING, List, Optional, Set

import torch
import torch.nn as nn

from ..framework import logging

if TYPE_CHECKING:
    from transformers import PreTrainedModel

    from ..hparams import FinetuningArguments

logger = logging.get_logger(__name__)


def get_forbidden_modules(finetuning_args: "FinetuningArguments") -> Set[str]:
    r"""
    Freezes network modules for tuning.
    """
    forbidden_modules = set()
    if finetuning_args.freeze_point_tower:
        forbidden_modules.add("point_backbone")
    if finetuning_args.freeze_language_tower:
        forbidden_modules.update({"model", "lm_head"})
    if finetuning_args.freeze_image_tower:
        forbidden_modules.add("image_backbone")
    if finetuning_args.freeze_image_proj:
        forbidden_modules.add("image_proj")
    if finetuning_args.freeze_point_proj:
        forbidden_modules.add("point_proj")
    if finetuning_args.freeze_motion_head:
        forbidden_modules.update({"motion_head", "shallow_fusion"})
    if finetuning_args.train_point_proj_only:
        forbidden_modules.update(
            {"point_backbone", "model", "lm_head", "image_backbone", "image_proj", "motion_head", "shallow_fusion"})
    if finetuning_args.train_image_proj_only:
        forbidden_modules.update(
            {"point_backbone", "model", "lm_head", "image_backbone", "point_proj", "motion_head", "shallow_fusion"})
    if finetuning_args.train_motion_head_only:
        forbidden_modules.update({"point_backbone", "model", "lm_head", "image_backbone", "point_proj", "image_proj"})

    return forbidden_modules


def _setup_full_tuning(
        model: "PreTrainedModel",
        finetuning_args: "FinetuningArguments",
        is_trainable: bool,
        cast_trainable_params_to_fp32: bool,
) -> None:
    if not is_trainable:
        return

    logger.info_rank0("Fine-tuning method: Full")
    forbidden_modules = get_forbidden_modules(finetuning_args)
    for name, param in model.named_parameters():
        if not any(forbidden_module in name for forbidden_module in forbidden_modules):
            if cast_trainable_params_to_fp32:
                param.data = param.data.to(torch.float32)
        else:
            param.data = param.data.to(torch.float32)
            param.requires_grad_(False)

    # Log trainable parameters summary for safety.
    trainable_names = [n for n, p in model.named_parameters() if p.requires_grad]
    trainable_numel = sum(p.numel() for _, p in model.named_parameters() if p.requires_grad)
    total_numel = sum(p.numel() for _, p in model.named_parameters())
    logger.info_rank0(
        "Trainable params: %d / %d (%.4f%%)",
        trainable_numel,
        total_numel,
        (100.0 * trainable_numel / max(total_numel, 1)),
    )
    if len(trainable_names) == 0:
        logger.warning_rank0("No trainable parameters found after applying freezing rules.")
    else:
        # Print a limited list to avoid flooding logs.
        max_to_print = 40
        preview = trainable_names[:max_to_print]
        logger.info_rank0(
            "Trainable parameter names (first %d/%d): %s",
            len(preview),
            len(trainable_names),
            ", ".join(preview),
        )
        if len(trainable_names) > max_to_print:
            logger.info_rank0(
                "... (%d more trainable parameters not shown)",
                len(trainable_names) - max_to_print,
            )

    # force point_backbone & motion_head to have float32
    model.set_point_backbone_dtype(torch.float32)
    model.set_motion_head_dtype(torch.float32)

    # Optional: force image_backbone to have float32 (e.g., when adding CLIP tower).
    # Only call if the model implements it and the vision tower exists.
    if hasattr(model, "set_image_backbone_dtype") and getattr(model, "image_backbone", None) is not None:
        model.set_image_backbone_dtype(torch.float32)


def _parse_csv(value: Optional[str]) -> Optional[List[str]]:
    if value is None:
        return None
    value = value.strip()
    if not value:
        return None
    return [v.strip() for v in value.split(",") if v.strip()]


def _set_requires_grad_by_module_prefix(
        model: "PreTrainedModel",
        module_prefix: str,
        requires_grad: bool,
        cast_to_fp32: bool,
) -> None:
    """Set requires_grad for all params whose name startswith `module_prefix + '.'` or equals module_prefix."""
    prefix = module_prefix + "."
    for name, param in model.named_parameters():
        if prefix in name:
            param.requires_grad_(requires_grad)
            if cast_to_fp32:
                param.data = param.data.to(torch.float32)


def _setup_lora_tuning(
        model: "PreTrainedModel",
        finetuning_args: "FinetuningArguments",
        is_trainable: bool,
        cast_trainable_params_to_fp32: bool,
) -> "PreTrainedModel":
    """Attach LoRA adapters to language tower while keeping existing freeze rules for other modules."""
    if not is_trainable:
        return model

    try:
        from peft import LoraConfig, TaskType, get_peft_model
    except Exception as e:
        raise RuntimeError(
            "LoRA finetuning requested but `peft` is not available. Please install peft."
        ) from e

    logger.info_rank0("Fine-tuning method: LoRA")

    # 1) Configure which modules receive LoRA.
    # We keep "short" names (q_proj, k_proj, ...) but enforce that LoRA is only
    # applied to the language tower by filtering on module full names.
    user_target_modules = _parse_csv(getattr(finetuning_args, "lora_target_modules", None))
    if user_target_modules is None:
        # target_modules = [
        #     "model.layers.*.self_attn.q_proj",
        #     "model.layers.*.self_attn.k_proj",
        #     "model.layers.*.self_attn.v_proj",
        #     "model.layers.*.self_attn.o_proj",
        #     "model.layers.*.mlp.gate_proj",
        #     "model.layers.*.mlp.up_proj",
        #     "model.layers.*.mlp.down_proj",
        # ]

        user_target_modules = [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ]

    # Build an explicit `target_modules` list by scanning ONLY the language tower.
    # This avoids regex/glob version differences in PEFT and guarantees we target
    # the intended submodules.
    if not hasattr(model, "model"):
        raise ValueError("LoRA mode expects the top-level model to have attribute `model` (language tower).")

    lang_tower = model.model
    target_modules: List[str] = []
    for module_name, _ in lang_tower.named_modules():
        if not module_name:
            continue
        # Match by containment to support patterns like "q_proj" or
        # "self_attn.q_proj" provided by the user.
        if any(pat in module_name for pat in user_target_modules):
            target_modules.append("model." + module_name)

    # De-duplicate while keeping order.
    seen = set()
    target_modules = [m for m in target_modules if not (m in seen or seen.add(m))]

    if len(target_modules) == 0:
        raise ValueError(
            "No target modules found under model.model for LoRA. "
            f"user_target_modules={user_target_modules}. "
            "Please verify the language tower architecture/module names."
        )

    modules_to_save = _parse_csv(getattr(finetuning_args, "lora_modules_to_save", None))

    # 2) Freeze rules: for LoRA, we still want to be able to freeze/enable
    # point/image towers and custom heads similarly to "full" mode.
    # forbidden_modules = get_forbidden_modules(finetuning_args)

    # NOTE: In LoRA mode, freezing the language tower base weights is the default
    # behavior of PEFT. Applying a blanket freeze on `model.*` here can
    # accidentally freeze LoRA adapter weights as well. Therefore we only apply
    # `forbidden_modules` freezing to non-language parts before/after LoRA.

    # 3) Attach LoRA adapters.
    # IMPORTANT: apply LoRA ONLY to the language tower (Qwen backbone) to avoid
    # accidentally matching similarly-named modules in other towers (e.g. CLIP).
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=int(finetuning_args.lora_r),
        lora_alpha=int(finetuning_args.lora_alpha),
        lora_dropout=float(finetuning_args.lora_dropout),
        target_modules=target_modules,
        bias=str(getattr(finetuning_args, "lora_bias", "none")),
        modules_to_save=modules_to_save,
    )

    # 3.0) Apply freezing rules to non-language modules BEFORE LoRA wrapping.
    # We intentionally skip the language tower prefix ("model") here:
    # PEFT will freeze base language weights by default and keep LoRA trainable.
    # pre_lora_forbidden_modules = {m for m in forbidden_modules if m not in {"model"}}
    # for name, param in model.named_parameters():
    #     if any(forbidden_module in name for forbidden_module in pre_lora_forbidden_modules):
    #         param.data = param.data.to(torch.float32)
    #         param.requires_grad_(False)

    # Wrap the *top-level* model for easier save/load/train/infer workflows,
    # but keep LoRA injection restricted by `target_modules`.
    model = get_peft_model(model, lora_config)
    for name, module in model.named_modules():
        if "image_backbone" in name:
            print(name, type(module))
    # Ensure modules_to_save remain trainable (PEFT will wrap them, but their
    # trainability can be affected by external freeze rules).
    # if modules_to_save:
    #     for module_prefix in modules_to_save:
    #         _set_requires_grad_by_module_prefix(
    #             model=model,
    #             module_prefix=str(module_prefix),
    #             requires_grad=True,
    #             cast_to_fp32=cast_trainable_params_to_fp32,
    #         )

    # 4) Keep lm_head freezing if requested.
    # Note: LoRA is applied only to `model.model`, so it will not affect lm_head.
    # if "lm_head" in forbidden_modules:
    #     for name, param in model.named_parameters():
    #         if name.startswith("lm_head"):
    #             param.data = param.data.to(torch.float32)
    #             param.requires_grad_(False)

    # 5) In LoRA mode, we do not implicitly unfreeze other modules.
    # Whether non-language modules (point/image towers, heads) are trainable is
    # controlled by their existing requires_grad state plus the forbidden_modules
    # freezing rules above.

    # 6) Upcast trainable params to fp32 if requested.
    if cast_trainable_params_to_fp32:
        for _, param in model.named_parameters():
            if param.requires_grad:
                param.data = param.data.to(torch.float32)

    # Keep dtype policies consistent for heavy towers.
    if hasattr(model, "set_point_backbone_dtype"):
        model.set_point_backbone_dtype(torch.float32)
    if hasattr(model, "set_motion_head_dtype"):
        model.set_motion_head_dtype(torch.float32)
    if hasattr(model, "set_image_backbone_dtype") and getattr(model, "image_backbone", None) is not None:
        model.set_image_backbone_dtype(torch.float32)

    # Log trainable parameters summary.
    trainable_names = [n for n, p in model.named_parameters() if p.requires_grad]
    trainable_numel = sum(p.numel() for _, p in model.named_parameters() if p.requires_grad)
    total_numel = sum(p.numel() for _, p in model.named_parameters())
    logger.info_rank0(
        "Trainable params: %d / %d (%.4f%%)",
        trainable_numel,
        total_numel,
        (100.0 * trainable_numel / max(total_numel, 1)),
    )
    if len(trainable_names) == 0:
        logger.warning_rank0("No trainable parameters found after applying LoRA/freezing rules.")
    else:
        max_to_print = 40
        preview = trainable_names[:max_to_print]
        logger.info_rank0(
            "Trainable parameter names (first %d/%d): %s",
            len(preview),
            len(trainable_names),
            ", ".join(preview),
        )
        if len(trainable_names) > max_to_print:
            logger.info_rank0(
                "... (%d more trainable parameters not shown)",
                len(trainable_names) - max_to_print,
            )

    return model


def init_adapter(
        model: "PreTrainedModel",
        finetuning_args: "FinetuningArguments",
        is_trainable: bool,
) -> "PreTrainedModel":
    r"""Initialize the adapters.

    Support only full-parameter training for now.

    Note that the trainable parameters must be cast to float32.
    """

    # cast trainable parameters to float32 if:
    # 1. is_trainable and not pure_bf16
    cast_trainable_params_to_fp32 = False
    if not is_trainable:
        pass
    elif finetuning_args.pure_bf16:
        logger.info_rank0(
            "Pure bf16 detected, remaining trainable params in half precision."
        )
    else:
        logger.info_rank0("Upcasting trainable params to float32.")
        cast_trainable_params_to_fp32 = True

    finetuning_type = str(getattr(finetuning_args, "finetuning_type", "full")).lower()
    if finetuning_type == "full":
        _setup_full_tuning(
            model, finetuning_args, is_trainable, cast_trainable_params_to_fp32
        )
        return model
    elif finetuning_type == "lora":
        return _setup_lora_tuning(
            model, finetuning_args, is_trainable, cast_trainable_params_to_fp32
        )
    else:
        raise ValueError(
            f"Unknown finetuning_type={finetuning_type}. Choices: ['full','lora']."
        )
