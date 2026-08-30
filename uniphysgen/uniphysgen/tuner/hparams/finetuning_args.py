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

from dataclasses import asdict, dataclass, field
from typing import Any, Literal, Optional


@dataclass
class SwanLabArguments:
    use_swanlab: bool = field(
        default=False,
        metadata={
            "help": "Whether or not to use the SwanLab (an experiment tracking and visualization tool)."
        },
    )
    swanlab_project: Optional[str] = field(
        default="llamafactory",
        metadata={"help": "The project name in SwanLab."},
    )
    swanlab_workspace: Optional[str] = field(
        default=None,
        metadata={"help": "The workspace name in SwanLab."},
    )
    swanlab_run_name: Optional[str] = field(
        default=None,
        metadata={"help": "The experiment name in SwanLab."},
    )
    swanlab_mode: Literal["cloud", "local"] = field(
        default="cloud",
        metadata={"help": "The mode of SwanLab."},
    )
    swanlab_api_key: Optional[str] = field(
        default=None,
        metadata={"help": "The API key for SwanLab."},
    )
    swanlab_logdir: Optional[str] = field(
        default=None,
        metadata={"help": "The log directory for SwanLab."},
    )
    swanlab_lark_webhook_url: Optional[str] = field(
        default=None,
        metadata={"help": "The Lark(飞书) webhook URL for SwanLab."},
    )
    swanlab_lark_secret: Optional[str] = field(
        default=None,
        metadata={"help": "The Lark(飞书) secret for SwanLab."},
    )


@dataclass
class FinetuningArguments(SwanLabArguments):
    r"""Arguments pertaining to which techniques we are going to fine-tuning with."""

    finetuning_type: Literal["full", "lora"] = field(
        default="full",
        metadata={"help": "Fine-tuning method. Choices: ['full', 'lora']."},
    )

    # LoRA / PEFT settings (used when finetuning_type='lora')
    lora_r: int = field(
        default=16,
        metadata={"help": "LoRA rank (r)."},
    )
    lora_alpha: int = field(
        default=32,
        metadata={"help": "LoRA alpha."},
    )
    lora_dropout: float = field(
        default=0.05,
        metadata={"help": "LoRA dropout."},
    )
    lora_target_modules: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Comma-separated module name patterns to apply LoRA on. "
                "If None, will use a Qwen-style default: q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj."
            )
        },
    )
    lora_bias: Literal["none", "all", "lora_only"] = field(
        default="none",
        metadata={"help": "Bias handling for LoRA. Choices: ['none','all','lora_only']."},
    )
    lora_modules_to_save: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Comma-separated module names to keep trainable and save alongside LoRA adapters. "
                "Useful for e.g. 'lm_head' or custom projectors/heads when desired."
            )
        },
    )


    pure_bf16: bool = field(
        default=False,
        metadata={
            "help": "Whether or not to train model in purely bf16 precision (without AMP)."
        },
    )
    freeze_language_tower: bool = field(
        default=False,
        metadata={"help": "Whether ot not to freeze language tower in training."},
    )
    freeze_point_tower: bool = field(
        default=False,
        metadata={"help": "Whether ot not to freeze point tower in training."},
    )
    freeze_image_tower: bool = field(
        default=False,
        metadata={"help": "Whether ot not to freeze image tower in training."},
    )
    train_point_proj_only: bool = field(
        default=False,
        metadata={"help": "Whether or not to train the projector only."},
    )
    train_image_proj_only: bool = field(
        default=False,
        metadata={"help": "Whether or not to train the projector only."},
    )
    train_motion_head_only: bool = field(
        default=False,
        metadata={"help": "Whether or not to train the motion head only."},
    )
    freeze_point_proj: bool = field(
        default=False,
        metadata={"help": "Whether or not to freeze the point projector."},
    )
    freeze_image_proj: bool = field(
        default=False,
        metadata={"help": "Whether or not to freeze the image projector."},
    )
    freeze_motion_head: bool = field(
        default=False,
        metadata={"help": "Whether or not to freeze the motion head."},
    )

    def to_dict(self) -> dict[str, Any]:
        args = asdict(self)
        args = {
            k: f"<{k.upper()}>" if k.endswith("api_key") else v for k, v in args.items()
        }
        return args
