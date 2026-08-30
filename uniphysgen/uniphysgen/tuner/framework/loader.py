import os
from typing import TYPE_CHECKING, Any, Optional, TypedDict, Type

import torch
import transformers.dynamic_module_utils
from transformers.dynamic_module_utils import get_relative_imports
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    AutoModelForTextToWaveform,
    AutoModelForVision2Seq,
    AutoProcessor,
    AutoTokenizer,
    GenerationConfig,
)

from . import logging
from .utils import check_version, count_parameters, is_env_enabled
from .patcher import patch_config, patch_model, patch_tokenizer, patch_processor
from .adapter import init_adapter

if TYPE_CHECKING:
    from transformers import (
        PretrainedConfig,
        PreTrainedModel,
        PreTrainedTokenizer,
        ProcessorMixin,
    )

    from ..hparams import DataArguments, ModelArguments, FinetuningArguments

logger = logging.get_logger(__name__)


def export_hf_checkpoint_without_motion_head(
        *,
        src_model_name_or_path: str,
        export_dir: str,
        motion_head_prefix: str = "motion_head.",
        trust_remote_code: bool = True,
        cache_dir: Optional[str] = None,
        revision: str = "main",
        token: Optional[str] = None,
        safe_serialization: bool = True,
) -> str:
    """Export a HF-style checkpoint while filtering out motion head parameters.

    This is useful when you frequently change `MotionHead` but want a stable "base"
    checkpoint for `AutoModelForCausalLM.from_pretrained()`.

    What gets exported:
      - model weights excluding keys starting with `motion_head_prefix`
      - config.json (copied from src)
      - generation_config.json (if exists)
      - tokenizer files (copied via save_pretrained)

    Note:
      - The resulting checkpoint still loads into the same architecture (it will
        simply have missing motion_head weights, which will be randomly initialized).
    """

    os.makedirs(export_dir, exist_ok=True)

    init_kwargs = {
        "trust_remote_code": trust_remote_code,
        "cache_dir": cache_dir,
        "revision": revision,
        "token": token,
    }

    model = AutoModelForCausalLM.from_pretrained(src_model_name_or_path, **init_kwargs)
    sd = model.state_dict()
    filtered_sd = {k: v for k, v in sd.items() if not k.startswith(motion_head_prefix)}

    # Save model weights with filtered state_dict.
    model.save_pretrained(export_dir, safe_serialization=safe_serialization, state_dict=filtered_sd)

    # Copy config + generation config (best-effort) and tokenizer.
    try:
        cfg = AutoConfig.from_pretrained(src_model_name_or_path, **init_kwargs)
        cfg.save_pretrained(export_dir)
    except Exception as e:
        logger.warning_rank0(f"Failed to copy config.json from {src_model_name_or_path}: {e}")

    try:
        gen_cfg = GenerationConfig.from_pretrained(src_model_name_or_path, **init_kwargs)
        gen_cfg.save_pretrained(export_dir)
    except Exception as e:
        logger.warning_rank0(f"Failed to copy generation_config.json from {src_model_name_or_path}: {e}")

    try:
        tok = AutoTokenizer.from_pretrained(src_model_name_or_path, **init_kwargs)
        tok.save_pretrained(export_dir)
    except Exception as e:
        logger.warning_rank0(f"Failed to copy tokenizer from {src_model_name_or_path}: {e}")

    logger.info_rank0(
        f"Exported checkpoint without '{motion_head_prefix}*' params to: {export_dir} (src={src_model_name_or_path})"
    )
    return export_dir


def load_base_checkpoint_with_fresh_motion_head(
        *,
        base_model_name_or_path: str,
        export_dir: Optional[str] = None,
        safe_serialization: bool = True,
        trust_remote_code: bool = True,
        cache_dir: Optional[str] = None,
        revision: str = "main",
        token: Optional[str] = None,
        torch_dtype: Optional[torch.dtype] = None,
        device_map: Optional[Any] = None,
) -> "Any":
    """Load a base checkpoint (exported without motion_head weights) and keep motion_head freshly initialized.

    Intended flow:
      1) You exported a base dir with `export_hf_checkpoint_without_motion_head(...)`.
      2) You changed `MotionHead` code/structure.
      3) Now you want to load the base model via `from_pretrained`, while the
         model's `motion_head` uses its default initialization (no weight loading).

    Notes:
      - Since the base checkpoint does not contain `motion_head.*` keys, HuggingFace
        will not load them, so the head should already be randomly initialized.
      - A fresh model is constructed from config, then only non-motion-head
        weights are loaded. `MotionHead` keeps its constructor initialization.

        If `export_dir` is provided, this function will also export a full HF-style
        checkpoint at `export_dir` (including the freshly initialized motion head).
    """

    init_kwargs = {
        "trust_remote_code": trust_remote_code,
        "cache_dir": cache_dir,
        "revision": revision,
        "token": token,
    }
    if torch_dtype is not None:
        init_kwargs["torch_dtype"] = torch_dtype
    if device_map is not None:
        init_kwargs["device_map"] = device_map

    # Two-step load to be extra defensive:
    #  1) Load weights from checkpoint to a temporary model (HF may run init + load).
    #  2) Rebuild a fresh model from config, then load ONLY non-motion_head weights.
    # This guarantees motion_head params come purely from fresh initialization.
    tmp = AutoModelForCausalLM.from_pretrained(base_model_name_or_path, **init_kwargs)
    sd = tmp.state_dict()
    filtered_sd = {k: v for k, v in sd.items() if not k.startswith("motion_head.")}

    cfg = AutoConfig.from_pretrained(base_model_name_or_path, **init_kwargs)
    cfg.use_projector_fusion = True
    cfg.motion_pooling = "attn"
    model = AutoModelForCausalLM.from_config(cfg, trust_remote_code=trust_remote_code)
    if torch_dtype is not None:
        model.to(dtype=torch_dtype)
    if device_map is not None:
        # Best-effort device placement through model.to().
        try:
            model = model.to(device_map)  # type: ignore[arg-type]
        except Exception:
            pass

    missing, unexpected = model.load_state_dict(filtered_sd, strict=False)
    if unexpected:
        logger.warning_rank0(
            f"Unexpected keys when loading base weights (filtered): {unexpected[:20]}"
            + (" ..." if len(unexpected) > 20 else "")
        )

    # motion_head is expected to be initialized by model post-init.
    # Since we intentionally do not load any motion_head.* weights, it will remain fresh.

    if missing:
        # Expected to include motion_head.* (and possibly other newly added params).
        logger.info_rank0(
            f"Missing keys when loading base weights (expected for fresh head): {missing[:20]}"
            + (" ..." if len(missing) > 20 else "")
        )

    # Optional: export a full checkpoint with the fresh motion head included.
    if export_dir is not None:
        os.makedirs(export_dir, exist_ok=True)
        model.save_pretrained(export_dir, safe_serialization=safe_serialization)

        # Best-effort: ensure config, generation config and tokenizer exist in export.
        try:
            cfg.save_pretrained(export_dir)
        except Exception as e:
            logger.warning_rank0(f"Failed to save config.json to {export_dir}: {e}")

        try:
            gen_cfg = GenerationConfig.from_pretrained(base_model_name_or_path, **init_kwargs)
            gen_cfg.save_pretrained(export_dir)
        except Exception as e:
            logger.warning_rank0(f"Failed to save generation_config.json to {export_dir}: {e}")

        try:
            tok = AutoTokenizer.from_pretrained(base_model_name_or_path, **init_kwargs)
            tok.save_pretrained(export_dir)
        except Exception as e:
            logger.warning_rank0(f"Failed to save tokenizer to {export_dir}: {e}")

        logger.info_rank0(
            f"Exported full checkpoint (with fresh motion_head) to: {export_dir} (base={base_model_name_or_path})"
        )

    return model


def assemble_and_export_qwen_sonata_pretrained(
        *,
        llm_name_or_path: str,
        sonata_state_dict_path: str,
        export_dir: str,
        use_image_backbone: bool = False,
        image_backbone: str = "clip",
        image_backbone_name_or_path: Optional[str] = None,
        trust_remote_code: bool = True,
        cache_dir: Optional[str] = None,
        revision: str = "main",
        token: Optional[str] = None,
        model_class: "PreTrainedModel" = None,
        config_class: "PretrainedConfig" = None,
        point_config: Optional[dict[str, Any]] = None,
        projector: str = "mlp",
        strict_llm: bool = False,
        strict_sonata: bool = True,
        dtype: Optional[torch.dtype] = None,
) -> str:
    """Assemble a UniPhysGen checkpoint (Qwen3 + Sonata + projector) and export it.

    Merge and export pretrained weights once, then fine-tune via
    `AutoModelForCausalLM.from_pretrained(export_dir, trust_remote_code=True)`.

    Inputs
    - llm_name_or_path: a Qwen3 HF repo/path (language-only pretrained).
    - sonata_state_dict_path: path to Sonata pretrained weights (.pt/.pth) containing a state_dict.
    - export_dir: output directory (will be created if missing).

    Notes
    - Pass UniPhysGenQwen3ForCausalLM as `model_class` and
      UniPhysGenQwen3Config as `config_class`.
    - `point_config` must match the Sonata variant you are loading (in_channels/order/...
      and embed dim for projector). If omitted, the caller must ensure the base config
      already includes `point_config`.
    - This function does not modify training-time loader behavior.
    """

    os.makedirs(export_dir, exist_ok=True)

    init_kwargs = {
        "trust_remote_code": trust_remote_code,
        "cache_dir": cache_dir,
        "revision": revision,
        "token": token,
    }

    # 1) Load Qwen3 settings through the supplied UniPhysGenQwen3Config class.
    base_cfg = config_class.from_pretrained(llm_name_or_path, **init_kwargs)

    # Multimodal config fields consumed by UniPhysGenQwen3ForCausalLM.
    if not hasattr(base_cfg, "point_config"):
        base_cfg.point_config = {}
    if point_config is not None:
        base_cfg.point_config.update(point_config)

    # Coordinate resolution used by the Sonata backbone's positional encoding.
    base_cfg.point_config.setdefault("num_bins", 400)

    # Required: which point backbone to build.
    if not hasattr(base_cfg, "point_backbone"):
        base_cfg.point_backbone = "sonata"

    # Optional: image backbone (e.g. CLIP).
    # UniPhysGenQwen3ForCausalLM enables images through `config.image_backbone`.
    if use_image_backbone:
        setattr(base_cfg, "image_backbone", image_backbone)
        # Persist the exact vision tower spec for model __init__.
        # UniPhysGenQwen3ForCausalLM prefers this over a default CLIPVisionConfig().
        if image_backbone_name_or_path is not None:
            setattr(base_cfg, "image_backbone_name_or_path", image_backbone_name_or_path)
            try:
                from transformers import CLIPVisionConfig

                setattr(
                    base_cfg,
                    "image_backbone_config",
                    CLIPVisionConfig.from_pretrained(image_backbone_name_or_path, **init_kwargs).to_dict(),
                )
            except Exception as e:
                logger.warning_rank0(
                    f"Failed to fetch CLIPVisionConfig from {image_backbone_name_or_path}: {e}"
                )

    # Optional: projector type (linear/mlp).
    setattr(base_cfg, "projector", projector)

    # Do NOT force-enable image backbone here.

    # 2) Ensure multimodal special tokens exist and ids are written into config
    # BEFORE building the model, because the model __init__ reads these fields.
    # These tokens are used to delimit and placeholder point cloud segments.
    part_point_start_tok = "<|part_point_start|>"
    part_point_end_tok = "<|part_point_end|>"
    obj_point_start_tok = "<|object_point_start|>"
    obj_point_end_tok = "<|object_point_end|>"
    point_pad_tok = "<|point_pad|>"  # placeholder to be replaced by point tokens if present

    img_start_tok = "<|vision_start|>"
    img_end_tok = "<|vision_end|>"
    img_pad_tok = "<|image_pad|>"

    tok = AutoTokenizer.from_pretrained(llm_name_or_path, **init_kwargs)

    def _token_exists(t, token: str) -> bool:
        tid = t.convert_tokens_to_ids(token)
        if tid is None:
            return False
        # # Many tokenizers return unk_token_id for unknown tokens.
        # unk = getattr(t, "unk_token_id", None)
        add_vocs = t.get_added_vocab()
        return token in add_vocs

    additional = []
    for s in (part_point_start_tok, part_point_end_tok,
              obj_point_start_tok, obj_point_end_tok,
              point_pad_tok,
              img_start_tok, img_end_tok, img_pad_tok):
        if not _token_exists(tok, s):
            additional.append(s)
    if additional:
        tok.add_special_tokens({"additional_special_tokens": additional}, replace_additional_special_tokens=False)

    # Record separate part/object boundaries and image ids for UniPhysGenQwen3ForCausalLM.

    base_cfg.part_point_start_token_id = tok.convert_tokens_to_ids(part_point_start_tok)
    base_cfg.part_point_end_token_id = tok.convert_tokens_to_ids(part_point_end_tok)
    base_cfg.object_point_start_token_id = tok.convert_tokens_to_ids(obj_point_start_tok)
    base_cfg.object_point_end_token_id = tok.convert_tokens_to_ids(obj_point_end_tok)
    # In this codebase `point_token_id` is the placeholder token (<|point_pad|>).
    base_cfg.point_token_id = tok.convert_tokens_to_ids(point_pad_tok)

    base_cfg.image_start_token_id = tok.convert_tokens_to_ids(img_start_tok)
    base_cfg.image_end_token_id = tok.convert_tokens_to_ids(img_end_tok)
    base_cfg.image_token_id = tok.convert_tokens_to_ids(img_pad_tok)

    # 3) Construct the supplied UniPhysGenQwen3ForCausalLM class directly.
    model = model_class(base_cfg)

    # 4) Load Qwen3 pretrained weights into the language tower and lm_head.
    # We intentionally load by submodules to avoid accidental key overlap with multimodal additions.
    llm = AutoModelForCausalLM.from_pretrained(llm_name_or_path, **init_kwargs)

    if not hasattr(model, "model"):
        raise AttributeError("Assembled model has no `model` attribute (language tower).")
    if not hasattr(llm, "model"):
        raise AttributeError("Pretrained LLM has no `model` attribute (language tower).")

    missing_tower, unexpected_tower = model.model.load_state_dict(
        llm.model.state_dict(), strict=strict_llm
    )
    if (missing_tower or unexpected_tower) and strict_llm:
        raise RuntimeError(
            "LLM tower strict load failed. "
            f"missing={missing_tower}, unexpected={unexpected_tower}"
        )

    # Copy the Qwen3ForCausalLM output projection as well as its language tower.
    if hasattr(model, "lm_head") and hasattr(llm, "lm_head"):
        missing_head, unexpected_head = model.lm_head.load_state_dict(
            llm.lm_head.state_dict(), strict=strict_llm
        )
        if (missing_head or unexpected_head) and strict_llm:
            raise RuntimeError(
                "LLM lm_head strict load failed. "
                f"missing={missing_head}, unexpected={unexpected_head}"
            )

    # If vocab expanded, align model embeddings / lm_head AFTER loading pretrained weights.
    # This avoids shape mismatches when the base LLM checkpoint uses the original vocab size.
    if hasattr(model, "resize_token_embeddings"):
        try:
            cur = model.get_input_embeddings().weight.shape[0]
        except Exception:
            cur = None
        if cur is not None and len(tok) > cur:
            model.resize_token_embeddings(len(tok))
        # NOTE: Some models (e.g. Qwen) can have `len(tokenizer) < vocab_size` due to
        # reserved/unused embedding slots. In that case we do nothing (never shrink).

    if getattr(base_cfg, "tie_word_embeddings", False) and hasattr(model, "tie_weights"):
        model.tie_weights()

    # 5) Load Sonata weights into the point backbone.
    if not hasattr(model, "point_backbone") or model.point_backbone is None:
        raise AttributeError(
            "Assembled model has no `point_backbone`. Ensure your model_class builds Sonata."
        )

    ckpt = torch.load(sonata_state_dict_path, map_location="cpu")
    sonata_sd = ckpt.get("state_dict", ckpt)
    if not isinstance(sonata_sd, dict):
        raise ValueError(
            "Sonata checkpoint must be a state_dict dict or have key 'state_dict'."
        )

    # Load the state_dict directly; parameter names must match the point backbone.

    missing_pb, unexpected_pb = model.point_backbone.load_state_dict(
        sonata_sd, strict=strict_sonata
    )

    if (missing_pb or unexpected_pb) and strict_sonata:
        raise RuntimeError(
            f"Sonata strict load failed. missing={missing_pb}, unexpected={unexpected_pb}"
        )

    # 6) Optional: load image backbone pretrained weights.
    if use_image_backbone:
        if not hasattr(model, "image_backbone") or model.image_backbone is None:
            raise AttributeError(
                "Assembled model has no `image_backbone` but use_image_backbone=True. "
                "Ensure your model_class builds the image tower when config.image_backbone is set."
            )

        # Default to CLIP vision tower weights if not specified.
        # This should be a CLIPVisionModel checkpoint (e.g. openai/clip-vit-base-patch32).
        vision_ckpt = image_backbone_name_or_path or "openai/clip-vit-base-patch32"
        from transformers import CLIPVisionModel

        vision = CLIPVisionModel.from_pretrained(vision_ckpt, **init_kwargs)
        missing_v, unexpected_v = model.image_backbone.load_state_dict(
            vision.state_dict(), strict=True
        )
        if missing_v or unexpected_v:
            raise RuntimeError(
                f"Image backbone strict load failed. missing={missing_v}, unexpected={unexpected_v}"
            )

    # 7) Optional dtype cast.
    if dtype is not None:
        model.to(dtype=dtype)

    # 8) Export HF-style checkpoint.
    model.save_pretrained(export_dir, safe_serialization=True)

    # Preserve base LLM generation defaults (sampling params, eos list, etc.).
    # Otherwise transformers may auto-generate a minimal generation_config.json
    # from the model config.
    try:
        gen_cfg = GenerationConfig.from_pretrained(llm_name_or_path, **init_kwargs)
        gen_cfg.save_pretrained(export_dir)
    except Exception as e:
        logger.warning_rank0(f"Failed to copy generation_config.json from base LLM: {e}")

    # Also export the (possibly updated) config so special-token ids are persisted.
    base_cfg.save_pretrained(export_dir)

    # Tokenizer: export tokenizer including our multimodal placeholder tokens.
    tok.save_pretrained(export_dir)

    logger.info_rank0(
        f"Exported assembled multimodal checkpoint to: {export_dir} (llm={llm_name_or_path}, sonata={sonata_state_dict_path})"
    )
    return export_dir


def assemble_and_export_physmeshllm_qwen_image_from_physmeshllm(
        *,
        use_lora: bool = False,
        merged_model_name_or_path: str,
        export_dir: str,
        use_image_backbone: bool = False,
        image_backbone: str = "clip",
        image_backbone_name_or_path: Optional[str] = None,
        trust_remote_code: bool = True,
        cache_dir: Optional[str] = None,
        revision: str = "main",
        token: Optional[str] = None,
        model_class: "PreTrainedModel" = None,
        config_class: "PretrainedConfig" = None,
        projector: str = "mlp",
        strict_merged: bool = False,
        dtype: Optional[torch.dtype] = None,
) -> str:
    os.makedirs(export_dir, exist_ok=True)

    init_kwargs = {
        "trust_remote_code": trust_remote_code,
        "cache_dir": cache_dir,
        "revision": revision,
        "token": token,
    }

    base_cfg = config_class.from_pretrained(merged_model_name_or_path, **init_kwargs)

    # Optional: image backbone (e.g. CLIP).
    # UniPhysGenQwen3ForCausalLM enables images through `config.image_backbone`.
    if use_image_backbone:
        setattr(base_cfg, "image_backbone", image_backbone)
        # Persist the exact vision tower spec for model __init__.
        # UniPhysGenQwen3ForCausalLM prefers this over a default CLIPVisionConfig().
        if image_backbone_name_or_path is not None:
            setattr(base_cfg, "image_backbone_name_or_path", image_backbone_name_or_path)
            try:
                from transformers import CLIPVisionConfig

                setattr(
                    base_cfg,
                    "image_backbone_config",
                    CLIPVisionConfig.from_pretrained(image_backbone_name_or_path, **init_kwargs).to_dict(),
                )
            except Exception as e:
                logger.warning_rank0(
                    f"Failed to fetch CLIPVisionConfig from {image_backbone_name_or_path}: {e}"
                )

    # Optional: projector type (linear/mlp).
    setattr(base_cfg, "projector", projector)
    setattr(base_cfg, "use_lora", use_lora)
    tok = AutoTokenizer.from_pretrained(merged_model_name_or_path, **init_kwargs)

    model = model_class(base_cfg)
    merged = AutoModelForCausalLM.from_pretrained(merged_model_name_or_path, **init_kwargs)

    missing, unexpected = model.load_state_dict(merged.state_dict(), strict=strict_merged)
    if (missing or unexpected) and strict_merged:
        raise RuntimeError(
            f"Merged checkpoint strict load failed. missing={missing}, unexpected={unexpected}"
        )

    if getattr(base_cfg, "tie_word_embeddings", False) and hasattr(model, "tie_weights"):
        model.tie_weights()

    # Optional: load image backbone pretrained weights.
    if use_image_backbone:
        if not hasattr(model, "image_backbone") or model.image_backbone is None:
            raise AttributeError(
                "Assembled model has no `image_backbone` but use_image_backbone=True. "
                "Ensure your model_class builds the image tower when config.image_backbone is set."
            )

        # Default to CLIP vision tower weights if not specified.
        # This should be a CLIPVisionModel checkpoint (e.g. openai/clip-vit-base-patch32).
        vision_ckpt = image_backbone_name_or_path or "openai/clip-vit-base-patch32"
        from transformers import CLIPVisionModel

        vision = CLIPVisionModel.from_pretrained(vision_ckpt, **init_kwargs)
        missing_v, unexpected_v = model.image_backbone.load_state_dict(
            vision.state_dict(), strict=True
        )
        if missing_v or unexpected_v:
            raise RuntimeError(
                f"Image backbone strict load failed. missing={missing_v}, unexpected={unexpected_v}"
            )

    # 5) Optional dtype cast.
    if dtype is not None:
        model.to(dtype=dtype)

    # 6) Export HF-style checkpoint.
    model.save_pretrained(export_dir, safe_serialization=True)

    # Preserve base LLM generation defaults (sampling params, eos list, etc.).
    # Otherwise transformers may auto-generate a minimal generation_config.json
    # from the model config.
    try:
        gen_cfg = GenerationConfig.from_pretrained(merged_model_name_or_path, **init_kwargs)
        gen_cfg.save_pretrained(export_dir)
    except Exception as e:
        logger.warning_rank0(f"Failed to copy generation_config.json from base LLM: {e}")

    # Also export the (possibly updated) config so special-token ids are persisted.
    base_cfg.save_pretrained(export_dir)

    # Tokenizer: export tokenizer including our multimodal placeholder tokens.
    tok.save_pretrained(export_dir)

    logger.info_rank0(
        f"Exported assembled multimodal checkpoint to: {export_dir} (llm={merged_model_name_or_path}, clip={image_backbone_name_or_path})"
    )
    return export_dir


class TokenizerModule(TypedDict):
    tokenizer: "PreTrainedTokenizer"
    processor: Optional["ProcessorMixin"]


def skip_check_imports() -> None:
    r"""Avoid flash attention import error in custom model files."""
    if not is_env_enabled("FORCE_CHECK_IMPORTS"):
        transformers.dynamic_module_utils.check_imports = get_relative_imports


def use_modelscope() -> bool:
    return is_env_enabled("USE_MODELSCOPE_HUB")


def try_download_model_from_other_hub(model_args: "ModelArguments") -> str:
    if not use_modelscope() or os.path.exists(model_args.model_name_or_path):
        return model_args.model_name_or_path

    if use_modelscope():
        check_version("modelscope>=1.11.0", mandatory=True)
        from modelscope import snapshot_download  # type: ignore

        revision = (
            "master"
            if model_args.model_revision == "main"
            else model_args.model_revision
        )
        return snapshot_download(
            model_args.model_name_or_path,
            revision=revision,
            cache_dir=model_args.cache_dir,
        )


def _get_init_kwargs(model_args: "ModelArguments") -> dict[str, Any]:
    r"""Get arguments to load config/tokenizer/model.

    Note: including inplace operation of model_args.
    """
    model_args.model_name_or_path = try_download_model_from_other_hub(model_args)
    return {
        "trust_remote_code": model_args.trust_remote_code,
        "cache_dir": model_args.cache_dir,
        "revision": model_args.model_revision,
        "token": model_args.hf_hub_token,
    }


def load_config(model_args: "ModelArguments") -> "PretrainedConfig":
    r"""Load model config."""
    init_kwargs = _get_init_kwargs(model_args)
    return AutoConfig.from_pretrained(model_args.model_name_or_path, **init_kwargs)


def load_tokenizer(model_args: "ModelArguments") -> "TokenizerModule":
    r"""Load pretrained tokenizer and optionally loads processor.

    Note: including inplace operation of model_args.
    """
    init_kwargs = _get_init_kwargs(model_args)
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            model_args.model_name_or_path,
            use_fast=model_args.use_fast_tokenizer,
            split_special_tokens=model_args.split_special_tokens,
            padding_side="right",
            **init_kwargs,
        )
    except ValueError:  # try the fast one
        tokenizer = AutoTokenizer.from_pretrained(
            model_args.model_name_or_path,
            use_fast=True,
            padding_side="right",
            **init_kwargs,
        )
    except Exception as e:
        raise OSError("Failed to load tokenizer.") from e

    patch_tokenizer(tokenizer, model_args)

    return {"tokenizer": tokenizer}


def register_autoclass(
        config: "PretrainedConfig",
        model: "PreTrainedModel",
        tokenizer: "PreTrainedTokenizer",
):
    if "AutoConfig" in getattr(config, "auto_map", {}):
        config.__class__.register_for_auto_class()
    if "AutoModelForCausalLM" in getattr(config, "auto_map", {}):
        model.__class__.register_for_auto_class()
    if "AutoTokenizer" in tokenizer.init_kwargs.get("auto_map", {}):
        tokenizer.__class__.register_for_auto_class()


def load_model(
        tokenizer: "PreTrainedTokenizer",
        data_args: "DataArguments",
        model_args: "ModelArguments",
        finetuning_args: "FinetuningArguments",
        is_trainable: bool = False,
) -> "PreTrainedModel":
    r"""Load pretrained model."""
    init_kwargs = _get_init_kwargs(model_args)
    config = load_config(model_args)
    config.point_config["num_bins"] = data_args.num_bins
    config.use_projector_fusion = model_args.use_projector_fusion
    config.motion_pooling = model_args.motion_pooling
    # Persist runtime task switches into config so they survive PEFT wrappers and
    # are saved into config.json when exporting checkpoints.
    # Otherwise custom model __init__ may fall back to default (e.g. "physics").
    config.task_name = model_args.task_name
    config.use_motion_head = model_args.use_motion_head
    # config.use_lora = finetuning_args.use_lora
    # if finetuning_args.use_lora:
    #     config.use_lora = finetuning_args.use_lora
    #     config.lora_r = finetuning_args.lora_r
    #     config.lora_alpha = finetuning_args.lora_alpha
    #     config.lora_dropout = finetuning_args.lora_dropout
    #     config.lora_target_modules = finetuning_args.lora_target_modules
    #     config.lora_bias = finetuning_args.lora_bias

    # config.image_backbone = model_args.image_backbone
    # config.point_backbone = model_args.point_backbone
    patch_config(config, model_args, init_kwargs, is_trainable)

    init_kwargs["config"] = config
    init_kwargs["pretrained_model_name_or_path"] = model_args.model_name_or_path

    if model_args.train_from_scratch:
        model = AutoModelForCausalLM.from_config(
            config, trust_remote_code=model_args.trust_remote_code
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(**init_kwargs)
        # model.motion_head._init_weights()
    patch_model(model, model_args, is_trainable)
    register_autoclass(config, model, tokenizer)
    model = init_adapter(model, finetuning_args, is_trainable)
    # Keep attributes on the runtime model as well (some code paths read from model).
    model.task_name = model_args.task_name
    model.use_motion_head = model_args.use_motion_head

    if not is_trainable:
        model.requires_grad_(False)
        for param in model.parameters():
            if (
                    param.data.dtype == torch.float32
                    and model_args.compute_dtype != torch.float32
            ):
                param.data = param.data.to(model_args.compute_dtype)

        model.eval()
    else:
        model.train()

    trainable_params, all_param = count_parameters(model)
    if is_trainable:
        param_stats = (
            f"trainable params: {trainable_params:,} || "
            f"all params: {all_param:,} || trainable%: {100 * trainable_params / all_param:.4f}"
        )
    else:
        param_stats = f"all params: {all_param:,}"

    logger.info_rank0(param_stats)

    if model_args.print_param_status and int(os.getenv("LOCAL_RANK", "0")) == 0:
        for name, param in model.named_parameters():
            print(
                f"name: {name}, dtype: {param.dtype}, device: {param.device}, trainable: {param.requires_grad}"
            )

    return model
