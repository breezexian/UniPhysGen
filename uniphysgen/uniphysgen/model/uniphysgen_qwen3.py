from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from torch import nn

from transformers import AutoConfig, AutoModelForCausalLM, Qwen3ForCausalLM, Qwen3Model
from transformers import CLIPVisionConfig, CLIPVisionModel
from transformers.cache_utils import Cache
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.models.qwen3.configuration_qwen3 import Qwen3Config
from transformers.utils import logging

try:
    import torchsparse
    from torchsparse.utils.collate import sparse_collate
except ImportError:
    torchsparse = None
    sparse_collate = None

from .common_structure import PointBackboneType, ProjectorType, ImageBackboneType

from .heads import MotionHead

logger = logging.get_logger(__name__)

IGNORE_INDEX = -100


class UniPhysGenQwen3Config(Qwen3Config):
    # Transformers registration identifier for UniPhysGenQwen3Config.
    model_type = "uniphysgen_qwen3"

    # Separate part/object boundary ids, populated when assembling the checkpoint.
    part_point_start_token_id: Optional[int] = None
    part_point_end_token_id: Optional[int] = None
    object_point_start_token_id: Optional[int] = None
    object_point_end_token_id: Optional[int] = None
    point_token_id: Optional[int] = None

    # Optional image placeholder token ids (must exist in tokenizer vocab)
    image_start_token_id: Optional[int] = None
    image_end_token_id: Optional[int] = None
    image_token_id: Optional[int] = None

    # Optional image feature settings
    # NOTE: image modality is enabled by `image_backbone` (not by `use_image`).
    image_feature_dim: int = 768
    image_num_tokens: int = 256  # number of image tokens to insert
    # How to turn vision backbone outputs into tokens to be inserted.
    # Choices: ["pool", "cls", "patch", "patch_first", "patch_uniform"]
    # - pool: use pooler_output (fallback to cls if missing), yields 1 token
    # - cls: use CLS embedding (last_hidden_state[:,0]), yields 1 token
    # - patch: use all patch tokens (last_hidden_state excluding cls), yields variable tokens
    # - patch_first: use first K patch tokens (excluding cls), where K=image_num_tokens
    # - patch_uniform: uniformly sample K patch tokens (excluding cls), where K=image_num_tokens
    image_token_strategy: str = "patch_uniform"


@dataclass
class PhysMeshLLMOutput3(CausalLMOutputWithPast):
    """UniPhysGen causal-LM output with optional motion-regression predictions."""

    motion: Optional[Dict[str, torch.Tensor]] = None


class UniPhysGenQwen3ForCausalLM(Qwen3ForCausalLM):

    config_class = UniPhysGenQwen3Config

    def __init__(self, config: UniPhysGenQwen3Config):
        super().__init__(config)

        self.model = Qwen3Model(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        self.task_name = getattr(config, "task_name", "physics")
        self.use_motion_head = getattr(config, "use_motion_head", False)
        self.point_backbone_type = PointBackboneType(config.point_backbone)
        self.point_backbone: Optional[nn.Module] = None
        point_config = config.point_config

        # shallow_features: encoder context pooled (pre-proj) -> motion head
        self.use_shallow_features = getattr(config, "use_shallow_features", True)
        self.shallow_feature_dim = getattr(config, "shallow_feature_dim", 256)

        if self.point_backbone_type == PointBackboneType.SONATA:
            from uniphysgen.model.sonata_encoder import Sonata

            self.point_backbone = Sonata(
                in_channels=point_config["in_channels"],
                order=point_config["order"],
                stride=point_config["stride"],
                enc_depths=point_config["enc_depths"],
                enc_channels=point_config["enc_channels"],
                enc_num_head=point_config["enc_num_head"],
                enc_patch_size=point_config["enc_patch_size"],
                mlp_ratio=point_config["mlp_ratio"],
                mask_token=point_config["mask_token"],
                enc_mode=point_config["enc_mode"],
                enable_fourier_encode=True,
                num_bins=point_config["num_bins"],
            )
            embed_channels = point_config["enc_channels"][-1]
        else:
            raise ValueError(f"Unknown point backbone type: {self.point_backbone_type}")

        # Optional built-in image encoder backbone (e.g. CLIP)
        self.image_backbone_type: Optional[ImageBackboneType] = None
        self.image_backbone: Optional[nn.Module] = None
        if getattr(config, "image_backbone", None) is not None:
            self.image_backbone_type = ImageBackboneType(config.image_backbone)
            if self.image_backbone_type == ImageBackboneType.CLIP:
                # NOTE: when constructing model (esp. training from scratch), do not call
                # `from_pretrained()` here because it triggers weight download/loading.
                # We construct a CLIP vision tower.
                # Prefer an explicit config saved in our exported `config.json` so the
                # architecture matches the intended CLIP variant (B/32, L/14, 336px, etc.).
                img_cfg_dict = getattr(config, "image_backbone_config", None)
                img_name_or_path = getattr(config, "image_backbone_name_or_path", None)

                if isinstance(img_cfg_dict, dict) and len(img_cfg_dict) > 0:
                    self.image_backbone = CLIPVisionModel(CLIPVisionConfig(**img_cfg_dict))
                elif isinstance(img_name_or_path, str) and len(img_name_or_path) > 0:
                    self.image_backbone = CLIPVisionModel(CLIPVisionConfig.from_pretrained(img_name_or_path))
                else:
                    # Fallback: default CLIP vision config.
                    self.image_backbone = CLIPVisionModel(CLIPVisionConfig())
            else:
                raise ValueError(f"Unknown image backbone type: {self.image_backbone_type}")

        # Whether to enable image tokens: decided purely by whether an image backbone is configured.
        self.enable_image = self.image_backbone is not None

        self.projector_type = ProjectorType(getattr(config, "projector", "mlp"))
        if self.projector_type == ProjectorType.LINEAR:
            self.point_proj = nn.Linear(embed_channels, config.hidden_size)
        elif self.projector_type == ProjectorType.MLP:
            self.point_proj = nn.Sequential(
                nn.Linear(embed_channels, embed_channels),
                nn.GELU(),
                nn.Linear(embed_channels, config.hidden_size),
            )
        else:
            raise ValueError(f"Unknown projector type: {self.projector_type}")

        # shallow 融合层：concat(part_shallow, obj_shallow) -> shallow_feature_dim
        if self.use_shallow_features:
            self.shallow_fusion = nn.Sequential(
                nn.Linear(embed_channels * 2, self.shallow_feature_dim),
                nn.GELU(),
            )

        # self.point_start_token_id = self.config.point_start_token_id
        # self.point_end_token_id = self.config.point_end_token_id

        # Optional separate placeholders
        self.part_point_start_token_id = self.config.part_point_start_token_id
        self.part_point_end_token_id = self.config.part_point_end_token_id
        self.object_point_start_token_id = self.config.object_point_start_token_id
        self.object_point_end_token_id = self.config.object_point_end_token_id
        self.point_token_id = self.config.point_token_id

        # image placeholder tokens (optional)
        self.image_start_token_id = self.config.image_start_token_id
        self.image_end_token_id = self.config.image_end_token_id
        self.image_token_id = self.config.image_token_id

        # image projection (only created when image backbone is enabled)
        self.image_num_tokens = int(config.image_num_tokens)
        if self.enable_image:
            # Prefer backbone config hidden_size when available.
            self.image_feature_dim = int(getattr(self.image_backbone.config, "hidden_size", config.image_feature_dim))
            # Choose image projector by (optional) image_projector override, otherwise fall back to `projector`.
            image_projector = getattr(self.config, "image_projector", None)
            image_projector_type = ProjectorType(
                str(image_projector)) if image_projector is not None else self.projector_type

            if image_projector_type == ProjectorType.LINEAR:
                self.image_proj = nn.Linear(self.image_feature_dim, self.config.hidden_size)
            elif image_projector_type == ProjectorType.MLP:
                self.image_proj = nn.Sequential(
                    nn.Linear(self.image_feature_dim, self.image_feature_dim),
                    nn.GELU(),
                    nn.Linear(self.image_feature_dim, self.config.hidden_size),
                )
            else:
                raise ValueError(f"Unknown image projector type: {image_projector_type}")
        else:
            self.image_feature_dim = int(getattr(self.config, "image_feature_dim", 768))
            self.image_proj = None

        self.motion_head = MotionHead(config)

        self.post_init()

    @property
    def embed_tokens(self) -> nn.Module:
        """Compatibility shim.

        Some PEFT/Trainer utilities expect a top-level `model.embed_tokens` or
        `embed_tokens` attribute (common in LLaMA-style models). Our language
        tower lives under `self.model` (Qwen3Model), so we expose the embedding
        layer here.
        """
        return self.model.embed_tokens

    def _sanitize_point_cloud(self, point_cloud: torch.Tensor) -> torch.Tensor:
        nan_mask = torch.isnan(point_cloud).any(dim=1)
        return point_cloud[~nan_mask]

    def forward_point_cloud(
            self,
            point_cloud: torch.Tensor,
            device: torch.device,
            dtype: torch.dtype,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """编码单个点云，同时返回 shallow_features。

        Returns:
            point_tokens: (1, T, H)  # 投影到 LLM hidden 后的 token 序列
            shallow: (1, D) or None  # encoder context 的 pooled 特征（用于 motion head）
        """
        assert self.point_backbone is not None
        self.point_backbone.to(torch.float32)

        point_cloud = self._sanitize_point_cloud(point_cloud)
        coords = point_cloud[:, :3].int()
        feats = point_cloud[:, 3:].float()

        physics_feat = None

        if self.point_backbone_type == PointBackboneType.SONATA:
            input_dict = {
                "coord": feats[:, :3].to(device),
                "grid_coord": coords.to(device),
                "feat": feats.to(device),
                "batch": torch.zeros(coords.shape[0], dtype=torch.long, device=device),
            }
            context = self.point_backbone(input_dict)  # (T, C)

            # context: (T, C)  # SONATA per-point features
            # physics_feat: (1, C)  # geometry-aware global feature
            physics_feat = context.mean(dim=0, keepdim=True).to(dtype)

            return self.point_proj(context.to(dtype)).unsqueeze(0), physics_feat

        raise ValueError(f"Unknown point backbone type: {self.point_backbone_type}")

    def _insert_point_tokens(
            self,
            input_ids: torch.LongTensor,
            inputs_embeds: torch.Tensor,
            attention_mask: torch.Tensor,
            point_tokens: List[torch.Tensor],
            start_token_id: Optional[int] = None,
            end_token_id: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, List[Tuple[int, int, int]]]:
        point_start_end_token_pos: List[Tuple[int, int, int]] = []
        new_input_embeds: List[torch.Tensor] = []
        new_attention_mask: List[torch.Tensor] = []

        start_token_id = self.point_start_token_id if start_token_id is None else start_token_id
        end_token_id = self.point_end_token_id if end_token_id is None else end_token_id

        if len(point_tokens) != input_ids.shape[0]:
            raise ValueError(
                f"len(point_tokens) must equal batch size {input_ids.shape[0]}, got {len(point_tokens)}"
            )

        max_len = 0
        for cur_ids, cur_emb, cur_mask, cur_pt in zip(input_ids, inputs_embeds, attention_mask, point_tokens):
            num_start = (cur_ids == start_token_id).sum().item()
            num_end = (cur_ids == end_token_id).sum().item()
            assert num_start == num_end == 1, "Only support exactly one start/end placeholder per sample"

            start_pos = torch.where(cur_ids == start_token_id)[0][0]
            end_pos = torch.where(cur_ids == end_token_id)[0][0]

            # normalize cur_pt -> (T, H)
            if cur_pt.ndim == 3:
                # (1, T, H)
                if cur_pt.shape[0] != 1:
                    raise ValueError(f"point token tensor must have batch=1 when 3D, got shape {tuple(cur_pt.shape)}")
                cur_pt = cur_pt.squeeze(0)
            if cur_pt.ndim != 2:
                raise ValueError(f"point token tensor must be 2D (T,H), got shape {tuple(cur_pt.shape)}")

            T = cur_pt.shape[0]
            cur_new_emb = torch.cat([cur_emb[: start_pos + 1], cur_pt, cur_emb[end_pos:]], dim=0)
            cur_new_mask = torch.cat(
                [cur_mask[: start_pos + 1], torch.ones(T, device=cur_mask.device), cur_mask[end_pos:]],
                dim=0,
            )

            new_input_embeds.append(cur_new_emb)
            new_attention_mask.append(cur_new_mask)
            point_start_end_token_pos.append((int(start_pos), int(T), int(end_pos)))
            max_len = max(max_len, cur_new_emb.shape[0])

        for i in range(len(new_input_embeds)):
            cur_emb = new_input_embeds[i]
            if cur_emb.shape[0] < max_len:
                last_row = cur_emb[-1]
                pad = last_row.repeat(max_len - cur_emb.shape[0], 1)
                new_input_embeds[i] = torch.cat([cur_emb, pad], dim=0)

            cur_mask = new_attention_mask[i]
            if cur_mask.shape[0] < max_len:
                new_attention_mask[i] = F.pad(cur_mask, (0, max_len - cur_mask.shape[0]), value=0)

        return torch.stack(new_input_embeds, dim=0), torch.stack(new_attention_mask, dim=0), point_start_end_token_pos

    @staticmethod
    def _mask_labels_for_inserted_region(
            labels: torch.Tensor,
            inserted_pos: List[Tuple[int, int, int]],
            max_len: int,
            ignore_index: int,
    ) -> torch.Tensor:
        """Mask labels for inserted token regions.

        inserted_pos: list of (start_pos, T_inserted, end_pos) in the *original* sequence.
        """
        if not inserted_pos:
            return labels
        new_labels: List[torch.Tensor] = []
        for i, (start_pos, T, end_pos) in enumerate(inserted_pos):
            cur_labels = labels[i]
            cur_new = torch.cat(
                [
                    cur_labels[: start_pos + 1],
                    torch.full((T,), ignore_index, device=cur_labels.device),
                    cur_labels[end_pos:],
                ],
                dim=0,
            )
            cur_new = F.pad(cur_new, (0, max_len - cur_new.shape[0]), value=ignore_index)
            new_labels.append(cur_new)
        return torch.stack(new_labels, dim=0)

    def _insert_image_tokens(
            self,
            input_ids: torch.LongTensor,
            inputs_embeds: torch.Tensor,
            attention_mask: torch.Tensor,
            image_tokens: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, List[Tuple[int, int, int]]]:
        """Insert image_tokens between the configured vision start/end tokens.

        image_tokens: (B, Timg, H)
        Returns:
            new_inputs_embeds, new_attention_mask, image_start_end_token_pos
        """
        if self.image_start_token_id is None or self.image_end_token_id is None:
            raise ValueError("image_start_token_id/image_end_token_id must be set in config when use_image=True")

        image_start_end_token_pos: List[Tuple[int, int, int]] = []
        new_input_embeds: List[torch.Tensor] = []
        new_attention_mask: List[torch.Tensor] = []

        max_len = 0
        for cur_ids, cur_emb, cur_mask, cur_img in zip(input_ids, inputs_embeds, attention_mask, image_tokens):
            num_start = (cur_ids == self.image_start_token_id).sum().item()
            num_end = (cur_ids == self.image_end_token_id).sum().item()
            assert num_start == num_end == 1, f"Only support exactly one <image_start>/<image_end> per sample, {num_start}, {num_end}, {self.image_start_token_id}"

            start_pos = torch.where(cur_ids == self.image_start_token_id)[0][0]
            end_pos = torch.where(cur_ids == self.image_end_token_id)[0][0]

            T = cur_img.shape[0]
            cur_new_emb = torch.cat([cur_emb[: start_pos + 1], cur_img, cur_emb[end_pos:]], dim=0)
            cur_new_mask = torch.cat(
                [cur_mask[: start_pos + 1], torch.ones(T, device=cur_mask.device), cur_mask[end_pos:]],
                dim=0,
            )

            new_input_embeds.append(cur_new_emb)
            new_attention_mask.append(cur_new_mask)
            image_start_end_token_pos.append((int(start_pos), int(T), int(end_pos)))
            max_len = max(max_len, cur_new_emb.shape[0])

        for i in range(len(new_input_embeds)):
            cur_emb = new_input_embeds[i]
            if cur_emb.shape[0] < max_len:
                last_row = cur_emb[-1]
                pad = last_row.repeat(max_len - cur_emb.shape[0], 1)
                new_input_embeds[i] = torch.cat([cur_emb, pad], dim=0)

            cur_mask = new_attention_mask[i]
            if cur_mask.shape[0] < max_len:
                new_attention_mask[i] = F.pad(cur_mask, (0, max_len - cur_mask.shape[0]), value=0)

        return torch.stack(new_input_embeds, dim=0), torch.stack(new_attention_mask, dim=0), image_start_end_token_pos

    def forward_image(
            self,
            images: torch.Tensor,
            dtype: torch.dtype,
    ) -> torch.Tensor:
        """Encode images and return projected image tokens for insertion.

        Args:
            images: tensor (B, 3, H, W) pixel_values for CLIP.
            dtype: target dtype (typically inputs_embeds.dtype).

        Returns:
            img_tokens: (B, Timg, H)
        """
        if not self.enable_image or self.image_backbone is None or self.image_proj is None:
            raise ValueError("Image backbone is not enabled, but images were provided.")

        # Only CLIP is supported currently.
        if self.image_backbone_type != ImageBackboneType.CLIP:
            raise ValueError(f"Unsupported image backbone type: {self.image_backbone_type}")

        strategy = str(getattr(self.config, "image_token_strategy", "patch_first")).lower()

        with torch.no_grad():
            out = self.image_backbone(pixel_values=images)
            last_hidden = out.last_hidden_state  # (B, 1+Npatch, D)

            if strategy == "pool":
                if getattr(out, "pooler_output", None) is not None:
                    feats = out.pooler_output.unsqueeze(1)  # (B, 1, D)
                else:
                    feats = last_hidden[:, 0:1, :]  # (B, 1, D)

            elif strategy == "cls":
                feats = last_hidden[:, 0:1, :]  # (B, 1, D)

            elif strategy == "patch":
                # all patch tokens (exclude CLS)
                feats = last_hidden[:, 1:, :]  # (B, Npatch, D)

            elif strategy == "patch_first":
                # first K patch tokens (exclude CLS)
                k = int(getattr(self.config, "image_num_tokens", 1))
                if k <= 0:
                    raise ValueError("config.image_num_tokens must be > 0 when image_token_strategy='patch_first'")
                feats = last_hidden[:, 1: 1 + k, :]  # (B, k, D)

            elif strategy == "patch_uniform":
                # uniformly sample K patch tokens (exclude CLS)
                k = int(getattr(self.config, "image_num_tokens", 1))
                if k <= 0:
                    raise ValueError("config.image_num_tokens must be > 0 when image_token_strategy='patch_uniform'")
                patches = last_hidden[:, 1:, :]  # (B, Npatch, D)
                n = patches.shape[1]
                if k >= n:
                    feats = patches
                else:
                    # deterministic uniform indices: round(linspace(0, n-1, k))
                    idx = torch.linspace(0, n - 1, steps=k, device=patches.device)
                    idx = idx.round().to(torch.long)
                    feats = patches.index_select(dim=1, index=idx)  # (B, k, D)

            else:
                raise ValueError(
                    f"Unknown image_token_strategy={strategy}. Choices: ['pool','cls','patch','patch_first','patch_uniform']"
                )

        if feats.shape[-1] != self.image_feature_dim:
            raise ValueError(
                f"image features last dim {feats.shape[-1]} != image_feature_dim {self.image_feature_dim}"
            )

        return self.image_proj(feats.to(dtype))

    def set_point_backbone_dtype(self, dtype: torch.dtype):
        for param in self.point_backbone.parameters():
            param.data = param.data.to(dtype)

    def set_motion_head_dtype(self, dtype: torch.dtype):
        for param in self.motion_head.parameters():
            param.data = param.data.to(dtype)

    def set_image_backbone_dtype(self, dtype: torch.dtype):
        for param in self.image_backbone.parameters():
            param.data = param.data.to(dtype)

    def get_model(self):
        return self.model

    def forward(
            self,
            input_ids: torch.LongTensor = None,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_values: Optional[Union[Cache, List[torch.FloatTensor]]] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            labels: Optional[torch.LongTensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
            cache_position: Optional[torch.LongTensor] = None,
            num_logits_to_keep: int = 0,
            part_point_clouds: Optional[torch.Tensor] = None,
            object_point_clouds: Optional[torch.Tensor] = None,
            # image inputs: provide raw images only; image tokens are produced by `forward_image`.
            images: Optional[torch.Tensor] = None,
            motion_labels: Optional[Dict[str, torch.Tensor]] = None,
            motion_types: Optional[torch.Tensor] = None,
                centroids: Optional[torch.Tensor] = None,
            **loss_kwargs,
    ) -> Union[Tuple, PhysMeshLLMOutput3]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)

        if inputs_embeds is None:
            inputs_embeds = self.model.embed_tokens(input_ids)

        part_point_start_end_token_pos: List[Tuple[int, int, int]] = []
        object_point_start_end_token_pos: List[Tuple[int, int, int]] = []
        image_start_end_token_pos: List[Tuple[int, int, int]] = []
        shallow_features: Optional[torch.Tensor] = None
        is_object_level_task = self.task_name in {"object_level"}

        # 1) Insert image tokens first (so point insertion sees updated sequence)
        if images is not None and (input_ids.shape[1] != 1 or self.training):
            # print("......lililili None==================================")
            img_tokens = self.forward_image(images=images, dtype=inputs_embeds.dtype)
            inputs_embeds, attention_mask, image_start_end_token_pos = self._insert_image_tokens(
                input_ids=input_ids,
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                image_tokens=img_tokens,
            )

        if (
                self.point_backbone is not None
                and (input_ids.shape[1] != 1 or self.training)
                # and part_point_clouds is not None
                and object_point_clouds is not None
        ):
            """
            在很多生成式推理（generate）里，首步会喂一整段 prompt（长度通常 > 1），后续每一步为了加速会变成 一次只喂 1 个新 token（长度 = 1），依赖 KV cache。
            这条判断的目的通常是：推理阶段当 seq_len==1（增量解码步）时，不再重复做点云编码/融合，因为点云特征在首步已经算过并缓存/注入过了；重复算既慢又可能导致重复注入。
            但在 训练阶段（self.training == True），即使某些情况下 seq_len 可能等于 1，也仍然允许走点云分支（训练逻辑可能需要每步都参与梯度/对齐）。
            """

            # B = part_point_clouds.shape[0]
            B = object_point_clouds.shape[0]
            part_tokens_list: List[torch.Tensor] = []
            obj_tokens_list: List[torch.Tensor] = []
            shallow_list: List[torch.Tensor] = []

            for i in range(B):
                part_shallow = None
                if not is_object_level_task and part_point_clouds is not None:
                    part_tok, part_shallow = self.forward_point_cloud(
                        part_point_clouds[i],
                        device=inputs_embeds.device,
                        dtype=inputs_embeds.dtype,
                    )
                    part_tokens_list.append(part_tok)
                obj_tok, obj_shallow = self.forward_point_cloud(
                    object_point_clouds[i],
                    device=inputs_embeds.device,
                    dtype=inputs_embeds.dtype,
                )
                obj_tokens_list.append(obj_tok)

                if part_shallow is not None and obj_shallow is not None:
                    shallow_list.append(self.shallow_fusion(torch.cat([part_shallow, obj_shallow], dim=-1)))

            if len(shallow_list) == B:
                shallow_features = torch.cat(shallow_list, dim=0)
            if not is_object_level_task:
                # Insert part tokens into <part_point_start>..<part_point_end>
                inputs_embeds, attention_mask, part_point_start_end_token_pos = self._insert_point_tokens(
                    input_ids=input_ids,
                    inputs_embeds=inputs_embeds,
                    attention_mask=attention_mask,
                    point_tokens=part_tokens_list,
                    start_token_id=self.part_point_start_token_id,
                    end_token_id=self.part_point_end_token_id,
                )

            # Insert object tokens into <object_point_start>..<object_point_end>
            inputs_embeds, attention_mask, object_point_start_end_token_pos = self._insert_point_tokens(
                input_ids=input_ids,
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                point_tokens=obj_tokens_list,
                start_token_id=self.object_point_start_token_id,
                end_token_id=self.object_point_end_token_id,
            )

        outputs = self.model(
            input_ids=None,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
        )

        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states[:, -num_logits_to_keep:, :])

        loss = None
        motion_out: Optional[Dict[str, torch.Tensor]] = None

        # 2) Optional MotionHead regression for the motion task.
        if self.task_name in {"motion"} and self.use_motion_head:
            motion_attention_mask: Optional[torch.Tensor] = None
            # Motion-head pooling mask (prompt-only) to avoid label leakage.
            #
            # In SFT training, `labels` typically uses IGNORE_INDEX (-100) on the prompt
            # tokens and real token ids on the response tokens. If motion head pools over
            # the full sequence (including response/GT JSON), it can "see the answer" and
            # overfit, yielding low train loss but poor inference.
            #
            # We therefore build a separate mask for motion pooling that keeps ONLY the
            # prompt portion (labels == IGNORE_INDEX) and also respects padding.
            if labels is not None:
                # IMPORTANT: `attention_mask` has already been expanded by point/image token insertion,
                # but `labels` is still in the *original* token space.
                # Align labels to the post-insertion length before building the prompt-only mask.
                max_len = attention_mask.shape[1]
                aligned_labels = labels
                aligned_labels = self._mask_labels_for_inserted_region(
                    labels=aligned_labels,
                    inserted_pos=part_point_start_end_token_pos,
                    max_len=max_len,
                    ignore_index=IGNORE_INDEX,
                )
                aligned_labels = self._mask_labels_for_inserted_region(
                    labels=aligned_labels,
                    inserted_pos=object_point_start_end_token_pos,
                    max_len=max_len,
                    ignore_index=IGNORE_INDEX,
                )
                if image_start_end_token_pos:
                    aligned_labels = self._mask_labels_for_inserted_region(
                        labels=aligned_labels,
                        inserted_pos=image_start_end_token_pos,
                        max_len=max_len,
                        ignore_index=IGNORE_INDEX,
                    )

                prompt_mask = (aligned_labels == IGNORE_INDEX)
                motion_attention_mask = (attention_mask.to(torch.bool) & prompt_mask).to(attention_mask.dtype)

            if motion_attention_mask is None:
                if self.training:
                    raise RuntimeError(
                        "motion_attention_mask is None. For motion-head training, pass `labels` with prompt tokens masked "
                        f"as IGNORE_INDEX={IGNORE_INDEX} so we can build a prompt-only pooling mask and avoid label leakage."
                    )
                # In inference, we typically don't have `labels`, so we fall back to the normal
                # attention mask (no leakage risk because there's no ground-truth response tokens in input).
                motion_attention_mask = attention_mask

            motion_out = self.motion_head(
                hidden_states=hidden_states,
                attention_mask=motion_attention_mask,
                shallow_features=shallow_features,
            )
            if motion_labels is not None:
                # Support dataloader-friendly packed tensor motion labels.
                # packed layout: [axis(3), pos(3), range(2)] -> (B, 8)
                if isinstance(motion_labels, torch.Tensor):
                    if motion_labels.ndim != 2 or motion_labels.shape[1] != 8:
                        raise ValueError(
                            f"motion_labels tensor must have shape (B,8), got {tuple(motion_labels.shape)}"
                        )
                    motion_labels = {
                        "axis": motion_labels[:, 0:3],
                        "pos": motion_labels[:, 3:6],
                        "range": motion_labels[:, 6:8],
                    }
                loss_dict = self.motion_head.compute_loss(
                    motion_out, motion_labels, motion_types=motion_types, centroids=centroids
                )
                loss = loss_dict["loss"]

        # Without a regression loss, all tasks use Qwen3 logits and causal-LM loss.
        # Task-specific prompts and labels define the generated targets.

        if labels is not None and loss is None:
            max_len = logits.shape[1]
            labels = self._mask_labels_for_inserted_region(
                labels=labels,
                inserted_pos=part_point_start_end_token_pos,
                max_len=max_len,
                ignore_index=IGNORE_INDEX,
            )
            labels = self._mask_labels_for_inserted_region(
                labels=labels,
                inserted_pos=object_point_start_end_token_pos,
                max_len=max_len,
                ignore_index=IGNORE_INDEX,
            )

            # also ignore image token region in labels
            if image_start_end_token_pos:
                labels = self._mask_labels_for_inserted_region(
                    labels=labels,
                    inserted_pos=image_start_end_token_pos,
                    max_len=max_len,
                    ignore_index=IGNORE_INDEX,
                )

            loss = self.loss_function(
                logits=logits,
                labels=labels,
                vocab_size=self.config.vocab_size,
                **loss_kwargs,
            )

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return PhysMeshLLMOutput3(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            motion=motion_out,
        )

    def prepare_inputs_for_generation(
            self,
            input_ids,
            past_key_values=None,
            attention_mask=None,
            inputs_embeds=None,
            **kwargs,
    ):
        if past_key_values:
            input_ids = input_ids[:, -1:]

        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs.update(
            {
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "attention_mask": attention_mask,
                "part_point_clouds": kwargs.get("part_point_clouds"),
                "object_point_clouds": kwargs.get("object_point_clouds"),
                "images": kwargs.get("images"),
                "centroids": kwargs.get("centroids"),
                # "task_name": kwargs.get("task_name"),
                # "motion_types": kwargs.get("motion_types"),
            }
        )
        return model_inputs


# Register model with transformers
AutoConfig.register("uniphysgen_qwen3", UniPhysGenQwen3Config)
AutoModelForCausalLM.register(UniPhysGenQwen3Config, UniPhysGenQwen3ForCausalLM)
