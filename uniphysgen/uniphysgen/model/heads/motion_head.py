# Copyright (c) PhysLLM Authors.
# All rights reserved.

"""
Motion Head for motion parameter regression.

This head predicts continuous motion parameters:
- axis_direction: [3] - unit vector for motion axis direction
- axis_position: [3] - a point on the motion axis (canonical coordinates)
- motion_range: [2] - (min, max) motion range
"""

from typing import Dict, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

from ...losses.motion_loss import MotionLoss, MotionLossBaseline


class MotionHead(nn.Module):
    """Motion head with explicit type branches.

    Two branches are used to avoid coupling different scales/constraints:
      - trans (B/prismatic): predicts axis + range(distance)
      - rot   (C/revolute):  predicts axis + pos(pivot) + range(angle)

    The forward returns both branches. Loss is computed with `motion_types` masks.
    """
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.motion_hidden_dim = getattr(config, "motion_head_hidden_dim", 512)
        self.use_shallow_features = getattr(config, "use_shallow_features", True)

        # Pooling strategy for turning token-level hidden states into a single vector.
        # Options:
        #   - "mean" (default): masked mean pooling (legacy behavior)
        #   - "last": last non-padding token hidden state
        #   - "attn": learned attention pooling over tokens
        self.motion_pooling = str(getattr(config, "motion_pooling", "mean")).lower()
        if self.motion_pooling not in {"mean", "last", "attn"}:
            raise ValueError(
                f"Unknown motion_pooling={self.motion_pooling!r}. Expected one of ['mean','last','attn']."
            )
        if self.motion_pooling == "attn":
            self.motion_attn_pool = nn.Sequential(
                nn.Linear(self.hidden_size, self.hidden_size // 2),
                nn.GELU(),
                nn.Linear(self.hidden_size // 2, 1),
            )

        # New fusion path (kept opt-in to avoid breaking pretrain state_dict loading)
        self.use_projector_fusion = bool(getattr(config, "use_projector_fusion", False))
        self.projector_fusion_mode = getattr(config, "projector_fusion_mode", "add")
        self.projector_use_shallow_gate = bool(getattr(config, "projector_use_shallow_gate", True))
        
        # Shallow feature dimension (from point encoder intermediate layers)
        self.shallow_feature_dim = getattr(config, "shallow_feature_dim", 256)

        # -----------------------------
        # Legacy concat fusion (KEEP as-is for checkpoint compatibility)
        # -----------------------------
        # Shared trunk (optional)
        if not self.use_projector_fusion:
            if self.use_shallow_features:
                fusion_input_dim = self.hidden_size + self.shallow_feature_dim
            else:
                fusion_input_dim = self.hidden_size

            self.motion_head_shared = nn.Sequential(
                nn.Linear(fusion_input_dim, self.motion_hidden_dim),
                nn.GELU(),
                nn.LayerNorm(self.motion_hidden_dim),
                nn.Dropout(0.1),
            )

        # -----------------------------
        # New projector fusion (OPT-IN, uses new parameter names)
        # -----------------------------
        if self.use_projector_fusion:
            if self.projector_fusion_mode not in {"add", "concat"}:
                raise ValueError(
                    f"Unknown projector_fusion_mode={self.projector_fusion_mode!r}. Expected 'add' or 'concat'."
                )

            # Projectors to align feature spaces + stabilize scale
            self.motion_llm_proj = nn.Sequential(
                nn.Linear(self.hidden_size, self.motion_hidden_dim),
                nn.LayerNorm(self.motion_hidden_dim),
            )
            if self.use_shallow_features:
                self.motion_shallow_proj = nn.Sequential(
                    nn.Linear(self.shallow_feature_dim, self.motion_hidden_dim),
                    nn.LayerNorm(self.motion_hidden_dim),
                )
                if self.projector_use_shallow_gate and self.projector_fusion_mode == "add":
                    # init near 0 so shallow won't dominate early
                    self.motion_shallow_gate = nn.Parameter(torch.tensor(0.0))

            # Separate trunk for projector fusion to avoid shape mismatch with legacy trunk
            if self.projector_fusion_mode == "concat":
                projector_fusion_in = self.motion_hidden_dim * (2 if self.use_shallow_features else 1)
            else:
                projector_fusion_in = self.motion_hidden_dim

            self.motion_head_shared_projector = nn.Sequential(
                nn.Linear(projector_fusion_in, self.motion_hidden_dim),
                nn.GELU(),
                nn.LayerNorm(self.motion_hidden_dim),
                nn.Dropout(0.1),
            )

        # Translation head (B): axis + range(distance)
        self.motion_head_trans = nn.ModuleDict(
            {
                "axis": nn.Sequential(
                    nn.Linear(self.motion_hidden_dim, self.motion_hidden_dim // 2),
                    nn.GELU(),
                    nn.Linear(self.motion_hidden_dim // 2, 3),
                ),
                "range": nn.Sequential(
                    nn.Linear(self.motion_hidden_dim, self.motion_hidden_dim // 2),
                    nn.GELU(),
                    nn.Linear(self.motion_hidden_dim // 2, 2),
                ),
            }
        )

        # Rotation head (C): axis + pos(pivot) + range(angle)
        self.motion_head_rot = nn.ModuleDict(
            {
                "axis": nn.Sequential(
                    nn.Linear(self.motion_hidden_dim, self.motion_hidden_dim // 2),
                    nn.GELU(),
                    nn.Linear(self.motion_hidden_dim // 2, 3),
                ),
                "pos": nn.Sequential(
                    nn.Linear(self.motion_hidden_dim, self.motion_hidden_dim // 2),
                    nn.GELU(),
                    nn.Linear(self.motion_hidden_dim // 2, 3),
                ),
                "range": nn.Sequential(
                    nn.Linear(self.motion_hidden_dim, self.motion_hidden_dim // 2),
                    nn.GELU(),
                    nn.Linear(self.motion_hidden_dim // 2, 2),
                ),
            }
        )

        # Loss for each branch. For translation, position_weight should be 0.
        # self._loss_trans = MotionLoss(
        #     direction_weight=float(getattr(config, "motion_loss_direction_weight_trans", 1.0)),
        #     position_weight=float(getattr(config, "motion_loss_position_weight_trans", 0.0)),
        #     range_weight=float(getattr(config, "motion_loss_range_weight_trans", 0.0)),
        #     use_axis_distance=bool(getattr(config, "motion_loss_use_axis_distance", False)),
        #     use_soft_min=bool(getattr(config, "motion_loss_use_soft_min", False)),
        #     # loss_type="mse"
        # )
        # self._loss_rot = MotionLoss(
        #     direction_weight=float(getattr(config, "motion_loss_direction_weight_rot", 1.0)),
        #     position_weight=float(getattr(config, "motion_loss_position_weight_rot", 1.0)),
        #     range_weight=float(getattr(config, "motion_loss_range_weight_rot", 0.0)),
        #     use_axis_distance=bool(getattr(config, "motion_loss_use_axis_distance", False)),
        #     use_soft_min=bool(getattr(config, "motion_loss_use_soft_min", False)),
        #     # loss_type="mse"
        # )

        # Loss for each branch. For translation, position_weight should be 0.
        self._loss_trans = MotionLossBaseline(
            direction_weight=float(getattr(config, "motion_loss_direction_weight_trans", 1.0)),
            position_weight=float(getattr(config, "motion_loss_position_weight_trans", 0.0)),
            range_weight=float(getattr(config, "motion_loss_range_weight_trans", 0.0)),
            
        )
        self._loss_rot = MotionLossBaseline(
            direction_weight=float(getattr(config, "motion_loss_direction_weight_rot", 1.0)),
            position_weight=float(getattr(config, "motion_loss_position_weight_rot", 1.0)),
            range_weight=float(getattr(config, "motion_loss_range_weight_rot", 0.0)),
            use_joint_position_loss=bool(getattr(config, "motion_loss_use_joint_position_loss", True)),
            joint_gt_pos_weight=float(getattr(config, "motion_loss_joint_gt_pos_weight", 0.1)),
        )
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights with small values for stable training."""
        def _init(m: nn.Module):
            for layer in m.modules():
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_uniform_(layer.weight, gain=0.1)
                    if layer.bias is not None:
                        nn.init.zeros_(layer.bias)

        # Legacy trunk + branches
        if hasattr(self, "motion_head_shared"):
            _init(self.motion_head_shared)
        _init(self.motion_head_trans)
        _init(self.motion_head_rot)

        # Projector fusion modules (optional)
        if hasattr(self, "motion_llm_proj"):
            _init(self.motion_llm_proj)
        if hasattr(self, "motion_shallow_proj"):
            _init(self.motion_shallow_proj)
        if hasattr(self, "motion_head_shared_projector"):
            _init(self.motion_head_shared_projector)

        if hasattr(self, "motion_attn_pool"):
            _init(self.motion_attn_pool)

        # Optional shallow gate: reset to 0 to avoid shallow dominating early.
        if hasattr(self, "motion_shallow_gate"):
            with torch.no_grad():
                self.motion_shallow_gate.fill_(0.0)
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        shallow_features: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Forward pass through motion head.
        
        Args:
            hidden_states: Hidden states from the language model of shape
                (batch_size, seq_len, hidden_size) or pooled features of
                shape (batch_size, hidden_size).
            shallow_features: Optional shallow features from point encoder
                of shape (batch_size, shallow_feature_dim).
            attention_mask: Optional attention mask for pooling.
        
        Returns:
            Dictionary containing:
                - axis_direction: (batch_size, 3) - normalized direction vector
                - axis_position: (batch_size, 3) - point on axis
                - motion_range: (batch_size, 2) - (min, max) range
        """
        # Pool hidden states if needed
        pooled: torch.Tensor
        if hidden_states.dim() == 3:
            bsz, seqlen, _ = hidden_states.shape
            if self.motion_pooling == "mean":
                if attention_mask is not None:
                    # Masked mean pooling
                    mask = attention_mask.unsqueeze(-1).float()
                    pooled = (hidden_states * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
                else:
                    pooled = hidden_states.mean(dim=1)

            elif self.motion_pooling == "last":
                if attention_mask is None:
                    pooled = hidden_states[:, -1, :]
                else:
                    # last index where mask==1
                    idx = attention_mask.to(torch.long).sum(dim=1) - 1
                    idx = idx.clamp(min=0)
                    pooled = hidden_states[torch.arange(bsz, device=hidden_states.device), idx]

            else:  # "attn"
                # Learned attention pooling.
                # scores: (B, T)
                # Ensure inputs match attention-pooling weights dtype.
                attn_dtype = next(self.motion_attn_pool.parameters()).dtype
                hs = hidden_states if hidden_states.dtype == attn_dtype else hidden_states.to(dtype=attn_dtype)
                scores = self.motion_attn_pool(hs).squeeze(-1)
                if attention_mask is not None:
                    scores = scores.masked_fill(attention_mask == 0, torch.finfo(scores.dtype).min)
                weights = torch.softmax(scores, dim=1).unsqueeze(-1)  # (B, T, 1)
                pooled = (hs * weights).sum(dim=1)

        else:
            # Already pooled features
            pooled = hidden_states
        
        # Fuse with shallow features if available
        if self.use_projector_fusion:
            # Ensure inputs match the projector weights dtype.
            # This prevents RuntimeError: mat1 and mat2 must have the same dtype.
            target_dtype = next(self.motion_llm_proj.parameters()).dtype
            if pooled.dtype != target_dtype:
                pooled = pooled.to(dtype=target_dtype)
            if self.use_shallow_features and shallow_features is not None and shallow_features.dtype != target_dtype:
                shallow_features = shallow_features.to(dtype=target_dtype)

            llm_feat = self.motion_llm_proj(pooled)
            if self.use_shallow_features and shallow_features is not None:
                shallow_feat = self.motion_shallow_proj(shallow_features)
                if self.projector_fusion_mode == "concat":
                    fused = torch.cat([llm_feat, shallow_feat], dim=-1)
                else:
                    if self.projector_use_shallow_gate and hasattr(self, "motion_shallow_gate"):
                        fused = llm_feat + torch.tanh(self.motion_shallow_gate) * shallow_feat
                    else:
                        fused = llm_feat + shallow_feat
            else:
                fused = llm_feat

            features = self.motion_head_shared_projector(fused)
        else:
            # Legacy behavior (concat raw features then shared trunk)
            # Keep dtype aligned with legacy trunk weights.
            target_dtype = next(self.motion_head_shared.parameters()).dtype
            if pooled.dtype != target_dtype:
                pooled = pooled.to(dtype=target_dtype)
            if self.use_shallow_features and shallow_features is not None and shallow_features.dtype != target_dtype:
                shallow_features = shallow_features.to(dtype=target_dtype)

            if self.use_shallow_features and shallow_features is not None:
                fused = torch.cat([pooled, shallow_features], dim=-1)
            else:
                fused = pooled
            features = self.motion_head_shared(fused)

        # Predict both branches
        trans_axis = F.normalize(self.motion_head_trans["axis"](features), p=2, dim=-1)
        trans_range = self.motion_head_trans["range"](features)

        rot_axis = F.normalize(self.motion_head_rot["axis"](features), p=2, dim=-1)
        rot_pos = self.motion_head_rot["pos"](features)
        rot_range = self.motion_head_rot["range"](features)

        return {
            # Keep mm_plugin naming inside branches
            "trans": {"axis": trans_axis, "range": trans_range},
            "rot": {"axis": rot_axis, "pos": rot_pos, "range": rot_range},
        }

    def compute_loss(
        self,
        pred: Dict[str, torch.Tensor],
        motion_labels: Dict[str, torch.Tensor],
        motion_types: Optional[torch.Tensor] = None,
        centroids: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Compute loss with explicit branch separation.

        motion_types: LongTensor[B], 0 -> B(trans), 1 -> C(rot)
        motion_labels: expected keys axis/pos/range (from mm_plugin)
        """
        if motion_types is None:
            raise ValueError("motion_types is required for branched MotionHead.")

        if motion_types.dtype != torch.long:
            motion_types = motion_types.to(torch.long)

        # normalize label keys
        if "axis" not in motion_labels:
            raise ValueError("motion_labels must contain keys: axis/pos/range")

        device = motion_labels["axis"].device
        motion_types = motion_types.to(device=device)

        mask_B = motion_types == 0
        mask_C = motion_types == 1

        total = torch.zeros((), device=device)
        out: Dict[str, torch.Tensor] = {}

        # Translation loss (B): axis + range only
        if mask_B.any():
            pred_B = {
                "axis_direction": pred["trans"]["axis"][mask_B],
                # dummy axis_position to satisfy MotionLoss API; weight=0 so it won't matter
                "axis_position": torch.zeros_like(motion_labels["pos"][mask_B]),
                "motion_range": pred["trans"]["range"][mask_B],
            }
            tgt_B = {
                "axis_direction": motion_labels["axis"][mask_B],
                "axis_position": torch.zeros_like(motion_labels["pos"][mask_B]),
                "motion_range": motion_labels["range"][mask_B],
            }
            loss_B = self._loss_trans(pred_B, tgt_B)
            out["loss_trans"] = loss_B["loss"]
            out["direction_loss_trans"] = loss_B["direction_loss"]
            out["range_loss_trans"] = loss_B["range_loss"]
            total = total + loss_B["loss"]
        else:
            out["loss_trans"] = torch.zeros((), device=device)
            # DDP safety: if this rank has no B samples, ensure trans-branch params
            # still participate in the autograd graph (with zero gradient).
            total = total + pred["trans"]["axis"].sum() * 0.0 + pred["trans"]["range"].sum() * 0.0

        # Rotation loss (C): axis + pos + range
        if mask_C.any():
            pred_C = {
                "axis_direction": pred["rot"]["axis"][mask_C],
                "axis_position": pred["rot"]["pos"][mask_C],
                "motion_range": pred["rot"]["range"][mask_C],
            }
            tgt_C = {
                "axis_direction": motion_labels["axis"][mask_C],
                "axis_position": motion_labels["pos"][mask_C],
                "motion_range": motion_labels["range"][mask_C],
            }
            # centroid is only needed when MotionLossBaseline.use_joint_position_loss=True
            centroid_C = None
            if centroids is not None:
                centroid_C = centroids.to(device=device)[mask_C]
            loss_C = self._loss_rot(pred_C, tgt_C, centroid=centroid_C)
            out["loss_rot"] = loss_C["loss"]
            out["direction_loss_rot"] = loss_C["direction_loss"]
            out["position_loss_rot"] = loss_C["position_loss"]
            out["range_loss_rot"] = loss_C["range_loss"]
            total = total + loss_C["loss"]
        else:
            out["loss_rot"] = torch.zeros((), device=device)
            # DDP safety: if this rank has no C samples, ensure rot-branch params
            # still participate in the autograd graph (with zero gradient).
            total = (
                total
                + pred["rot"]["axis"].sum() * 0.0
                + pred["rot"]["pos"].sum() * 0.0
                + pred["rot"]["range"].sum() * 0.0
            )

        out["loss"] = total
        return out
