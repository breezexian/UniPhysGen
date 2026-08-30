# Copyright (c) PhysLLM Authors.
# All rights reserved.

"""
Motion Loss for motion parameter regression.

This module provides the loss function for the motion head,
handling axis direction/position ambiguities and range flip consistency.
"""

from typing import Dict
import torch
import torch.nn as nn
import torch.nn.functional as F


class MotionLoss(nn.Module):
    """Loss function for motion parameter regression.
    
    Handles the ambiguities in axis representation:
    - Direction sign ambiguity: d and -d represent the same axis
    - Position ambiguity: any point on the axis is valid
    - Range flip consistency: when direction flips, range should flip too
    
    Args:
        direction_weight: Weight for direction loss.
        position_weight: Weight for position loss.
        range_weight: Weight for range loss.
        use_axis_distance: If True, use axis-to-axis distance for position loss.
            Otherwise, use simple L1 loss.
    """

    def __init__(
            self,
            direction_weight: float = 1.0,
            position_weight: float = 1.0,
            range_weight: float = 1.0,
            use_axis_distance: bool = True,
            use_soft_min: bool = False,
            soft_min_tau: float = 0.1,
            loss_type: str = "complex",
    ):
        super().__init__()
        self.direction_weight = direction_weight
        self.position_weight = position_weight
        self.range_weight = range_weight
        self.use_axis_distance = use_axis_distance
        self.use_soft_min = use_soft_min
        self.soft_min_tau = soft_min_tau
        self.loss_type = loss_type

    def forward(
            self,
            pred: Dict[str, torch.Tensor],
            target: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """Compute motion loss.
        
        Args:
            pred: Dictionary containing predicted:
                - axis_direction: (B, 3) unit vector
                - axis_position: (B, 3) point on axis
                - motion_range: (B, 2) (min, max)
            target: Dictionary containing ground truth with same keys.
        
        Returns:
            Dictionary containing:
                - loss: Total loss
                - direction_loss: Direction component
                - position_loss: Position component
                - range_loss: Range component
        """
        pred_dir = pred["axis_direction"]
        pred_pos = pred["axis_position"]
        pred_range = pred["motion_range"]

        gt_dir = target["axis_direction"]
        gt_pos = target["axis_position"]
        gt_range = target["motion_range"]

        # Simple sanity-check loss: plain MSE on raw targets.
        # Useful to quickly verify the model can learn at all.
        if self.loss_type.lower() in {"mse", "simple", "baseline_mse"}:
            direction_loss = F.mse_loss(pred_dir, gt_dir)
            position_loss = F.mse_loss(pred_pos, gt_pos)
            range_loss = F.mse_loss(pred_range, gt_range)
            total_loss = (
                    self.direction_weight * direction_loss
                    + self.position_weight * position_loss
                    + self.range_weight * range_loss
            )
            return {
                "loss": total_loss,
                "direction_loss": direction_loss,
                "position_loss": position_loss,
                "range_loss": range_loss,
            }

        # # Normalize directions for numerical stability and to avoid magnitude
        # # affecting axis-related losses.
        # pred_dir = F.normalize(pred_dir, p=2, dim=-1, eps=1e-8)
        # gt_dir = F.normalize(gt_dir, p=2, dim=-1, eps=1e-8)

        # Joint flip handling (paired axis+range):
        # (gt_dir, gt_range)  <->  (-gt_dir, [-max, -min])
        # We compute both paired losses and take the per-sample minimum.

        # Case A: normal GT
        dir_loss_a = (1 - F.cosine_similarity(pred_dir, gt_dir, dim=-1))  # (B,)
        if self.use_axis_distance:
            pos_loss_a = self._axis_distance_loss_per_sample(
                pred_pos, pred_dir, gt_pos, gt_dir
            )  # (B,)
        else:
            pos_loss_a = F.smooth_l1_loss(pred_pos, gt_pos, reduction="none").mean(dim=-1)
        range_loss_a = F.smooth_l1_loss(pred_range, gt_range, reduction="none").mean(dim=-1)

        total_a = (
                self.direction_weight * dir_loss_a +
                self.position_weight * pos_loss_a +
                self.range_weight * range_loss_a
        )  # (B,)

        # Case B: flipped GT (paired)
        gt_dir_b = -gt_dir
        gt_range_b = -gt_range[:, [1, 0]]
        dir_loss_b = (1 - F.cosine_similarity(pred_dir, gt_dir_b, dim=-1))  # (B,)
        if self.use_axis_distance:
            pos_loss_b = self._axis_distance_loss_per_sample(
                pred_pos, pred_dir, gt_pos, gt_dir_b
            )  # (B,)
        else:
            pos_loss_b = F.smooth_l1_loss(pred_pos, gt_pos, reduction="none").mean(dim=-1)
        range_loss_b = F.smooth_l1_loss(pred_range, gt_range_b, reduction="none").mean(dim=-1)

        total_b = (
                self.direction_weight * dir_loss_b +
                self.position_weight * pos_loss_b +
                self.range_weight * range_loss_b
        )  # (B,)

        # Aggregate the two equivalent pairings:
        # - hard min: per-sample choose the smaller one
        # - soft min: smooth approximation (better gradients near ties)
        if self.use_soft_min:
            tau = float(self.soft_min_tau)
            if tau <= 0:
                raise ValueError(f"soft_min_tau must be > 0, got {tau}")

            stacked = torch.stack([total_a, total_b], dim=0)  # (2, B)
            # total = -tau * log( exp(-a/tau) + exp(-b/tau) )
            total_per_sample = -tau * torch.logsumexp(-stacked / tau, dim=0)  # (B,)
            total_loss = total_per_sample.mean()

            # Soft assignment weights for reporting component losses
            # w_i = softmax(-total_i/tau)
            w = torch.softmax(-stacked / tau, dim=0)  # (2, B)
            w_a, w_b = w[0], w[1]
            direction_loss = (w_a * dir_loss_a + w_b * dir_loss_b).mean()
            position_loss = (w_a * pos_loss_a + w_b * pos_loss_b).mean()
            range_loss = (w_a * range_loss_a + w_b * range_loss_b).mean()
        else:
            use_b = (total_b < total_a)
            total_loss = torch.where(use_b, total_b, total_a).mean()
            direction_loss = torch.where(use_b, dir_loss_b, dir_loss_a).mean()
            position_loss = torch.where(use_b, pos_loss_b, pos_loss_a).mean()
            range_loss = torch.where(use_b, range_loss_b, range_loss_a).mean()

        return {
            "loss": total_loss,
            "direction_loss": direction_loss,
            "position_loss": position_loss,
            "range_loss": range_loss,
        }

    def _axis_distance_loss(
            self,
            pred_pos: torch.Tensor,
            pred_dir: torch.Tensor,
            gt_pos: torch.Tensor,
            gt_dir: torch.Tensor,
            eps: float = 1e-8,
    ) -> torch.Tensor:
        """Compute axis-to-axis distance loss.
        
        The distance between two 3D lines (skew or parallel).
        
        For skew lines: d = |w · n| / |n| where n = d1 × d2
        For parallel lines: d = point-to-line distance
        """
        # Normalize directions
        d1 = F.normalize(pred_dir, p=2, dim=-1)
        d2 = F.normalize(gt_dir, p=2, dim=-1)

        # Cross product of directions
        n = torch.cross(d1, d2, dim=-1)
        n_norm = n.norm(dim=-1, keepdim=True)

        # Vector between points on the two lines
        w = pred_pos - gt_pos

        # Check if lines are parallel (cross product is near zero)
        is_parallel = n_norm.squeeze(-1) < eps

        # Skew line distance: |w · n| / |n|
        skew_dist = (w * n).sum(dim=-1).abs() / n_norm.squeeze(-1).clamp(min=eps)

        # Parallel line distance: point to line
        proj_len = (w * d2).sum(dim=-1, keepdim=True)
        perp = w - proj_len * d2
        parallel_dist = perp.norm(dim=-1)

        # Select based on parallelism
        distance = torch.where(is_parallel, parallel_dist, skew_dist)

        return distance.mean()

    def _axis_distance_loss_per_sample(
            self,
            pred_pos: torch.Tensor,
            pred_dir: torch.Tensor,
            gt_pos: torch.Tensor,
            gt_dir: torch.Tensor,
            eps: float = 1e-8,
    ) -> torch.Tensor:
        """Per-sample axis-to-axis distance (returns shape (B,))."""
        d1 = F.normalize(pred_dir, p=2, dim=-1)
        d2 = F.normalize(gt_dir, p=2, dim=-1)

        n = torch.cross(d1, d2, dim=-1)
        n_norm = n.norm(dim=-1, keepdim=True)

        w = pred_pos - gt_pos
        is_parallel = n_norm.squeeze(-1) < eps

        skew_dist = (w * n).sum(dim=-1).abs() / n_norm.squeeze(-1).clamp(min=eps)

        proj_len = (w * d2).sum(dim=-1, keepdim=True)
        perp = w - proj_len * d2
        parallel_dist = perp.norm(dim=-1)

        return torch.where(is_parallel, parallel_dist, skew_dist)


class MotionLossBaseline(nn.Module):
    """Motion loss with sign-invariant directions and optional geometric position loss.
    
    The default loss uses:
    - Absolute cosine similarity for direction (axis sign is ignored)
    - SmoothL1 or geometric distance for position
    - SmoothL1 for range
    Unlike MotionLoss, this class does not jointly flip the axis and range.
    The mse/simple modes instead use MSE for all three components.
    
    Args:
        direction_weight: Weight for direction loss.
        position_weight: Weight for position loss.
        range_weight: Weight for range loss.
    """

    def __init__(
            self,
            direction_weight: float = 1.0,
            position_weight: float = 1.0,
            range_weight: float = 1.0,
            loss_type: str = "smoothl1",
            use_joint_position_loss: bool = False,
            joint_gt_pos_weight: float = 0.0,
            use_axis_distance: bool = False,
    ):
        super().__init__()
        self.direction_weight = direction_weight
        self.position_weight = position_weight
        self.range_weight = range_weight
        self.loss_type = loss_type
        self.use_joint_position_loss = use_joint_position_loss
        self.joint_gt_pos_weight = float(joint_gt_pos_weight)
        self.use_axis_distance = use_axis_distance

    def _axis_distance_loss(
            self,
            pred_pos: torch.Tensor,
            pred_dir: torch.Tensor,
            gt_pos: torch.Tensor,
            gt_dir: torch.Tensor,
            eps: float = 1e-3,  # 放大 eps 避免中间状态梯度消失
    ) -> torch.Tensor:
        """
        Compute axis-to-axis distance with smooth handling of skew & parallel lines.
        """
        # Normalize directions
        d1 = F.normalize(pred_dir, dim=-1, eps=eps)
        d2 = F.normalize(gt_dir, dim=-1, eps=eps)

        # Vector between points
        w = pred_pos - gt_pos

        # Skew distance component
        n = torch.cross(d1, d2, dim=-1)
        n_norm = n.norm(dim=-1)
        skew_dist = (w * n).sum(dim=-1).abs() / (n_norm + eps)

        # Parallel/orthogonal distance component
        orth_dist = torch.cross(w, d2, dim=-1).norm(dim=-1)

        # Soft interpolation (smooth transition)
        alpha = n_norm / (n_norm + eps)
        distance = alpha * skew_dist + (1 - alpha) * orth_dist

        return distance.mean()

    def forward(
            self,
            pred: Dict[str, torch.Tensor],
            target: Dict[str, torch.Tensor],
            *,
            centroid: torch.Tensor | None = None,
    ) -> Dict[str, torch.Tensor]:
        """Compute baseline motion loss."""
        pred_dir = pred["axis_direction"]
        pred_pos = pred["axis_position"]
        pred_range = pred["motion_range"]

        gt_dir = target["axis_direction"]
        gt_pos = target["axis_position"]
        gt_range = target["motion_range"]

        if self.loss_type.lower() in {"mse", "simple"}:
            direction_loss = F.mse_loss(pred_dir, gt_dir)
            position_loss = F.mse_loss(pred_pos, gt_pos)
            range_loss = F.mse_loss(pred_range, gt_range)
            total_loss = (
                    self.direction_weight * direction_loss
                    + self.position_weight * position_loss
                    + self.range_weight * range_loss
            )
            return {
                "loss": total_loss,
                "direction_loss": direction_loss,
                "position_loss": position_loss,
                "range_loss": range_loss,
            }

        # Direction loss: 1 - |cos(theta)|.
        # Use absolute cosine similarity to ignore axis direction sign ambiguity.
        cos_sim = F.cosine_similarity(pred_dir, gt_dir, dim=-1).abs()
        direction_loss = (1 - cos_sim).mean()

        # Position loss: geometric supervision when enabled, otherwise SmoothL1.
        if self.use_joint_position_loss:
            if self.use_axis_distance:
                # 2. Position loss (axis distance)
                position_loss = self._axis_distance_loss(pred_pos, pred_dir, gt_pos, gt_dir)
            else:
                if centroid is None:
                    raise ValueError(
                        "centroid must be provided when use_joint_position_loss=True. "
                        "Pass centroid=... with shape (B, 3)."
                    )
                # Geometric-equivalence supervision:
                # We do NOT require pred_pos == gt_pos.
                # We only require pred_pos to lie on the GT axis line.
                # Axis line is defined by (gt_pos, gt_dir). Distance point->line:
                #   dist = || (pred_pos - gt_pos) x d_hat ||
                d_hat = F.normalize(gt_dir, p=2, dim=-1, eps=1e-8)
                v = pred_pos - gt_pos
                dist = torch.cross(v, d_hat, dim=-1).norm(dim=-1)
                geo_loss = dist.mean()
                # geo_loss = (dist * cos_sim.detach()).mean()
                position_loss = geo_loss
                # print("进来这里了")
            if self.joint_gt_pos_weight > 0:
                # print("进来这里了2")
                position_loss = (
                        position_loss
                        + self.joint_gt_pos_weight * F.smooth_l1_loss(pred_pos, gt_pos)
                )
        else:
            position_loss = F.smooth_l1_loss(pred_pos, gt_pos)

        # Range loss: SmoothL1
        range_loss = F.smooth_l1_loss(pred_range, gt_range)
        # print("loss:", direction_loss, position_loss, F.smooth_l1_loss(pred_pos, gt_pos), self.position_weight, self.range_weight)
        total_loss = (
                self.direction_weight * direction_loss +
                self.position_weight * position_loss +
                self.range_weight * range_loss
        )

        return {
            "loss": total_loss,
            "direction_loss": direction_loss,
            "position_loss": position_loss,
            "range_loss": range_loss,
        }
