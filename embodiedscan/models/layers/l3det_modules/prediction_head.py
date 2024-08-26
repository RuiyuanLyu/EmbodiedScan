import numpy as np
import torch.nn as nn
import torch.nn.functional as F


class ThreeLayerMLP(nn.Module):
    """A 3-layer MLP with normalization and dropout."""

    def __init__(self, dim, out_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(dim, dim, 1, bias=False),
            nn.BatchNorm1d(dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Conv1d(dim, dim, 1, bias=False),
            nn.BatchNorm1d(dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Conv1d(dim, out_dim, 1)
        )

    def forward(self, x):
        """Forward pass, x can be (B, dim, N)."""
        return self.net(x)


class ClsAgnosticPredictHead(nn.Module):
    def __init__(self, num_class, num_heading_bin, num_proposal,
                 seed_feat_dim=256, objectness=True, heading=False,
                 compute_sem_scores=True):
        super().__init__()

        self.num_class = num_class
        self.num_heading_bin = num_heading_bin
        self.num_proposal = num_proposal
        self.seed_feat_dim = seed_feat_dim
        self.objectness = objectness
        self.heading = heading
        self.compute_sem_scores = compute_sem_scores

        if objectness:
            self.objectness_scores_head = ThreeLayerMLP(seed_feat_dim, 1)
        self.center_residual_head = ThreeLayerMLP(seed_feat_dim, 3)
        if heading:
            self.heading_class_head = nn.Conv1d(seed_feat_dim, num_heading_bin, 1)
            self.heading_residual_head = nn.Conv1d(seed_feat_dim, num_heading_bin, 1)
        self.size_pred_head = ThreeLayerMLP(seed_feat_dim, 3)
        if compute_sem_scores:
            self.sem_cls_scores_head = ThreeLayerMLP(seed_feat_dim, self.num_class)

    def forward(self, features, base_xyz, end_points, prefix=''):
        """
        Args:
            features: (B,C,num_proposal)
        Returns:
            scores: (B,num_proposal,2+3+NH*2+NS*4)
        """

        batch_size = features.shape[0]
        num_proposal = features.shape[-1]
        net = features

        # objectness
        if self.objectness:
            objectness_scores = self.objectness_scores_head(net).transpose(2, 1)  # (batch_size, num_proposal, 1)
            end_points[f'{prefix}obj_scores'] = objectness_scores.squeeze(-1)

        # center
        center_residual = self.center_residual_head(net).transpose(2, 1)
        center = base_xyz + center_residual  # (batch_size, num_proposal, 3)

        # heading
        if self.heading:
            heading_scores = self.heading_class_head(net).transpose(2, 1)
            # (batch_size, num_proposal, num_heading_bin) (should be -1 to 1)
            heading_residuals_normalized = self.heading_residual_head(net).transpose(2, 1)
            heading_residuals = heading_residuals_normalized * (np.pi / self.num_heading_bin)
            end_points[f'{prefix}heading_scores'] = heading_scores
            end_points[f'{prefix}heading_residuals_normalized'] = heading_residuals_normalized
            end_points[f'{prefix}heading_residuals'] = heading_residuals

        # size
        pred_size = self.size_pred_head(net).transpose(2, 1).view(
            [batch_size, num_proposal, 3])  # (batch_size, num_proposal, 3)

        # class
        if self.compute_sem_scores:
            sem_cls_scores = self.sem_cls_scores_head(features).transpose(2, 1)  # (batch_size, num_proposal, num_class)

        end_points[f'{prefix}base_xyz'] = base_xyz
        end_points[f'{prefix}center'] = center
        end_points[f'{prefix}pred_size'] = pred_size

        if self.compute_sem_scores:
            end_points[f'{prefix}sem_scores'] = sem_cls_scores
        return center, pred_size
