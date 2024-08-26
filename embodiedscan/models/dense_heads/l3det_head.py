# Copyright (c) OpenMMLab. All rights reserved.
import copy
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from mmcv.cnn import ConvModule
from mmcv.ops import PointsSampler as Points_Sampler
from mmcv.ops import gather_points
from mmdet.models.utils import multi_apply
from mmengine.model import BaseModule, xavier_init
from mmengine.structures import InstanceData
from torch import Tensor
from torch import nn as nn
from torch.nn import functional as F

from embodiedscan.models.layers import aligned_3d_nms
from embodiedscan.registry import MODELS
from embodiedscan.structures import BaseInstance3DBoxes, Det3DDataSample
from embodiedscan.structures.det3d_data_sample import SampleList
from embodiedscan.models.layers.l3det_modules import (BiDecoderLayer, ClsAgnosticPredictHead,
                                                 BiEncoderLayer, BiEncoder, PositionEmbeddingLearned)

EPS = 1e-6


class PointsObjClsModule(BaseModule):
    """object candidate point prediction from seed point features.

    Args:
        in_channel (int): number of channels of seed point features.
        num_convs (int, optional): number of conv layers.
            Default: 3.
        conv_cfg (dict, optional): Config of convolution.
            Default: dict(type='Conv1d').
        norm_cfg (dict, optional): Config of normalization.
            Default: dict(type='BN1d').
        act_cfg (dict, optional): Config of activation.
            Default: dict(type='ReLU').
    """

    def __init__(self,
                 in_channel: int,
                 num_convs: int = 3,
                 conv_cfg: dict = dict(type='Conv1d'),
                 norm_cfg: dict = dict(type='BN1d'),
                 act_cfg: dict = dict(type='ReLU'),
                 init_cfg: Optional[dict] = None):
        super().__init__(init_cfg=init_cfg)
        conv_channels = [in_channel for _ in range(num_convs - 1)]
        conv_channels.append(1)

        self.mlp = nn.Sequential()
        prev_channels = in_channel
        for i in range(num_convs):
            self.mlp.add_module(
                f'layer{i}',
                ConvModule(
                    prev_channels,
                    conv_channels[i],
                    1,
                    padding=0,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg if i < num_convs - 1 else None,
                    act_cfg=act_cfg if i < num_convs - 1 else None,
                    bias=True,
                    inplace=True))
            prev_channels = conv_channels[i]

    def forward(self, seed_features):
        """Forward pass.

        Args:
            seed_features (torch.Tensor): seed features, dims:
                (batch_size, feature_dim, num_seed)

        Returns:
            torch.Tensor: objectness logits, dim:
                (batch_size, 1, num_seed)
        """
        return self.mlp(seed_features)


class GeneralSamplingModule(nn.Module):
    """Sampling Points.

    Sampling points with given index.
    """

    def forward(self, xyz: Tensor, features: Tensor,
                sample_inds: Tensor) -> Tuple[Tensor]:
        """Forward pass.

        Args:
            xyz (Tensor)： (B, N, 3) the coordinates of the features.
            features (Tensor): (B, C, N) features to sample.
            sample_inds (Tensor): (B, M) the given index,
                where M is the number of points.

        Returns:
            Tensor: (B, M, 3) coordinates of sampled features
            Tensor: (B, C, M) the sampled features.
            Tensor: (B, M) the given index.
        """
        xyz_t = xyz.transpose(1, 2).contiguous()
        new_xyz = gather_points(xyz_t, sample_inds).transpose(1,
                                                              2).contiguous()
        new_features = gather_points(features, sample_inds).contiguous()

        return new_xyz, new_features, sample_inds


@MODELS.register_module()
class L3DetHead(BaseModule):
    r"""Bbox head of L3Det.

    Args:
        num_classes (int): The number of class.
        in_channels (int): The dims of input features from backbone.
        bbox_coder (:obj:`BaseBBoxCoder`): Bbox coder for encoding and
            decoding boxes.
        num_decoder_layers (int): The number of transformer decoder layers.
        transformerlayers (dict): Config for transformer decoder.
        train_cfg (dict, optional): Config for training.
        test_cfg (dict, optional): Config for testing.
        num_proposal (int): The number of initial sampling candidates.
        pred_layer_cfg (dict, optional): Config of classfication and regression
            prediction layers.
        size_cls_agnostic (bool): Whether the predicted size is class-agnostic.
        sampling_objectness_loss (dict, optional): Config of initial sampling
            objectness loss.
        objectness_loss (dict, optional): Config of objectness loss.
        center_loss (dict, optional): Config of center loss.
        dir_class_loss (dict, optional): Config of direction classification
            loss.
        dir_res_loss (dict, optional): Config of direction residual
            regression loss.
        size_class_loss (dict, optional): Config of size classification loss.
        size_res_loss (dict, optional): Config of size residual
            regression loss.
        size_reg_loss (dict, optional): Config of class-agnostic size
            regression loss.
        semantic_loss (dict, optional): Config of point-wise semantic
            segmentation loss.
    """

    def __init__(self,
                 num_classes: int,
                 in_channels: int,  # 288
                 num_decoder_layers: int,
                 self_position_embedding='loc_learned',
                 self_attend=True,
                 butd=True,
                 num_obj_class=485,
                 train_cfg: Optional[dict] = None,
                 test_cfg: Optional[dict] = None,
                 num_proposal: int = 256,
                 size_cls_agnostic: bool = True,
                 sampling_objectness_loss: Optional[dict] = None,
                 objectness_loss: Optional[dict] = None,
                 center_loss: Optional[dict] = None,
                 dir_class_loss: Optional[dict] = None,
                 dir_res_loss: Optional[dict] = None,
                 size_class_loss: Optional[dict] = None,
                 size_res_loss: Optional[dict] = None,
                 size_reg_loss: Optional[dict] = None,
                 semantic_loss: Optional[dict] = None,
                 contrastive_align_loss: Optional[dict] = True,
                 init_cfg: Optional[dict] = None):
        super(L3DetHead, self).__init__(init_cfg=init_cfg)
        self.num_classes = num_classes
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.num_proposal = num_proposal
        self.in_channels = in_channels
        self.num_decoder_layers = num_decoder_layers
        self.size_cls_agnostic = size_cls_agnostic
        self.contrastive_align_loss = contrastive_align_loss
        self.self_position_embedding = self_position_embedding
        self.sample_mode = 'kps'
        self.butd = butd

        # Box encoder
        if self.butd:
            self.butd_class_embeddings = nn.Embedding(num_obj_class, 768)
            saved_embeddings = torch.from_numpy(np.load(
                'data/scanrefer/class_embeddings3d.npy', allow_pickle=True
            ))
            self.butd_class_embeddings.weight.data.copy_(saved_embeddings)
            self.butd_class_embeddings.requires_grad = False
            self.class_embeddings = nn.Linear(768, in_channels - 128)
            self.box_embeddings = PositionEmbeddingLearned(6, 128)

        self.decoder = nn.ModuleList()
        for i in range(self.num_decoder_layers):
            self.decoder.append(BiDecoderLayer(
                in_channels, n_heads=8, dim_feedforward=256,
                dropout=0.1, activation="relu",
                self_position_embedding='loc_learned', butd=self.butd
            ))
        self.embed_dims = in_channels  # 288

        # Extra layers for contrastive losses
        if contrastive_align_loss:
            self.contrastive_align_projection_image = nn.Sequential(
                nn.Linear(in_channels, in_channels),
                nn.ReLU(),
                nn.Linear(in_channels, in_channels),
                nn.ReLU(),
                nn.Linear(in_channels, 64)
            )
            self.contrastive_align_projection_text = nn.Sequential(
                nn.Linear(in_channels, in_channels),
                nn.ReLU(),
                nn.Linear(in_channels, in_channels),
                nn.ReLU(),
                nn.Linear(in_channels, 64)
            )

        # Cross-encoder
        self.pos_embed = PositionEmbeddingLearned(3, self.embed_dims)
        bi_layer = BiEncoderLayer(
            self.embed_dims, dropout=0.1, activation="relu",
            n_heads=8, dim_feedforward=256,
            self_attend_lang=self_attend, self_attend_vis=self_attend,
            use_butd_enc_attn=self.butd
        )
        self.cross_encoder = BiEncoder(bi_layer, 3)

        # Initial object candidate sampling
        self.gsample_module = GeneralSamplingModule()
        self.fps_module = Points_Sampler([self.num_proposal])
        self.points_obj_cls = PointsObjClsModule(self.in_channels)

        # Proposal (layer for size and center)
        self.proposal_head = ClsAgnosticPredictHead(
            num_classes, 1, num_proposal, self.embed_dims,
            objectness=False, heading=False,
            compute_sem_scores=True
        )

        # query proj and key proj
        self.decoder_query_proj = nn.Conv1d(
            self.embed_dims, self.embed_dims, kernel_size=1)
        # self.decoder_key_proj = nn.Conv1d(
        #     self.embed_dims, self.embed_dims, kernel_size=1)

        # Prediction Heads
        self.prediction_heads = nn.ModuleList()
        for i in range(self.num_decoder_layers):
            self.prediction_heads.append(ClsAgnosticPredictHead(
                num_classes, 1, num_proposal, self.embed_dims,
                objectness=False, heading=False,
                compute_sem_scores=True
            ))

        self.loss_sampling_objectness = MODELS.build(sampling_objectness_loss)
        self.loss_objectness = MODELS.build(objectness_loss)
        self.loss_center = MODELS.build(center_loss)
        self.loss_dir_res = MODELS.build(dir_res_loss)
        self.loss_dir_class = MODELS.build(dir_class_loss)
        self.loss_semantic = MODELS.build(semantic_loss)
        # self.loss_contrastive_align = MODELS.build(contrastive_align_loss)

        if self.size_cls_agnostic:
            self.loss_size_reg = MODELS.build(size_reg_loss)
        else:
            self.loss_size_res = MODELS.build(size_res_loss)
            self.loss_size_class = MODELS.build(size_class_loss)


    def _extract_input(self, feat_dict: dict) -> Tuple[Tensor]:
        """Extract inputs from features dictionary.

        Args:
            feat_dict (dict): Feature dict from backbone.

        Returns:
            Tuple[Tensor]:

            - seed_points (Tensor): Coordinates of input points.
            - seed_features (Tensor): Features of input points.
            - seed_indices (Tensor): Indices of input points.
        """

        seed_points = feat_dict['fp_xyz'][-1]  # (B, points, 3)
        seed_features = feat_dict['fp_features'][-1]  # (B, F, points)
        seed_indices = feat_dict['fp_indices'][-1]  # (B, points)
        text_feats = feat_dict['text_feats']  # (B, L, F)  
        text_padding_mask = feat_dict['text_attention_mask']  # (B, L)


        return seed_points, seed_features, seed_indices, text_feats, text_padding_mask
    

    # @property
    # def sample_mode(self):
    #     """
    #     Returns:
    #         str: Sample mode for initial candidates sampling.
    #     """
    #     if self.training:
    #         sample_mode = self.train_cfg.sample_mode
    #     else:
    #         sample_mode = self.test_cfg.sample_mode
    #     assert sample_mode in ['fps', 'kps']
    #     return sample_mode

    def forward(self, feat_dict: dict) -> dict:
        """Forward pass.

        Note:
            The forward of GroupFree3DHead is divided into 2 steps:

                1. Initial object candidates sampling.
                2. Iterative object box prediction by transformer decoder.

        Args:
            feat_dict (dict): Feature dict from backbone.


        Returns:
            results (dict): Predictions of GroupFree3D head.
        """
        sample_mode = self.sample_mode

        # Extract the point cloud and text feature respectively
        # (B, N, C)
        # (B, L, F)  (B, L) attention mask 是用来判断这个 token 是不是 padding 过的
        seed_xyz, seed_features, seed_indices, text_feats, text_padding_mask = self._extract_input(feat_dict)


        # Box encoding
        if self.butd:
            # attend on those features
            detected_mask = ~feat_dict['det_bbox_label_mask']
            detected_feats = torch.cat([
                self.box_embeddings(feat_dict['det_bboxes']),
                self.class_embeddings(self.butd_class_embeddings(
                    feat_dict['det_bbox_class_ids']
                )).transpose(1, 2)  # 92.5, 84.9
            ], 1).transpose(1, 2).contiguous()


        # Cross-modality encoding and update seed_features and text_feats
        seed_features, text_feats = self.cross_encoder(
            vis_feats=seed_features.transpose(1, 2).contiguous(),
            pos_feats=self.pos_embed(seed_xyz).transpose(1, 2).contiguous(),
            padding_mask=torch.zeros(
                len(seed_xyz), seed_xyz.size(1)
            ).to(seed_xyz.device).bool(),
            text_feats=text_feats,
            text_padding_mask=text_padding_mask,
            detected_feats=detected_feats,
            detected_mask=detected_mask
        )
        seed_features = seed_features.transpose(1, 2).contiguous()

        results = dict(
            seed_points=seed_xyz,
            seed_features=seed_features,
            seed_indices=seed_indices,
            text_memory=text_feats)

        if self.contrastive_align_loss:
            proj_tokens = F.normalize(
                self.contrastive_align_projection_text(text_feats), p=2, dim=-1
            )
            results['proj_tokens'] = proj_tokens

        # 1. Initial object candidates sampling.
        if sample_mode == 'fps':
            sample_inds = self.fps_module(seed_xyz, seed_features)
        elif sample_mode == 'kps':  # 默认使用 kps 进行 sample，这里选取了 topk 个 num query
            points_obj_cls_logits = self.points_obj_cls(seed_features)  # (batch_size, 1, num_seed),
            points_obj_cls_scores = points_obj_cls_logits.sigmoid().squeeze(1)
            sample_inds = torch.topk(points_obj_cls_scores,
                                     self.num_proposal)[1].int()
            results['seeds_obj_cls_logits'] = points_obj_cls_logits
        else:
            raise NotImplementedError(
                f'Sample mode {sample_mode} is not supported!')

        candidate_xyz, candidate_features, sample_inds = self.gsample_module(
            seed_xyz, seed_features, sample_inds)

        results['query_points_xyz'] = candidate_xyz  # (B, M, 3)
        results['query_points_feature'] = candidate_features  # (B, C, M)
        results['query_points_sample_inds'] = sample_inds.long()  # (B, M)

        # Proposals (one for each query)
        proposal_center, proposal_size = self.proposal_head(
            candidate_features,
            base_xyz=candidate_xyz,
            end_points=results,
            prefix='proposal_'
        )

        base_xyz = proposal_center.detach().clone()  # (B, #V, 3)
        base_size = proposal_size.detach().clone()  # (B, V, 3)
        query_mask = None

        # check the feature dimension
        query = self.decoder_query_proj(candidate_features)
        query = query.transpose(1, 2).contiguous()  # (B, V, F)

        if self.contrastive_align_loss:
            results['proposal_proj_queries'] = F.normalize(
                self.contrastive_align_projection_image(query), p=2, dim=-1
            )

        # key = self.decoder_key_proj(seed_features).permute(2, 0, 1)
        # value = key

        # transformer decoder
        results['num_decoder_layers'] = 0
        for i in range(self.num_decoder_layers):
            prefix = 'last_' if i == self.num_decoder_layers-1 else f'{i}head_'

            # Position Embedding for Self-Attention
            if self.self_position_embedding == 'none':
                query_pos = None
            elif self.self_position_embedding == 'xyz_learned':
                query_pos = base_xyz
            elif self.self_position_embedding == 'loc_learned':
                query_pos = torch.cat([base_xyz, base_size], -1)
            else:
                raise NotImplementedError

            query = self.decoder[i](
                query, seed_features.transpose(1, 2).contiguous(), 
                text_feats, query_pos,
                query_mask,
                text_padding_mask,
                detected_feats=detected_feats if self.butd else None,
                detected_mask=detected_mask if self.butd else None)  # (B, V, F)

            results[f'{prefix}query'] = query

            if self.contrastive_align_loss:
                results[f'{prefix}proj_queries'] = F.normalize(
                    self.contrastive_align_projection_image(query), p=2, dim=-1
                )

            # Prediction
            base_xyz, base_size = self.prediction_heads[i](
                query.transpose(1, 2).contiguous(),  # (B, F, V)
                base_xyz=candidate_xyz,
                end_points=results,
                prefix=prefix
            )
            bbox3d = torch.cat([base_xyz, base_size], -1)

            results[f'{prefix}bbox3d'] = bbox3d
            base_xyz = base_xyz.detach().clone()
            base_size = base_size.detach().clone()

            results['num_decoder_layers'] += 1

        for key in results:
            if 'bbox3d' in key:
                pred_size = torch.clamp(results[key][:, :, 3:], min=1e-6)
                pred_center = results[key][:, :, :3]
                results[key] = torch.cat([pred_center, pred_size], -1)
                

        return results

    def predict(self, points: List[torch.Tensor],
                feats_dict: Dict[str, torch.Tensor],
                batch_data_samples: List[Det3DDataSample],
                **kwargs) -> List[InstanceData]:
        """
        Args:
            points (list[tensor]): Point clouds of multiple samples.
            feats_dict (dict): Features from FPN or backbone.
            batch_data_samples (List[:obj:`Det3DDataSample`]): The Data
                Samples. It usually includes meta information of data.

        Returns:
            list[:obj:`InstanceData`]: List of processed predictions. Each
            InstanceData contains 3d Bounding boxes and corresponding
            scores and labels.
        """

        preds_dict = self(feats_dict)
        batch_input_metas = []
        batch_gt_instance_3d = []
        for data_sample in batch_data_samples:
            batch_input_metas.append(data_sample.metainfo)
            batch_gt_instance_3d.append(data_sample.gt_instances_3d)

        results_list = self.predict_by_feat(points, preds_dict,
                                            batch_gt_instance_3d,
                                            batch_input_metas, **kwargs)

        return results_list

    def predict_by_feat(self,
                        points: List[torch.Tensor],
                        bbox_preds_dict: dict,
                        batch_gt_instances_3d: List[InstanceData],
                        batch_input_metas: List[dict],
                        task: str = 'vg',   # 'vg' or 'det'
                        use_nms: bool = True,
                        **kwargs) -> List[InstanceData]:
        """Generate bboxes from vote head predictions.

        Args:
            points (List[torch.Tensor]): Input points of multiple samples.
            bbox_preds_dict (dict): Predictions from groupfree3d head.
            batch_input_metas (list[dict]): Each item
                contains the meta information of each sample.
            use_nms (bool): Whether to apply NMS, skip nms postprocessing
                while using vote head in rpn stage.

        Returns:
            list[:obj:`InstanceData`]: List of processed predictions. Each
            InstanceData cantains 3d Bounding boxes and corresponding
            scores and labels.
        """
        # support multi-stage predictions  我们默认使用 last
        assert self.test_cfg['prediction_stages'] in \
            ['last', 'all', 'last_three']

        if self.test_cfg['prediction_stages'] == 'last':
            prefixes = ['last_']
        elif self.test_cfg['prediction_stages'] == 'all':
            prefixes = ['proposal_'] + \
                [f'{i}head_' for i in range(self.num_decoder_layers-1)] + ['last_']
        elif self.test_cfg['prediction_stages'] == 'last_three':
            prefixes = [
                f'{i}head_' for i in range(self.num_decoder_layers -
                                        3, self.num_decoder_layers-1) + ['last_']
            ]
        else:
            raise NotImplementedError
    
        batch_gt_token_maps = [
            gt_instances_3d.pred_pos_map
            for gt_instances_3d in batch_gt_instances_3d
        ]

        gt_token_maps = torch.stack(batch_gt_token_maps)   # (B, 1, 256)

        proj_tokens = bbox_preds_dict['proj_tokens']  # (B, tokens, 64)
        sem_scores = list()
        proj_queries = list()
        bbox3d = list()
        for prefix in prefixes:
            proj_query = bbox_preds_dict[f'{prefix}proj_queries']  # (b, n, 64)
            sem_score = bbox_preds_dict[f'{prefix}sem_scores'].softmax(-1)
            bbox = bbox_preds_dict[f'{prefix}bbox3d']   #（b, n, 6)
            proj_queries.append(proj_query)
            sem_scores.append(sem_score)
            bbox3d.append(bbox)

        sem_scores = torch.cat(sem_scores, dim=1)
        proj_queries = torch.cat(proj_queries, dim=1)
        bbox3d = torch.cat(bbox3d, dim=1)
        batch_size = bbox3d.shape[0]
        stack_points = torch.stack(points)
        results_list = list()

        if task == 'det':
            if use_nms:
                temp_results = InstanceData()
                for b in range(batch_size):
                    bbox_selected, score_selected, labels = \
                        self.multiclass_nms_single(None,
                                                sem_scores[b],
                                                bbox3d[b],
                                                stack_points[b, ..., :3],
                                                batch_input_metas[b])
                    bbox = batch_input_metas[b]['box_type_3d'](
                        bbox_selected,
                        box_dim=bbox_selected.shape[-1],
                        with_yaw=False)
                    temp_results.bboxes_3d = bbox
                    temp_results.scores_3d = score_selected
                    temp_results.labels_3d = labels
                    results_list.append(temp_results)
                return results_list
            else:
                return bbox3d
        elif task == 'vg':
            align_scores = self.compute_alignment_scores(proj_tokens, proj_queries, gt_token_maps)  # (B. N)
            for b in range(batch_size):
                temp_results = InstanceData()
                bbox_selected = bbox3d[b]
                bbox = batch_input_metas[b]['box_type_3d'](
                    bbox_selected,
                    box_dim=bbox_selected.shape[-1],
                    with_yaw=False,
                    origin=(0.5, 0.5, 0.5))
                temp_results.bboxes_3d = bbox
                temp_results.scores_3d = align_scores[b]
                results_list.append(temp_results)
            return results_list
        else:
            raise NotImplementedError


    def compute_alignment_scores(self, proj_tokens, proj_queries, gt_token_maps):
        """Compute the alignment socres of each bboxes in one batch.

        Args:
            sem_scores (torch.Tensor): semantic class score of bounding boxes. (B, N*num_stage, 256) 这里默认 num_stage == 1
            proj_tokens (torch.Tensor):  (B, tokens, 64)
            proj_queris (torch.Tensor):  (B, N*num_stage, 64)
            bbox (torch.Tensor): Predicted bounding boxes. (B, N*num_stage, 6)
            gt_token_maps (torch.Tensor): positive map for each text query in one batch. (B, 1, 256)

        Returns:
            torch.Tensor: Bounding boxes, scores. 
        """     
        # gt_token_map 是预先算好的, 存储在了 pickle 文件里面  (B, 1, 256)
        align_scores = torch.matmul(proj_queries, proj_tokens.transpose(-1, -2))
        align_scores_ = (align_scores / 0.07).softmax(-1)  # (B, N, tokens)  tokens < 256
        align_scores = torch.zeros(align_scores_.size(0), align_scores_.size(1), 256)
        align_scores = align_scores.to(align_scores_.device)
        align_scores[:, :align_scores_.size(1), :align_scores_.size(2)] = align_scores_  # (B, Q, 256)
        align_scores = (align_scores*gt_token_maps).sum(-1)  # (B, Q)

        return align_scores
 

    def multiclass_nms_single(self, obj_scores, sem_scores, bbox, points,
                              input_meta):
        """Multi-class nms in single batch.

        Args:
            obj_scores (torch.Tensor): Objectness score of bounding boxes.
            sem_scores (torch.Tensor): semantic class score of bounding boxes.
            bbox (torch.Tensor): Predicted bounding boxes.
            points (torch.Tensor): Input points.
            input_meta (dict): Point cloud and image's meta info.

        Returns:
            tuple[torch.Tensor]: Bounding boxes, scores and labels.
        """
        bbox = input_meta['box_type_3d'](
            bbox,
            box_dim=bbox.shape[-1],
            with_yaw=False,
            origin=(0.5, 0.5, 0.5))
        box_indices = bbox.points_in_boxes_all(points)

        corner3d = bbox.corners
        minmax_box3d = corner3d.new(torch.Size((corner3d.shape[0], 6)))
        minmax_box3d[:, :3] = torch.min(corner3d, dim=1)[0]
        minmax_box3d[:, 3:] = torch.max(corner3d, dim=1)[0]

        nonempty_box_mask = box_indices.T.sum(1) > 5

        bbox_classes = torch.argmax(sem_scores, -1)
        nms_selected = aligned_3d_nms(minmax_box3d[nonempty_box_mask],
                                      obj_scores[nonempty_box_mask],
                                      bbox_classes[nonempty_box_mask],
                                      self.test_cfg.nms_thr)

        # filter empty boxes and boxes with low score
        scores_mask = (obj_scores > self.test_cfg.score_thr)
        nonempty_box_inds = torch.nonzero(
            nonempty_box_mask, as_tuple=False).flatten()
        nonempty_mask = torch.zeros_like(bbox_classes).scatter(
            0, nonempty_box_inds[nms_selected], 1)
        selected = (nonempty_mask.bool() & scores_mask.bool())

        if self.test_cfg.per_class_proposal:
            bbox_selected, score_selected, labels = [], [], []
            for k in range(sem_scores.shape[-1]):
                bbox_selected.append(bbox[selected].tensor)
                score_selected.append(obj_scores[selected] *
                                      sem_scores[selected][:, k])
                labels.append(
                    torch.zeros_like(bbox_classes[selected]).fill_(k))
            bbox_selected = torch.cat(bbox_selected, 0)
            score_selected = torch.cat(score_selected, 0)
            labels = torch.cat(labels, 0)
        else:
            bbox_selected = bbox[selected].tensor
            score_selected = obj_scores[selected]
            labels = bbox_classes[selected]

        return bbox_selected, score_selected, labels

