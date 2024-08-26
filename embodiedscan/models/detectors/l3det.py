# ------------------------------------------------------------------------
# BEAUTY DETR
# Copyright (c) 2022 Ayush Jain & Nikolaos Gkanatsios
# Licensed under CC-BY-NC [see LICENSE for details]
# All Rights Reserved
# ------------------------------------------------------------------------
# Parts adapted from Group-Free
# Copyright (c) 2021 Ze Liu. All Rights Reserved.
# Licensed under the MIT License.
# ------------------------------------------------------------------------

import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
from transformers import RobertaModel, RobertaTokenizerFast
from .single_stage import SingleStage3DDetector
from embodiedscan.registry import MODELS
from ...structures.det3d_data_sample import SampleList
from torch import Tensor
from typing import Dict, Tuple, Union


@MODELS.register_module()
class L3Det(SingleStage3DDetector):
    """Language-guided 3D Detector.

    Args:
        num_class (int): number of semantics classes to predict
        num_obj_class (int): number of object classes
        input_feature_dim (int): feat_dim of pointcloud (without xyz)
        num_queries (int): Number of queries generated
        num_decoder_layers (int): number of decoder layers
        self_position_embedding (str or None): how to compute pos embeddings
        contrastive_align_loss (bool): contrast queries and token features
        d_model (int): dimension of features
        butd (bool): use detected box stream
        pointnet_ckpt (str or None): path to pre-trained pp++ checkpoint
        self_attend (bool): add self-attention in encoder
    """

    def __init__(self, 
                 backbone,
                 bbox_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 init_cfg=None,
                 **kwargs):
        """Initialize layers."""
        super(L3Det, self).__init__(
            backbone=backbone,
            bbox_head=bbox_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            init_cfg=init_cfg,
            **kwargs)

        # Text Encoder
        # t_type = "roberta-base"
        t_type = "/mnt/afs/xurunsen/projects/IndoorPerceptron/huggingface_models/roberta-base"
        self.tokenizer = RobertaTokenizerFast.from_pretrained(t_type)
        self.text_encoder = RobertaModel.from_pretrained(t_type)
        for param in self.text_encoder.parameters():
            param.requires_grad = False

        self.text_projector = nn.Sequential(
            nn.Linear(self.text_encoder.config.hidden_size, bbox_head['in_channels']),
            nn.LayerNorm(bbox_head['in_channels'], eps=1e-12),
            nn.Dropout(0.1)
        )

        # Init
        self.init_bn_momentum()
    
    def extract_feat(
        self, batch_inputs_dict: Dict[str, Tensor]
    ) -> Union[Tuple[torch.Tensor], Dict[str, Tensor]]:
        """Directly extract features from the backbone+neck.

        Args:
            batch_inputs_dict (dict): The model input dict which include
                'points', 'img' keys.

                    - points (list[torch.Tensor]): Point cloud of each sample.
                    - imgs (torch.Tensor, optional): Image of each sample.

        Returns:
            tuple[Tensor] | dict:  For outside 3D object detection, we
                typically obtain a tuple of features from the backbone + neck,
                and for inside 3D object detection, usually a dict containing
                features will be obtained.
        """
        points = batch_inputs_dict['points']
        stack_points = torch.stack(points)
        x = self.backbone(stack_points) # feats_dict
        if self.with_neck:
            x = self.neck(x)
        tokenized = self.tokenizer.batch_encode_plus(
            batch_inputs_dict['text'], padding="longest", return_tensors="pt"
        ).to(stack_points.device)
        encoded_text = self.text_encoder(**tokenized)
        text_feats = self.text_projector(encoded_text.last_hidden_state)
        # Invert attention mask that we get from huggingface
        # because its the opposite in pytorch transformer
        text_attention_mask = tokenized.attention_mask.ne(1).bool()
        x['text_feats'] = text_feats
        x['text_attention_mask'] = text_attention_mask
        x['tokenized'] = tokenized
        x['det_bboxes'] = batch_inputs_dict['det_bboxes']
        x['det_bbox_label_mask'] = batch_inputs_dict['det_bbox_label_mask']
        x['det_bbox_class_ids'] = batch_inputs_dict['det_bbox_class_ids']
        # return the feats_dict including point feature and text feature
        return x

    def init_bn_momentum(self):
        """Initialize batch-norm momentum."""
        for m in self.modules():
            if isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                m.momentum = 0.1

    def predict(self, batch_inputs_dict: dict, batch_data_samples: SampleList,
                **kwargs) -> SampleList:
        """Predict results from a batch of inputs and data samples with post-
        processing.

        Args:
            batch_inputs_dict (dict): The model input dict which include
                'points', 'imgs' keys.

                    - points (list[torch.Tensor]): Point cloud of each sample.
                    - imgs (torch.Tensor, optional): Image of each sample.

            batch_data_samples (List[:obj:`Det3DDataSample`]): The Data
                Samples. It usually includes information such as
                `gt_instance_3d`, `gt_pts_seg`.
            rescale (bool): Whether to rescale the results.
                Defaults to True.

        Returns:
            list[:obj:`Det3DDataSample`]: Detection results of the
            input images. Each Det3DDataSample usually contain
            'pred_instances_3d'. And the ``pred_instances_3d`` usually
            contains following keys.

            - scores_3d (Tensor): Classification scores, has a shape
                (num_instance, )
            - labels_3d (Tensor): Labels of bboxes, has a shape
                (num_instances, ).
            - bboxes_3d (Tensor): Contains a tensor with shape
                (num_instances, C) where C >=7.
        """
        x = self.extract_feat(batch_inputs_dict)
        points = batch_inputs_dict['points']
        results_list = self.bbox_head.predict(points, x, batch_data_samples,
                                              **kwargs)
        predictions = self.add_pred_to_datasample(batch_data_samples,
                                                  results_list)
        return predictions