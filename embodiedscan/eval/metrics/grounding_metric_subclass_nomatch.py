# Copyright (c) OpenRobotLab. All rights reserved.
import os
from typing import Dict, List, Optional, Sequence

import mmengine
from mmengine.evaluator import BaseMetric
from mmengine.logging import MMLogger, print_log
from terminaltables import AsciiTable

from embodiedscan.registry import METRICS
from embodiedscan.structures import EulerDepthInstance3DBoxes

import numpy as np
from scipy.optimize import linear_sum_assignment

@METRICS.register_module()
class GroundingMetricMod(BaseMetric):
    """Lanuage grounding evaluation metric. We calculate the grounding
    performance based on the alignment score of each bbox with the input
    prompt.

    Args:
        iou_thr (float or List[float]): List of iou threshold when calculate
            the metric. Defaults to [0.25, 0.5].
        collect_device (str): Device name used for collecting results from
            different ranks during distributed training. Must be 'cpu' or
            'gpu'. Defaults to 'cpu'.
        prefix (str, optional): The prefix that will be added in the metric
            names to disambiguate homonymous metrics of different evaluators.
            If prefix is not provided in the argument, self.default_prefix will
            be used instead. Defaults to None.
        format_only (bool): Whether to only inference the predictions without
            evaluation. Defaults to False.
        result_dir (str): Dir to save results, e.g., if result_dir = './',
            the result file will be './test_results.json'. Defaults to ''.
    """  

    def __init__(self,
                 iou_thr: List[float] = [0.25, 0.5],
                 collect_device: str = 'cpu',
                 prefix: Optional[str] = None,
                 format_only=False,
                 result_dir='') -> None:
        super(GroundingMetricMod, self).__init__(prefix=prefix,
                                              collect_device=collect_device)
        self.iou_thr = [iou_thr] if isinstance(iou_thr, float) else iou_thr
        self.prefix = prefix
        self.format_only = format_only
        self.result_dir = result_dir
        self.mapping = {
            'direct_attribute_o_individual': 'dir_attr_indi',
            'direct_attribute_o_common': 'dir_attr_com',
            'direct_eq': 'dir_eq',
            'indirect_or': 'indir_or',
            'indirect_space_oo': 'indir_space',
            'indirect_attribute_oo': 'indir_attr',
            'other': 'other',
            'overall': 'overall'
        }

    def process(self, data_batch: dict, data_samples: Sequence[dict]) -> None:
        """Process one batch of data samples and predictions.

        The processed results should be stored in ``self.results``, which will
        be used to compute the metrics when all batches have been processed.

        Args:
            data_batch (dict): A batch of data from the dataloader.
            data_samples (Sequence[dict]): A batch of outputs from the model.
        """
        for data_sample in data_samples:
            pred_3d = data_sample['pred_instances_3d']
            eval_ann_info = data_sample['eval_ann_info']
            cpu_pred_3d = dict()
            for k, v in pred_3d.items():
                if hasattr(v, 'to'):
                    cpu_pred_3d[k] = v.to('cpu')
                else:
                    cpu_pred_3d[k] = v
            self.results.append((eval_ann_info, cpu_pred_3d))

    def ground_eval_subset(self, gt_annos, det_annos, logger=None, prefix=''):

        assert len(det_annos) == len(gt_annos)

        num_samples = len(gt_annos) # each sample contains multiple pred boxes
        total_gt_boxes = 0
        # these lists records for each sample, whether a gt box is matched or not
        gt_matched_records = [[] for _ in self.iou_thr]
        # these lists records for each pred box, NOT for each sample        
        sample_indices = [] # each pred box belongs to which sample
        confidences = [] # each pred box has a confidence score
        ious = [] # each pred box has a ious, shape (num_gt) in the corresponding sample
        # record the indices of each reference type

        for sample_idx in range(num_samples):
            det_anno = det_annos[sample_idx]
            gt_anno = gt_annos[sample_idx]

            target_scores = det_anno['target_scores_3d']  # (num_query, )
            top_idxs =  np.argsort(-target_scores) 
            target_scores = target_scores[top_idxs]
            pred_bboxes = det_anno['bboxes_3d'][top_idxs]
            gt_bboxes = gt_anno['gt_bboxes_3d']
            pred_bboxes = EulerDepthInstance3DBoxes(pred_bboxes.tensor,
                                               origin=(0.5, 0.5, 0.5))
            gt_bboxes = EulerDepthInstance3DBoxes(gt_bboxes.tensor,
                                                  origin=(0.5, 0.5, 0.5))
            num_preds = len(pred_bboxes)
            num_gts = len(gt_bboxes)
            total_gt_boxes += num_gts
            for iou_idx, _ in enumerate(self.iou_thr):
                gt_matched_records[iou_idx].append(np.zeros(num_gts, dtype=bool))

            iou_mat = pred_bboxes.overlaps(pred_bboxes, gt_bboxes).cpu().numpy()  # (num_query, num_gt)
            for i, score in enumerate(target_scores):
                sample_indices.append(sample_idx)
                confidences.append(score)
                ious.append(iou_mat[i])
        
        confidences = np.array(confidences)
        sorted_inds = np.argsort(-confidences)
        sample_indices = [sample_indices[i] for i in sorted_inds]
        ious = [ious[i] for i in sorted_inds]

        tp_thr = {}
        fp_thr = {}
        for thr in self.iou_thr:
            tp_thr[f'{prefix}@{thr}'] = np.zeros(len(sample_indices))
            fp_thr[f'{prefix}@{thr}'] = np.zeros(len(sample_indices))

        for d, sample_idx in enumerate(sample_indices):
            iou_max = -np.inf
            num_gts = len(gt_annos[sample_idx]['gt_bboxes_3d'])
            cur_iou = ious[d]
            if num_gts > 0:
                for j in range(num_gts):
                    iou = cur_iou[j]
                    if iou > iou_max:
                        iou_max = iou
                        jmax = j
            
            for iou_idx, thr in enumerate(self.iou_thr):
                if iou_max >= thr:
                    if not gt_matched_records[iou_idx][sample_idx][jmax]:
                        gt_matched_records[iou_idx][sample_idx][jmax] = True
                        tp_thr[f'{prefix}@{thr}'][d] = 1.0
                    else:
                        fp_thr[f'{prefix}@{thr}'][d] = 1.0
                else:
                    fp_thr[f'{prefix}@{thr}'][d] = 1.0
        ret = {}
        for t in self.iou_thr:
            metric = prefix + '@' + str(t)
            fp = np.cumsum(fp_thr[metric])
            tp = np.cumsum(tp_thr[metric])
            recall = tp / float(total_gt_boxes)
            precision = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
            ap = average_precision(recall, precision)
            ret[metric] = float(ap)
            best_recall = recall[-1] if len(recall) > 0 else 0
            ret[metric + '_rec'] = float(best_recall)
        return ret

    def ground_eval(self, gt_annos, det_annos, logger=None):
        reference_options = [v for k, v in self.mapping.items()]
        assert len(det_annos) == len(gt_annos)
        results = {}
        for ref in reference_options:
            indices = [i for i, gt_anno in enumerate(gt_annos) if self.mapping[gt_anno.get('sub_class', 'other').lower().strip('vg_')] == ref]
            sub_gt_annos = [gt_annos[i] for i in indices]
            sub_det_annos = [det_annos[i] for i in indices]
            print(str(ref) + str(len(indices)))
            ret = self.ground_eval_subset(sub_gt_annos, sub_det_annos, logger=logger, prefix=ref)
            for k, v in ret.items():
                results[k] = v
        overall_ret = self.ground_eval_subset(gt_annos, det_annos, logger=logger, prefix='overall')
        for k, v in overall_ret.items():
            results[k] = v


        # Compute the precision and recall for each iou threshold
        header = ['Type']
        header.extend(reference_options)
        table_columns = [[] for _ in range(len(header))]
        for t in self.iou_thr:
            table_columns[0].append('AP  '+str(t))
            table_columns[0].append('Rec '+str(t))            
            for i, ref in enumerate(reference_options):
                metric = ref + '@' + str(t)
                ap = results[metric]
                best_recall = results[metric + '_rec']
                table_columns[i+1].append(f'{float(ap):.4f}')
                table_columns[i+1].append(f'{float(best_recall):.4f}')

        table_data = [header]
        table_rows = list(zip(*table_columns))
        table_data += table_rows
        table = AsciiTable(table_data)
        table.inner_footing_row_border = True
        print_log('\n' + table.table, logger=logger)
        return results

    def compute_metrics(self, results: list) -> Dict[str, float]:
        """Compute the metrics from processed results after all batches have
        been processed.

        Args:
            results (list): The processed results of each batch.

        Returns:
            Dict[str, float]: The computed metrics. The keys are the names of
            the metrics, and the values are corresponding results.
        """
        logger: MMLogger = MMLogger.get_current_instance()  # noqa
        annotations, preds = zip(*results)
        ret_dict = {}
        if self.format_only:
            # preds is a list of dict
            results = []
            for pred in preds:
                result = dict()
                # convert the Euler boxes to the numpy array to save
                bboxes_3d = pred['bboxes_3d'].tensor
                scores_3d = pred['scores_3d']
                # Note: hard-code save top-20 predictions
                # eval top-10 predictions during the test phase by default
                box_index = scores_3d.argsort(dim=-1, descending=True)[:20]
                top_bboxes_3d = bboxes_3d[box_index]
                top_scores_3d = scores_3d[box_index]
                result['bboxes_3d'] = top_bboxes_3d.numpy()
                result['scores_3d'] = top_scores_3d.numpy()
                results.append(result)
            mmengine.dump(results,
                          os.path.join(self.result_dir, 'test_results.json'))
            return ret_dict

        ret_dict = self.ground_eval(annotations, preds)

        return ret_dict


def average_precision(recalls, precisions, mode='area'):
    """Calculate average precision (for single or multiple scales).

    Args:
        recalls (np.ndarray): Recalls with shape of (num_scales, num_dets)
            or (num_dets, ).
        precisions (np.ndarray): Precisions with shape of
            (num_scales, num_dets) or (num_dets, ).
        mode (str): 'area' or '11points', 'area' means calculating the area
            under precision-recall curve, '11points' means calculating
            the average precision of recalls at [0, 0.1, ..., 1]

    Returns:
        float or np.ndarray: Calculated average precision.
    """
    if recalls.ndim == 1:
        recalls = recalls[np.newaxis, :]
        precisions = precisions[np.newaxis, :]

    assert recalls.shape == precisions.shape
    assert recalls.ndim == 2

    num_scales = recalls.shape[0]
    ap = np.zeros(num_scales, dtype=np.float32)
    if mode == 'area':
        zeros = np.zeros((num_scales, 1), dtype=recalls.dtype)
        ones = np.ones((num_scales, 1), dtype=recalls.dtype)
        mrec = np.hstack((zeros, recalls, ones))
        mpre = np.hstack((zeros, precisions, zeros))
        for i in range(mpre.shape[1] - 1, 0, -1):
            mpre[:, i - 1] = np.maximum(mpre[:, i - 1], mpre[:, i])
        for i in range(num_scales):
            ind = np.where(mrec[i, 1:] != mrec[i, :-1])[0]
            ap[i] = np.sum(
                (mrec[i, ind + 1] - mrec[i, ind]) * mpre[i, ind + 1])
    elif mode == '11points':
        for i in range(num_scales):
            for thr in np.arange(0, 1 + 1e-3, 0.1):
                precs = precisions[i, recalls[i, :] >= thr]
                prec = precs.max() if precs.size > 0 else 0
                ap[i] += prec
            ap /= 11
    else:
        raise ValueError(
            'Unrecognized mode, only "area" and "11points" are supported')
    return ap


def matcher(preds, gts, cost_fns):
    """
    Matcher function that uses the Hungarian algorithm to find the best match
    between predictions and ground truths.

    Parameters:
    - preds: predicted bounding boxes (num_preds) 
    - gts: ground truth bounding boxes (num_gts)
    - cost_fn: a function that computes the cost matrix between preds and gts

    Returns:
    - matched_pred_inds: indices of matched predictions
    - matched_gt_inds: indices of matched ground truths
    - costs: cost of each matched pair
    """
    # Compute the cost matrix
    num_preds = len(preds) if not isinstance(preds, (list, tuple)) else len(preds[0])
    num_gts = len(gts) if not isinstance(gts, (list, tuple)) else len(gts[0])
    cost_matrix = np.zeros((num_preds, num_gts))
    for cost_fn in cost_fns:
        cost_matrix += cost_fn(preds, gts) #shape (num_preds, num_gts)

    # Perform linear sum assignment to minimize the total cost
    matched_pred_inds, matched_gt_inds = linear_sum_assignment(cost_matrix)
    costs = cost_matrix[matched_pred_inds, matched_gt_inds]
    return matched_pred_inds, matched_gt_inds, costs

def iou_cost_fn(pred_boxes, gt_boxes):
    iou = pred_boxes.overlaps(pred_boxes, gt_boxes)  # (num_query, num_gt)
    iou = iou.cpu().numpy()
    return 1.0 - iou
