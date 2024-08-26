from pytorch3d.ops import box3d_overlap, _check_nonzero, _check_coplanar
from embodiedscan.structures.bbox_3d.euler_depth_box3d import EulerDepthInstance3DBoxes
from embodiedscan.models.losses.chamfer_distance import bbox_to_corners
import torch
import numpy as np


# create a tensor of shape num_boxes, 9: num_boxes boxes of 9 parameters each

def check_consistency(num_boxes=1, verbose=False):
    boxes = torch.zeros(num_boxes, 9)
    boxes[:, :3] = torch.rand(num_boxes, 3)  # center
    boxes[:, 3:6] = torch.rand(num_boxes, 3) + 0.1  # size
    max_angle = 2 * np.pi
    boxes[:, 6:9] = torch.rand(num_boxes, 3) * max_angle - max_angle / 2  # orientation

    # boxes = torch.zeros(num_boxes, 9)
    # boxes[:, 3:6] = torch.arange(1, 4).repeat(num_boxes, 1)  # size
    if verbose: 
        print(boxes)
    corners1 = bbox_to_corners(boxes)
    boxes2 = EulerDepthInstance3DBoxes(boxes)
    corners2 = boxes2.corners
    vol, iou = box3d_overlap(corners1, corners2)
    diag_iou = [iou[i, i] for i in range(num_boxes)]
    iou = torch.tensor(diag_iou)
    if verbose:
        print(corners1)
        print(corners2)
        print(iou)
    
    assert (iou >= 1 - 1e-4).all(), "iou < 1"
    

if __name__ == '__main__':
    for i in range(10000):
        check_consistency(num_boxes=1, verbose=False)
        if (i+1) % 1000 == 0:
            print(f"Checked {i+1} boxes")