import torch
import torch.nn as nn
import torch.nn.functional as F

class Folder(nn.Module):

    def __init__(self, kernel_size=3):
        super().__init__()
        self.kernel_size = kernel_size

    def forward(self, feature_map):
        N,_,H,W = feature_map.size()
        feature_map = F.unfold(
            feature_map,kernel_size=self.kernel_size,padding=1)
        feature_map = feature_map.view(N, -1, H, W)
        return feature_map

def compute_locations(h, w, stride, device):
    shifts_x = torch.arange(0,
        w * stride,
        step=stride,
        dtype=torch.float32,
        device=device)
    shifts_y = torch.arange(0,
        h * stride,
        step=stride,
        dtype=torch.float32,
        device=device)
    shift_y, shift_x = torch.meshgrid(shifts_y, shifts_x)
    shift_x = shift_x.reshape(-1)
    shift_y = shift_y.reshape(-1)
    locations = torch.stack((shift_x, shift_y), dim=1) + stride // 2
    return locations

def relative_coordinate_maps(feat_sizes, centers, strides, parsing_stride, range=8.0):
    _, _, H, W = feat_sizes
    locations = compute_locations(H,
                                  W,
                                  stride=parsing_stride,
                                  device=centers.device)
    relative_coordinates = centers.reshape(-1, 1, 2) - locations.reshape(1, -1, 2)
    relative_coordinates = relative_coordinates.permute(0, 2, 1).float()
    relative_coordinates = relative_coordinates / (strides[:, None, None] * range)
    return relative_coordinates

def parse_dynamic_params(params, channels, final_channels, weight_nums, bias_nums):
    assert params.dim() == 2
    assert len(weight_nums) == len(bias_nums)
    assert params.size(1) == sum(weight_nums) + sum(bias_nums)
    num_instances = params.size(0)
    num_layers = len(weight_nums)

    params_splits = list(
        torch.split_with_sizes(params, weight_nums + bias_nums, dim=1))

    weight_splits = params_splits[:num_layers]
    bias_splits = params_splits[num_layers:]

    for l in range(num_layers):
        if l < num_layers - 1:
            # out_channels x in_channels x 1 x 1
            weight_splits[l] = weight_splits[l].reshape(
                num_instances * channels, -1, 1, 1)
            bias_splits[l] = bias_splits[l].reshape(num_instances * channels)
        else:
            # out_channels x in_channels x 1 x 1
            weight_splits[l] = weight_splits[l].reshape(
                num_instances * final_channels, -1, 1, 1)
            bias_splits[l] = bias_splits[l].reshape(num_instances * final_channels)

    return weight_splits, bias_splits

def dynamic_forward(features, weights, biases, num_instances, extra_relu=False):
    '''
    :param features
    :param weights: [w0, w1, ...]
    :param bias: [b0, b1, ...]
    :return:
    '''
    assert features.dim() == 4
    n_layers = len(weights)
    x = features
    for i, (w, b) in enumerate(zip(weights, biases)):
        x = F.conv2d(x,
                     w,
                     bias=b,
                     stride=1,
                     padding=0,
                     groups=num_instances)
        if i < n_layers - 1:
            x = F.relu(x)
    if extra_relu:
        return F.relu(x)
    return x

def aligned_bilinear(tensor, factor):
    assert tensor.dim() == 4
    assert factor >= 1
    assert int(factor) == factor
    if factor == 1:
        return tensor

    h, w = tensor.size()[2:]
    tensor = F.pad(tensor, pad=(0, 1, 0, 1), mode="replicate")
    oh = factor * h + 1
    ow = factor * w + 1
    tensor = F.interpolate(tensor,
                        size=(oh, ow),
                        mode='bilinear',
                        align_corners=True)
    tensor = F.pad(tensor,
                pad=(factor // 2, 0, factor // 2, 0),
                mode="replicate")
    return tensor[:, :, :oh - 1, :ow - 1]

def get_point_offset_target(gt_parse_points, centers, gt_bboxes):
    """get part point targets.

        Args:
            gt_parse_points (Tensor): N * C * 2, N is the number of pos samples,
                C is the category of dataset
            pos_centers (Tensor): N * 2.
            pos_gt_bboxes (list[Tensor]): centerness for each scale level, each
                is a 4D-tensor, the channel number is num_points * 1.
            valid_mask (Tensor): N * C * 1

        Returns:
            return the gt_points offsets targets
    """
    
    offsets = gt_parse_points - centers.unsqueeze(1)
    w = gt_bboxes[:, 2] - gt_bboxes[:, 0]
    h = gt_bboxes[:, 3] - gt_bboxes[:, 1]
    offsets[..., 0] /= w.unsqueeze(1)
    offsets[..., 1] /= h.unsqueeze(1)

    return offsets

def decode_point(pred_offsets, centers, gt_bboxes):
    """get part point targets.

        Args:
            gt_parse_points (Tensor): N * C * 2, N is the number of pos samples,
                C is the category of dataset
            pos_centers (Tensor): N * 2.
            pos_gt_bboxes (list[Tensor]): centerness for each scale level, each
                is a 4D-tensor, the channel number is num_points * 1.
            valid_mask (Tensor): N * C * 1

        Returns:
            return the gt_points offsets targets
    """
    
    w = gt_bboxes[:, 2] - gt_bboxes[:, 0]
    h = gt_bboxes[:, 3] - gt_bboxes[:, 1]
    pred_offsets[..., 0] *= w.unsqueeze(1)
    pred_offsets[..., 1] *= h.unsqueeze(1)
    pred_part_points = centers.unsqueeze(1) + pred_offsets

    return pred_part_points

def points2bbox(pts, y_first=True):
    """Converting the points set into bounding box.
    :param pts: the input points sets (fields), each points
        set (fields) is represented as 2n scalar.
    :param y_first: if y_first=True, the point set is represented as
        [y1, x1, y2, x2 ... yn, xn], otherwise the point set is
        represented as [x1, y1, x2, y2 ... xn, yn].
    :return: each points set is converting to a bbox [x1, y1, x2, y2].
    """
    pts_reshape = pts.view(pts.shape[0], -1, 2, *pts.shape[2:])
    pts_y = pts_reshape[:, :, 0, ...] if y_first else pts_reshape[:, :, 1,
                                                                    ...]
    pts_x = pts_reshape[:, :, 1, ...] if y_first else pts_reshape[:, :, 0,
                                                                    ...]

    bbox_left = pts_x.min(dim=1, keepdim=True)[0]
    bbox_right = pts_x.max(dim=1, keepdim=True)[0]
    bbox_up = pts_y.min(dim=1, keepdim=True)[0]
    bbox_bottom = pts_y.max(dim=1, keepdim=True)[0]
    bbox = torch.cat([bbox_left, bbox_up, bbox_right, bbox_bottom],
                        dim=1)
    return bbox
