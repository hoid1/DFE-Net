'''
本文件由 0526 版迁移而来 —— 检测头本体，逐字保留，未作任何修改
来源：nn/extra_modules/head.py:1656-1817
0526 版 FGIF 检测头。

请勿直接在 yaml 中引用本文件的类，yaml 请使用同目录 FGIF.py 中的同名适配类。
'''

import os, sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../../../..')

import warnings
warnings.filterwarnings('ignore')

import math
import torch.nn as nn
import torch
from ultralytics.nn.extra_modules.head.LSPCD import Conv_GN
from ultralytics.nn.modules.block import DFL
from ultralytics.utils.tal import dist2bbox, make_anchors


# ---- 原样迁移自 nn/extra_modules/head.py:1656-1817 ----
class Detect_FGIF(nn.Module):
    # Fine-grained iteration Free
    """YOLOv8 Detect head for detection models."""

    dynamic = False  # force grid reconstruction
    export = False  # export mode
    end2end = True  # end2end
    FGIF = True
    max_det = 300  # max_det
    shape = None
    anchors = torch.empty(0)  # init
    strides = torch.empty(0)  # init

    def __init__(self, nc=80, hidc=256, ch=()):
        """Initializes the YOLOv8 detection layer with specified number of classes and channels."""
        super().__init__()
        self.nc = nc  # number of classes
        self.nl = len(ch)  # number of detection layers
        self.reg_max = 16  # DFL channels (ch[0] // 16 to scale 4/8/12/16/20 for n/s/m/l/x)
        self.no = nc + self.reg_max * 4  # number of outputs per anchor
        self.stride = torch.zeros(self.nl)  # strides computed during build
        self.fg_layers = 3
        self.conv_align = nn.ModuleList(nn.Sequential(Conv_GN(x, hidc, 3)) for x in ch)
        self.conv_fineGrained = nn.ModuleList(nn.Sequential(Conv_GN(hidc, hidc, 3, g=8), Conv_GN(hidc, hidc, 1)) for _ in range(3))
        
        self.cv2_head = nn.ModuleList(
            nn.ModuleList(nn.Sequential(nn.Conv2d(hidc, 4 * self.reg_max, 1)) for x in ch) for _ in range(self.fg_layers)
        )
        self.cv3_head = nn.ModuleList(
            nn.ModuleList(nn.Sequential(nn.Conv2d(hidc, self.nc, 1)) for x in ch) for _ in range(self.fg_layers)
        )
        
        self.dfl = DFL(self.reg_max) if self.reg_max > 1 else nn.Identity()

    def forward(self, x):
        for i in range(self.nl):
            x[i] = self.conv_align[i](x[i])
        output_dict = {}
        last_box_pred, last_cls_pred = [None for i in range(self.nl)], [None for i in range(self.nl)]
        for fg_layer_id in range(self.fg_layers):
            out = []
            for i in range(self.nl):
                x[i] = self.conv_fineGrained[fg_layer_id](x[i])
                box_pred = self.cv2_head[fg_layer_id][i](x[i])
                cls_pred = self.cv3_head[fg_layer_id][i](x[i])
                if fg_layer_id == 0:
                    last_box_pred[i] = box_pred
                    last_cls_pred[i] = cls_pred
                else:
                    last_box_pred[i] = last_box_pred[i] + box_pred
                    last_cls_pred[i] = last_cls_pred[i] + cls_pred
                out.append(torch.cat((last_box_pred[i], last_cls_pred[i]), 1))
            output_dict[f'output_{fg_layer_id}'] = out
        if self.training:
            return output_dict
        
        y = self._inference(output_dict[f'output_{self.fg_layers - 1}'])
        y = self.postprocess(y.permute(0, 2, 1), self.max_det, self.nc)
        return y if self.export else (y, output_dict)
    
    # def forward(self, x):
    #     """Concatenates and returns predicted bounding boxes and class probabilities."""
    #     if self.end2end:
    #         return self.forward_end2end(x)

    #     for i in range(self.nl):
    #         x[i] = torch.cat((self.cv2[i](x[i]), self.cv3[i](x[i])), 1)
    #     if self.training:  # Training path
    #         return x
    #     y = self._inference(x)
    #     return y if self.export else (y, x)

    # def forward_end2end(self, x):
    #     """
    #     Performs forward pass of the v10Detect module.

    #     Args:
    #         x (tensor): Input tensor.

    #     Returns:
    #         (dict, tensor): If not in training mode, returns a dictionary containing the outputs of both one2many and one2one detections.
    #                        If in training mode, returns a dictionary containing the outputs of one2many and one2one detections separately.
    #     """
    #     # x_detach = [xi.detach() for xi in x]
    #     one2one = [
    #         torch.cat((self.one2one_cv2[i](x[i]), self.one2one_cv3[i](x[i])), 1) for i in range(self.nl)
    #     ]
    #     if hasattr(self, 'cv2') and hasattr(self, 'cv3'):
    #         for i in range(self.nl):
    #             x[i] = torch.cat((self.cv2[i](x[i]), self.cv3[i](x[i])), 1)
    #     if self.training:  # Training path
    #         return {"one2many": x, "one2one": one2one}

    #     y = self._inference(one2one)
    #     y = self.postprocess(y.permute(0, 2, 1), self.max_det, self.nc)
    #     return y if self.export else (y, {"one2many": x, "one2one": one2one})

    def _inference(self, x):
        """Decode predicted bounding boxes and class probabilities based on multiple-level feature maps."""
        # Inference path
        shape = x[0].shape  # BCHW
        x_cat = torch.cat([xi.view(shape[0], self.no, -1) for xi in x], 2)
        if self.dynamic or self.shape != shape:
            self.anchors, self.strides = (x.transpose(0, 1) for x in make_anchors(x, self.stride, 0.5))
            self.shape = shape

        if self.export and self.format in {"saved_model", "pb", "tflite", "edgetpu", "tfjs"}:  # avoid TF FlexSplitV ops
            box = x_cat[:, : self.reg_max * 4]
            cls = x_cat[:, self.reg_max * 4 :]
        else:
            box, cls = x_cat.split((self.reg_max * 4, self.nc), 1)

        if self.export and self.format in {"tflite", "edgetpu"}:
            # Precompute normalization factor to increase numerical stability
            # See https://github.com/ultralytics/ultralytics/issues/7371
            grid_h = shape[2]
            grid_w = shape[3]
            grid_size = torch.tensor([grid_w, grid_h, grid_w, grid_h], device=box.device).reshape(1, 4, 1)
            norm = self.strides / (self.stride[0] * grid_size)
            dbox = self.decode_bboxes(self.dfl(box) * norm, self.anchors.unsqueeze(0) * norm[:, :2])
        else:
            dbox = self.decode_bboxes(self.dfl(box), self.anchors.unsqueeze(0)) * self.strides

        return torch.cat((dbox, cls.sigmoid()), 1)

    def bias_init(self):
        """Initialize Detect() biases, WARNING: requires stride availability."""
        m = self  # self.model[-1]  # Detect() module
        # cf = torch.bincount(torch.tensor(np.concatenate(dataset.labels, 0)[:, 0]).long(), minlength=nc) + 1
        # ncf = math.log(0.6 / (m.nc - 0.999999)) if cf is None else torch.log(cf / cf.sum())  # nominal class frequency
        for a_s, b_s in zip(m.cv2_head, m.cv3_head):  # from
            for a, b, s in zip(a_s, b_s, m.stride):  # from
                a[-1].bias.data[:] = 1.0  # box
                b[-1].bias.data[: m.nc] = math.log(5 / m.nc / (640 / s) ** 2)  # cls (.01 objects, 80 classes, 640 img)

    def decode_bboxes(self, bboxes, anchors):
        """Decode bounding boxes."""
        return dist2bbox(bboxes, anchors, xywh=not self.end2end, dim=1)

    @staticmethod
    def postprocess(preds: torch.Tensor, max_det: int, nc: int = 80):
        """
        Post-processes YOLO model predictions.

        Args:
            preds (torch.Tensor): Raw predictions with shape (batch_size, num_anchors, 4 + nc) with last dimension
                format [x, y, w, h, class_probs].
            max_det (int): Maximum detections per image.
            nc (int, optional): Number of classes. Default: 80.

        Returns:
            (torch.Tensor): Processed predictions with shape (batch_size, min(max_det, num_anchors), 6) and last
                dimension format [x, y, w, h, max_class_prob, class_index].
        """
        batch_size, anchors, _ = preds.shape  # i.e. shape(16,8400,84)
        boxes, scores = preds.split([4, nc], dim=-1)
        index = scores.amax(dim=-1).topk(min(max_det, anchors))[1].unsqueeze(-1)
        boxes = boxes.gather(dim=1, index=index.repeat(1, 1, 4))
        scores = scores.gather(dim=1, index=index.repeat(1, 1, nc))
        scores, index = scores.flatten(1).topk(min(max_det, anchors))
        i = torch.arange(batch_size)[..., None]  # batch indices
        return torch.cat([boxes[i, index // nc], scores[..., None], (index % nc)[..., None].float()], dim=-1)
