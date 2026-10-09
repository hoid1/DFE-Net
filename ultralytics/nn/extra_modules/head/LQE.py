# LQE思想来源：https://arxiv.org/pdf/2011.12885
    
import os, sys   
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../../../..')

import math, copy
import numpy as np

import torch
import torch.nn as nn   
import torch.nn.functional as F   
import torch.nn.init as init

from ultralytics.nn.extra_modules.conv_module.pconv import Partial_Conv
   
from ultralytics.nn.modules.conv import Conv, DWConv, DSConv, autopad  
from ultralytics.nn.modules.head import Proto, Proto26, RealNVP, dist2rbox, NOT_MACOS14
from ultralytics.nn.modules.head import Detect, Segment, Segment26, Pose, Pose26, OBB, OBB26     
     
class MLP(nn.Module):  
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):  
        super().__init__() 
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        # self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))     
        self.layers = nn.ModuleList(nn.Conv2d(n, k, 1) for n, k in zip([input_dim] + h, h + [output_dim]))     
        self.act = nn.ReLU()

    def forward(self, x):   
        for i, layer in enumerate(self.layers):
            x = self.act(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x     
   
class LQE(nn.Module):
    """
    位置质量估计器 (Location Quality Estimator, LQE)
    用于评估和调整边界框预测的质量分数，结合分布统计信息提升精度     
    """     
    def __init__(self, k, hidden_dim, num_layers, reg_max):  
        """ 
        初始化 LQE 模块    
        参数:   
            k: 前 k 个最高概率值的数量，用于统计分析 
            hidden_dim: MLP隐藏层维度
            num_layers: MLP层数
            reg_max: 回归的最大值（边界框分布的最大范围）
        """
        super(LQE, self).__init__()  
        self.k = min(k, reg_max)
        self.reg_max = reg_max
        # 定义一个多层感知机（MLP），输入维度为 4*(k+1)，输出为 1    
        self.reg_conf = MLP(4 * (self.k + 1), hidden_dim, 1, num_layers)     
        # 初始化最后一层的偏置和权重为 0    
        init.constant_(self.reg_conf.layers[-1].bias, 0)  
        init.constant_(self.reg_conf.layers[-1].weight, 0)

    def forward(self, scores, pred_corners):
        """
        前向传播
        参数:
            scores: 初始分类得分 [B, num_classes, h, w]
            pred_corners: 预测的边界框角点分布 [B, 4*(reg_max), h, w]   
        返回: 
            调整后的质量分数
        """   
        # 计算 softmax 概率    
        B, C, H, W = pred_corners.size()
        prob = F.softmax(pred_corners.reshape(B, 4, self.reg_max, H, W), dim=2)
        # 提取前 k 个最高概率值及其索引    
        prob_topk, _ = prob.topk(self.k, dim=2)  
        # 将 top-k 概率及其均值拼接，作为统计特征     
        stat = torch.cat([prob_topk, prob_topk.mean(dim=2, keepdim=True)], dim=2) 
        # 通过 MLP 计算质量分数调整值  
        quality_score = self.reg_conf(stat.reshape(B, -1, H, W))
        # 将初始得分与质量调整值相加
        return scores + quality_score    
     
class Detect_LQE(Detect):     
    def __init__(self, nc = 80, reg_max=16, end2end=False, ch = ...):    
        super().__init__(nc, reg_max, end2end, ch)

        self.lqe = nn.ModuleList(LQE(4, 64, 2, self.reg_max) for x in ch)

        if end2end:
            self.one2one_lqe = copy.deepcopy(self.lqe)   
 
    @property     
    def one2many(self):    
        """Returns the one-to-many head components, here for v5/v5/v8/v9/11 backward compatibility."""    
        return dict(box_head=self.cv2, cls_head=self.cv3, lqe_head=self.lqe)

    @property
    def one2one(self):
        """Returns the one-to-one head components.""" 
        return dict(box_head=self.one2one_cv2, cls_head=self.one2one_cv3, lqe_head=self.one2one_lqe)  
     
    def forward_lqe_head( 
            self, x: list[torch.Tensor], box_head: torch.nn.Module = None, 
            cls_head: torch.nn.Module = None, lqe_head: torch.nn.Module = None 
        ) -> dict[str, torch.Tensor]:     
        if box_head is None or cls_head is None or lqe_head is None:  # for fused inference
            return dict()
    
        bs = x[0].shape[0]  # batch size
        boxes, scores = [], []
        for i in range(self.nl): 
            pred_corners = box_head[i](x[i])    
            pred_scores = lqe_head[i](cls_head[i](x[i]), pred_corners)     
     
            boxes.append(pred_corners.view(bs, 4 * self.reg_max, -1)) 
            scores.append(pred_scores.view(bs, self.nc, -1))
        boxes, scores = torch.cat(boxes, dim=-1), torch.cat(scores, dim=-1)

        return dict(boxes=boxes, scores=scores, feats=x)   
    
    def forward(  
        self, x: list[torch.Tensor]
    ) -> dict[str, torch.Tensor] | torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]:    
        """Concatenates and returns predicted bounding boxes and class probabilities."""  
        preds = self.forward_lqe_head(x, **self.one2many)
        if self.end2end:
            x_detach = [xi.detach() for xi in x]  
            one2one = self.forward_lqe_head(x_detach, **self.one2one)    
            preds = {"one2many": preds, "one2one": one2one}
        if self.training:   
            return preds    
        y = self._inference(preds["one2one"] if self.end2end else preds)     
        if self.end2end:    
            y = self.postprocess(y.permute(0, 2, 1))   
        return y if self.export else (y, preds)
  

class Segment_LQE(Detect_LQE):
    """YOLO Segment head with LQE-enhanced classification scores."""    
    
    def __init__(self, nc: int = 80, nm: int = 32, npr: int = 256, reg_max=16, end2end=False, ch: tuple = ()):
        super().__init__(nc, reg_max, end2end, ch)
        self.nm = nm   
        self.npr = npr
        self.proto = Proto(ch[0], self.npr, self.nm)

        c4 = max(ch[0] // 4, self.nm)   
        self.cv4 = nn.ModuleList(nn.Sequential(Conv(x, c4, 3), Conv(c4, c4, 3), nn.Conv2d(c4, self.nm, 1)) for x in ch)    
        if end2end:     
            self.one2one_cv4 = copy.deepcopy(self.cv4)

    @property    
    def one2many(self):
        return dict(box_head=self.cv2, cls_head=self.cv3, lqe_head=self.lqe, mask_head=self.cv4)     

    @property
    def one2one(self):
        return dict(
            box_head=self.one2one_cv2,    
            cls_head=self.one2one_cv3,
            lqe_head=self.one2one_lqe,   
            mask_head=self.one2one_cv4,
        )
    
    def forward(self, x: list[torch.Tensor]) -> tuple | list[torch.Tensor] | dict[str, torch.Tensor]:
        outputs = super().forward(x)     
        preds = outputs[1] if isinstance(outputs, tuple) else outputs  
        proto = self.proto(x[0])     
        if isinstance(preds, dict): 
            if self.end2end:   
                preds["one2many"]["proto"] = proto
                preds["one2one"]["proto"] = proto.detach()     
            else:   
                preds["proto"] = proto
        if self.training:
            return preds
        return (outputs, proto) if self.export else ((outputs[0], proto), preds)     
     
    def _inference(self, x: dict[str, torch.Tensor]) -> torch.Tensor:
        preds = super()._inference(x) 
        return torch.cat([preds, x["mask_coefficient"]], dim=1)

    def forward_lqe_head(    
        self,   
        x: list[torch.Tensor],     
        box_head: torch.nn.Module,    
        cls_head: torch.nn.Module,
        lqe_head: torch.nn.Module,    
        mask_head: torch.nn.Module,
    ) -> torch.Tensor:
        preds = super().forward_lqe_head(x, box_head, cls_head, lqe_head)
        if mask_head is not None:
            bs = x[0].shape[0]
            preds["mask_coefficient"] = torch.cat([mask_head[i](x[i]).view(bs, self.nm, -1) for i in range(self.nl)], 2)
        return preds

    def postprocess(self, preds: torch.Tensor) -> torch.Tensor:
        boxes, scores, mask_coefficient = preds.split([4, self.nc, self.nm], dim=-1)
        scores, conf, idx = self.get_topk_index(scores, self.max_det)    
        boxes = boxes.gather(dim=1, index=idx.repeat(1, 1, 4))   
        mask_coefficient = mask_coefficient.gather(dim=1, index=idx.repeat(1, 1, self.nm))  
        return torch.cat([boxes, scores, conf, mask_coefficient], dim=-1) 
    
    def fuse(self) -> None:
        self.cv2 = self.cv3 = self.cv4 = self.lqe = None

  
class Segment26_LQE(Segment_LQE):
    """YOLO26 Segment head with LQE-enhanced classification scores."""
    
    def __init__(self, nc: int = 80, nm: int = 32, npr: int = 256, reg_max=16, end2end=False, ch: tuple = ()):     
        super().__init__(nc, nm, npr, reg_max, end2end, ch)     
        self.proto = Proto26(ch, self.npr, self.nm, nc)    

    def forward(self, x: list[torch.Tensor]) -> tuple | list[torch.Tensor] | dict[str, torch.Tensor]:   
        outputs = Detect_LQE.forward(self, x)   
        preds = outputs[1] if isinstance(outputs, tuple) else outputs  
        proto = self.proto(x) 
        if isinstance(preds, dict): 
            if self.end2end:
                preds["one2many"]["proto"] = proto
                preds["one2one"]["proto"] = (
                    tuple(p.detach() for p in proto) if isinstance(proto, tuple) else proto.detach()
                )   
            else:     
                preds["proto"] = proto
        if self.training:
            return preds 
        return (outputs, proto) if self.export else ((outputs[0], proto), preds)

    def fuse(self) -> None:    
        super().fuse()
        if hasattr(self.proto, "fuse"):
            self.proto.fuse()    
  
 
class OBB_LQE(Detect_LQE):    
    """YOLO OBB head with LQE-enhanced classification scores."""     
  
    def __init__(self, nc: int = 80, ne: int = 1, reg_max=16, end2end=False, ch: tuple = ()):
        super().__init__(nc, reg_max, end2end, ch)   
        self.ne = ne

        c4 = max(ch[0] // 4, self.ne) 
        self.cv4 = nn.ModuleList(nn.Sequential(Conv(x, c4, 3), Conv(c4, c4, 3), nn.Conv2d(c4, self.ne, 1)) for x in ch)
        if end2end:     
            self.one2one_cv4 = copy.deepcopy(self.cv4) 

    @property  
    def one2many(self):  
        return dict(box_head=self.cv2, cls_head=self.cv3, lqe_head=self.lqe, angle_head=self.cv4)
     
    @property  
    def one2one(self): 
        return dict(  
            box_head=self.one2one_cv2,
            cls_head=self.one2one_cv3,    
            lqe_head=self.one2one_lqe,
            angle_head=self.one2one_cv4, 
        )

    def _inference(self, x: dict[str, torch.Tensor]) -> torch.Tensor:
        self.angle = x["angle"]    
        preds = super()._inference(x)
        return torch.cat([preds, x["angle"]], dim=1)

    def forward_lqe_head(     
        self,   
        x: list[torch.Tensor],
        box_head: torch.nn.Module,
        cls_head: torch.nn.Module,
        lqe_head: torch.nn.Module, 
        angle_head: torch.nn.Module,
    ) -> torch.Tensor:     
        preds = Detect_LQE.forward_lqe_head(self, x, box_head, cls_head, lqe_head)  
        if angle_head is not None:
            bs = x[0].shape[0]
            angle = torch.cat([angle_head[i](x[i]).view(bs, self.ne, -1) for i in range(self.nl)], 2)
            preds["angle"] = (angle.sigmoid() - 0.25) * math.pi
        return preds
     
    def decode_bboxes(self, bboxes: torch.Tensor, anchors: torch.Tensor) -> torch.Tensor:     
        return dist2rbox(bboxes, self.angle, anchors, dim=1)   
     
    def postprocess(self, preds: torch.Tensor) -> torch.Tensor:
        boxes, scores, angle = preds.split([4, self.nc, self.ne], dim=-1)
        scores, conf, idx = self.get_topk_index(scores, self.max_det)   
        boxes = boxes.gather(dim=1, index=idx.repeat(1, 1, 4))    
        angle = angle.gather(dim=1, index=idx.repeat(1, 1, self.ne))
        return torch.cat([boxes, scores, conf, angle], dim=-1)
 
    def fuse(self) -> None:
        self.cv2 = self.cv3 = self.cv4 = self.lqe = None
  

class OBB26_LQE(OBB_LQE):    
    """YOLO26 OBB head with raw-angle output and LQE-enhanced classification scores."""

    def forward_lqe_head(  
        self,   
        x: list[torch.Tensor],
        box_head: torch.nn.Module,
        cls_head: torch.nn.Module,
        lqe_head: torch.nn.Module,
        angle_head: torch.nn.Module,
    ) -> torch.Tensor:
        preds = Detect_LQE.forward_lqe_head(self, x, box_head, cls_head, lqe_head)
        if angle_head is not None:
            bs = x[0].shape[0]    
            preds["angle"] = torch.cat([angle_head[i](x[i]).view(bs, self.ne, -1) for i in range(self.nl)], 2)
        return preds    


class Pose_LQE(Detect_LQE):
    """YOLO Pose head with LQE-enhanced classification scores."""  

    def __init__(self, nc: int = 80, kpt_shape: tuple = (17, 3), reg_max=16, end2end=False, ch: tuple = ()):   
        super().__init__(nc, reg_max, end2end, ch)
        self.kpt_shape = kpt_shape    
        self.nk = kpt_shape[0] * kpt_shape[1] 
  
        c4 = max(ch[0] // 4, self.nk)
        self.cv4 = nn.ModuleList(nn.Sequential(Conv(x, c4, 3), Conv(c4, c4, 3), nn.Conv2d(c4, self.nk, 1)) for x in ch)     
        if end2end:   
            self.one2one_cv4 = copy.deepcopy(self.cv4)     

    @property    
    def one2many(self):  
        return dict(box_head=self.cv2, cls_head=self.cv3, lqe_head=self.lqe, pose_head=self.cv4)
    
    @property
    def one2one(self):  
        return dict(     
            box_head=self.one2one_cv2,
            cls_head=self.one2one_cv3,
            lqe_head=self.one2one_lqe,
            pose_head=self.one2one_cv4,   
        )   

    def _inference(self, x: dict[str, torch.Tensor]) -> torch.Tensor:
        preds = super()._inference(x) 
        return torch.cat([preds, self.kpts_decode(x["kpts"])], dim=1)
     
    def forward_lqe_head(
        self,
        x: list[torch.Tensor],  
        box_head: torch.nn.Module,  
        cls_head: torch.nn.Module,
        lqe_head: torch.nn.Module,
        pose_head: torch.nn.Module,
    ) -> torch.Tensor:  
        preds = Detect_LQE.forward_lqe_head(self, x, box_head, cls_head, lqe_head)
        if pose_head is not None:
            bs = x[0].shape[0]     
            preds["kpts"] = torch.cat([pose_head[i](x[i]).view(bs, self.nk, -1) for i in range(self.nl)], 2)    
        return preds 

    def postprocess(self, preds: torch.Tensor) -> torch.Tensor:
        boxes, scores, kpts = preds.split([4, self.nc, self.nk], dim=-1)
        scores, conf, idx = self.get_topk_index(scores, self.max_det)  
        boxes = boxes.gather(dim=1, index=idx.repeat(1, 1, 4))    
        kpts = kpts.gather(dim=1, index=idx.repeat(1, 1, self.nk))   
        return torch.cat([boxes, scores, conf, kpts], dim=-1)

    def fuse(self) -> None:
        self.cv2 = self.cv3 = self.cv4 = self.lqe = None  

    def kpts_decode(self, kpts: torch.Tensor) -> torch.Tensor:
        ndim = self.kpt_shape[1]     
        bs = kpts.shape[0] 
        if self.export:
            y = kpts.view(bs, *self.kpt_shape, -1)
            a = (y[:, :, :2] * 2.0 + (self.anchors - 0.5)) * self.strides     
            if ndim == 3:
                a = torch.cat((a, y[:, :, 2:3].sigmoid()), 2)
            return a.view(bs, self.nk, -1)    
        y = kpts.clone()   
        if ndim == 3:  
            if NOT_MACOS14:
                y[:, 2::ndim].sigmoid_()
            else:
                y[:, 2::ndim] = y[:, 2::ndim].sigmoid()   
        y[:, 0::ndim] = (y[:, 0::ndim] * 2.0 + (self.anchors[0] - 0.5)) * self.strides   
        y[:, 1::ndim] = (y[:, 1::ndim] * 2.0 + (self.anchors[1] - 0.5)) * self.strides
        return y
     
     
class Pose26_LQE(Pose_LQE):
    """YOLO26 Pose head with LQE-enhanced classification scores."""     

    def __init__(self, nc: int = 80, kpt_shape: tuple = (17, 3), reg_max=16, end2end=False, ch: tuple = ()):   
        super().__init__(nc, kpt_shape, reg_max, end2end, ch)
        self.flow_model = RealNVP()    

        c4 = max(ch[0] // 4, kpt_shape[0] * (kpt_shape[1] + 2))
        self.cv4 = nn.ModuleList(nn.Sequential(Conv(x, c4, 3), Conv(c4, c4, 3)) for x in ch) 
        self.cv4_kpts = nn.ModuleList(nn.Conv2d(c4, self.nk, 1) for _ in ch)
        self.nk_sigma = kpt_shape[0] * 2
        self.cv4_sigma = nn.ModuleList(nn.Conv2d(c4, self.nk_sigma, 1) for _ in ch)
   
        if end2end:
            self.one2one_cv4 = copy.deepcopy(self.cv4)
            self.one2one_cv4_kpts = copy.deepcopy(self.cv4_kpts) 
            self.one2one_cv4_sigma = copy.deepcopy(self.cv4_sigma)

    @property   
    def one2many(self):
        return dict(
            box_head=self.cv2, 
            cls_head=self.cv3,
            lqe_head=self.lqe,
            pose_head=self.cv4, 
            kpts_head=self.cv4_kpts,    
            kpts_sigma_head=self.cv4_sigma, 
        ) 

    @property
    def one2one(self): 
        return dict(
            box_head=self.one2one_cv2,
            cls_head=self.one2one_cv3, 
            lqe_head=self.one2one_lqe, 
            pose_head=self.one2one_cv4,  
            kpts_head=self.one2one_cv4_kpts,
            kpts_sigma_head=self.one2one_cv4_sigma,
        ) 
 
    def forward_lqe_head(
        self,   
        x: list[torch.Tensor],
        box_head: torch.nn.Module,
        cls_head: torch.nn.Module,
        lqe_head: torch.nn.Module,     
        pose_head: torch.nn.Module,  
        kpts_head: torch.nn.Module,
        kpts_sigma_head: torch.nn.Module,
    ) -> torch.Tensor:    
        preds = Detect_LQE.forward_lqe_head(self, x, box_head, cls_head, lqe_head)   
        if pose_head is not None:
            bs = x[0].shape[0]    
            features = [pose_head[i](x[i]) for i in range(self.nl)]
            preds["kpts"] = torch.cat([kpts_head[i](features[i]).view(bs, self.nk, -1) for i in range(self.nl)], 2)  
            if self.training:
                preds["kpts_sigma"] = torch.cat(
                    [kpts_sigma_head[i](features[i]).view(bs, self.nk_sigma, -1) for i in range(self.nl)], 2
                )  
        return preds

    def fuse(self) -> None:    
        super().fuse()
        self.cv4_kpts = self.cv4_sigma = self.flow_model = self.one2one_cv4_sigma = None   

    def kpts_decode(self, kpts: torch.Tensor) -> torch.Tensor:
        ndim = self.kpt_shape[1]
        bs = kpts.shape[0]    
        if self.export:
            y = kpts.view(bs, *self.kpt_shape, -1) 
            a = (y[:, :, :2] + self.anchors) * self.strides
            if ndim == 3:
                a = torch.cat((a, y[:, :, 2:3].sigmoid()), 2)  
            return a.view(bs, self.nk, -1)
        y = kpts.clone()  
        if ndim == 3:     
            if NOT_MACOS14:
                y[:, 2::ndim].sigmoid_()
            else:
                y[:, 2::ndim] = y[:, 2::ndim].sigmoid()
        y[:, 0::ndim] = (y[:, 0::ndim] + self.anchors[0]) * self.strides    
        y[:, 1::ndim] = (y[:, 1::ndim] + self.anchors[1]) * self.strides
        return y 
