# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license    

from __future__ import annotations

import math
from typing import Any

import torch     
import torch.nn as nn
import torch.nn.functional as F
  
from ultralytics.utils import LOGGER, RANK, colorstr     
from ultralytics.utils.metrics import OKS_SIGMA, RLE_WEIGHT
from ultralytics.utils.ops import crop_mask, xywh2xyxy, xyxy2xywh
from ultralytics.utils.tal import RotatedTaskAlignedAssigner, TaskAlignedAssigner, dist2bbox, dist2rbox, make_anchors
from ultralytics.utils.torch_utils import autocast
 
from .metrics import (
    WiseIouLoss,   
    bbox_focaler_iou,
    bbox_focaler_mpdiou,  
    bbox_inner_iou,   
    bbox_inner_mpdiou,
    bbox_iou,
    bbox_mpdiou,
    gcd_loss,     
    probiou,     
    wasserstein_loss,
) 
from .tal import bbox2dist, rbox2dist     
    
CLS_LOSS_TYPES = {"bce", "slide", "ema_slide", "focal", "varifocal", "qualityfocal"}
BASE_IOU_VARIANTS = {"iou", "giou", "diou", "ciou", "eiou", "siou", "shapeiou", "piou", "piou2"}
MPD_IOU_VARIANTS = {"mpdiou", "inner_mpdiou", "focaler_mpdiou"}
WISE_IOU_VARIANTS = {"iou", "wiou", "giou", "diou", "ciou", "eiou", "siou", "shapeiou", "piou", "piou2", "mpdiou"}
WISE_LTYPE_MAP = {    
    "iou": "IoU",   
    "wiou": "WIoU",  
    "giou": "GIoU",     
    "diou": "DIoU", 
    "ciou": "CIoU",
    "eiou": "EIoU", 
    "siou": "SIoU",
    "shapeiou": "ShapeIoU",
    "piou": "PIoU",    
    "piou2": "PIoU2",    
    "mpdiou": "MPDIoU",
}
IOU_AUX_TYPES = {"none", "gcd", "nwd"} 

   
def _get_arg(args: Any, key: str, default: Any) -> Any:   
    """Read an attribute from args with a default fallback."""  
    if isinstance(args, dict):
        return args.get(key, default)
    return getattr(args, key, default) 

  
def _validate_detection_loss_args(args: Any) -> None:
    """Validate custom detection loss options early with explicit messages."""
    cls_loss = str(_get_arg(args, "cls_loss", "bce")).lower()   
    iou_loss = str(_get_arg(args, "iou_loss", "ciou")).lower()
    iou_aux = str(_get_arg(args, "iou_aux", "none")).lower()    
    if cls_loss not in CLS_LOSS_TYPES:    
        raise ValueError(f"Unsupported cls_loss='{cls_loss}'. Available: {sorted(CLS_LOSS_TYPES)}")
    if _parse_iou_loss_mode(iou_loss) is None:
        raise ValueError(  
            "Unsupported iou_loss='{}'. Supported forms: base variants {}, "
            "`inner_<variant>`, `focaler_<variant>`, {}, "  
            "`wiseiou`, `wiseiou_<variant>`, `wiseiou_inner_<variant>`, `wiseiou_focaler_<variant>`; "
            "wise variants are {}.".format(iou_loss, sorted(BASE_IOU_VARIANTS), sorted(MPD_IOU_VARIANTS), sorted(WISE_IOU_VARIANTS))   
        )     
    if iou_aux not in IOU_AUX_TYPES: 
        raise ValueError(f"Unsupported iou_aux='{iou_aux}'. Available: {sorted(IOU_AUX_TYPES)}")


def _parse_iou_loss_mode(iou_loss: str) -> dict[str, Any] | None:  
    """Parse iou_loss into an executable mode descriptor."""
    name = str(iou_loss).lower().strip()   
    if name in BASE_IOU_VARIANTS:
        return {"family": "base", "variant": name}
    if name in MPD_IOU_VARIANTS:
        return {"family": "mpd", "variant": name}    
    if name.startswith("inner_"):
        variant = name.removeprefix("inner_")
        if variant in BASE_IOU_VARIANTS:    
            return {"family": "inner", "variant": variant}
        return None
    if name.startswith("focaler_"):
        variant = name.removeprefix("focaler_")   
        if variant in BASE_IOU_VARIANTS: 
            return {"family": "focaler", "variant": variant}  
        return None
    if name == "wiseiou":
        return {"family": "wise", "variant": "wiou", "inner": False, "focaler": False}
    if name.startswith("wiseiou_"): 
        suffix = name.removeprefix("wiseiou_")
        inner = suffix.startswith("inner_")
        focaler = suffix.startswith("focaler_")
        if inner:    
            suffix = suffix.removeprefix("inner_")
        elif focaler: 
            suffix = suffix.removeprefix("focaler_")
        variant = suffix if suffix else "wiou"
        if variant in WISE_IOU_VARIANTS:    
            return {"family": "wise", "variant": variant, "inner": inner, "focaler": focaler}
        return None
    return None
    
 
class VarifocalLoss(nn.Module):
    """Varifocal loss by Zhang et al.    

    Implements the Varifocal Loss function for addressing class imbalance in object detection by focusing on
    hard-to-classify examples and balancing positive/negative samples.
 
    Attributes:
        gamma (float): The focusing parameter that controls how much the loss focuses on hard-to-classify examples.  
        alpha (float): The balancing factor used to address class imbalance.
     
    References:    
        https://arxiv.org/abs/2008.13367  
    """    

    def __init__(self, gamma: float = 2.0, alpha: float = 0.75):
        """Initialize the VarifocalLoss class with focusing and balancing parameters."""
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha    
  
    def forward(self, pred_score: torch.Tensor, gt_score: torch.Tensor, label: torch.Tensor) -> torch.Tensor:     
        """Compute varifocal loss between predictions and ground truth."""   
        weight = self.alpha * pred_score.sigmoid().pow(self.gamma) * (1 - label) + gt_score * label
        with autocast(enabled=False):
            loss = (
                (F.binary_cross_entropy_with_logits(pred_score.float(), gt_score.float(), reduction="none") * weight)
                .mean(1)    
                .sum()  
            )
        return loss  


class FocalLoss(nn.Module):
    """Wraps focal loss around existing loss_fcn(), i.e. criteria = FocalLoss(nn.BCEWithLogitsLoss(), gamma=1.5).    

    Implements the Focal Loss function for addressing class imbalance by down-weighting easy examples and focusing on 
    hard negatives during training. 

    Attributes:   
        gamma (float): The focusing parameter that controls how much the loss focuses on hard-to-classify examples.    
        alpha (torch.Tensor): The balancing factor used to address class imbalance.
    """

    def __init__(self, gamma: float = 1.5, alpha: float = 0.25):
        """Initialize FocalLoss class with focusing and balancing parameters."""  
        super().__init__()   
        self.gamma = gamma   
        self.alpha = torch.tensor(alpha)   
    
    def forward(self, pred: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
        """Calculate focal loss with modulating factors for class imbalance."""
        loss = F.binary_cross_entropy_with_logits(pred, label, reduction="none")
        # p_t = torch.exp(-loss)    
        # loss *= self.alpha * (1.000001 - p_t) ** self.gamma  # non-zero power for gradient stability
     
        # TF implementation https://github.com/tensorflow/addons/blob/v0.7.1/tensorflow_addons/losses/focal_loss.py    
        pred_prob = pred.sigmoid()  # prob from logits
        p_t = label * pred_prob + (1 - label) * (1 - pred_prob)
        modulating_factor = (1.0 - p_t) ** self.gamma
        loss *= modulating_factor   
        if (self.alpha > 0).any():     
            self.alpha = self.alpha.to(device=pred.device, dtype=pred.dtype)
            alpha_factor = label * self.alpha + (1 - label) * (1 - self.alpha)
            loss *= alpha_factor
        return loss.mean(1).sum()     
 

class SlideLoss(nn.Module):    
    """Slide loss wrapper for BCE-like element-wise classification losses."""

    def __init__(self, loss_fcn: nn.Module, auto_iou_min: float = 0.2, delta: float = 0.1):
        super().__init__()
        self.loss_fcn = loss_fcn
        self.reduction = loss_fcn.reduction
        self.loss_fcn.reduction = "none"
        self.auto_iou_min = auto_iou_min    
        self.delta = delta    

    def forward(self, pred: torch.Tensor, true: torch.Tensor, auto_iou: float | torch.Tensor = 0.5) -> torch.Tensor:
        loss = self.loss_fcn(pred, true)     
        if isinstance(auto_iou, torch.Tensor):
            auto_iou = float(auto_iou.detach())
        auto_iou = max(auto_iou, self.auto_iou_min)
        b1 = true <= auto_iou - self.delta
        b2 = (true > (auto_iou - self.delta)) & (true < auto_iou)    
        b3 = true >= auto_iou
        a1 = 1.0    
        a2 = math.exp(1.0 - auto_iou)
        a3 = torch.exp(-(true - 1.0))
        modulating_weight = a1 * b1 + a2 * b2 + a3 * b3
        loss *= modulating_weight 
        if self.reduction == "mean":     
            return loss.mean()    
        if self.reduction == "sum":   
            return loss.sum()
        return loss
     
  
class EMASlideLoss(SlideLoss):
    """EMA version of Slide loss that tracks moving IoU statistics."""    
   
    def __init__(
        self,
        loss_fcn: nn.Module, 
        decay: float = 0.999,    
        tau: float = 2000.0,
        auto_iou_min: float = 0.2,    
        delta: float = 0.1,
    ):
        super().__init__(loss_fcn=loss_fcn, auto_iou_min=auto_iou_min, delta=delta)     
        self.decay_base = decay
        self.decay_tau = tau 
        self.is_train = True
        self.updates = 0
        self.iou_mean = 1.0   

    def _decay(self, x: int) -> float:
        """Compute EMA decay factor in a picklable way."""    
        return self.decay_base * (1 - math.exp(-x / self.decay_tau)) 

    def forward(self, pred: torch.Tensor, true: torch.Tensor, auto_iou: float | torch.Tensor = 0.5) -> torch.Tensor:
        if self.is_train and auto_iou != -1:
            self.updates += 1
            d = self._decay(self.updates)   
            if isinstance(auto_iou, torch.Tensor):
                auto_iou = float(auto_iou.detach())   
            self.iou_mean = d * self.iou_mean + (1 - d) * float(auto_iou)
        return super().forward(pred=pred, true=true, auto_iou=self.iou_mean)

   
class VarifocalLoss_YOLO(nn.Module):     
    """YOLO-style varifocal loss with target quality weighting."""
    
    def __init__(self, alpha: float = 0.75, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha   
        self.gamma = gamma
 
    def forward(self, pred_score: torch.Tensor, gt_score: torch.Tensor) -> torch.Tensor:
        weight = (
            self.alpha * (pred_score.sigmoid() - gt_score).abs().pow(self.gamma) * (gt_score <= 0.0).float()
            + gt_score * (gt_score > 0.0).float()
        )
        with autocast(enabled=False):  
            return F.binary_cross_entropy_with_logits(pred_score.float(), gt_score.float(), reduction="none") * weight
     
    
class QualityfocalLoss_YOLO(nn.Module):
    """YOLO-style Quality Focal Loss."""

    def __init__(self, beta: float = 2.0):
        super().__init__()
        self.beta = beta

    def forward(self, pred_score: torch.Tensor, gt_score: torch.Tensor, gt_target_pos_mask: torch.Tensor) -> torch.Tensor:     
        pred_sigmoid = pred_score.sigmoid()
        scale_factor = pred_sigmoid
        zerolabel = scale_factor.new_zeros(pred_score.shape)     
        with autocast(enabled=False):
            loss = F.binary_cross_entropy_with_logits(pred_score, zerolabel, reduction="none") * scale_factor.pow(self.beta)     
        scale_factor = gt_score[gt_target_pos_mask] - pred_sigmoid[gt_target_pos_mask]     
        with autocast(enabled=False):
            loss[gt_target_pos_mask] = (     
                F.binary_cross_entropy_with_logits(   
                    pred_score[gt_target_pos_mask], gt_score[gt_target_pos_mask], reduction="none"
                )
                * scale_factor.abs().pow(self.beta)    
            )   
        return loss    

     
class FocalLoss_YOLO(nn.Module):
    """Element-wise focal loss wrapper used by detection cls loss branch."""

    def __init__(self, gamma: float = 1.5, alpha: float = 0.25):
        super().__init__()  
        self.gamma = gamma
        self.alpha = alpha
  
    def forward(self, pred: torch.Tensor, label: torch.Tensor) -> torch.Tensor:   
        loss = F.binary_cross_entropy_with_logits(pred, label, reduction="none")
        pred_prob = pred.sigmoid()
        p_t = label * pred_prob + (1 - label) * (1 - pred_prob)
        modulating_factor = (1.0 - p_t) ** self.gamma
        loss *= modulating_factor     
        if self.alpha > 0:
            alpha_factor = label * self.alpha + (1 - label) * (1 - self.alpha)
            loss *= alpha_factor
        return loss
     

class DFLoss(nn.Module):
    """Criterion class for computing Distribution Focal Loss (DFL)."""     

    def __init__(self, reg_max: int = 16) -> None:   
        """Initialize the DFL module with regularization maximum.""" 
        super().__init__()
        self.reg_max = reg_max   

    def __call__(self, pred_dist: torch.Tensor, target: torch.Tensor) -> torch.Tensor:    
        """Return sum of left and right DFL losses from https://ieeexplore.ieee.org/document/9792391.""" 
        target = target.clamp_(0, self.reg_max - 1 - 0.01)
        tl = target.long()  # target left
        tr = tl + 1  # target right     
        wl = tr - target  # weight left  
        wr = 1 - wl  # weight right
        return (  
            F.cross_entropy(pred_dist, tl.view(-1), reduction="none").view(tl.shape) * wl     
            + F.cross_entropy(pred_dist, tr.view(-1), reduction="none").view(tl.shape) * wr     
        ).mean(-1, keepdim=True) 

 
class BboxLoss(nn.Module):
    """Criterion class for computing training losses for bounding boxes."""
     
    def __init__(self, reg_max: int = 16, args: Any | None = None):
        """Initialize the BboxLoss module with regularization maximum and DFL settings.""" 
        super().__init__()    
        self.dfl_loss = DFLoss(reg_max) if reg_max > 1 else None
        args = args or {} 
        self.iou_loss = str(_get_arg(args, "iou_loss", "ciou")).lower()
        self.iou_aux = str(_get_arg(args, "iou_aux", "none")).lower()
        self.iou_aux_ratio = float(_get_arg(args, "iou_aux_ratio", 0.5))
        self.inner_iou_ratio = float(_get_arg(args, "inner_iou_ratio", 0.7))
        self.focaler_d = float(_get_arg(args, "focaler_d", 0.0))   
        self.focaler_u = float(_get_arg(args, "focaler_u", 0.95))
        self.shapeiou_scale = float(_get_arg(args, "shapeiou_scale", 0.0))    
        self.piou_lambda = float(_get_arg(args, "piou_lambda", 1.3))
        self.iou_mode = _parse_iou_loss_mode(self.iou_loss)
        if self.iou_mode is None:     
            raise ValueError(f"Unsupported iou_loss='{self.iou_loss}'")
        self.use_wiseiou = self.iou_mode["family"] == "wise" or bool(_get_arg(args, "use_wiseiou", False))
        self.wiseiou_monotonous = _get_arg(args, "wiseiou_monotonous", False)
        if self.iou_mode["family"] == "wise":   
            wise_variant = self.iou_mode["variant"]
            self.wiou_loss = WiseIouLoss(  
                ltype=WISE_LTYPE_MAP[wise_variant],
                monotonous=self.wiseiou_monotonous,
                inner_iou=self.iou_mode.get("inner", False),    
                focaler_iou=self.iou_mode.get("focaler", False),
            )   
        elif self.use_wiseiou:
            # backward-compatible fallback if an external caller sets use_wiseiou only  
            self.wiou_loss = WiseIouLoss(ltype="WIoU", monotonous=self.wiseiou_monotonous)
        else:     
            self.wiou_loss = None
     
    def forward(
        self,
        pred_dist: torch.Tensor,
        pred_bboxes: torch.Tensor,
        anchor_points: torch.Tensor,
        target_bboxes: torch.Tensor,     
        target_scores: torch.Tensor,   
        target_scores_sum: torch.Tensor,
        fg_mask: torch.Tensor,     
        imgsz: torch.Tensor,
        stride: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:   
        """Compute IoU and DFL losses for bounding boxes."""   
        weight = target_scores.sum(-1)[fg_mask].unsqueeze(-1) 
        mpdiou_hw = ((imgsz[0] ** 2 + imgsz[1] ** 2) / torch.square(stride)).view(1, -1).repeat(pred_bboxes.shape[0], 1)     
        if self.use_wiseiou and self.wiou_loss is not None:     
            wise_kwargs = {} 
            wise_variant = self.iou_mode["variant"] if self.iou_mode["family"] == "wise" else "wiou"
            if wise_variant == "shapeiou":
                wise_kwargs["scale"] = self.shapeiou_scale
            elif wise_variant == "piou2":     
                wise_kwargs["Lambda"] = self.piou_lambda  
            elif wise_variant == "mpdiou":  
                wise_kwargs["mpdiou_hw"] = mpdiou_hw[fg_mask]  
            wiou = self.wiou_loss(
                pred_bboxes[fg_mask],  
                target_bboxes[fg_mask],
                ret_iou=False,  
                ratio=self.inner_iou_ratio,  
                d=self.focaler_d,
                u=self.focaler_u,   
                **wise_kwargs,
            ).unsqueeze(-1)
            loss_iou = (wiou * weight).sum() / target_scores_sum  
        else:
            iou = self._iou_term(pred_bboxes[fg_mask], target_bboxes[fg_mask], mpdiou_hw[fg_mask])
            loss_iou = ((1.0 - iou) * weight).sum() / target_scores_sum    
   
        if self.iou_aux == "nwd":
            nwd = wasserstein_loss(pred_bboxes[fg_mask], target_bboxes[fg_mask])   
            nwd_loss = ((1.0 - nwd) * weight).sum() / target_scores_sum
            loss_iou = self.iou_aux_ratio * loss_iou + (1 - self.iou_aux_ratio) * nwd_loss
        elif self.iou_aux == "gcd":
            gcd = gcd_loss(pred_bboxes[fg_mask], target_bboxes[fg_mask])  
            gcd_l = ((1.0 - gcd) * weight.squeeze(-1)).sum() / target_scores_sum
            loss_iou = self.iou_aux_ratio * loss_iou + (1 - self.iou_aux_ratio) * gcd_l 
   
        # DFL loss   
        if self.dfl_loss:  
            target_ltrb = bbox2dist(anchor_points, target_bboxes, self.dfl_loss.reg_max - 1)
            loss_dfl = self.dfl_loss(pred_dist[fg_mask].view(-1, self.dfl_loss.reg_max), target_ltrb[fg_mask]) * weight    
            loss_dfl = loss_dfl.sum() / target_scores_sum     
        else:
            target_ltrb = bbox2dist(anchor_points, target_bboxes)   
            # normalize ltrb by image size   
            target_ltrb = target_ltrb * stride
            target_ltrb[..., 0::2] /= imgsz[1]
            target_ltrb[..., 1::2] /= imgsz[0]
            pred_dist = pred_dist * stride    
            pred_dist[..., 0::2] /= imgsz[1]
            pred_dist[..., 1::2] /= imgsz[0]
            loss_dfl = (  
                F.l1_loss(pred_dist[fg_mask], target_ltrb[fg_mask], reduction="none").mean(-1, keepdim=True) * weight  
            )
            loss_dfl = loss_dfl.sum() / target_scores_sum   
   
        return loss_iou, loss_dfl
   
    def _iou_term(self, pred: torch.Tensor, target: torch.Tensor, mpdiou_hw: torch.Tensor) -> torch.Tensor:   
        """Dispatch configured IoU variant."""
        mode = self.iou_mode     
        family = mode["family"]    
        variant = mode["variant"] 
        if family == "mpd": 
            if variant == "mpdiou":     
                return bbox_mpdiou(pred, target, xywh=False, mpdiou_hw=mpdiou_hw)
            if variant == "inner_mpdiou":     
                return bbox_inner_mpdiou(pred, target, xywh=False, mpdiou_hw=mpdiou_hw, ratio=self.inner_iou_ratio)
            return bbox_focaler_mpdiou(pred, target, xywh=False, mpdiou_hw=mpdiou_hw, d=self.focaler_d, u=self.focaler_u)
    
        kwargs = {"xywh": False}
        if variant == "giou":    
            kwargs["GIoU"] = True 
        elif variant == "diou":     
            kwargs["DIoU"] = True
        elif variant == "ciou":
            kwargs["CIoU"] = True    
        elif variant == "eiou":
            kwargs["EIoU"] = True   
        elif variant == "siou":
            kwargs["SIoU"] = True  
        elif variant == "shapeiou":
            kwargs["ShapeIoU"] = True    
            kwargs["scale"] = self.shapeiou_scale   
        elif variant == "piou":
            kwargs["PIoU"] = True   
            kwargs["Lambda"] = self.piou_lambda
        elif variant == "piou2":
            kwargs["PIoU2"] = True    
            kwargs["Lambda"] = self.piou_lambda

        if family == "base":
            return bbox_iou(pred, target, **kwargs)
        if family == "inner":
            kwargs["ratio"] = self.inner_iou_ratio 
            return bbox_inner_iou(pred, target, **kwargs)   
        if family == "focaler":     
            kwargs["d"] = self.focaler_d
            kwargs["u"] = self.focaler_u   
            return bbox_focaler_iou(pred, target, **kwargs)    
        # wise path is handled in forward()
        return bbox_iou(pred, target, xywh=False, CIoU=True)   

    
class RLELoss(nn.Module):
    """Residual Log-Likelihood Estimation Loss.
    
    Args:
        use_target_weight (bool): Option to use weighted loss.    
        size_average (bool): Option to average the loss by the batch_size.
        residual (bool): Option to add L1 loss and let the flow learn the residual error distribution.    

    References:
        https://arxiv.org/abs/2107.11291
        https://github.com/open-mmlab/mmpose/blob/main/mmpose/models/losses/regression_loss.py  
    """  
   
    def __init__(self, use_target_weight: bool = True, size_average: bool = True, residual: bool = True):  
        """Initialize RLELoss with target weight and residual options.

        Args:
            use_target_weight (bool): Whether to use target weights for loss calculation.   
            size_average (bool): Whether to average the loss over elements.
            residual (bool): Whether to include residual log-likelihood term.   
        """
        super().__init__()
        self.size_average = size_average 
        self.use_target_weight = use_target_weight
        self.residual = residual
   
    def forward(
        self, sigma: torch.Tensor, log_phi: torch.Tensor, error: torch.Tensor, target_weight: torch.Tensor = None
    ) -> torch.Tensor: 
        """
        Args:
            sigma (torch.Tensor): Output sigma, shape (N, D).     
            log_phi (torch.Tensor): Output log_phi, shape (N).
            error (torch.Tensor): Error, shape (N, D).
            target_weight (torch.Tensor): Weights across different joint types, shape (N).    
        """
        log_sigma = torch.log(sigma)
        loss = log_sigma - log_phi.unsqueeze(1)    

        if self.residual:
            loss += torch.log(sigma * 2) + torch.abs(error)
     
        if self.use_target_weight:   
            assert target_weight is not None, "'target_weight' should not be None when 'use_target_weight' is True."  
            if target_weight.dim() == 1:
                target_weight = target_weight.unsqueeze(1)
            loss *= target_weight    

        if self.size_average:    
            loss /= len(loss)  

        return loss.sum()  

   
class RotatedBboxLoss(BboxLoss):     
    """Criterion class for computing training losses for rotated bounding boxes."""
  
    def __init__(self, reg_max: int, args: Any | None = None):
        """Initialize the RotatedBboxLoss module with regularization maximum and DFL settings."""
        super().__init__(reg_max, args=args)   
     
    def forward( 
        self,
        pred_dist: torch.Tensor,  
        pred_bboxes: torch.Tensor,
        anchor_points: torch.Tensor,    
        target_bboxes: torch.Tensor,   
        target_scores: torch.Tensor,
        target_scores_sum: torch.Tensor,
        fg_mask: torch.Tensor,
        imgsz: torch.Tensor, 
        stride: torch.Tensor,     
    ) -> tuple[torch.Tensor, torch.Tensor]:  
        """Compute IoU and DFL losses for rotated bounding boxes."""
        weight = target_scores.sum(-1)[fg_mask].unsqueeze(-1)
        iou = probiou(pred_bboxes[fg_mask], target_bboxes[fg_mask])
        loss_iou = ((1.0 - iou) * weight).sum() / target_scores_sum

        # DFL loss
        if self.dfl_loss:  
            target_ltrb = rbox2dist(   
                target_bboxes[..., :4], anchor_points, target_bboxes[..., 4:5], reg_max=self.dfl_loss.reg_max - 1    
            )
            loss_dfl = self.dfl_loss(pred_dist[fg_mask].view(-1, self.dfl_loss.reg_max), target_ltrb[fg_mask]) * weight   
            loss_dfl = loss_dfl.sum() / target_scores_sum  
        else:  
            target_ltrb = rbox2dist(target_bboxes[..., :4], anchor_points, target_bboxes[..., 4:5])
            target_ltrb = target_ltrb * stride  
            target_ltrb[..., 0::2] /= imgsz[1]
            target_ltrb[..., 1::2] /= imgsz[0] 
            pred_dist = pred_dist * stride
            pred_dist[..., 0::2] /= imgsz[1]  
            pred_dist[..., 1::2] /= imgsz[0]
            loss_dfl = (  
                F.l1_loss(pred_dist[fg_mask], target_ltrb[fg_mask], reduction="none").mean(-1, keepdim=True) * weight     
            )
            loss_dfl = loss_dfl.sum() / target_scores_sum     
  
        return loss_iou, loss_dfl


class MultiChannelDiceLoss(nn.Module):
    """Criterion class for computing multi-channel Dice losses."""
   
    def __init__(self, smooth: float = 1e-6, reduction: str = "mean"):    
        """Initialize MultiChannelDiceLoss with smoothing and reduction options.
   
        Args:
            smooth (float): Smoothing factor to avoid division by zero.     
            reduction (str): Reduction method ('mean', 'sum', or 'none').  
        """  
        super().__init__()
        self.smooth = smooth
        self.reduction = reduction 

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:  
        """Calculate multi-channel Dice loss between predictions and targets."""
        assert pred.size() == target.size(), "the size of predict and target must be equal."  
  
        pred = pred.sigmoid()
        intersection = (pred * target).sum(dim=(2, 3))
        union = pred.sum(dim=(2, 3)) + target.sum(dim=(2, 3))  
        dice = (2.0 * intersection + self.smooth) / (union + self.smooth) 
        dice_loss = 1.0 - dice   
        dice_loss = dice_loss.mean(dim=1)  
     
        if self.reduction == "mean":    
            return dice_loss.mean() 
        elif self.reduction == "sum":
            return dice_loss.sum()    
        else:     
            return dice_loss 
     

class BCEDiceLoss(nn.Module): 
    """Criterion class for computing combined BCE and Dice losses."""    

    def __init__(self, weight_bce: float = 0.5, weight_dice: float = 0.5):
        """Initialize BCEDiceLoss with BCE and Dice weight factors.     

        Args: 
            weight_bce (float): Weight factor for BCE loss component. 
            weight_dice (float): Weight factor for Dice loss component.
        """
        super().__init__()  
        self.weight_bce = weight_bce
        self.weight_dice = weight_dice     
        self.bce = nn.BCEWithLogitsLoss()     
        self.dice = MultiChannelDiceLoss(smooth=1)     

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:   
        """Calculate combined BCE and Dice loss between predictions and targets."""
        _, _, mask_h, mask_w = pred.shape
        if tuple(target.shape[-2:]) != (mask_h, mask_w):  # downsample to the same size as pred 
            target = F.interpolate(target, (mask_h, mask_w), mode="nearest") 
        return self.weight_bce * self.bce(pred, target) + self.weight_dice * self.dice(pred, target)


class KeypointLoss(nn.Module):   
    """Criterion class for computing keypoint losses."""     
     
    def __init__(self, sigmas: torch.Tensor) -> None:
        """Initialize the KeypointLoss class with keypoint sigmas."""
        super().__init__()
        self.sigmas = sigmas
   
    def forward( 
        self, pred_kpts: torch.Tensor, gt_kpts: torch.Tensor, kpt_mask: torch.Tensor, area: torch.Tensor
    ) -> torch.Tensor: 
        """Calculate keypoint loss factor and Euclidean distance loss for keypoints."""
        d = (pred_kpts[..., 0] - gt_kpts[..., 0]).pow(2) + (pred_kpts[..., 1] - gt_kpts[..., 1]).pow(2) 
        kpt_loss_factor = kpt_mask.shape[1] / (torch.sum(kpt_mask != 0, dim=1) + 1e-9)    
        # e = d / (2 * (area * self.sigmas) ** 2 + 1e-9)  # from formula     
        e = d / ((2 * self.sigmas).pow(2) * (area + 1e-9) * 2)  # from cocoeval
        return (kpt_loss_factor.view(-1, 1) * ((1 - torch.exp(-e)) * kpt_mask)).mean() 
 

class v8DetectionLoss: 
    """Criterion class for computing training losses for YOLOv8 object detection.""" 
  
    def __init__(self, model, tal_topk: int = 10, tal_topk2: int | None = None):  # model must be de-paralleled   
        """Initialize v8DetectionLoss with model parameters and task-aligned assignment settings."""
        device = next(model.parameters()).device  # get model device     
        h = model.args  # hyperparameters 

        m = model.model[-1]  # Detect() module     
        _validate_detection_loss_args(h)
        self.bce = self._build_cls_loss(h)
        self.hyp = h
        self.stride = m.stride  # model strides
        self.nc = m.nc  # number of classes 
        self.no = m.nc + m.reg_max * 4
        self.reg_max = m.reg_max
        self.device = device

        self.use_dfl = m.reg_max > 1     
   
        self.assigner = TaskAlignedAssigner(
            topk=tal_topk,    
            num_classes=self.nc,     
            alpha=0.5,   
            beta=6.0,     
            stride=self.stride.tolist(),
            topk2=tal_topk2,
        )    
        self.bbox_loss = BboxLoss(m.reg_max, args=h).to(device)
        self.proj = torch.arange(m.reg_max, dtype=torch.float, device=device)
        self._log_selected_losses()  

    def _format_selected_losses(self) -> str:    
        """Build concise one-line summary for configured cls/iou losses."""    
        cls_loss = str(_get_arg(self.hyp, "cls_loss", "bce")).lower()
        iou_loss = str(_get_arg(self.hyp, "iou_loss", "ciou")).lower()
        iou_aux = str(_get_arg(self.hyp, "iou_aux", "none")).lower()
        mode = self.bbox_loss.iou_mode
        extras = [f"iou_family={mode['family']}", f"iou_variant={mode['variant']}"]

        if cls_loss in {"slide", "ema_slide"}:   
            extras.append(f"slide_auto_iou_min={float(_get_arg(self.hyp, 'slide_auto_iou_min', 0.2)):.3f}")   
            extras.append(f"slide_delta={float(_get_arg(self.hyp, 'slide_delta', 0.1)):.3f}") 
        if cls_loss == "ema_slide":  
            extras.append(f"ema_decay={float(_get_arg(self.hyp, 'ema_decay', 0.999)):.3f}")
            extras.append(f"ema_tau={float(_get_arg(self.hyp, 'ema_tau', 2000.0)):.1f}")
        if cls_loss == "focal": 
            extras.append(f"focal_gamma={float(_get_arg(self.hyp, 'focal_gamma', 1.5)):.3f}")
            extras.append(f"focal_alpha={float(_get_arg(self.hyp, 'focal_alpha', 0.25)):.3f}")
        if cls_loss == "varifocal":
            extras.append(f"varifocal_alpha={float(_get_arg(self.hyp, 'varifocal_alpha', 0.75)):.3f}")  
            extras.append(f"varifocal_gamma={float(_get_arg(self.hyp, 'varifocal_gamma', 2.0)):.3f}") 
        if cls_loss == "qualityfocal": 
            extras.append(f"qfl_beta={float(_get_arg(self.hyp, 'qfl_beta', 2.0)):.3f}") 

        if iou_aux != "none":
            extras.append(f"iou_aux_ratio={float(_get_arg(self.hyp, 'iou_aux_ratio', 0.5)):.3f}")
        if mode["family"] == "inner" or (mode["family"] == "mpd" and mode["variant"] == "inner_mpdiou") or (  
            mode["family"] == "wise" and mode.get("inner", False)  
        ):   
            extras.append(f"inner_iou_ratio={float(_get_arg(self.hyp, 'inner_iou_ratio', 0.7)):.3f}")     
        if mode["family"] == "focaler" or (mode["family"] == "mpd" and mode["variant"] == "focaler_mpdiou") or (
            mode["family"] == "wise" and mode.get("focaler", False)    
        ):
            extras.append(f"focaler_d={float(_get_arg(self.hyp, 'focaler_d', 0.0)):.3f}")
            extras.append(f"focaler_u={float(_get_arg(self.hyp, 'focaler_u', 0.95)):.3f}")
        if mode["variant"] == "shapeiou":
            extras.append(f"shapeiou_scale={float(_get_arg(self.hyp, 'shapeiou_scale', 0.0)):.3f}") 
        if mode["variant"] in {"piou", "piou2"}:
            extras.append(f"piou_lambda={float(_get_arg(self.hyp, 'piou_lambda', 1.3)):.3f}")
        if mode["family"] == "wise" or bool(_get_arg(self.hyp, "use_wiseiou", False)):
            extras.append(f"wiseiou_monotonous={_get_arg(self.hyp, 'wiseiou_monotonous', False)}")

        return (
            f"{colorstr('bold', 'cyan', 'LossConfig')} "
            f"cls_loss={cls_loss}, iou_loss={iou_loss}, iou_aux={iou_aux}; " + ", ".join(extras)   
        )
    
    def _log_selected_losses(self) -> None:  
        """Print selected loss configuration once on main process.""" 
        if RANK in {-1, 0}:
            LOGGER.info(self._format_selected_losses())
    
    def preprocess(self, targets: torch.Tensor, batch_size: int, scale_tensor: torch.Tensor) -> torch.Tensor:    
        """Preprocess targets by converting to tensor format and scaling coordinates."""
        nl, ne = targets.shape
        if nl == 0:
            out = torch.zeros(batch_size, 0, ne - 1, device=self.device)
        else:    
            i = targets[:, 0]  # image index 
            _, counts = i.unique(return_counts=True)
            counts = counts.to(dtype=torch.int32)
            out = torch.zeros(batch_size, counts.max(), ne - 1, device=self.device)
            for j in range(batch_size):
                matches = i == j
                if n := matches.sum():     
                    out[j, :n] = targets[matches, 1:]
            out[..., 1:5] = xywh2xyxy(out[..., 1:5].mul_(scale_tensor))
        return out     

    def bbox_decode(self, anchor_points: torch.Tensor, pred_dist: torch.Tensor) -> torch.Tensor:     
        """Decode predicted object bounding box coordinates from anchor points and distribution."""
        if self.use_dfl:   
            b, a, c = pred_dist.shape  # batch, anchors, channels   
            pred_dist = pred_dist.view(b, a, 4, c // 4).softmax(3).matmul(self.proj.type(pred_dist.dtype))
            # pred_dist = pred_dist.view(b, a, c // 4, 4).transpose(2,3).softmax(3).matmul(self.proj.type(pred_dist.dtype)) 
            # pred_dist = (pred_dist.view(b, a, c // 4, 4).softmax(2) * self.proj.type(pred_dist.dtype).view(1, 1, -1, 1)).sum(2)   
        return dist2bbox(pred_dist, anchor_points, xywh=False)

    def _build_cls_loss(self, args: Any) -> nn.Module:   
        """Build detection cls loss from unified config."""  
        cls_loss = str(_get_arg(args, "cls_loss", "bce")).lower()     
        if cls_loss == "slide":  
            return SlideLoss(   
                nn.BCEWithLogitsLoss(reduction="none"),     
                auto_iou_min=float(_get_arg(args, "slide_auto_iou_min", 0.2)),    
                delta=float(_get_arg(args, "slide_delta", 0.1)),
            )   
        if cls_loss == "ema_slide": 
            return EMASlideLoss( 
                nn.BCEWithLogitsLoss(reduction="none"),    
                decay=float(_get_arg(args, "ema_decay", 0.999)),
                tau=float(_get_arg(args, "ema_tau", 2000.0)),
                auto_iou_min=float(_get_arg(args, "slide_auto_iou_min", 0.2)),
                delta=float(_get_arg(args, "slide_delta", 0.1)),
            ) 
        if cls_loss == "focal":
            return FocalLoss_YOLO(  
                gamma=float(_get_arg(args, "focal_gamma", 1.5)), 
                alpha=float(_get_arg(args, "focal_alpha", 0.25)),
            )
        if cls_loss == "varifocal":
            return VarifocalLoss_YOLO(   
                alpha=float(_get_arg(args, "varifocal_alpha", 0.75)),  
                gamma=float(_get_arg(args, "varifocal_gamma", 2.0)),
            )
        if cls_loss == "qualityfocal":
            return QualityfocalLoss_YOLO(beta=float(_get_arg(args, "qfl_beta", 2.0)))
        return nn.BCEWithLogitsLoss(reduction="none")     
   
    def _compute_cls_loss(
        self,  
        pred_scores: torch.Tensor,  
        target_scores: torch.Tensor,
        target_labels: torch.Tensor,
        target_bboxes: torch.Tensor,  
        pred_bboxes: torch.Tensor,    
        fg_mask: torch.Tensor,    
        stride_tensor: torch.Tensor,   
        target_scores_sum: torch.Tensor,
        dtype: torch.dtype,
    ) -> torch.Tensor: 
        """Compute cls loss for configured cls criterion."""
        if isinstance(self.bce, (nn.BCEWithLogitsLoss, FocalLoss_YOLO)):
            return self.bce(pred_scores, target_scores.to(dtype)).sum() / target_scores_sum
 
        if isinstance(self.bce, (VarifocalLoss_YOLO, QualityfocalLoss_YOLO)):
            if fg_mask.sum():  
                pos_ious = bbox_iou(pred_bboxes, target_bboxes / stride_tensor, xywh=False).clamp(min=1e-6).detach()     
                targets_onehot = torch.zeros(
                    (target_labels.shape[0], target_labels.shape[1], self.nc), dtype=torch.int64, device=target_labels.device  
                )
                targets_onehot.scatter_(2, target_labels.long().clamp_(0, self.nc - 1).unsqueeze(-1), 1)   
                fg_scores_mask = fg_mask[:, :, None].repeat(1, 1, self.nc)  
                cls_iou_targets = torch.where(fg_scores_mask > 0, pos_ious * targets_onehot, 0) 
                if isinstance(self.bce, QualityfocalLoss_YOLO):
                    targets_onehot_pos = torch.where(fg_scores_mask > 0, targets_onehot, 0)
                    return (
                        self.bce(pred_scores, cls_iou_targets.to(dtype), targets_onehot_pos.to(torch.bool)).sum()
                        / max(fg_mask.sum(), 1)
                    )
                return self.bce(pred_scores, cls_iou_targets.to(dtype)).sum() / max(fg_mask.sum(), 1)   
            cls_iou_targets = torch.zeros(  
                (target_labels.shape[0], target_labels.shape[1], self.nc), dtype=torch.float, device=target_labels.device   
            )
            if isinstance(self.bce, QualityfocalLoss_YOLO):
                zero_pos = torch.zeros_like(cls_iou_targets, dtype=torch.bool)  
                return self.bce(pred_scores, cls_iou_targets.to(dtype), zero_pos).sum() / max(fg_mask.sum(), 1)
            return self.bce(pred_scores, cls_iou_targets.to(dtype)).sum() / max(fg_mask.sum(), 1) 

        if isinstance(self.bce, (SlideLoss, EMASlideLoss)):   
            if fg_mask.sum():
                auto_iou = bbox_iou(pred_bboxes[fg_mask], (target_bboxes / stride_tensor)[fg_mask], xywh=False, CIoU=True).mean()     
            else:
                auto_iou = -1  
            return self.bce(pred_scores, target_scores.to(dtype), auto_iou).sum() / target_scores_sum     

        return nn.BCEWithLogitsLoss(reduction="none")(pred_scores, target_scores.to(dtype)).sum() / target_scores_sum  

    def get_assigned_targets_and_loss(self, preds: dict[str, torch.Tensor], batch: dict[str, Any]) -> tuple:
        """Calculate the sum of the loss for box, cls and dfl multiplied by batch size and return foreground mask and 
        target indices.
        """     
        loss = torch.zeros(3, device=self.device)  # box, cls, dfl  
        pred_distri, pred_scores = (   
            preds["boxes"].permute(0, 2, 1).contiguous(),    
            preds["scores"].permute(0, 2, 1).contiguous(),
        )  
        anchor_points, stride_tensor = make_anchors(preds["feats"], self.stride, 0.5)
   
        dtype = pred_scores.dtype  
        batch_size = pred_scores.shape[0]     
        imgsz = torch.tensor(preds["feats"][0].shape[2:], device=self.device, dtype=dtype) * self.stride[0]  
    
        # Targets
        targets = torch.cat((batch["batch_idx"].view(-1, 1), batch["cls"].view(-1, 1), batch["bboxes"]), 1)  
        targets = self.preprocess(targets.to(self.device), batch_size, scale_tensor=imgsz[[1, 0, 1, 0]])
        gt_labels, gt_bboxes = targets.split((1, 4), 2)  # cls, xyxy     
        mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0.0)   

        # Pboxes
        pred_bboxes = self.bbox_decode(anchor_points, pred_distri)  # xyxy, (b, h*w, 4)

        target_labels, target_bboxes, target_scores, fg_mask, target_gt_idx = self.assigner( 
            pred_scores.detach().sigmoid(),
            (pred_bboxes.detach() * stride_tensor).type(gt_bboxes.dtype),
            anchor_points * stride_tensor,  
            gt_labels,
            gt_bboxes,     
            mask_gt,
        )    

        target_scores_sum = max(target_scores.sum(), 1)
 
        # Cls loss
        loss[1] = self._compute_cls_loss(     
            pred_scores=pred_scores,
            target_scores=target_scores,     
            target_labels=target_labels,
            target_bboxes=target_bboxes,
            pred_bboxes=pred_bboxes,
            fg_mask=fg_mask, 
            stride_tensor=stride_tensor,     
            target_scores_sum=target_scores_sum, 
            dtype=dtype,
        )     
  
        # Bbox loss
        if fg_mask.sum():   
            loss[0], loss[2] = self.bbox_loss(    
                pred_distri,    
                pred_bboxes,
                anchor_points,  
                target_bboxes / stride_tensor,
                target_scores,  
                target_scores_sum,    
                fg_mask,
                imgsz,
                stride_tensor,    
            )   
    
        loss[0] *= self.hyp.box  # box gain   
        loss[1] *= self.hyp.cls  # cls gain 
        loss[2] *= self.hyp.dfl  # dfl gain
        return (
            (fg_mask, target_gt_idx, target_bboxes, anchor_points, stride_tensor),
            loss, 
            loss.detach(),     
        )  # loss(box, cls, dfl)

    def parse_output(  
        self, preds: dict[str, torch.Tensor] | tuple[torch.Tensor, dict[str, torch.Tensor]]
    ) -> torch.Tensor: 
        """Parse model predictions to extract features."""
        return preds[1] if isinstance(preds, tuple) else preds 

    def __call__(
        self,
        preds: dict[str, torch.Tensor] | tuple[torch.Tensor, dict[str, torch.Tensor]],
        batch: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:    
        """Calculate the sum of the loss for box, cls and dfl multiplied by batch size.""" 
        return self.loss(self.parse_output(preds), batch) 
  
    def loss(self, preds: dict[str, torch.Tensor], batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        """A wrapper for get_assigned_targets_and_loss and parse_output."""
        batch_size = preds["boxes"].shape[0]
        loss, loss_detach = self.get_assigned_targets_and_loss(preds, batch)[1:] 
        return loss * batch_size, loss_detach   
  
   
class v8SegmentationLoss(v8DetectionLoss): 
    """Criterion class for computing training losses for YOLOv8 segmentation."""

    def __init__(self, model, tal_topk: int = 10, tal_topk2: int | None = None):  # model must be de-paralleled
        """Initialize the v8SegmentationLoss class with model parameters and mask overlap setting."""  
        super().__init__(model, tal_topk, tal_topk2)   
        self.overlap = model.args.overlap_mask
        self.bcedice_loss = BCEDiceLoss(weight_bce=0.5, weight_dice=0.5)
 
    def loss(self, preds: dict[str, torch.Tensor], batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        """Calculate and return the combined loss for detection and segmentation."""
        pred_masks, proto = preds["mask_coefficient"].permute(0, 2, 1).contiguous(), preds["proto"]     
        loss = torch.zeros(5, device=self.device)  # box, seg, cls, dfl
        if isinstance(proto, tuple) and len(proto) == 2:   
            proto, pred_semseg = proto   
        else:   
            pred_semseg = None 
        (fg_mask, target_gt_idx, target_bboxes, _, _), det_loss, _ = self.get_assigned_targets_and_loss(preds, batch) 
        # NOTE: re-assign index for consistency for now. Need to be removed in the future. 
        loss[0], loss[2], loss[3] = det_loss[0], det_loss[1], det_loss[2]
  
        batch_size, _, mask_h, mask_w = proto.shape  # batch size, number of masks, mask height, mask width     
        if fg_mask.sum(): 
            # Masks loss    
            masks = batch["masks"].to(self.device).float()
            if tuple(masks.shape[-2:]) != (mask_h, mask_w):  # downsample
                # masks = F.interpolate(masks[None], (mask_h, mask_w), mode="nearest")[0]    
                proto = F.interpolate(proto, masks.shape[-2:], mode="bilinear", align_corners=False) 
   
            imgsz = (
                torch.tensor(preds["feats"][0].shape[2:], device=self.device, dtype=pred_masks.dtype) * self.stride[0]
            )
            loss[1] = self.calculate_segmentation_loss(     
                fg_mask,  
                masks,
                target_gt_idx,
                target_bboxes,
                batch["batch_idx"].view(-1, 1),    
                proto,
                pred_masks,   
                imgsz,   
            )
            if pred_semseg is not None: 
                sem_masks = batch["sem_masks"].to(self.device)  # NxHxW   
                sem_masks = F.one_hot(sem_masks.long(), num_classes=self.nc).permute(0, 3, 1, 2).float()  # NxCxHxW     

                if self.overlap:     
                    mask_zero = masks == 0  # NxHxW
                    sem_masks[mask_zero.unsqueeze(1).expand_as(sem_masks)] = 0 
                else:  
                    batch_idx = batch["batch_idx"].view(-1)  # [total_instances]     
                    for i in range(batch_size):
                        instance_mask_i = masks[batch_idx == i]  # [num_instances_i, H, W]  
                        if len(instance_mask_i) == 0:
                            continue
                        sem_masks[i, :, instance_mask_i.sum(dim=0) == 0] = 0

                loss[4] = self.bcedice_loss(pred_semseg, sem_masks)
                loss[4] *= self.hyp.box  # seg gain     

        # WARNING: lines below prevent Multi-GPU DDP 'unused gradient' PyTorch errors, do not remove
        else:  
            loss[1] += (proto * 0).sum() + (pred_masks * 0).sum()  # inf sums may lead to nan loss   
            if pred_semseg is not None:    
                loss[4] += (pred_semseg * 0).sum()
   
        loss[1] *= self.hyp.box  # seg gain  
        return loss * batch_size, loss.detach()  # loss(box, cls, dfl)     
    
    @staticmethod 
    def single_mask_loss(
        gt_mask: torch.Tensor, pred: torch.Tensor, proto: torch.Tensor, xyxy: torch.Tensor, area: torch.Tensor
    ) -> torch.Tensor:  
        """Compute the instance segmentation loss for a single image.
  
        Args:
            gt_mask (torch.Tensor): Ground truth mask of shape (N, H, W), where N is the number of objects.
            pred (torch.Tensor): Predicted mask coefficients of shape (N, 32).   
            proto (torch.Tensor): Prototype masks of shape (32, H, W).
            xyxy (torch.Tensor): Ground truth bounding boxes in xyxy format, normalized to [0, 1], of shape (N, 4).
            area (torch.Tensor): Area of each ground truth bounding box of shape (N,).
 
        Returns:     
            (torch.Tensor): The calculated mask loss for a single image.

        Notes:
            The function uses the equation pred_mask = torch.einsum('in,nhw->ihw', pred, proto) to produce the  
            predicted masks from the prototype masks and predicted mask coefficients.
        """  
        pred_mask = torch.einsum("in,nhw->ihw", pred, proto)  # (n, 32) @ (32, 80, 80) -> (n, 80, 80)
        loss = F.binary_cross_entropy_with_logits(pred_mask, gt_mask, reduction="none")
        return (crop_mask(loss, xyxy).mean(dim=(1, 2)) / area).sum()

    def calculate_segmentation_loss(
        self,
        fg_mask: torch.Tensor,
        masks: torch.Tensor,     
        target_gt_idx: torch.Tensor, 
        target_bboxes: torch.Tensor,     
        batch_idx: torch.Tensor,
        proto: torch.Tensor,
        pred_masks: torch.Tensor,
        imgsz: torch.Tensor,
    ) -> torch.Tensor:
        """Calculate the loss for instance segmentation.
     
        Args:     
            fg_mask (torch.Tensor): A binary tensor of shape (BS, N_anchors) indicating which anchors are positive.
            masks (torch.Tensor): Ground truth masks of shape (BS, H, W) if `overlap` is False, otherwise (BS, ?, H, W).
            target_gt_idx (torch.Tensor): Indexes of ground truth objects for each anchor of shape (BS, N_anchors).
            target_bboxes (torch.Tensor): Ground truth bounding boxes for each anchor of shape (BS, N_anchors, 4).
            batch_idx (torch.Tensor): Batch indices of shape (N_labels_in_batch, 1).
            proto (torch.Tensor): Prototype masks of shape (BS, 32, H, W).  
            pred_masks (torch.Tensor): Predicted masks for each anchor of shape (BS, N_anchors, 32).
            imgsz (torch.Tensor): Size of the input image as a tensor of shape (2), i.e., (H, W).    

        Returns:   
            (torch.Tensor): The calculated loss for instance segmentation.

        Notes:
            The batch loss can be computed for improved speed at higher memory usage.     
            For example, pred_mask can be computed as follows:  
                pred_mask = torch.einsum('in,nhw->ihw', pred, proto)  # (i, 32) @ (32, 160, 160) -> (i, 160, 160)
        """
        _, _, mask_h, mask_w = proto.shape    
        loss = 0
 
        # Normalize to 0-1     
        target_bboxes_normalized = target_bboxes / imgsz[[1, 0, 1, 0]]
 
        # Areas of target bboxes 
        marea = xyxy2xywh(target_bboxes_normalized)[..., 2:].prod(2) 
 
        # Normalize to mask size
        mxyxy = target_bboxes_normalized * torch.tensor([mask_w, mask_h, mask_w, mask_h], device=proto.device)
   
        for i, single_i in enumerate(zip(fg_mask, target_gt_idx, pred_masks, proto, mxyxy, marea, masks)):
            fg_mask_i, target_gt_idx_i, pred_masks_i, proto_i, mxyxy_i, marea_i, masks_i = single_i   
            if fg_mask_i.any():     
                mask_idx = target_gt_idx_i[fg_mask_i]
                if self.overlap:
                    gt_mask = masks_i == (mask_idx + 1).view(-1, 1, 1) 
                    gt_mask = gt_mask.float()
                else: 
                    gt_mask = masks[batch_idx.view(-1) == i][mask_idx]

                loss += self.single_mask_loss(
                    gt_mask, pred_masks_i[fg_mask_i], proto_i, mxyxy_i[fg_mask_i], marea_i[fg_mask_i]
                )

            # WARNING: lines below prevents Multi-GPU DDP 'unused gradient' PyTorch errors, do not remove
            else:   
                loss += (proto * 0).sum() + (pred_masks * 0).sum()  # inf sums may lead to nan loss  
  
        return loss / fg_mask.sum()  
   
   
class v8PoseLoss(v8DetectionLoss):
    """Criterion class for computing training losses for YOLOv8 pose estimation."""   
   
    def __init__(self, model, tal_topk: int = 10, tal_topk2: int = 10):  # model must be de-paralleled 
        """Initialize v8PoseLoss with model parameters and keypoint-specific loss functions."""     
        super().__init__(model, tal_topk, tal_topk2)   
        self.kpt_shape = model.model[-1].kpt_shape
        self.bce_pose = nn.BCEWithLogitsLoss()
        is_pose = self.kpt_shape == [17, 3]    
        nkpt = self.kpt_shape[0]  # number of keypoints  
        sigmas = torch.from_numpy(OKS_SIGMA).to(self.device) if is_pose else torch.ones(nkpt, device=self.device) / nkpt     
        self.keypoint_loss = KeypointLoss(sigmas=sigmas)

    def loss(self, preds: dict[str, torch.Tensor], batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        """Calculate the total loss and detach it for pose estimation."""
        pred_kpts = preds["kpts"].permute(0, 2, 1).contiguous()    
        loss = torch.zeros(5, device=self.device)  # box, cls, dfl, kpt_location, kpt_visibility   
        (fg_mask, target_gt_idx, target_bboxes, anchor_points, stride_tensor), det_loss, _ = (
            self.get_assigned_targets_and_loss(preds, batch)
        )     
        # NOTE: re-assign index for consistency for now. Need to be removed in the future.   
        loss[0], loss[3], loss[4] = det_loss[0], det_loss[1], det_loss[2]

        batch_size = pred_kpts.shape[0]
        imgsz = torch.tensor(preds["feats"][0].shape[2:], device=self.device, dtype=pred_kpts.dtype) * self.stride[0] 

        # Pboxes
        pred_kpts = self.kpts_decode(anchor_points, pred_kpts.view(batch_size, -1, *self.kpt_shape))  # (b, h*w, 17, 3)  

        # Bbox loss
        if fg_mask.sum():
            keypoints = batch["keypoints"].to(self.device).float().clone()
            keypoints[..., 0] *= imgsz[1]
            keypoints[..., 1] *= imgsz[0]
     
            loss[1], loss[2] = self.calculate_keypoints_loss(  
                fg_mask,   
                target_gt_idx,
                keypoints,
                batch["batch_idx"].view(-1, 1),   
                stride_tensor,    
                target_bboxes,
                pred_kpts,  
            )

        loss[1] *= self.hyp.pose  # pose gain 
        loss[2] *= self.hyp.kobj  # kobj gain   

        return loss * batch_size, loss.detach()  # loss(box, pose, kobj, cls, dfl)

    @staticmethod
    def kpts_decode(anchor_points: torch.Tensor, pred_kpts: torch.Tensor) -> torch.Tensor:  
        """Decode predicted keypoints to image coordinates."""
        y = pred_kpts.clone()
        y[..., :2] *= 2.0 
        y[..., 0] += anchor_points[:, [0]] - 0.5
        y[..., 1] += anchor_points[:, [1]] - 0.5
        return y 
    
    def _select_target_keypoints(
        self,
        keypoints: torch.Tensor, 
        batch_idx: torch.Tensor,
        target_gt_idx: torch.Tensor,
        masks: torch.Tensor,
    ) -> torch.Tensor:
        """Select target keypoints for each anchor based on batch index and target ground truth index.    
 
        Args:    
            keypoints (torch.Tensor): Ground truth keypoints, shape (N_kpts_in_batch, N_kpts_per_object, kpts_dim).  
            batch_idx (torch.Tensor): Batch index tensor for keypoints, shape (N_kpts_in_batch, 1).     
            target_gt_idx (torch.Tensor): Index tensor mapping anchors to ground truth objects, shape (BS, N_anchors).     
            masks (torch.Tensor): Binary mask tensor indicating object presence, shape (BS, N_anchors).   
  
        Returns:
            (torch.Tensor): Selected keypoints tensor, shape (BS, N_anchors, N_kpts_per_object, kpts_dim).
        """   
        batch_idx = batch_idx.flatten()
        batch_size = len(masks)     
   
        # Find the maximum number of keypoints in a single image
        max_kpts = torch.unique(batch_idx, return_counts=True)[1].max()
 
        # Create a tensor to hold batched keypoints 
        batched_keypoints = torch.zeros(     
            (batch_size, max_kpts, keypoints.shape[1], keypoints.shape[2]), device=keypoints.device
        )     

        # TODO: any idea how to vectorize this?
        # Fill batched_keypoints with keypoints based on batch_idx
        for i in range(batch_size):
            keypoints_i = keypoints[batch_idx == i]
            batched_keypoints[i, : keypoints_i.shape[0]] = keypoints_i   

        # Expand dimensions of target_gt_idx to match the shape of batched_keypoints
        target_gt_idx_expanded = target_gt_idx.unsqueeze(-1).unsqueeze(-1)

        # Use target_gt_idx_expanded to select keypoints from batched_keypoints   
        selected_keypoints = batched_keypoints.gather(
            1, target_gt_idx_expanded.expand(-1, -1, keypoints.shape[1], keypoints.shape[2])    
        )

        return selected_keypoints  
 
    def calculate_keypoints_loss(
        self,   
        masks: torch.Tensor,
        target_gt_idx: torch.Tensor,    
        keypoints: torch.Tensor,     
        batch_idx: torch.Tensor,
        stride_tensor: torch.Tensor,
        target_bboxes: torch.Tensor,
        pred_kpts: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:    
        """Calculate the keypoints loss for the model.   
 
        This function calculates the keypoints loss and keypoints object loss for a given batch. The keypoints loss is
        based on the difference between the predicted keypoints and ground truth keypoints. The keypoints object loss is     
        a binary classification loss that classifies whether a keypoint is present or not. 

        Args:    
            masks (torch.Tensor): Binary mask tensor indicating object presence, shape (BS, N_anchors).
            target_gt_idx (torch.Tensor): Index tensor mapping anchors to ground truth objects, shape (BS, N_anchors).   
            keypoints (torch.Tensor): Ground truth keypoints, shape (N_kpts_in_batch, N_kpts_per_object, kpts_dim).
            batch_idx (torch.Tensor): Batch index tensor for keypoints, shape (N_kpts_in_batch, 1).
            stride_tensor (torch.Tensor): Stride tensor for anchors, shape (N_anchors, 1).
            target_bboxes (torch.Tensor): Ground truth boxes in (x1, y1, x2, y2) format, shape (BS, N_anchors, 4).
            pred_kpts (torch.Tensor): Predicted keypoints, shape (BS, N_anchors, N_kpts_per_object, kpts_dim).

        Returns:    
            kpts_loss (torch.Tensor): The keypoints loss.   
            kpts_obj_loss (torch.Tensor): The keypoints object loss. 
        """    
        # Select target keypoints using helper method
        selected_keypoints = self._select_target_keypoints(keypoints, batch_idx, target_gt_idx, masks)
   
        # Divide coordinates by stride
        selected_keypoints[..., :2] /= stride_tensor.view(1, -1, 1, 1)  
 
        kpts_loss = 0    
        kpts_obj_loss = 0

        if masks.any():     
            target_bboxes /= stride_tensor
            gt_kpt = selected_keypoints[masks]    
            area = xyxy2xywh(target_bboxes[masks])[:, 2:].prod(1, keepdim=True)
            pred_kpt = pred_kpts[masks]
            kpt_mask = gt_kpt[..., 2] != 0 if gt_kpt.shape[-1] == 3 else torch.full_like(gt_kpt[..., 0], True)  
            kpts_loss = self.keypoint_loss(pred_kpt, gt_kpt, kpt_mask, area)  # pose loss

            if pred_kpt.shape[-1] == 3:
                kpts_obj_loss = self.bce_pose(pred_kpt[..., 2], kpt_mask.float())  # keypoint obj loss  

        return kpts_loss, kpts_obj_loss  

   
class PoseLoss26(v8PoseLoss):
    """Criterion class for computing training losses for YOLOv8 pose estimation with RLE loss support."""  

    def __init__(self, model, tal_topk: int = 10, tal_topk2: int | None = None):  # model must be de-paralleled
        """Initialize PoseLoss26 with model parameters and keypoint-specific loss functions including RLE loss."""
        super().__init__(model, tal_topk, tal_topk2)
        is_pose = self.kpt_shape == [17, 3]    
        nkpt = self.kpt_shape[0]  # number of keypoints     
        self.rle_loss = None     
        self.flow_model = model.model[-1].flow_model if hasattr(model.model[-1], "flow_model") else None  
        if self.flow_model is not None:
            self.rle_loss = RLELoss(use_target_weight=True).to(self.device)     
            self.target_weights = (
                torch.from_numpy(RLE_WEIGHT).to(self.device) if is_pose else torch.ones(nkpt, device=self.device)
            )
 
    def loss(self, preds: dict[str, torch.Tensor], batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:     
        """Calculate the total loss and detach it for pose estimation."""
        pred_kpts = preds["kpts"].permute(0, 2, 1).contiguous()    
        loss = torch.zeros(6 if self.rle_loss else 5, device=self.device)  # box, cls, dfl, kpt_location, kpt_visibility    
        (fg_mask, target_gt_idx, target_bboxes, anchor_points, stride_tensor), det_loss, _ = (     
            self.get_assigned_targets_and_loss(preds, batch)
        ) 
        # NOTE: re-assign index for consistency for now. Need to be removed in the future.     
        loss[0], loss[3], loss[4] = det_loss[0], det_loss[1], det_loss[2]   
  
        batch_size = pred_kpts.shape[0]  
        imgsz = torch.tensor(preds["feats"][0].shape[2:], device=self.device, dtype=pred_kpts.dtype) * self.stride[0]

        pred_kpts = pred_kpts.view(batch_size, -1, *self.kpt_shape)  # (b, h*w, 17, 3)
  
        if self.rle_loss and preds.get("kpts_sigma", None) is not None:
            pred_sigma = preds["kpts_sigma"].permute(0, 2, 1).contiguous()    
            pred_sigma = pred_sigma.view(batch_size, -1, self.kpt_shape[0], 2)  # (b, h*w, 17, 2)
            pred_kpts = torch.cat([pred_kpts, pred_sigma], dim=-1)  # (b, h*w, 17, 5)     

        pred_kpts = self.kpts_decode(anchor_points, pred_kpts)   
  
        # Bbox loss
        if fg_mask.sum():  
            keypoints = batch["keypoints"].to(self.device).float().clone()
            keypoints[..., 0] *= imgsz[1]
            keypoints[..., 1] *= imgsz[0]

            keypoints_loss = self.calculate_keypoints_loss(
                fg_mask,  
                target_gt_idx,   
                keypoints,
                batch["batch_idx"].view(-1, 1),
                stride_tensor,  
                target_bboxes,
                pred_kpts,
            )
            loss[1] = keypoints_loss[0]
            loss[2] = keypoints_loss[1]    
            if self.rle_loss is not None:  
                loss[5] = keypoints_loss[2]    

        loss[1] *= self.hyp.pose  # pose gain    
        loss[2] *= self.hyp.kobj  # kobj gain  
        if self.rle_loss is not None:
            loss[5] *= self.hyp.rle  # rle gain

        return loss * batch_size, loss.detach()  # loss(box, cls, dfl, kpt_location, kpt_visibility) 
 
    @staticmethod   
    def kpts_decode(anchor_points: torch.Tensor, pred_kpts: torch.Tensor) -> torch.Tensor:
        """Decode predicted keypoints to image coordinates."""
        y = pred_kpts.clone()     
        y[..., 0] += anchor_points[:, [0]] 
        y[..., 1] += anchor_points[:, [1]]     
        return y

    def calculate_rle_loss(self, pred_kpt: torch.Tensor, gt_kpt: torch.Tensor, kpt_mask: torch.Tensor) -> torch.Tensor:
        """Calculate the RLE (Residual Log-likelihood Estimation) loss for keypoints.  
    
        Args:
            pred_kpt (torch.Tensor): Predicted keypoints with sigma, shape (N, kpts_dim) where kpts_dim >= 4.
            gt_kpt (torch.Tensor): Ground truth keypoints, shape (N, kpts_dim).   
            kpt_mask (torch.Tensor): Mask for valid keypoints, shape (N, num_keypoints).  
    
        Returns:
            (torch.Tensor): The RLE loss.    
        """     
        pred_kpt_visible = pred_kpt[kpt_mask]
        gt_kpt_visible = gt_kpt[kpt_mask] 
        pred_coords = pred_kpt_visible[:, 0:2]
        pred_sigma = pred_kpt_visible[:, -2:] 
        gt_coords = gt_kpt_visible[:, 0:2]

        target_weights = self.target_weights.unsqueeze(0).repeat(kpt_mask.shape[0], 1) 
        target_weights = target_weights[kpt_mask]   
     
        pred_sigma = pred_sigma.sigmoid()    
        error = (pred_coords - gt_coords) / (pred_sigma + 1e-9)

        # Filter out NaN and Inf values to prevent MultivariateNormal validation errors    
        valid_mask = ~(torch.isnan(error) | torch.isinf(error)).any(dim=-1)
        if not valid_mask.any():
            return torch.tensor(0.0, device=pred_kpt.device)
 
        error = error[valid_mask]
        error = error.clamp(-100, 100)  # Prevent numerical instability
        pred_sigma = pred_sigma[valid_mask]
        target_weights = target_weights[valid_mask]   

        log_phi = self.flow_model.log_prob(error)    
  
        return self.rle_loss(pred_sigma, log_phi, error, target_weights)

    def calculate_keypoints_loss(     
        self, 
        masks: torch.Tensor,    
        target_gt_idx: torch.Tensor,
        keypoints: torch.Tensor,
        batch_idx: torch.Tensor,  
        stride_tensor: torch.Tensor, 
        target_bboxes: torch.Tensor,  
        pred_kpts: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]: 
        """Calculate the keypoints loss for the model.

        This function calculates the keypoints loss and keypoints object loss for a given batch. The keypoints loss is  
        based on the difference between the predicted keypoints and ground truth keypoints. The keypoints object loss is 
        a binary classification loss that classifies whether a keypoint is present or not. 
     
        Args:   
            masks (torch.Tensor): Binary mask tensor indicating object presence, shape (BS, N_anchors).
            target_gt_idx (torch.Tensor): Index tensor mapping anchors to ground truth objects, shape (BS, N_anchors).
            keypoints (torch.Tensor): Ground truth keypoints, shape (N_kpts_in_batch, N_kpts_per_object, kpts_dim).
            batch_idx (torch.Tensor): Batch index tensor for keypoints, shape (N_kpts_in_batch, 1).
            stride_tensor (torch.Tensor): Stride tensor for anchors, shape (N_anchors, 1).
            target_bboxes (torch.Tensor): Ground truth boxes in (x1, y1, x2, y2) format, shape (BS, N_anchors, 4).  
            pred_kpts (torch.Tensor): Predicted keypoints, shape (BS, N_anchors, N_kpts_per_object, kpts_dim).     
     
        Returns:
            kpts_loss (torch.Tensor): The keypoints loss.
            kpts_obj_loss (torch.Tensor): The keypoints object loss.
            rle_loss (torch.Tensor): The RLE loss.
        """
        # Select target keypoints using inherited helper method
        selected_keypoints = self._select_target_keypoints(keypoints, batch_idx, target_gt_idx, masks) 
     
        # Divide coordinates by stride
        selected_keypoints[..., :2] /= stride_tensor.view(1, -1, 1, 1) 
     
        kpts_loss = 0   
        kpts_obj_loss = 0  
        rle_loss = 0  
    
        if masks.any():
            target_bboxes /= stride_tensor     
            gt_kpt = selected_keypoints[masks]
            area = xyxy2xywh(target_bboxes[masks])[:, 2:].prod(1, keepdim=True)   
            pred_kpt = pred_kpts[masks]     
            kpt_mask = gt_kpt[..., 2] != 0 if gt_kpt.shape[-1] == 3 else torch.full_like(gt_kpt[..., 0], True)
            kpts_loss = self.keypoint_loss(pred_kpt, gt_kpt, kpt_mask, area)  # pose loss
   
            if self.rle_loss is not None and (pred_kpt.shape[-1] == 4 or pred_kpt.shape[-1] == 5):   
                rle_loss = self.calculate_rle_loss(pred_kpt, gt_kpt, kpt_mask) 
            if pred_kpt.shape[-1] == 3 or pred_kpt.shape[-1] == 5:
                kpts_obj_loss = self.bce_pose(pred_kpt[..., 2], kpt_mask.float())  # keypoint obj loss     

        return kpts_loss, kpts_obj_loss, rle_loss   

     
class v8ClassificationLoss:     
    """Criterion class for computing training losses for classification."""
  
    def __call__(self, preds: Any, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute the classification loss between predictions and true labels."""
        preds = preds[1] if isinstance(preds, (list, tuple)) else preds
        loss = F.cross_entropy(preds, batch["cls"], reduction="mean") 
        return loss, loss.detach()
   
  
class v8OBBLoss(v8DetectionLoss):  
    """Calculates losses for object detection, classification, and box distribution in rotated YOLO models."""
    
    def __init__(self, model, tal_topk=10, tal_topk2: int | None = None):
        """Initialize v8OBBLoss with model, assigner, and rotated bbox loss; model must be de-paralleled."""     
        super().__init__(model, tal_topk=tal_topk, tal_topk2=tal_topk2)
        self.assigner = RotatedTaskAlignedAssigner(
            topk=tal_topk, 
            num_classes=self.nc,
            alpha=0.5,
            beta=6.0,     
            stride=self.stride.tolist(), 
            topk2=tal_topk2,  
        )   
        self.bbox_loss = RotatedBboxLoss(self.reg_max, args=self.hyp).to(self.device)

    def preprocess(self, targets: torch.Tensor, batch_size: int, scale_tensor: torch.Tensor) -> torch.Tensor:
        """Preprocess targets for oriented bounding box detection.""" 
        if targets.shape[0] == 0:
            out = torch.zeros(batch_size, 0, 6, device=self.device)
        else: 
            i = targets[:, 0]  # image index
            _, counts = i.unique(return_counts=True)     
            counts = counts.to(dtype=torch.int32)
            out = torch.zeros(batch_size, counts.max(), 6, device=self.device)
            for j in range(batch_size):  
                matches = i == j
                if n := matches.sum():
                    bboxes = targets[matches, 2:]    
                    bboxes[..., :4].mul_(scale_tensor)
                    out[j, :n] = torch.cat([targets[matches, 1:2], bboxes], dim=-1) 
        return out

    def loss(self, preds: dict[str, torch.Tensor], batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:     
        """Calculate and return the loss for oriented bounding box detection."""
        loss = torch.zeros(4, device=self.device)  # box, cls, dfl, angle
        pred_distri, pred_scores, pred_angle = (
            preds["boxes"].permute(0, 2, 1).contiguous(),  
            preds["scores"].permute(0, 2, 1).contiguous(),   
            preds["angle"].permute(0, 2, 1).contiguous(),
        )   
        anchor_points, stride_tensor = make_anchors(preds["feats"], self.stride, 0.5)
        batch_size = pred_angle.shape[0]  # batch size, number of masks, mask height, mask width

        dtype = pred_scores.dtype     
        imgsz = torch.tensor(preds["feats"][0].shape[2:], device=self.device, dtype=dtype) * self.stride[0]
 
        # targets
        try:    
            batch_idx = batch["batch_idx"].view(-1, 1)   
            targets = torch.cat((batch_idx, batch["cls"].view(-1, 1), batch["bboxes"].view(-1, 5)), 1)
            rw, rh = targets[:, 4] * float(imgsz[1]), targets[:, 5] * float(imgsz[0])
            targets = targets[(rw >= 2) & (rh >= 2)]  # filter rboxes of tiny size to stabilize training    
            targets = self.preprocess(targets.to(self.device), batch_size, scale_tensor=imgsz[[1, 0, 1, 0]])
            gt_labels, gt_bboxes = targets.split((1, 5), 2)  # cls, xywhr
            mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0.0)  
        except RuntimeError as e:  
            raise TypeError(     
                "ERROR ❌ OBB dataset incorrectly formatted or not a OBB dataset.\n"     
                "This error can occur when incorrectly training a 'OBB' model on a 'detect' dataset, " 
                "i.e. 'yolo train model=yolo26n-obb.pt data=dota8.yaml'.\nVerify your dataset is a " 
                "correctly formatted 'OBB' dataset using 'data=dota8.yaml' " 
                "as an example.\nSee https://docs.ultralytics.com/datasets/obb/ for help."     
            ) from e    

        # Pboxes   
        pred_bboxes = self.bbox_decode(anchor_points, pred_distri, pred_angle)  # xyxy, (b, h*w, 4)

        bboxes_for_assigner = pred_bboxes.clone().detach()
        # Only the first four elements need to be scaled
        bboxes_for_assigner[..., :4] *= stride_tensor   
        target_labels, target_bboxes, target_scores, fg_mask, _ = self.assigner(
            pred_scores.detach().sigmoid(),     
            bboxes_for_assigner.type(gt_bboxes.dtype),
            anchor_points * stride_tensor,   
            gt_labels,
            gt_bboxes,
            mask_gt, 
        )

        target_scores_sum = max(target_scores.sum(), 1) 
   
        # Cls loss
        pred_bboxes_xyxy = xywh2xyxy(pred_bboxes[..., :4])
        target_bboxes_xyxy = xywh2xyxy(target_bboxes[..., :4])
        loss[1] = self._compute_cls_loss(
            pred_scores=pred_scores,    
            target_scores=target_scores,     
            target_labels=target_labels,
            target_bboxes=target_bboxes_xyxy,
            pred_bboxes=pred_bboxes_xyxy,
            fg_mask=fg_mask,     
            stride_tensor=stride_tensor,
            target_scores_sum=target_scores_sum,  
            dtype=dtype, 
        )

        # Bbox loss
        if fg_mask.sum():    
            target_bboxes[..., :4] /= stride_tensor
            loss[0], loss[2] = self.bbox_loss(
                pred_distri,   
                pred_bboxes,
                anchor_points,  
                target_bboxes,
                target_scores,   
                target_scores_sum,   
                fg_mask, 
                imgsz,
                stride_tensor,     
            )    
            weight = target_scores.sum(-1)[fg_mask]
            loss[3] = self.calculate_angle_loss(
                pred_bboxes, target_bboxes, fg_mask, weight, target_scores_sum
            )  # angle loss  
        else: 
            loss[0] += (pred_angle * 0).sum()   
   
        loss[0] *= self.hyp.box  # box gain
        loss[1] *= self.hyp.cls  # cls gain
        loss[2] *= self.hyp.dfl  # dfl gain
        loss[3] *= self.hyp.angle  # angle gain

        return loss * batch_size, loss.detach()  # loss(box, cls, dfl, angle)   

    def bbox_decode(   
        self, anchor_points: torch.Tensor, pred_dist: torch.Tensor, pred_angle: torch.Tensor
    ) -> torch.Tensor:
        """Decode predicted object bounding box coordinates from anchor points and distribution. 

        Args: 
            anchor_points (torch.Tensor): Anchor points, (h*w, 2).     
            pred_dist (torch.Tensor): Predicted rotated distance, (bs, h*w, 4). 
            pred_angle (torch.Tensor): Predicted angle, (bs, h*w, 1).

        Returns:  
            (torch.Tensor): Predicted rotated bounding boxes with angles, (bs, h*w, 5).
        """
        if self.use_dfl:
            b, a, c = pred_dist.shape  # batch, anchors, channels
            pred_dist = pred_dist.view(b, a, 4, c // 4).softmax(3).matmul(self.proj.type(pred_dist.dtype))
        return torch.cat((dist2rbox(pred_dist, pred_angle, anchor_points), pred_angle), dim=-1)
    
    def calculate_angle_loss(self, pred_bboxes, target_bboxes, fg_mask, weight, target_scores_sum, lambda_val=3):   
        """Calculate oriented angle loss.   

        Args:
            pred_bboxes: [N, 5] (x, y, w, h, theta).
            target_bboxes: [N, 5] (x, y, w, h, theta).   
            fg_mask: Foreground mask indicating valid predictions.
            weight: Loss weights for each prediction.  
            target_scores_sum: Sum of target scores for normalization.   
            lambda_val: control the sensitivity to aspect ratio. 
        """
        w_gt = target_bboxes[..., 2]
        h_gt = target_bboxes[..., 3]
        pred_theta = pred_bboxes[..., 4]     
        target_theta = target_bboxes[..., 4]
    
        log_ar = torch.log((w_gt + 1e-9) / (h_gt + 1e-9))  
        scale_weight = torch.exp(-(log_ar**2) / (lambda_val**2))

        delta_theta = pred_theta - target_theta
        delta_theta_wrapped = delta_theta - torch.round(delta_theta / math.pi) * math.pi
        ang_loss = torch.sin(2 * delta_theta_wrapped[fg_mask]) ** 2
    
        ang_loss = scale_weight[fg_mask] * ang_loss
        ang_loss = ang_loss * weight

        return ang_loss.sum() / target_scores_sum   
    
     
class E2EDetectLoss:  
    """Criterion class for computing training losses for end-to-end detection."""
     
    def __init__(self, model):
        """Initialize E2EDetectLoss with one-to-many and one-to-one detection losses using the provided model.""" 
        self.one2many = v8DetectionLoss(model, tal_topk=10)     
        self.one2one = v8DetectionLoss(model, tal_topk=1)
 
    def __call__(self, preds: Any, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        """Calculate the sum of the loss for box, cls and dfl multiplied by batch size.""" 
        preds = preds[1] if isinstance(preds, tuple) else preds
        one2many = preds["one2many"]   
        loss_one2many = self.one2many(one2many, batch)
        one2one = preds["one2one"]    
        loss_one2one = self.one2one(one2one, batch)
        return loss_one2many[0] + loss_one2one[0], loss_one2many[1] + loss_one2one[1]   
     
    
class E2ELoss:
    """Criterion class for computing training losses for end-to-end detection."""  

    def __init__(self, model, loss_fn=v8DetectionLoss):
        """Initialize E2ELoss with one-to-many and one-to-one detection losses using the provided model.""" 
        self.one2many = loss_fn(model, tal_topk=10)    
        self.one2one = loss_fn(model, tal_topk=7, tal_topk2=1)
        self.updates = 0  
        self.total = 1.0
        # init gain   
        self.o2m = 0.8 
        self.o2o = self.total - self.o2m
        self.o2m_copy = self.o2m     
        # final gain     
        self.final_o2m = 0.1

    def __call__(self, preds: Any, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        """Calculate the sum of the loss for box, cls and dfl multiplied by batch size."""  
        preds = self.one2many.parse_output(preds)
        one2many, one2one = preds["one2many"], preds["one2one"]
        loss_one2many = self.one2many.loss(one2many, batch)
        loss_one2one = self.one2one.loss(one2one, batch)     
        return loss_one2many[0] * self.o2m + loss_one2one[0] * self.o2o, loss_one2one[1]  
   
    def update(self) -> None: 
        """Update the weights for one-to-many and one-to-one losses based on the decay schedule.""" 
        self.updates += 1
        self.o2m = self.decay(self.updates) 
        self.o2o = max(self.total - self.o2m, 0)     

    def decay(self, x) -> float:
        """Calculate the decayed weight for one-to-many loss based on the current update step."""
        return max(1 - x / max(self.one2one.hyp.epochs - 1, 1), 0) * (self.o2m_copy - self.final_o2m) + self.final_o2m   
    

class TVPDetectLoss: 
    """Criterion class for computing training losses for text-visual prompt detection."""
     
    def __init__(self, model, tal_topk=10, tal_topk2: int | None = None):
        """Initialize TVPDetectLoss with task-prompt and visual-prompt criteria using the provided model."""  
        self.vp_criterion = v8DetectionLoss(model, tal_topk, tal_topk2)   
        # NOTE: store following info as it's changeable in __call__  
        self.hyp = self.vp_criterion.hyp    
        self.ori_nc = self.vp_criterion.nc  
        self.ori_no = self.vp_criterion.no   
        self.ori_reg_max = self.vp_criterion.reg_max

    def parse_output(self, preds) -> dict[str, torch.Tensor]:   
        """Parse model predictions to extract features."""    
        return self.vp_criterion.parse_output(preds)  
 
    def __call__(self, preds: Any, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        """Calculate the loss for text-visual prompt detection."""    
        return self.loss(self.parse_output(preds), batch)
     
    def loss(self, preds: dict[str, torch.Tensor], batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        """Calculate the loss for text-visual prompt detection."""    
        if self.ori_nc == preds["scores"].shape[1]:
            loss = torch.zeros(3, device=self.vp_criterion.device, requires_grad=True)
            return loss, loss.detach()

        preds["scores"] = self._get_vp_features(preds)
        vp_loss = self.vp_criterion(preds, batch)  
        box_loss = vp_loss[0][1]
        return box_loss, vp_loss[1] 
 
    def _get_vp_features(self, preds: dict[str, torch.Tensor]) -> list[torch.Tensor]:    
        """Extract visual-prompt features from the model output."""
        scores = preds["scores"]    
        vnc = scores.shape[1]

        self.vp_criterion.nc = vnc
        self.vp_criterion.no = vnc + self.vp_criterion.reg_max * 4  
        self.vp_criterion.assigner.num_classes = vnc 
        return scores   
   
    
class TVPSegmentLoss(TVPDetectLoss):
    """Criterion class for computing training losses for text-visual prompt segmentation."""     
  
    def __init__(self, model, tal_topk=10):     
        """Initialize TVPSegmentLoss with task-prompt and visual-prompt criteria using the provided model."""   
        super().__init__(model)
        self.vp_criterion = v8SegmentationLoss(model, tal_topk)
        self.hyp = self.vp_criterion.hyp   
    
    def __call__(self, preds: Any, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]: 
        """Calculate the loss for text-visual prompt segmentation."""
        return self.loss(self.parse_output(preds), batch)  
    
    def loss(self, preds: Any, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:  
        """Calculate the loss for text-visual prompt detection."""
        if self.ori_nc == preds["scores"].shape[1]:     
            loss = torch.zeros(4, device=self.vp_criterion.device, requires_grad=True)     
            return loss, loss.detach()

        preds["scores"] = self._get_vp_features(preds)     
        vp_loss = self.vp_criterion(preds, batch)     
        cls_loss = vp_loss[0][2] 
        return cls_loss, vp_loss[1]
