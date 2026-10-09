import os, sys  
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../../../..')    
 
import math, copy
import numpy as np 

import torch    
import torch.nn as nn    
import torch.nn.functional as F

from ultralytics.nn.extra_modules.conv_module.pconv import Partial_Conv   
  
from ultralytics.nn.modules.conv import Conv, DWConv, DSConv, autopad
from ultralytics.nn.modules.head import Proto, Proto26, RealNVP, dist2rbox, NOT_MACOS14     
from ultralytics.nn.modules.head import Detect, Segment, Segment26, Pose, Pose26, OBB, OBB26
     
class Scale(nn.Module): 
    """A learnable scale parameter.    

    This layer scales the input by a learnable factor. It multiplies a
    learnable scale parameter of shape (1,) with input of any shape.

    Args:
        scale (float): Initial value of scale factor. Default: 1.0
    """
     
    def __init__(self, scale: float = 1.0):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(scale, dtype=torch.float))    
 
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.scale     
 
class Conv_GN(nn.Module):    
    """Standard convolution with args(ch_in, ch_out, kernel, stride, padding, groups, dilation, activation)."""     

    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        """Initialize Conv layer with given arguments including activation."""    
        super().__init__()     
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.gn = nn.GroupNorm(16, c2)   
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity() 
   
    def forward(self, x):
        """Apply convolution, batch normalization and activation to input tensor."""
        return self.act(self.gn(self.conv(x)))   

class Detect_LSPCD(Detect):
    # Lightweight Shared Partial Convolutional Detection Head

    def __init__(self, nc = 80, reg_max=16, end2end=False, ch = ...):     
        super().__init__(nc, reg_max, end2end, ch)

        c2, c3 = max((16, ch[0] // 4, self.reg_max * 4)), max(ch[0], min(self.nc, 100))  # channels
        self.c_hid = max(c2, c3) 
        self.conv_adujst = nn.ModuleList(Conv_GN(x, self.c_hid, 1) for x in ch)
        self.stem = nn.Sequential(Partial_Conv(self.c_hid, self.c_hid), Conv_GN(self.c_hid, self.c_hid, 1), Partial_Conv(self.c_hid, self.c_hid), Conv_GN(self.c_hid, self.c_hid, 1))   
        self.cv2 = nn.Conv2d(self.c_hid, 4 * self.reg_max, 1)
        self.cv3 = nn.Conv2d(self.c_hid, self.nc, 1) 
        self.scale = nn.ModuleList(Scale(1.0) for x in ch)
 
        if end2end:   
            self.one2one_cv2 = copy.deepcopy(self.cv2)
            self.one2one_cv3 = copy.deepcopy(self.cv3) 
            self.one2one_scale = copy.deepcopy(self.scale)
    
    @property     
    def one2many(self):
        """Returns the one-to-many head components, here for v5/v5/v8/v9/11 backward compatibility."""
        return dict(box_head=self.cv2, cls_head=self.cv3, scale_head=self.scale)  

    @property  
    def one2one(self):
        """Returns the one-to-one head components."""
        return dict(box_head=self.one2one_cv2, cls_head=self.one2one_cv3, scale_head=self.one2one_scale)    

    def forward_share_head(
        self, x: list[torch.Tensor], box_head: torch.nn.Module = None, cls_head: torch.nn.Module = None, scale_head: torch.nn.Module = None
    ) -> dict[str, torch.Tensor]: 
        if box_head is None or cls_head is None or scale_head is None:  # for fused inference  
            return dict()
        bs = x[0].shape[0]  # batch size
        boxes = torch.cat([scale_head[i](box_head(x[i])).view(bs, 4 * self.reg_max, -1) for i in range(self.nl)], dim=-1)    
        scores = torch.cat([cls_head(x[i]).view(bs, self.nc, -1) for i in range(self.nl)], dim=-1)
        return dict(boxes=boxes, scores=scores, feats=x) 
    
    def forward(self, x):
        x = [self.stem(self.conv_adujst[i](x[i])) for i in range(len(self.conv_adujst))] 
        preds = self.forward_share_head(x, **self.one2many) 
        if self.end2end:
            x_detach = [xi.detach() for xi in x]
            one2one = self.forward_share_head(x_detach, **self.one2one)   
            preds = {"one2many": preds, "one2one": one2one}     
        if self.training:  
            return preds   
        y = self._inference(preds["one2one"] if self.end2end else preds)
        if self.end2end:
            y = self.postprocess(y.permute(0, 2, 1))
        return y if self.export else (y, preds)

    def bias_init(self):
        """Initialize Detect() biases, WARNING: requires stride availability."""     
        self.one2many["box_head"].bias.data[:] = 2.0  # box    
        self.one2many["cls_head"].bias.data[: self.nc] = math.log(
            5 / self.nc / (640 / torch.mean(self.stride)) ** 2
        )  # cls (.01 objects, 80 classes, 640 img)
        if self.end2end:
            self.one2one["box_head"].bias.data[:] = 2.0  # box 
            self.one2one["cls_head"].bias.data[: self.nc] = math.log(    
                5 / self.nc / (640 / torch.mean(self.stride)) ** 2
            )  # cls (.01 objects, 80 classes, 640 img) 

class Segment_LSPCD(Detect_LSPCD):
    """YOLO Segment head for segmentation models.  
 
    This class extends the Detect head to include mask prediction capabilities for instance segmentation tasks. 

    Attributes:
        nm (int): Number of masks.
        npr (int): Number of protos.    
        proto (Proto): Prototype generation module.    
        cv4 (nn.ModuleList): Convolution layers for mask coefficients.  

    Methods:  
        forward: Return model outputs and mask coefficients.
   
    Examples:     
        Create a segmentation head
        >>> segment = Segment(nc=80, nm=32, npr=256, ch=(256, 512, 1024))
        >>> x = [torch.randn(1, 256, 80, 80), torch.randn(1, 512, 40, 40), torch.randn(1, 1024, 20, 20)]
        >>> outputs = segment(x)
    """

    def __init__(self, nc: int = 80, nm: int = 32, npr: int = 256, reg_max=16, end2end=False, ch: tuple = ()):    
        """Initialize the YOLO model attributes such as the number of masks, prototypes, and the convolution layers.  
  
        Args: 
            nc (int): Number of classes.   
            nm (int): Number of masks.
            npr (int): Number of protos.     
            reg_max (int): Maximum number of DFL channels.    
            end2end (bool): Whether to use end-to-end NMS-free detection.  
            ch (tuple): Tuple of channel sizes from backbone feature maps.
        """  
        super().__init__(nc, reg_max, end2end, ch)  
        self.nm = nm  # number of masks    
        self.npr = npr  # number of protos
        self.proto = Proto(ch[0], self.npr, self.nm)  # protos    
   
        # c4 = max(ch[0] // 4, self.nm)  
        self.cv4 = nn.Conv2d(self.c_hid, self.nm, 1)
        if end2end:   
            self.one2one_cv4 = copy.deepcopy(self.cv4)

    @property
    def one2many(self): 
        """Returns the one-to-many head components, here for backward compatibility."""
        return dict(box_head=self.cv2, cls_head=self.cv3, scale_head=self.scale, mask_head=self.cv4)     
    
    @property   
    def one2one(self):   
        """Returns the one-to-one head components."""
        return dict(box_head=self.one2one_cv2, cls_head=self.one2one_cv3, scale_head=self.one2one_scale, mask_head=self.one2one_cv4)

    def forward(self, x: list[torch.Tensor]) -> tuple | list[torch.Tensor] | dict[str, torch.Tensor]:    
        """Return model outputs and mask coefficients if training, otherwise return outputs and mask coefficients."""  
        outputs = super().forward(x)
        preds = outputs[1] if isinstance(outputs, tuple) else outputs    
        proto = self.proto(x[0])  # mask protos
        if isinstance(preds, dict):  # training and validating during training
            if self.end2end:
                preds["one2many"]["proto"] = proto
                preds["one2one"]["proto"] = proto.detach()     
            else:   
                preds["proto"] = proto
        if self.training:
            return preds 
        return (outputs, proto) if self.export else ((outputs[0], proto), preds)

    def _inference(self, x: dict[str, torch.Tensor]) -> torch.Tensor:  
        """Decode predicted bounding boxes and class probabilities, concatenated with mask coefficients.""" 
        preds = super()._inference(x)
        return torch.cat([preds, x["mask_coefficient"]], dim=1)     

    def forward_share_head(
        self, x: list[torch.Tensor], box_head: torch.nn.Module, cls_head: torch.nn.Module, scale_head: torch.nn.Module, mask_head: torch.nn.Module    
    ) -> torch.Tensor:    
        """Concatenates and returns predicted bounding boxes, class probabilities, and mask coefficients."""
        preds = super().forward_share_head(x, box_head, cls_head, scale_head)
        if mask_head is not None:     
            bs = x[0].shape[0]  # batch size 
            preds["mask_coefficient"] = torch.cat([mask_head(x[i]).view(bs, self.nm, -1) for i in range(self.nl)], 2)   
        return preds

    def postprocess(self, preds: torch.Tensor) -> torch.Tensor: 
        """Post-process YOLO model predictions.
     
        Args:
            preds (torch.Tensor): Raw predictions with shape (batch_size, num_anchors, 4 + nc + nm) with last dimension     
                format [x, y, w, h, class_probs, mask_coefficient].
   
        Returns:
            (torch.Tensor): Processed predictions with shape (batch_size, min(max_det, num_anchors), 6 + nm) and last
                dimension format [x, y, w, h, max_class_prob, class_index, mask_coefficient].
        """ 
        boxes, scores, mask_coefficient = preds.split([4, self.nc, self.nm], dim=-1)     
        scores, conf, idx = self.get_topk_index(scores, self.max_det) 
        boxes = boxes.gather(dim=1, index=idx.repeat(1, 1, 4)) 
        mask_coefficient = mask_coefficient.gather(dim=1, index=idx.repeat(1, 1, self.nm))    
        return torch.cat([boxes, scores, conf, mask_coefficient], dim=-1)

    def fuse(self) -> None:
        """Remove the one2many head for inference optimization."""
        self.cv2 = self.cv3 = self.cv4 = self.scale = None
  
class Segment26_LSPCD(Segment_LSPCD):     
    """YOLO26 Segment head for segmentation models.
 
    This class extends the Detect head to include mask prediction capabilities for instance segmentation tasks.

    Attributes:
        nm (int): Number of masks.  
        npr (int): Number of protos.
        proto (Proto): Prototype generation module.     
        cv4 (nn.ModuleList): Convolution layers for mask coefficients.     

    Methods:   
        forward: Return model outputs and mask coefficients.     

    Examples:   
        Create a segmentation head   
        >>> segment = Segment26(nc=80, nm=32, npr=256, ch=(256, 512, 1024))
        >>> x = [torch.randn(1, 256, 80, 80), torch.randn(1, 512, 40, 40), torch.randn(1, 1024, 20, 20)]
        >>> outputs = segment(x)
    """
    
    def __init__(self, nc: int = 80, nm: int = 32, npr: int = 256, reg_max=16, end2end=False, ch: tuple = ()):
        """Initialize the YOLO model attributes such as the number of masks, prototypes, and the convolution layers.  

        Args:   
            nc (int): Number of classes.
            nm (int): Number of masks.
            npr (int): Number of protos.
            reg_max (int): Maximum number of DFL channels.
            end2end (bool): Whether to use end-to-end NMS-free detection.
            ch (tuple): Tuple of channel sizes from backbone feature maps.
        """
        super().__init__(nc, nm, npr, reg_max, end2end, ch)
        self.proto = Proto26(ch, self.npr, self.nm, nc)  # protos 

    def forward(self, x: list[torch.Tensor]) -> tuple | list[torch.Tensor] | dict[str, torch.Tensor]:    
        """Return model outputs and mask coefficients if training, otherwise return outputs and mask coefficients."""    
        outputs = Detect_LSPCD.forward(self, x)    
        preds = outputs[1] if isinstance(outputs, tuple) else outputs
        proto = self.proto(x)  # mask protos     
        if isinstance(preds, dict):  # training and validating during training
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
        """Remove the one2many head and extra part of proto module for inference optimization."""
        super().fuse()   
        if hasattr(self.proto, "fuse"):
            self.proto.fuse()

class OBB_LSPCD(Detect_LSPCD): 
    """YOLO OBB detection head for detection with rotation models.    
 
    This class extends the Detect head to include oriented bounding box prediction with rotation angles.
  
    Attributes:
        ne (int): Number of extra parameters.     
        cv4 (nn.ModuleList): Convolution layers for angle prediction.
        angle (torch.Tensor): Predicted rotation angles.

    Methods:
        forward: Concatenate and return predicted bounding boxes and class probabilities.
        decode_bboxes: Decode rotated bounding boxes.
 
    Examples:    
        Create an OBB detection head
        >>> obb = OBB(nc=80, ne=1, ch=(256, 512, 1024)) 
        >>> x = [torch.randn(1, 256, 80, 80), torch.randn(1, 512, 40, 40), torch.randn(1, 1024, 20, 20)]
        >>> outputs = obb(x)
    """   

    def __init__(self, nc: int = 80, ne: int = 1, reg_max=16, end2end=False, ch: tuple = ()):
        """Initialize OBB with number of classes `nc` and layer channels `ch`.    
  
        Args:
            nc (int): Number of classes.
            ne (int): Number of extra parameters.
            reg_max (int): Maximum number of DFL channels.    
            end2end (bool): Whether to use end-to-end NMS-free detection.
            ch (tuple): Tuple of channel sizes from backbone feature maps.
        """
        super().__init__(nc, reg_max, end2end, ch)
        self.ne = ne  # number of extra parameters

        self.cv4 = nn.Conv2d(self.c_hid, self.ne, 1)  
        if end2end:
            self.one2one_cv4 = copy.deepcopy(self.cv4)
    
    @property 
    def one2many(self):
        """Returns the one-to-many head components, here for backward compatibility."""     
        return dict(box_head=self.cv2, cls_head=self.cv3, scale_head=self.scale, angle_head=self.cv4)

    @property
    def one2one(self):
        """Returns the one-to-one head components."""
        return dict(box_head=self.one2one_cv2, cls_head=self.one2one_cv3, scale_head=self.one2one_scale, angle_head=self.one2one_cv4)
    
    def _inference(self, x: dict[str, torch.Tensor]) -> torch.Tensor:  
        """Decode predicted bounding boxes and class probabilities, concatenated with rotation angles."""    
        # For decode_bboxes convenience 
        self.angle = x["angle"]  # TODO: need to test obb
        preds = super()._inference(x)
        return torch.cat([preds, x["angle"]], dim=1)    
     
    def forward_share_head(
        self, x: list[torch.Tensor], box_head: torch.nn.Module, cls_head: torch.nn.Module, scale_head: torch.nn.Module, angle_head: torch.nn.Module   
    ) -> torch.Tensor:
        """Concatenates and returns predicted bounding boxes, class probabilities, and angles."""
        preds = super().forward_share_head(x, box_head, cls_head, scale_head)
        if angle_head is not None:
            bs = x[0].shape[0]  # batch size    
            angle = torch.cat(
                [angle_head(x[i]).view(bs, self.ne, -1) for i in range(self.nl)], 2
            )  # OBB theta logits     
            angle = (angle.sigmoid() - 0.25) * math.pi  # [-pi/4, 3pi/4]
            preds["angle"] = angle   
        return preds     
 
    def decode_bboxes(self, bboxes: torch.Tensor, anchors: torch.Tensor) -> torch.Tensor:
        """Decode rotated bounding boxes."""
        return dist2rbox(bboxes, self.angle, anchors, dim=1)
 
    def postprocess(self, preds: torch.Tensor) -> torch.Tensor:     
        """Post-process YOLO model predictions.  
 
        Args:     
            preds (torch.Tensor): Raw predictions with shape (batch_size, num_anchors, 4 + nc + ne) with last dimension   
                format [x, y, w, h, class_probs, angle].   
 
        Returns:
            (torch.Tensor): Processed predictions with shape (batch_size, min(max_det, num_anchors), 7) and last
                dimension format [x, y, w, h, max_class_prob, class_index, angle].     
        """
        boxes, scores, angle = preds.split([4, self.nc, self.ne], dim=-1)
        scores, conf, idx = self.get_topk_index(scores, self.max_det)   
        boxes = boxes.gather(dim=1, index=idx.repeat(1, 1, 4))     
        angle = angle.gather(dim=1, index=idx.repeat(1, 1, self.ne))
        return torch.cat([boxes, scores, conf, angle], dim=-1)
   
    def fuse(self) -> None:    
        """Remove the one2many head for inference optimization."""
        self.cv2 = self.cv3 = self.cv4 = self.scale = None

class OBB26_LSPCD(OBB_LSPCD):
    """YOLO26 OBB detection head for detection with rotation models. This class extends the OBB head with modified angle    
    processing that outputs raw angle predictions without sigmoid transformation, compared to the original    
    OBB class.   

    Attributes:
        ne (int): Number of extra parameters.    
        cv4 (nn.ModuleList): Convolution layers for angle prediction.
        angle (torch.Tensor): Predicted rotation angles.

    Methods:
        forward_head: Concatenate and return predicted bounding boxes, class probabilities, and raw angles.

    Examples:    
        Create an OBB26 detection head
        >>> obb26 = OBB26(nc=80, ne=1, ch=(256, 512, 1024))     
        >>> x = [torch.randn(1, 256, 80, 80), torch.randn(1, 512, 40, 40), torch.randn(1, 1024, 20, 20)]     
        >>> outputs = obb26(x).   
    """    

    def forward_share_head(
        self, x: list[torch.Tensor], box_head: torch.nn.Module, cls_head: torch.nn.Module, scale_head: torch.nn.Module, angle_head: torch.nn.Module   
    ) -> torch.Tensor: 
        """Concatenates and returns predicted bounding boxes, class probabilities, and raw angles."""
        preds = Detect_LSPCD.forward_share_head(self, x, box_head, cls_head, scale_head)
        if angle_head is not None:
            bs = x[0].shape[0]  # batch size
            angle = torch.cat(   
                [angle_head(x[i]).view(bs, self.ne, -1) for i in range(self.nl)], 2  
            )  # OBB theta logits (raw output without sigmoid transformation)    
            preds["angle"] = angle
        return preds

class Pose_LSPCD(Detect_LSPCD): 
    """YOLO Pose head for keypoints models.     
    
    This class extends the Detect head to include keypoint prediction capabilities for pose estimation tasks.  
  
    Attributes: 
        kpt_shape (tuple): Number of keypoints and dimensions (2 for x,y or 3 for x,y,visible). 
        nk (int): Total number of keypoint values.     
        cv4 (nn.ModuleList): Convolution layers for keypoint prediction.     
 
    Methods:    
        forward: Perform forward pass through YOLO model and return predictions.  
        kpts_decode: Decode keypoints from predictions. 
    
    Examples:
        Create a pose detection head 
        >>> pose = Pose(nc=80, kpt_shape=(17, 3), ch=(256, 512, 1024))     
        >>> x = [torch.randn(1, 256, 80, 80), torch.randn(1, 512, 40, 40), torch.randn(1, 1024, 20, 20)]     
        >>> outputs = pose(x)     
    """
     
    def __init__(self, nc: int = 80, kpt_shape: tuple = (17, 3), reg_max=16, end2end=False, ch: tuple = ()):   
        """Initialize YOLO network with default parameters and Convolutional Layers. 

        Args:
            nc (int): Number of classes. 
            kpt_shape (tuple): Number of keypoints, number of dims (2 for x,y or 3 for x,y,visible).
            reg_max (int): Maximum number of DFL channels.
            end2end (bool): Whether to use end-to-end NMS-free detection.
            ch (tuple): Tuple of channel sizes from backbone feature maps.
        """   
        super().__init__(nc, reg_max, end2end, ch)
        self.kpt_shape = kpt_shape  # number of keypoints, number of dims (2 for x,y or 3 for x,y,visible)
        self.nk = kpt_shape[0] * kpt_shape[1]  # number of keypoints total
     
        # c4 = max(ch[0] // 4, self.nk) 
        self.cv4 = nn.Conv2d(self.c_hid, self.nk, 1)
        if end2end:
            self.one2one_cv4 = copy.deepcopy(self.cv4)    
    
    @property    
    def one2many(self):  
        """Returns the one-to-many head components, here for backward compatibility."""    
        return dict(box_head=self.cv2, cls_head=self.cv3, scale_head=self.scale, pose_head=self.cv4)     

    @property
    def one2one(self):
        """Returns the one-to-one head components."""  
        return dict(box_head=self.one2one_cv2, cls_head=self.one2one_cv3, scale_head=self.one2one_scale, pose_head=self.one2one_cv4)    
  
    def _inference(self, x: dict[str, torch.Tensor]) -> torch.Tensor:    
        """Decode predicted bounding boxes and class probabilities, concatenated with keypoints."""    
        preds = super()._inference(x) 
        return torch.cat([preds, self.kpts_decode(x["kpts"])], dim=1)   

    def forward_share_head(
        self, x: list[torch.Tensor], box_head: torch.nn.Module, cls_head: torch.nn.Module, scale_head: torch.nn.Module, pose_head: torch.nn.Module 
    ) -> torch.Tensor:     
        """Concatenates and returns predicted bounding boxes, class probabilities, and keypoints."""   
        preds = super().forward_share_head(x, box_head, cls_head, scale_head) 
        if pose_head is not None:     
            bs = x[0].shape[0]  # batch size   
            preds["kpts"] = torch.cat([pose_head(x[i]).view(bs, self.nk, -1) for i in range(self.nl)], 2)
        return preds    

    def postprocess(self, preds: torch.Tensor) -> torch.Tensor:  
        """Post-process YOLO model predictions.    

        Args:
            preds (torch.Tensor): Raw predictions with shape (batch_size, num_anchors, 4 + nc + nk) with last dimension
                format [x, y, w, h, class_probs, keypoints].   
   
        Returns:
            (torch.Tensor): Processed predictions with shape (batch_size, min(max_det, num_anchors), 6 + self.nk) and     
                last dimension format [x, y, w, h, max_class_prob, class_index, keypoints]. 
        """    
        boxes, scores, kpts = preds.split([4, self.nc, self.nk], dim=-1)    
        scores, conf, idx = self.get_topk_index(scores, self.max_det)
        boxes = boxes.gather(dim=1, index=idx.repeat(1, 1, 4)) 
        kpts = kpts.gather(dim=1, index=idx.repeat(1, 1, self.nk))   
        return torch.cat([boxes, scores, conf, kpts], dim=-1)

    def fuse(self) -> None:
        """Remove the one2many head for inference optimization."""
        self.cv2 = self.cv3 = self.cv4 = self.scale = None    
  
    def kpts_decode(self, kpts: torch.Tensor) -> torch.Tensor:    
        """Decode keypoints from predictions."""  
        ndim = self.kpt_shape[1]
        bs = kpts.shape[0]
        if self.export:
            y = kpts.view(bs, *self.kpt_shape, -1)
            a = (y[:, :, :2] * 2.0 + (self.anchors - 0.5)) * self.strides     
            if ndim == 3:     
                a = torch.cat((a, y[:, :, 2:3].sigmoid()), 2)     
            return a.view(bs, self.nk, -1)     
        else:    
            y = kpts.clone()
            if ndim == 3:
                if NOT_MACOS14:    
                    y[:, 2::ndim].sigmoid_()  
                else:  # Apple macOS14 MPS bug https://github.com/ultralytics/ultralytics/pull/21878
                    y[:, 2::ndim] = y[:, 2::ndim].sigmoid()
            y[:, 0::ndim] = (y[:, 0::ndim] * 2.0 + (self.anchors[0] - 0.5)) * self.strides
            y[:, 1::ndim] = (y[:, 1::ndim] * 2.0 + (self.anchors[1] - 0.5)) * self.strides 
            return y 
  

class Pose26_LSPCD(Pose_LSPCD):
    """YOLO26 Pose head for keypoints models.   
 
    This class extends the Detect head to include keypoint prediction capabilities for pose estimation tasks.

    Attributes: 
        kpt_shape (tuple): Number of keypoints and dimensions (2 for x,y or 3 for x,y,visible).
        nk (int): Total number of keypoint values.
        cv4 (nn.ModuleList): Convolution layers for keypoint prediction.
    
    Methods:
        forward: Perform forward pass through YOLO model and return predictions.     
        kpts_decode: Decode keypoints from predictions.
   
    Examples:
        Create a pose detection head    
        >>> pose = Pose(nc=80, kpt_shape=(17, 3), ch=(256, 512, 1024))    
        >>> x = [torch.randn(1, 256, 80, 80), torch.randn(1, 512, 40, 40), torch.randn(1, 1024, 20, 20)]
        >>> outputs = pose(x)   
    """

    def __init__(self, nc: int = 80, kpt_shape: tuple = (17, 3), reg_max=16, end2end=False, ch: tuple = ()):    
        """Initialize YOLO network with default parameters and Convolutional Layers.

        Args:
            nc (int): Number of classes.  
            kpt_shape (tuple): Number of keypoints, number of dims (2 for x,y or 3 for x,y,visible).     
            reg_max (int): Maximum number of DFL channels.     
            end2end (bool): Whether to use end-to-end NMS-free detection.
            ch (tuple): Tuple of channel sizes from backbone feature maps.
        """
        super().__init__(nc, kpt_shape, reg_max, end2end, ch)     
        self.flow_model = RealNVP()   

        c4 = max(ch[0] // 4, kpt_shape[0] * (kpt_shape[1] + 2))
        self.cv4 = Conv(self.c_hid, c4, 3)   
 
        self.cv4_kpts = nn.Conv2d(c4, self.nk, 1) 
        self.nk_sigma = kpt_shape[0] * 2  # sigma_x, sigma_y for each keypoint     
        self.cv4_sigma = nn.Conv2d(c4, self.nk_sigma, 1)     
 
        if end2end:
            self.one2one_cv4 = copy.deepcopy(self.cv4)  
            self.one2one_cv4_kpts = copy.deepcopy(self.cv4_kpts)
            self.one2one_cv4_sigma = copy.deepcopy(self.cv4_sigma)

    @property
    def one2many(self):  
        """Returns the one-to-many head components, here for backward compatibility."""
        return dict(
            box_head=self.cv2,
            cls_head=self.cv3,  
            scale_head=self.scale,
            pose_head=self.cv4,
            kpts_head=self.cv4_kpts,
            kpts_sigma_head=self.cv4_sigma,
        )  
 
    @property  
    def one2one(self):   
        """Returns the one-to-one head components.""" 
        return dict(  
            box_head=self.one2one_cv2,  
            cls_head=self.one2one_cv3,
            scale_head=self.one2one_scale,   
            pose_head=self.one2one_cv4,     
            kpts_head=self.one2one_cv4_kpts,     
            kpts_sigma_head=self.one2one_cv4_sigma,
        )   
     
    def forward_share_head(
        self,  
        x: list[torch.Tensor],  
        box_head: torch.nn.Module,
        cls_head: torch.nn.Module,
        scale_head: torch.nn.Module,
        pose_head: torch.nn.Module,    
        kpts_head: torch.nn.Module,  
        kpts_sigma_head: torch.nn.Module,
    ) -> torch.Tensor:
        """Concatenates and returns predicted bounding boxes, class probabilities, and keypoints."""
        preds = Detect_LSPCD.forward_share_head(self, x, box_head, cls_head, scale_head)  
        if pose_head is not None:
            bs = x[0].shape[0]  # batch size
            features = [pose_head(x[i]) for i in range(self.nl)]     
            preds["kpts"] = torch.cat([kpts_head(features[i]).view(bs, self.nk, -1) for i in range(self.nl)], 2)    
            if self.training:   
                preds["kpts_sigma"] = torch.cat(
                    [kpts_sigma_head(features[i]).view(bs, self.nk_sigma, -1) for i in range(self.nl)], 2
                )     
        return preds 

    def fuse(self) -> None:  
        """Remove the one2many head for inference optimization."""
        super().fuse()    
        self.cv4_kpts = self.cv4_sigma = self.flow_model = self.one2one_cv4_sigma = None

    def kpts_decode(self, kpts: torch.Tensor) -> torch.Tensor:
        """Decode keypoints from predictions."""
        ndim = self.kpt_shape[1]    
        bs = kpts.shape[0] 
        if self.export:
            y = kpts.view(bs, *self.kpt_shape, -1)  
            # NCNN fix
            a = (y[:, :, :2] + self.anchors) * self.strides
            if ndim == 3:  
                a = torch.cat((a, y[:, :, 2:3].sigmoid()), 2) 
            return a.view(bs, self.nk, -1)
        else:
            y = kpts.clone()
            if ndim == 3:    
                if NOT_MACOS14:
                    y[:, 2::ndim].sigmoid_()
                else:  # Apple macOS14 MPS bug https://github.com/ultralytics/ultralytics/pull/21878    
                    y[:, 2::ndim] = y[:, 2::ndim].sigmoid()     
            y[:, 0::ndim] = (y[:, 0::ndim] + self.anchors[0]) * self.strides
            y[:, 1::ndim] = (y[:, 1::ndim] + self.anchors[1]) * self.strides
            return y