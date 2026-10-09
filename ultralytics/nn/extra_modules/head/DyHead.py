import copy     
import math  
  
import torch
import torch.nn as nn     
import torch.nn.functional as F
   
from ultralytics.nn.modules.conv import Conv, DWConv
from ultralytics.nn.modules.head import NOT_MACOS14, Detect, Proto, Proto26, RealNVP, dist2rbox    
from ultralytics.nn.legacy_compat import kwargs_adapter, legacy_positional  # 0526 版签名兼容

try:
    from mmcv.cnn import build_activation_layer, build_norm_layer     
    from mmcv.ops.modulated_deform_conv import ModulatedDeformConv2d  
    from mmengine.model import constant_init, normal_init
except ImportError as e:  # pragma: no cover - exercised only in missing optional dependency environments
    build_activation_layer = build_norm_layer = ModulatedDeformConv2d = None
    constant_init = normal_init = None    
    _DYHEAD_IMPORT_ERROR = e
else:
    _DYHEAD_IMPORT_ERROR = None
  
__all__ = (    
    "Detect_DyHead",
    "Segment_DyHead",   
    "Segment26_DyHead",
    "OBB_DyHead",
    "OBB26_DyHead",     
    "Pose_DyHead", 
    "Pose26_DyHead",     
)
   
 
def _check_dyhead_deps():
    if _DYHEAD_IMPORT_ERROR is not None:
        raise ImportError("DyHead requires mmcv and mmengine with ModulatedDeformConv2d support.") from _DYHEAD_IMPORT_ERROR

    
def _make_divisible(v, divisor, min_value=None):  
    if min_value is None:  
        min_value = divisor   
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)    
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v     


class h_sigmoid(nn.Module):  
    def __init__(self, inplace=True, h_max=1):
        super().__init__()
        self.relu = nn.ReLU6(inplace=inplace)    
        self.h_max = h_max  

    def forward(self, x):
        return self.relu(x + 3) * self.h_max / 6    
   
  
class DyReLU(nn.Module):     
    def __init__(  
        self,   
        inp,  
        reduction=4,
        lambda_a=1.0,
        K2=True,
        use_bias=True,
        use_spatial=False,
        init_a=(1.0, 0.0),     
        init_b=(0.0, 0.0),
    ):
        super().__init__()   
        self.oup = inp     
        self.lambda_a = lambda_a * 2    
        self.K2 = K2
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.use_bias = use_bias     
        self.exp = 4 if K2 and use_bias else 2 if K2 or use_bias else 1
        self.init_a = init_a
        self.init_b = init_b    
        squeeze = inp // reduction if reduction == 4 else _make_divisible(inp // reduction, 4)     
        self.fc = nn.Sequential(nn.Linear(inp, squeeze), nn.ReLU(inplace=True), nn.Linear(squeeze, self.oup * self.exp), h_sigmoid())  
        self.spa = nn.Sequential(nn.Conv2d(inp, 1, kernel_size=1), nn.BatchNorm2d(1)) if use_spatial else None  

    def forward(self, x): 
        if isinstance(x, list):     
            x_in, x_out = x
        else:
            x_in = x_out = x 
        b, c, h, w = x_in.size()
        y = self.avg_pool(x_in).view(b, c)
        y = self.fc(y).view(b, self.oup * self.exp, 1, 1)
        if self.exp == 4:
            a1, b1, a2, b2 = torch.split(y, self.oup, dim=1)
            a1 = (a1 - 0.5) * self.lambda_a + self.init_a[0]
            a2 = (a2 - 0.5) * self.lambda_a + self.init_a[1]
            b1 = b1 - 0.5 + self.init_b[0]
            b2 = b2 - 0.5 + self.init_b[1]    
            out = torch.max(x_out * a1 + b1, x_out * a2 + b2)
        elif self.exp == 2:
            a1, a2 = torch.split(y, self.oup, dim=1)     
            a1 = (a1 - 0.5) * self.lambda_a + self.init_a[0]     
            if self.use_bias:    
                a2 = a2 - 0.5 + self.init_b[0]   
                out = x_out * a1 + a2
            else:  
                a2 = (a2 - 0.5) * self.lambda_a + self.init_a[1]    
                out = torch.max(x_out * a1, x_out * a2)    
        else:
            out = x_out * ((y - 0.5) * self.lambda_a + self.init_a[0])

        if self.spa:
            ys = self.spa(x_in).view(b, -1)
            ys = F.softmax(ys, dim=1).view(b, 1, h, w) * h * w   
            ys = F.hardtanh(ys, 0, 3, inplace=True) / 3 
            out = out * ys  
        return out


class DyDCNv2(nn.Module):  
    def __init__(self, in_channels, out_channels, stride=1, norm_cfg=None):  
        super().__init__()
        _check_dyhead_deps()    
        if norm_cfg is None:    
            norm_cfg = dict(type="GN", num_groups=16, requires_grad=True)
        self.with_norm = norm_cfg is not None   
        bias = not self.with_norm   
        self.conv = ModulatedDeformConv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=bias)    
        if self.with_norm:
            self.norm = build_norm_layer(norm_cfg, out_channels)[1] 
 
    def forward(self, x, offset, mask):
        x = self.conv(x.contiguous(), offset, mask)  
        if self.with_norm:
            x = self.norm(x)   
        return x     
    
 
class DyHeadBlock(nn.Module):
    def __init__(self, in_channels, norm_type="GN", zero_init_offset=True, act_cfg=None):     
        super().__init__()
        _check_dyhead_deps()
        self.zero_init_offset = zero_init_offset
        self.offset_and_mask_dim = 3 * 3 * 3
        self.offset_dim = 2 * 3 * 3     
        act_cfg = act_cfg or dict(type="HSigmoid", bias=3.0, divisor=6.0)
        if norm_type == "GN":  
            norm_dict = dict(type="GN", num_groups=16, requires_grad=True)
        elif norm_type == "BN":
            norm_dict = dict(type="BN", requires_grad=True)   
        else:
            norm_dict = None

        self.spatial_conv_high = DyDCNv2(in_channels, in_channels, norm_cfg=norm_dict)
        self.spatial_conv_mid = DyDCNv2(in_channels, in_channels, norm_cfg=norm_dict)
        self.spatial_conv_low = DyDCNv2(in_channels, in_channels, stride=2, norm_cfg=norm_dict)
        self.spatial_conv_offset = nn.Conv2d(in_channels, self.offset_and_mask_dim, 3, padding=1)
        self.scale_attn_module = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, 1, 1), 
            nn.ReLU(inplace=True),
            build_activation_layer(act_cfg),  
        )
        self.task_attn_module = DyReLU(in_channels) 
        self._init_weights()
     
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):    
                normal_init(m, 0, 0.01) 
        if self.zero_init_offset:
            constant_init(self.spatial_conv_offset, 0)
  
    def forward(self, x):
        outs = []     
        for level in range(len(x)):  
            offset_and_mask = self.spatial_conv_offset(x[level])
            offset = offset_and_mask[:, : self.offset_dim, :, :]
            mask = offset_and_mask[:, self.offset_dim :, :, :].sigmoid() 

            mid_feat = self.spatial_conv_mid(x[level], offset, mask)    
            sum_feat = mid_feat * self.scale_attn_module(mid_feat)
            summed_levels = 1     
            if level > 0:
                low_feat = self.spatial_conv_low(x[level - 1], offset, mask)    
                sum_feat += low_feat * self.scale_attn_module(low_feat)
                summed_levels += 1  
            if level < len(x) - 1:   
                high_feat = F.interpolate(
                    self.spatial_conv_high(x[level + 1], offset, mask),    
                    size=x[level].shape[-2:],  
                    mode="bilinear",  
                    align_corners=True,   
                )    
                sum_feat += high_feat * self.scale_attn_module(high_feat)
                summed_levels += 1     
            outs.append(self.task_attn_module(sum_feat / summed_levels))
        return outs


class Detect_DyHead(Detect):   
    # 0526 版子类 Detect_DyHeadWithDCNV3 / V4 按 (nc, hidc, block_num, ch) 做位置调用
    @legacy_positional(
        sentinels=("reg_max", "end2end"),
        adapter=kwargs_adapter(("nc", "hidc", "block_num", "ch")),
    )
    def __init__(self, nc=80, hidc=256, block_num=2, reg_max=16, end2end=False, ch=()):
        super().__init__(nc, reg_max, end2end, ch)
        self.hidc = hidc
        self.conv = nn.ModuleList(Conv(c, hidc, 1) for c in ch)  
        self.dyhead = nn.Sequential(*[DyHeadBlock(hidc) for _ in range(block_num)])
        c2, c3 = max((16, ch[0] // 4, self.reg_max * 4)), max(hidc, min(self.nc, 100))    
        self.cv2 = nn.ModuleList(nn.Sequential(Conv(hidc, c2, 3), Conv(c2, c2, 3), nn.Conv2d(c2, 4 * self.reg_max, 1)) for _ in ch)    
        self.cv3 = (
            nn.ModuleList(nn.Sequential(Conv(hidc, c3, 3), Conv(c3, c3, 3), nn.Conv2d(c3, self.nc, 1)) for _ in ch)
            if self.legacy
            else nn.ModuleList(  
                nn.Sequential(
                    nn.Sequential(DWConv(hidc, hidc, 3), Conv(hidc, c3, 1)), 
                    nn.Sequential(DWConv(c3, c3, 3), Conv(c3, c3, 1)),   
                    nn.Conv2d(c3, self.nc, 1),     
                )
                for _ in ch    
            ) 
        )
        if end2end:
            self.one2one_cv2 = copy.deepcopy(self.cv2)  
            self.one2one_cv3 = copy.deepcopy(self.cv3) 

    @property
    def one2many(self):
        return dict(box_head=self.cv2, cls_head=self.cv3)
    
    @property
    def one2one(self):     
        return dict(box_head=self.one2one_cv2, cls_head=self.one2one_cv3)   

    def forward_dyhead_features(self, x): 
        x = [self.conv[i](x[i]) for i in range(self.nl)]
        return self.dyhead(x)     
     
    def forward(self, x):   
        feats = self.forward_dyhead_features(x) 
        preds = self.forward_head(feats, **self.one2many)    
        if self.end2end:  
            one2one = self.forward_head([xi.detach() for xi in feats], **self.one2one)     
            preds = {"one2many": preds, "one2one": one2one}
        if self.training:
            return preds  
        y = self._inference(preds["one2one"] if self.end2end else preds)   
        if self.end2end: 
            y = self.postprocess(y.permute(0, 2, 1))   
        return y if self.export else (y, preds)
  
     
class Segment_DyHead(Detect_DyHead):     
    def __init__(self, nc=80, nm=32, npr=256, hidc=256, block_num=2, reg_max=16, end2end=False, ch=()):
        super().__init__(nc, hidc, block_num, reg_max, end2end, ch)    
        self.nm = nm
        self.npr = npr
        self.proto = Proto(ch[0], self.npr, self.nm) 
        c4 = max(hidc // 4, self.nm)   
        self.cv4 = nn.ModuleList(nn.Sequential(Conv(hidc, c4, 3), Conv(c4, c4, 3), nn.Conv2d(c4, self.nm, 1)) for _ in ch)
        if end2end:  
            self.one2one_cv4 = copy.deepcopy(self.cv4)

    @property    
    def one2many(self):    
        return dict(box_head=self.cv2, cls_head=self.cv3, mask_head=self.cv4)   
 
    @property
    def one2one(self): 
        return dict(box_head=self.one2one_cv2, cls_head=self.one2one_cv3, mask_head=self.one2one_cv4)
 
    def forward(self, x):  
        outputs = Detect_DyHead.forward(self, x) 
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

    def _inference(self, x):
        preds = super()._inference(x)
        return torch.cat([preds, x["mask_coefficient"]], dim=1) 

    def forward_head(self, x, box_head, cls_head, mask_head): 
        preds = super().forward_head(x, box_head, cls_head)
        if mask_head is not None:     
            bs = x[0].shape[0] 
            preds["mask_coefficient"] = torch.cat([mask_head[i](x[i]).view(bs, self.nm, -1) for i in range(self.nl)], 2)    
        return preds
  
    def postprocess(self, preds):
        boxes, scores, mask_coefficient = preds.split([4, self.nc, self.nm], dim=-1)
        scores, conf, idx = self.get_topk_index(scores, self.max_det)
        boxes = boxes.gather(dim=1, index=idx.repeat(1, 1, 4))     
        mask_coefficient = mask_coefficient.gather(dim=1, index=idx.repeat(1, 1, self.nm))     
        return torch.cat([boxes, scores, conf, mask_coefficient], dim=-1)    
    
    def fuse(self):   
        self.cv2 = self.cv3 = self.cv4 = None
 

class Segment26_DyHead(Segment_DyHead):   
    def __init__(self, nc=80, nm=32, npr=256, hidc=256, block_num=2, reg_max=16, end2end=False, ch=()): 
        super().__init__(nc, nm, npr, hidc, block_num, reg_max, end2end, ch)
        self.proto = Proto26(ch, self.npr, self.nm, nc)

    def forward(self, x):   
        outputs = Detect_DyHead.forward(self, x)
        preds = outputs[1] if isinstance(outputs, tuple) else outputs     
        proto = self.proto(x)     
        if isinstance(preds, dict):
            if self.end2end:
                preds["one2many"]["proto"] = proto
                preds["one2one"]["proto"] = tuple(p.detach() for p in proto) if isinstance(proto, tuple) else proto.detach()
            else:
                preds["proto"] = proto
        if self.training:
            return preds  
        return (outputs, proto) if self.export else ((outputs[0], proto), preds)

    def fuse(self):
        super().fuse()
        if hasattr(self.proto, "fuse"):
            self.proto.fuse()
    

class OBB_DyHead(Detect_DyHead):
    def __init__(self, nc=80, ne=1, hidc=256, block_num=2, reg_max=16, end2end=False, ch=()):   
        super().__init__(nc, hidc, block_num, reg_max, end2end, ch) 
        self.ne = ne
        c4 = max(hidc // 4, self.ne)
        self.cv4 = nn.ModuleList(nn.Sequential(Conv(hidc, c4, 3), Conv(c4, c4, 3), nn.Conv2d(c4, self.ne, 1)) for _ in ch)   
        if end2end:
            self.one2one_cv4 = copy.deepcopy(self.cv4)
    
    @property     
    def one2many(self):
        return dict(box_head=self.cv2, cls_head=self.cv3, angle_head=self.cv4)  
   
    @property     
    def one2one(self):
        return dict(box_head=self.one2one_cv2, cls_head=self.one2one_cv3, angle_head=self.one2one_cv4)
 
    def _inference(self, x):     
        self.angle = x["angle"]     
        preds = super()._inference(x)
        return torch.cat([preds, x["angle"]], dim=1)
    
    def forward_head(self, x, box_head, cls_head, angle_head):
        preds = super().forward_head(x, box_head, cls_head)
        if angle_head is not None:
            bs = x[0].shape[0]     
            angle = torch.cat([angle_head[i](x[i]).view(bs, self.ne, -1) for i in range(self.nl)], 2)     
            preds["angle"] = (angle.sigmoid() - 0.25) * math.pi 
        return preds

    def decode_bboxes(self, bboxes, anchors):
        return dist2rbox(bboxes, self.angle, anchors, dim=1)

    def postprocess(self, preds):  
        boxes, scores, angle = preds.split([4, self.nc, self.ne], dim=-1)   
        scores, conf, idx = self.get_topk_index(scores, self.max_det)     
        boxes = boxes.gather(dim=1, index=idx.repeat(1, 1, 4))     
        angle = angle.gather(dim=1, index=idx.repeat(1, 1, self.ne))
        return torch.cat([boxes, scores, conf, angle], dim=-1)     
 
    def fuse(self):
        self.cv2 = self.cv3 = self.cv4 = None
  
 
class OBB26_DyHead(OBB_DyHead):
    def forward_head(self, x, box_head, cls_head, angle_head):   
        preds = Detect_DyHead.forward_head(self, x, box_head, cls_head)     
        if angle_head is not None:
            bs = x[0].shape[0]
            preds["angle"] = torch.cat([angle_head[i](x[i]).view(bs, self.ne, -1) for i in range(self.nl)], 2)   
        return preds
    
   
class Pose_DyHead(Detect_DyHead):     
    def __init__(self, nc=80, kpt_shape=(17, 3), hidc=256, block_num=2, reg_max=16, end2end=False, ch=()):    
        super().__init__(nc, hidc, block_num, reg_max, end2end, ch) 
        self.kpt_shape = kpt_shape     
        self.nk = kpt_shape[0] * kpt_shape[1]   
        c4 = max(hidc // 4, self.nk)
        self.cv4 = nn.ModuleList(nn.Sequential(Conv(hidc, c4, 3), Conv(c4, c4, 3), nn.Conv2d(c4, self.nk, 1)) for _ in ch)
        if end2end:   
            self.one2one_cv4 = copy.deepcopy(self.cv4)     

    @property
    def one2many(self):
        return dict(box_head=self.cv2, cls_head=self.cv3, pose_head=self.cv4)    

    @property
    def one2one(self):
        return dict(box_head=self.one2one_cv2, cls_head=self.one2one_cv3, pose_head=self.one2one_cv4)
  
    def _inference(self, x):
        preds = super()._inference(x)
        return torch.cat([preds, self.kpts_decode(x["kpts"])], dim=1)

    def forward_head(self, x, box_head, cls_head, pose_head):  
        preds = super().forward_head(x, box_head, cls_head)
        if pose_head is not None:
            bs = x[0].shape[0] 
            preds["kpts"] = torch.cat([pose_head[i](x[i]).view(bs, self.nk, -1) for i in range(self.nl)], 2)  
        return preds  
    
    def postprocess(self, preds):
        boxes, scores, kpts = preds.split([4, self.nc, self.nk], dim=-1) 
        scores, conf, idx = self.get_topk_index(scores, self.max_det)
        boxes = boxes.gather(dim=1, index=idx.repeat(1, 1, 4))
        kpts = kpts.gather(dim=1, index=idx.repeat(1, 1, self.nk))    
        return torch.cat([boxes, scores, conf, kpts], dim=-1)    

    def fuse(self):   
        self.cv2 = self.cv3 = self.cv4 = None

    def kpts_decode(self, kpts):
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

   
class Pose26_DyHead(Pose_DyHead):
    def __init__(self, nc=80, kpt_shape=(17, 3), hidc=256, block_num=2, reg_max=16, end2end=False, ch=()):   
        super().__init__(nc, kpt_shape, hidc, block_num, reg_max, end2end, ch)  
        self.flow_model = RealNVP()     
        c4 = max(hidc // 4, kpt_shape[0] * (kpt_shape[1] + 2))     
        self.cv4 = nn.ModuleList(nn.Sequential(Conv(hidc, c4, 3), Conv(c4, c4, 3)) for _ in ch)
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
            pose_head=self.cv4,
            kpts_head=self.cv4_kpts,
            kpts_sigma_head=self.cv4_sigma,
        )

    @property
    def one2one(self):    
        return dict(
            box_head=self.one2one_cv2, 
            cls_head=self.one2one_cv3,     
            pose_head=self.one2one_cv4,   
            kpts_head=self.one2one_cv4_kpts,
            kpts_sigma_head=self.one2one_cv4_sigma,    
        )  
    
    def forward_head(self, x, box_head, cls_head, pose_head, kpts_head, kpts_sigma_head):     
        preds = Detect_DyHead.forward_head(self, x, box_head, cls_head)
        if pose_head is not None:
            bs = x[0].shape[0]
            features = [pose_head[i](x[i]) for i in range(self.nl)]  
            preds["kpts"] = torch.cat([kpts_head[i](features[i]).view(bs, self.nk, -1) for i in range(self.nl)], 2)
            if self.training:     
                preds["kpts_sigma"] = torch.cat(   
                    [kpts_sigma_head[i](features[i]).view(bs, self.nk_sigma, -1) for i in range(self.nl)], 2    
                ) 
        return preds

    def fuse(self):
        super().fuse()    
        self.cv4_kpts = self.cv4_sigma = self.flow_model = self.one2one_cv4_sigma = None
 
    def kpts_decode(self, kpts):    
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
