# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
    
import timm 
import contextlib  
import pickle   
import re     
import types   
from copy import deepcopy
from pathlib import Path 
from functools import partial

import torch
import torch.nn as nn
    
from ultralytics.nn.autobackend import check_class_names  
from ultralytics.nn.modules import *  
from ultralytics.nn.extra_modules import *    
from ultralytics.utils import DEFAULT_CFG_DICT, LOGGER, YAML, colorstr, emojis
from ultralytics.utils.checks import check_requirements, check_suffix, check_yaml  
from ultralytics.utils.loss import (
    E2ELoss,
    PoseLoss26,     
    v8ClassificationLoss,
    v8DetectionLoss,  
    v8OBBLoss,     
    v8PoseLoss,
    v8SegmentationLoss,
) 
from ultralytics.utils.ops import make_divisible    
from ultralytics.utils.patches import torch_load     
from ultralytics.utils.plotting import feature_visualization    
from ultralytics.utils.torch_utils import (  
    fuse_conv_and_bn,
    fuse_deconv_and_bn,
    initialize_weights,   
    intersect_dicts,
    model_info,     
    scale_img,     
    smart_inference_mode,  
    time_sync,
    get_num_params,    
)

RED, GREEN, BLUE, YELLOW, ORANGE, CYAN, MAGENTA, BOLD, RESET = "\033[91m", "\033[92m", "\033[94m", "\033[93m", "\033[38;5;208m", "\033[96m", "\033[95m", "\033[1m", "\033[0m"
  
class BaseModel(torch.nn.Module): 
    """Base class for all YOLO models in the Ultralytics family.

    This class provides common functionality for YOLO models including forward pass handling, model fusion, information
    display, and weight loading capabilities.
    
    Attributes:
        model (torch.nn.Module): The neural network model.
        save (list): List of layer indices to save outputs from.
        stride (torch.Tensor): Model stride values.     
     
    Methods: 
        forward: Perform forward pass for training or inference.  
        predict: Perform inference on input tensor.
        fuse: Fuse Conv2d and BatchNorm2d layers for optimization.
        info: Print model information. 
        load: Load weights into the model.
        loss: Compute loss for training.     
  
    Examples:    
        Create a BaseModel instance
        >>> model = BaseModel()
        >>> model.info()  # Display model information
    """
     
    def forward(self, x, *args, **kwargs):
        """Perform forward pass of the model for either training or inference.

        If x is a dict, calculates and returns the loss for training. Otherwise, returns predictions for inference.
  
        Args: 
            x (torch.Tensor | dict): Input tensor for inference, or dict with image tensor and labels for training. 
            *args (Any): Variable length argument list.
            **kwargs (Any): Arbitrary keyword arguments.
     
        Returns:   
            (torch.Tensor): Loss if x is a dict (training), or network predictions (inference).     
        """
        if isinstance(x, dict):  # for cases of training and validating while training.  
            return self.loss(x, *args, **kwargs)
        return self.predict(x, *args, **kwargs)    
   
    def predict(self, x, profile=False, visualize=False, augment=False, embed=None):
        """Perform a forward pass through the network.
    
        Args:
            x (torch.Tensor): The input tensor to the model.
            profile (bool): Print the computation time of each layer if True.
            visualize (bool): Save the feature maps of the model if True.
            augment (bool): Augment image during prediction.    
            embed (list, optional): A list of feature vectors/embeddings to return.

        Returns:
            (torch.Tensor): The last output of the model.  
        """
        if augment:
            return self._predict_augment(x) 
        return self._predict_once(x, profile, visualize, embed)
    
    def _predict_once(self, x, profile=False, visualize=False, embed=None):    
        """Perform a forward pass through the network.     

        Args:
            x (torch.Tensor): The input tensor to the model. 
            profile (bool): Print the computation time of each layer if True.
            visualize (bool): Save the feature maps of the model if True. 
            embed (list, optional): A list of feature vectors/embeddings to return.
     
        Returns:
            (torch.Tensor): The last output of the model.
        """    
        y, dt, embeddings = [], [], []  # outputs     
        embed = frozenset(embed) if embed is not None else {-1}
        max_idx = max(embed)
        for m in self.model: 
            if m.f != -1:  # if not from previous layer
                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]  # from earlier layers  
            if profile:
                self._profile_one_layer(m, x, dt)    
            x = m(x)  # run
            y.append(x if m.i in self.save else None)  # save output 
            if visualize:     
                feature_visualization(x, m.type, m.i, save_dir=visualize)
            if m.i in embed:
                embeddings.append(torch.nn.functional.adaptive_avg_pool2d(x, (1, 1)).squeeze(-1).squeeze(-1))  # flatten
                if m.i == max_idx:   
                    return torch.unbind(torch.cat(embeddings, 1), dim=0)  
        return x     

    def _predict_augment(self, x): 
        """Perform augmentations on input image x and return augmented inference."""    
        LOGGER.warning(  
            f"{self.__class__.__name__} does not support 'augment=True' prediction. " 
            f"Reverting to single-scale prediction."   
        )
        return self._predict_once(x)
    
    def _profile_one_layer(self, m, x, dt):
        """Profile the computation time and FLOPs of a single layer of the model on a given input.  
 
        Args:
            m (torch.nn.Module): The layer to be profiled.  
            x (torch.Tensor): The input data to the layer.
            dt (list): A list to store the computation time of the layer.
        """
        try:
            import thop    
        except ImportError:
            thop = None  # conda support without 'ultralytics-thop' installed
        
        warm_up_time = 100 
        test_time = 200     

        c = m == self.model[-1] and isinstance(x, list)  # is final layer list, copy input as inplace fix   
        flops = thop.profile(m, inputs=[x.copy() if c else x], verbose=False)[0] / 1e9 * 2 if thop else 0  # GFLOPs
        params = get_num_params(m)
        for _ in range(warm_up_time): m(x.copy() if c else x)
        t = time_sync()
        for _ in range(test_time): m(x.copy() if c else x)
        dt.append((time_sync() - t) / test_time * 100)
     
        if m == self.model[0]:    
            self._profile_total_params = 0
            self._profile_total_flops = 0

            # 打印表头，带有分隔线和颜色   
            LOGGER.info(f"\n{BOLD}{CYAN}{'='*100}{RESET}")
            LOGGER.info(   
                f"{BOLD}{ORANGE}{'Layer Profiling Results':^100}{RESET}"
            )   
            LOGGER.info(f"{BOLD}{CYAN}{'='*100}{RESET}")    
            LOGGER.info(
                f"{BOLD}{YELLOW}{'time (ms)':>12s} {'GFLOPs':>12s} {'params':>12s}  "
                f"{'module':<50}{RESET}"
            )    
            LOGGER.info(f"{CYAN}{'-'*100}{RESET}")     
 
        # 累加参数量和计算量
        self._profile_total_params += params     
        self._profile_total_flops += flops
 
        # 打印层信息     
        LOGGER.info(    
            f"{GREEN}{dt[-1]:12.4f}{RESET} "
            f"{ORANGE}{flops:12.3f}{RESET} "   
            f"{BLUE}{params:12.0f}{RESET}  "
            f"{CYAN}{m.type:<50}{RESET}"
        )
        
        if c:
            # 打印总计信息
            LOGGER.info(f"{CYAN}{'-'*100}{RESET}")
            LOGGER.info(   
                f"{BOLD}{MAGENTA}{sum(dt):12.4f} "     
                f"{self._profile_total_flops:12.3f} {self._profile_total_params:12.0f}{RESET}  " 
                f"{BOLD}{YELLOW}{'Overall Performance':<50}{RESET}"   
            )
            LOGGER.info(f"{BOLD}{CYAN}{'='*100}{RESET}\n")   

            # 清理临时变量     
            delattr(self, '_profile_total_params') 
            delattr(self, '_profile_total_flops')
            

    def fuse(self, verbose=True): 
        """Fuse the `Conv2d()` and `BatchNorm2d()` layers of the model into a single layer for improved computation
        efficiency.

        Returns:     
            (torch.nn.Module): The fused model is returned.
        """
        if not self.is_fused():  
            for m in self.model.modules():     
                if isinstance(m, (Conv, Conv2, DWConv)) and hasattr(m, "bn"):    
                    if isinstance(m, Conv2):   
                        m.fuse_convs()   
                    m.conv = fuse_conv_and_bn(m.conv, m.bn)  # update conv   
                    delattr(m, "bn")  # remove batchnorm     
                    m.forward = m.forward_fuse  # update forward     
                if isinstance(m, ConvTranspose) and hasattr(m, "bn"):     
                    m.conv_transpose = fuse_deconv_and_bn(m.conv_transpose, m.bn)
                    delattr(m, "bn")  # remove batchnorm
                    m.forward = m.forward_fuse  # update forward 
                if isinstance(m, RepConv):
                    m.fuse_convs()  
                    m.forward = m.forward_fuse  # update forward     
                if isinstance(m, RepVGGDW):  
                    m.fuse() 
                    m.forward = m.forward_fuse
                if isinstance(m, Detect) and getattr(m, "end2end", False):
                    m.fuse()  # remove one2many head
            self.info(verbose=verbose)    
     
        return self
    
    def is_fused(self, thresh=10):  
        """Check if the model has less than a certain threshold of BatchNorm layers.     

        Args:     
            thresh (int, optional): The threshold number of BatchNorm layers.
   
        Returns:
            (bool): True if the number of BatchNorm layers in the model is less than the threshold, False otherwise.    
        """
        bn = tuple(v for k, v in torch.nn.__dict__.items() if "Norm" in k)  # normalization layers, i.e. BatchNorm2d() 
        return sum(isinstance(v, bn) for v in self.modules()) < thresh  # True if < 'thresh' BatchNorm layers in model
     
    def info(self, detailed=False, verbose=True, imgsz=640):
        """Print model information.
     
        Args:    
            detailed (bool): If True, prints out detailed information about the model.   
            verbose (bool): If True, prints out the model information.
            imgsz (int): The size of the image that the model will be trained on. 
        """     
        return model_info(self, detailed=detailed, verbose=verbose, imgsz=imgsz)

    def _apply(self, fn): 
        """Apply a function to all tensors in the model that are not parameters or registered buffers.

        Args:   
            fn (function): The function to apply to the model.   

        Returns:
            (BaseModel): An updated BaseModel object. 
        """ 
        self = super()._apply(fn) 
        m = self.model[-1]  # Detect()     
        if isinstance(
            m, Detect 
        ):  # includes all Detect subclasses like Segment, Pose, OBB, WorldDetect, YOLOEDetect, YOLOESegment
            m.stride = fn(m.stride) 
            m.anchors = fn(m.anchors)    
            m.strides = fn(m.strides)  
        return self  
  
    def load(self, weights, verbose=True):
        """Load weights into the model.

        Args:
            weights (dict | torch.nn.Module): The pre-trained weights to be loaded.  
            verbose (bool, optional): Whether to log the transfer progress.
        """    
        model = weights["model"] if isinstance(weights, dict) else weights  # torchvision models are not dicts
        csd = model.float().state_dict()  # checkpoint state_dict as FP32
        updated_csd = intersect_dicts(csd, self.state_dict())  # intersect 
        self.load_state_dict(updated_csd, strict=False)  # load
        len_updated_csd = len(updated_csd) 
        first_conv = "model.0.conv.weight"  # hard-coded to yolo models for now     
        # mostly used to boost multi-channel training 
        state_dict = self.state_dict()
        if first_conv not in updated_csd and first_conv in state_dict: 
            c1, c2, h, w = state_dict[first_conv].shape
            cc1, cc2, ch, cw = csd[first_conv].shape
            if ch == h and cw == w:
                c1, c2 = min(c1, cc1), min(c2, cc2)
                state_dict[first_conv][:c1, :c2] = csd[first_conv][:c1, :c2]     
                len_updated_csd += 1
        if verbose:  
            LOGGER.info(f"Transferred {len_updated_csd}/{len(self.model.state_dict())} items from pretrained weights")
  
    def loss(self, batch, preds=None): 
        """Compute loss.

        Args: 
            batch (dict): Batch to compute loss on.    
            preds (torch.Tensor | list[torch.Tensor], optional): Predictions.     
        """ 
        if getattr(self, "criterion", None) is None:
            self.criterion = self.init_criterion()
   
        if preds is None:
            preds = self.forward(batch["img"])
        return self.criterion(preds, batch)

    def init_criterion(self):
        """Initialize the loss criterion for the BaseModel."""
        raise NotImplementedError("compute_loss() needs to be implemented by task heads")
    
   
class DetectionModel(BaseModel):
    """YOLO detection model.    
   
    This class implements the YOLO detection architecture, handling model initialization, forward pass, augmented    
    inference, and loss computation for object detection tasks.

    Attributes:
        yaml (dict): Model configuration dictionary.
        model (torch.nn.Sequential): The neural network model.
        save (list): List of layer indices to save outputs from.    
        names (dict): Class names dictionary. 
        inplace (bool): Whether to use inplace operations.
        end2end (bool): Whether the model uses end-to-end detection. 
        stride (torch.Tensor): Model stride values.     
 
    Methods:
        __init__: Initialize the YOLO detection model.  
        _predict_augment: Perform augmented inference.  
        _descale_pred: De-scale predictions following augmented inference. 
        _clip_augmented: Clip YOLO augmented inference tails.  
        init_criterion: Initialize the loss criterion.     
  
    Examples:     
        Initialize a detection model     
        >>> model = DetectionModel("yolo26n.yaml", ch=3, nc=80)  
        >>> results = model.predict(image_tensor)   
    """
  
    def __init__(self, cfg="yolo26n.yaml", ch=3, nc=None, verbose=True):     
        """Initialize the YOLO detection model with the given config and parameters.  

        Args:    
            cfg (str | dict): Model configuration file path or dictionary.
            ch (int): Number of input channels.
            nc (int, optional): Number of classes.
            verbose (bool): Whether to display model information.  
        """  
        super().__init__()   
        self.yaml = cfg if isinstance(cfg, dict) else yaml_model_load(cfg)  # cfg dict  
        if self.yaml["backbone"][0][2] == "Silence":     
            LOGGER.warning(
                "YOLOv9 `Silence` module is deprecated in favor of torch.nn.Identity. " 
                "Please delete local *.pt file and re-download the latest model checkpoint."
            )
            self.yaml["backbone"][0][2] = "nn.Identity"   

        # ---- 0526 版 KernelWarehouse 接线（新增，默认关闭，不影响其他配置）----
        warehouse_manager_flag = self.yaml.get("Warehouse_Manager", False)
        self.warehouse_manager = None
        if warehouse_manager_flag:
            self.warehouse_manager = Warehouse_Manager(
                cell_num_ratio=self.yaml.get("Warehouse_Manager_Ratio", 1.0))

        # Define model  
        self.yaml["channels"] = ch  # save channels
        if nc and nc != self.yaml["nc"]:
            LOGGER.info(f"Overriding model.yaml nc={self.yaml['nc']} with nc={nc}")
            self.yaml["nc"] = nc  # override YAML value   
        self.model, self.save = parse_model(deepcopy(self.yaml), ch=ch, verbose=verbose,
                                            warehouse_manager=self.warehouse_manager)  # model, savelist

        if warehouse_manager_flag:
            self.warehouse_manager.store()
            self.warehouse_manager.allocate(self)
            self.net_update_temperature(0)
        self.names = {i: f"{i}" for i in range(self.yaml["nc"])}  # default names dict
        self.inplace = self.yaml.get("inplace", True)

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)     

        # Build strides   
        m = self.model[-1]  # Detect()
        if isinstance(m, Detect):  # includes all Detect subclasses like Segment, Pose, OBB, YOLOEDetect, YOLOESegment  
            s = 256  # 2x min stride
            m.inplace = self.inplace
    
            def _forward(x):
                """Perform a forward pass through the model, handling different Detect subclass types accordingly."""     
                output = self.forward(x)    
                if self.end2end:
                    output = output["one2many"]
                return output["feats"] 
   
            self.model.eval()  # Avoid changing batch statistics until training begins    
            m.training = True  # Setting it to True to properly return strides
            m.stride = torch.tensor([s / x.shape[-2] for x in _forward(torch.zeros(2, ch, s, s).to(self.device))])  # forward   
            LOGGER.info(f'{BOLD}{MAGENTA}m.stride:{m.stride}{RESET}')  
            self.stride = m.stride  
            self.model.train()  # Set model back to training(default) mode
            m.bias_init()  # only run once
        else:    
            self.stride = torch.Tensor([32])  # default stride, e.g., RTDETR

        # Init weights, biases
        initialize_weights(self)    
        if verbose:
            self.info()
            LOGGER.info("") 
   
    @property     
    def end2end(self): 
        """Return whether the model uses end-to-end NMS-free detection."""    
        return getattr(self.model[-1], "end2end", False)
   
    @end2end.setter    
    def end2end(self, value):  
        """Override the end-to-end detection mode."""
        self.set_head_attr(end2end=value)

    def set_head_attr(self, **kwargs):
        """Set attributes of the model head (last layer).

        Args:
            **kwargs: Arbitrary keyword arguments representing attributes to set.     
        """ 
        head = self.model[-1]
        for k, v in kwargs.items():
            if not hasattr(head, k):
                LOGGER.warning(f"Head has no attribute '{k}'.")    
                continue
            setattr(head, k, v)   
     
    def net_update_temperature(self, temp):
        """0526 版 KernelWarehouse 用：逐层更新温度系数。"""
        for m in self.modules():
            if hasattr(m, "update_temperature"):
                m.update_temperature(temp)

    def _predict_augment(self, x):   
        """Perform augmentations on input image x and return augmented inference and train outputs. 
    
        Args:
            x (torch.Tensor): Input image tensor.

        Returns: 
            (torch.Tensor): Augmented inference output.  
        """  
        if getattr(self, "end2end", False) or self.__class__.__name__ != "DetectionModel":
            LOGGER.warning("Model does not support 'augment=True', reverting to single-scale prediction.")
            return self._predict_once(x) 
        img_size = x.shape[-2:]  # height, width     
        s = [1, 0.83, 0.67]  # scales  
        f = [None, 3, None]  # flips (2-ud, 3-lr)
        y = []  # outputs     
        for si, fi in zip(s, f):   
            xi = scale_img(x.flip(fi) if fi else x, si, gs=int(self.stride.max()))
            yi = super().predict(xi)[0]  # forward     
            yi = self._descale_pred(yi, fi, si, img_size)
            y.append(yi)
        y = self._clip_augmented(y)  # clip augmented tails 
        return torch.cat(y, -1), None  # augmented inference, train   
     
    @staticmethod    
    def _descale_pred(p, flips, scale, img_size, dim=1):
        """De-scale predictions following augmented inference (inverse operation).
    
        Args:  
            p (torch.Tensor): Predictions tensor.  
            flips (int): Flip type (0=none, 2=ud, 3=lr). 
            scale (float): Scale factor.
            img_size (tuple): Original image size (height, width).
            dim (int): Dimension to split at. 
 
        Returns: 
            (torch.Tensor): De-scaled predictions.
        """    
        p[:, :4] /= scale  # de-scale     
        x, y, wh, cls = p.split((1, 1, 2, p.shape[dim] - 4), dim)
        if flips == 2:    
            y = img_size[0] - y  # de-flip ud 
        elif flips == 3:
            x = img_size[1] - x  # de-flip lr
        return torch.cat((x, y, wh, cls), dim)     

    def _clip_augmented(self, y):
        """Clip YOLO augmented inference tails.
     
        Args:     
            y (list[torch.Tensor]): List of detection tensors.

        Returns:
            (list[torch.Tensor]): Clipped detection tensors.   
        """  
        nl = self.model[-1].nl  # number of detection layers (P3-P5)
        g = sum(4**x for x in range(nl))  # grid points
        e = 1  # exclude layer count  
        i = (y[0].shape[-1] // g) * sum(4**x for x in range(e))  # indices     
        y[0] = y[0][..., :-i]  # large
        i = (y[-1].shape[-1] // g) * sum(4 ** (nl - 1 - x) for x in range(e))  # indices    
        y[-1] = y[-1][..., i:]  # small
        return y 
   
    def init_criterion(self):
        """Initialize the loss criterion for the DetectionModel."""  
        return E2ELoss(self) if getattr(self, "end2end", False) else v8DetectionLoss(self)   

   
class OBBModel(DetectionModel):    
    """YOLO Oriented Bounding Box (OBB) model.    

    This class extends DetectionModel to handle oriented bounding box detection tasks, providing specialized loss
    computation for rotated object detection.

    Methods: 
        __init__: Initialize YOLO OBB model.  
        init_criterion: Initialize the loss criterion for OBB detection.

    Examples:
        Initialize an OBB model 
        >>> model = OBBModel("yolo26n-obb.yaml", ch=3, nc=80)     
        >>> results = model.predict(image_tensor)    
    """   

    def __init__(self, cfg="yolo26n-obb.yaml", ch=3, nc=None, verbose=True):
        """Initialize YOLO OBB model with given config and parameters.

        Args: 
            cfg (str | dict): Model configuration file path or dictionary.   
            ch (int): Number of input channels.
            nc (int, optional): Number of classes.
            verbose (bool): Whether to display model information.  
        """     
        super().__init__(cfg=cfg, ch=ch, nc=nc, verbose=verbose)

    def init_criterion(self):
        """Initialize the loss criterion for the model.""" 
        return E2ELoss(self, v8OBBLoss) if getattr(self, "end2end", False) else v8OBBLoss(self)

     
class SegmentationModel(DetectionModel):   
    """YOLO segmentation model.
     
    This class extends DetectionModel to handle instance segmentation tasks, providing specialized loss computation for
    pixel-level object detection and segmentation.

    Methods:    
        __init__: Initialize YOLO segmentation model.   
        init_criterion: Initialize the loss criterion for segmentation.
    
    Examples:
        Initialize a segmentation model
        >>> model = SegmentationModel("yolo26n-seg.yaml", ch=3, nc=80)  
        >>> results = model.predict(image_tensor)
    """    
  
    def __init__(self, cfg="yolo26n-seg.yaml", ch=3, nc=None, verbose=True):
        """Initialize Ultralytics YOLO segmentation model with given config and parameters.
  
        Args:
            cfg (str | dict): Model configuration file path or dictionary.  
            ch (int): Number of input channels.
            nc (int, optional): Number of classes. 
            verbose (bool): Whether to display model information.
        """
        super().__init__(cfg=cfg, ch=ch, nc=nc, verbose=verbose)  

    def init_criterion(self):  
        """Initialize the loss criterion for the SegmentationModel."""    
        return E2ELoss(self, v8SegmentationLoss) if getattr(self, "end2end", False) else v8SegmentationLoss(self) 


class PoseModel(DetectionModel):
    """YOLO pose model.  

    This class extends DetectionModel to handle human pose estimation tasks, providing specialized loss computation for 
    keypoint detection and pose estimation.

    Attributes:   
        kpt_shape (tuple): Shape of keypoints data (num_keypoints, num_dimensions).    

    Methods:   
        __init__: Initialize YOLO pose model.
        init_criterion: Initialize the loss criterion for pose estimation.
 
    Examples:  
        Initialize a pose model
        >>> model = PoseModel("yolo26n-pose.yaml", ch=3, nc=1, data_kpt_shape=(17, 3))  
        >>> results = model.predict(image_tensor) 
    """     

    def __init__(self, cfg="yolo26n-pose.yaml", ch=3, nc=None, data_kpt_shape=(None, None), verbose=True):
        """Initialize Ultralytics YOLO Pose model. 
 
        Args:
            cfg (str | dict): Model configuration file path or dictionary.
            ch (int): Number of input channels.  
            nc (int, optional): Number of classes.     
            data_kpt_shape (tuple): Shape of keypoints data. 
            verbose (bool): Whether to display model information. 
        """   
        if not isinstance(cfg, dict):
            cfg = yaml_model_load(cfg)  # load model YAML
        if any(data_kpt_shape) and list(data_kpt_shape) != list(cfg["kpt_shape"]):   
            LOGGER.info(f"Overriding model.yaml kpt_shape={cfg['kpt_shape']} with kpt_shape={data_kpt_shape}")  
            cfg["kpt_shape"] = data_kpt_shape 
        super().__init__(cfg=cfg, ch=ch, nc=nc, verbose=verbose)
 
    def init_criterion(self):   
        """Initialize the loss criterion for the PoseModel."""  
        return E2ELoss(self, PoseLoss26) if getattr(self, "end2end", False) else v8PoseLoss(self)  


class ClassificationModel(BaseModel):  
    """YOLO classification model.

    This class implements the YOLO classification architecture for image classification tasks, providing model    
    initialization, configuration, and output reshaping capabilities.
   
    Attributes:
        yaml (dict): Model configuration dictionary.
        model (torch.nn.Sequential): The neural network model.   
        stride (torch.Tensor): Model stride values.    
        names (dict): Class names dictionary.
     
    Methods:
        __init__: Initialize ClassificationModel.
        _from_yaml: Set model configurations and define architecture.  
        reshape_outputs: Update model to specified class count.
        init_criterion: Initialize the loss criterion. 
    
    Examples:  
        Initialize a classification model  
        >>> model = ClassificationModel("yolo26n-cls.yaml", ch=3, nc=1000)     
        >>> results = model.predict(image_tensor)
    """

    def __init__(self, cfg="yolo26n-cls.yaml", ch=3, nc=None, verbose=True):   
        """Initialize ClassificationModel with YAML, channels, number of classes, verbose flag.

        Args:    
            cfg (str | dict): Model configuration file path or dictionary.    
            ch (int): Number of input channels.   
            nc (int, optional): Number of classes.
            verbose (bool): Whether to display model information.    
        """
        super().__init__() 
        self._from_yaml(cfg, ch, nc, verbose)
  
    def _from_yaml(self, cfg, ch, nc, verbose):
        """Set Ultralytics YOLO model configurations and define the model architecture.

        Args:
            cfg (str | dict): Model configuration file path or dictionary. 
            ch (int): Number of input channels.    
            nc (int, optional): Number of classes.     
            verbose (bool): Whether to display model information.
        """
        self.yaml = cfg if isinstance(cfg, dict) else yaml_model_load(cfg)  # cfg dict  

        # Define model  
        ch = self.yaml["channels"] = self.yaml.get("channels", ch)  # input channels  
        if nc and nc != self.yaml["nc"]:    
            LOGGER.info(f"Overriding model.yaml nc={self.yaml['nc']} with nc={nc}")
            self.yaml["nc"] = nc  # override YAML value
        elif not nc and not self.yaml.get("nc", None):
            raise ValueError("nc not specified. Must specify nc in model.yaml or function arguments.")     
        self.model, self.save = parse_model(deepcopy(self.yaml), ch=ch, verbose=verbose)  # model, savelist   
        self.stride = torch.Tensor([1])  # no stride constraints    
        self.names = {i: f"{i}" for i in range(self.yaml["nc"])}  # default names dict
        self.info()
     
    @staticmethod    
    def reshape_outputs(model, nc):     
        """Update a TorchVision classification model to class count 'n' if required.

        Args:
            model (torch.nn.Module): Model to update.
            nc (int): New number of classes.
        """
        name, m = list((model.model if hasattr(model, "model") else model).named_children())[-1]  # last module
        if isinstance(m, Classify):  # YOLO Classify() head
            if m.linear.out_features != nc:
                m.linear = torch.nn.Linear(m.linear.in_features, nc)   
        elif isinstance(m, torch.nn.Linear):  # ResNet, EfficientNet
            if m.out_features != nc:
                setattr(model, name, torch.nn.Linear(m.in_features, nc))
        elif isinstance(m, torch.nn.Sequential): 
            types = [type(x) for x in m]
            if torch.nn.Linear in types:     
                i = len(types) - 1 - types[::-1].index(torch.nn.Linear)  # last torch.nn.Linear index  
                if m[i].out_features != nc:
                    m[i] = torch.nn.Linear(m[i].in_features, nc)
            elif torch.nn.Conv2d in types:
                i = len(types) - 1 - types[::-1].index(torch.nn.Conv2d)  # last torch.nn.Conv2d index   
                if m[i].out_channels != nc:
                    m[i] = torch.nn.Conv2d(
                        m[i].in_channels, nc, m[i].kernel_size, m[i].stride, bias=m[i].bias is not None    
                    )
  
    def init_criterion(self):   
        """Initialize the loss criterion for the ClassificationModel."""    
        return v8ClassificationLoss()    


class RTDETRDetectionModel(DetectionModel):
    """RTDETR (Real-time DEtection and Tracking using Transformers) Detection Model class.
     
    This class is responsible for constructing the RTDETR architecture, defining loss functions, and facilitating both 
    the training and inference processes. RTDETR is an object detection and tracking model that extends from the  
    DetectionModel base class.

    Attributes:  
        nc (int): Number of classes for detection.
        criterion (RTDETRDetectionLoss): Loss function for training.
     
    Methods: 
        __init__: Initialize the RTDETRDetectionModel.     
        init_criterion: Initialize the loss criterion.
        loss: Compute loss for training.     
        predict: Perform forward pass through the model.

    Examples:    
        Initialize an RTDETR model
        >>> model = RTDETRDetectionModel("rtdetr-l.yaml", ch=3, nc=80)
        >>> results = model.predict(image_tensor)  
    """
 
    def __init__(self, cfg="rtdetr-l.yaml", ch=3, nc=None, verbose=True):     
        """Initialize the RTDETRDetectionModel.
   
        Args:
            cfg (str | dict): Configuration file name or path.
            ch (int): Number of input channels.     
            nc (int, optional): Number of classes.  
            verbose (bool): Print additional information during initialization.
        """ 
        super().__init__(cfg=cfg, ch=ch, nc=nc, verbose=verbose) 

    def _apply(self, fn):
        """Apply a function to all tensors in the model that are not parameters or registered buffers.
    
        Args:    
            fn (function): The function to apply to the model.   
   
        Returns: 
            (RTDETRDetectionModel): An updated BaseModel object.   
        """
        self = super()._apply(fn)
        m = self.model[-1]     
        m.anchors = fn(m.anchors)  
        m.valid_mask = fn(m.valid_mask) 
        return self

    def init_criterion(self):
        """Initialize the loss criterion for the RTDETRDetectionModel."""
        from ultralytics.models.utils.loss import RTDETRDetectionLoss
    
        return RTDETRDetectionLoss(nc=self.nc, use_vfl=True)
    
    def loss(self, batch, preds=None):    
        """Compute the loss for the given batch of data.  
     
        Args:
            batch (dict): Dictionary containing image and label data.    
            preds (torch.Tensor, optional): Precomputed model predictions.
  
        Returns:   
            loss_sum (torch.Tensor): Total loss value. 
            loss_items (torch.Tensor): Main three losses in a tensor.
        """ 
        if not hasattr(self, "criterion"):  
            self.criterion = self.init_criterion()    
     
        img = batch["img"]
        # NOTE: preprocess gt_bbox and gt_labels to list. 
        bs = img.shape[0]
        batch_idx = batch["batch_idx"]
        gt_groups = [(batch_idx == i).sum().item() for i in range(bs)]
        targets = {
            "cls": batch["cls"].to(img.device, dtype=torch.long).view(-1),
            "bboxes": batch["bboxes"].to(device=img.device),  
            "batch_idx": batch_idx.to(img.device, dtype=torch.long).view(-1), 
            "gt_groups": gt_groups,     
        }     
 
        if preds is None:
            preds = self.predict(img, batch=targets) 
        dec_bboxes, dec_scores, enc_bboxes, enc_scores, dn_meta = preds if self.training else preds[1]  
        if dn_meta is None:
            dn_bboxes, dn_scores = None, None 
        else:
            dn_bboxes, dec_bboxes = torch.split(dec_bboxes, dn_meta["dn_num_split"], dim=2)
            dn_scores, dec_scores = torch.split(dec_scores, dn_meta["dn_num_split"], dim=2)     
   
        dec_bboxes = torch.cat([enc_bboxes.unsqueeze(0), dec_bboxes])  # (7, bs, 300, 4)     
        dec_scores = torch.cat([enc_scores.unsqueeze(0), dec_scores])

        loss = self.criterion(
            (dec_bboxes, dec_scores), targets, dn_bboxes=dn_bboxes, dn_scores=dn_scores, dn_meta=dn_meta 
        )
        # NOTE: There are like 12 losses in RTDETR, backward with all losses but only show the main three losses.
        return sum(loss.values()), torch.as_tensor(
            [loss[k].detach() for k in ["loss_giou", "loss_class", "loss_bbox"]], device=img.device
        )

    def predict(self, x, profile=False, visualize=False, batch=None, augment=False, embed=None):
        """Perform a forward pass through the model.

        Args:
            x (torch.Tensor): The input tensor. 
            profile (bool): If True, profile the computation time for each layer. 
            visualize (bool): If True, save feature maps for visualization.    
            batch (dict, optional): Ground truth data for evaluation.  
            augment (bool): If True, perform data augmentation during inference.   
            embed (list, optional): A list of feature vectors/embeddings to return.     
   
        Returns:
            (torch.Tensor): Model's output tensor.
        """
        y, dt, embeddings = [], [], []  # outputs   
        embed = frozenset(embed) if embed is not None else {-1}
        max_idx = max(embed)
        for m in self.model[:-1]:  # except the head part     
            if m.f != -1:  # if not from previous layer
                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]  # from earlier layers    
            if profile:  
                self._profile_one_layer(m, x, dt)   
            x = m(x)  # run    
            y.append(x if m.i in self.save else None)  # save output
            if visualize:     
                feature_visualization(x, m.type, m.i, save_dir=visualize)   
            if m.i in embed:     
                embeddings.append(torch.nn.functional.adaptive_avg_pool2d(x, (1, 1)).squeeze(-1).squeeze(-1))  # flatten
                if m.i == max_idx:
                    return torch.unbind(torch.cat(embeddings, 1), dim=0) 
        head = self.model[-1]
        x = head([y[j] for j in head.f], batch)  # head inference
        return x   
  
    
class WorldModel(DetectionModel):
    """YOLOv8 World Model.   
  
    This class implements the YOLOv8 World model for open-vocabulary object detection, supporting text-based class   
    specification and CLIP model integration for zero-shot detection capabilities.

    Attributes:    
        txt_feats (torch.Tensor): Text feature embeddings for classes.
        clip_model (torch.nn.Module): CLIP model for text encoding.   

    Methods:
        __init__: Initialize YOLOv8 world model.
        set_classes: Set classes for offline inference.
        get_text_pe: Get text positional embeddings.
        predict: Perform forward pass with text features.
        loss: Compute loss with text features.    
     
    Examples: 
        Initialize a world model    
        >>> model = WorldModel("yolov8s-world.yaml", ch=3, nc=80)  
        >>> model.set_classes(["person", "car", "bicycle"])
        >>> results = model.predict(image_tensor)   
    """    
   
    def __init__(self, cfg="yolov8s-world.yaml", ch=3, nc=None, verbose=True):
        """Initialize YOLOv8 world model with given config and parameters.   

        Args: 
            cfg (str | dict): Model configuration file path or dictionary.   
            ch (int): Number of input channels.
            nc (int, optional): Number of classes.
            verbose (bool): Whether to display model information. 
        """
        self.txt_feats = torch.randn(1, nc or 80, 512)  # features placeholder 
        self.clip_model = None  # CLIP model placeholder
        super().__init__(cfg=cfg, ch=ch, nc=nc, verbose=verbose) 
    
    def set_classes(self, text, batch=80, cache_clip_model=True):   
        """Set classes in advance so that model could do offline-inference without clip model.    
   
        Args:
            text (list[str]): List of class names.
            batch (int): Batch size for processing text tokens.
            cache_clip_model (bool): Whether to cache the CLIP model.
        """
        self.txt_feats = self.get_text_pe(text, batch=batch, cache_clip_model=cache_clip_model)
        self.model[-1].nc = len(text)     
    
    def get_text_pe(self, text, batch=80, cache_clip_model=True):
        """Get text positional embeddings for offline inference without CLIP model.
   
        Args:
            text (list[str]): List of class names.
            batch (int): Batch size for processing text tokens.  
            cache_clip_model (bool): Whether to cache the CLIP model.
     
        Returns:     
            (torch.Tensor): Text positional embeddings.
        """    
        from ultralytics.nn.text_model import build_text_model

        device = next(self.model.parameters()).device
        if not getattr(self, "clip_model", None) and cache_clip_model:
            # For backwards compatibility of models lacking clip_model attribute    
            self.clip_model = build_text_model("clip:ViT-B/32", device=device)
        model = self.clip_model if cache_clip_model else build_text_model("clip:ViT-B/32", device=device)   
        text_token = model.tokenize(text)    
        txt_feats = [model.encode_text(token).detach() for token in text_token.split(batch)]
        txt_feats = txt_feats[0] if len(txt_feats) == 1 else torch.cat(txt_feats, dim=0)   
        return txt_feats.reshape(-1, len(text), txt_feats.shape[-1]) 
 
    def predict(self, x, profile=False, visualize=False, txt_feats=None, augment=False, embed=None):  
        """Perform a forward pass through the model.

        Args: 
            x (torch.Tensor): The input tensor.   
            profile (bool): If True, profile the computation time for each layer.
            visualize (bool): If True, save feature maps for visualization.   
            txt_feats (torch.Tensor, optional): The text features, use it if it's given.     
            augment (bool): If True, perform data augmentation during inference.    
            embed (list, optional): A list of feature vectors/embeddings to return.   

        Returns:
            (torch.Tensor): Model's output tensor.   
        """    
        txt_feats = (self.txt_feats if txt_feats is None else txt_feats).to(device=x.device, dtype=x.dtype)
        if txt_feats.shape[0] != x.shape[0] or self.model[-1].export:  
            txt_feats = txt_feats.expand(x.shape[0], -1, -1) 
        ori_txt_feats = txt_feats.clone()
        y, dt, embeddings = [], [], []  # outputs
        embed = frozenset(embed) if embed is not None else {-1}
        max_idx = max(embed)
        for m in self.model:  # except the head part
            if m.f != -1:  # if not from previous layer
                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]  # from earlier layers
            if profile:
                self._profile_one_layer(m, x, dt)   
            if isinstance(m, C2fAttn):
                x = m(x, txt_feats)     
            elif isinstance(m, WorldDetect):
                x = m(x, ori_txt_feats) 
            elif isinstance(m, ImagePoolingAttn):
                txt_feats = m(x, txt_feats)
            else:  
                x = m(x)  # run
  
            y.append(x if m.i in self.save else None)  # save output 
            if visualize:
                feature_visualization(x, m.type, m.i, save_dir=visualize)
            if m.i in embed:  
                embeddings.append(torch.nn.functional.adaptive_avg_pool2d(x, (1, 1)).squeeze(-1).squeeze(-1))  # flatten
                if m.i == max_idx: 
                    return torch.unbind(torch.cat(embeddings, 1), dim=0)
        return x

    def loss(self, batch, preds=None):    
        """Compute loss.     
 
        Args:     
            batch (dict): Batch to compute loss on.  
            preds (torch.Tensor | list[torch.Tensor], optional): Predictions. 
        """ 
        if not hasattr(self, "criterion"):
            self.criterion = self.init_criterion()
   
        if preds is None:
            preds = self.forward(batch["img"], txt_feats=batch["txt_feats"])    
        return self.criterion(preds, batch)
   
   
class YOLOEModel(DetectionModel):   
    """YOLOE detection model.

    This class implements the YOLOE architecture for efficient object detection with text and visual prompts, supporting
    both prompt-based and prompt-free inference modes.
 
    Attributes:    
        pe (torch.Tensor): Prompt embeddings for classes.
        clip_model (torch.nn.Module): CLIP model for text encoding.     
     
    Methods:
        __init__: Initialize YOLOE model.  
        get_text_pe: Get text positional embeddings.    
        get_visual_pe: Get visual embeddings.
        set_vocab: Set vocabulary for prompt-free model.
        get_vocab: Get fused vocabulary layer.
        set_classes: Set classes for offline inference.    
        get_cls_pe: Get class positional embeddings.
        predict: Perform forward pass with prompts.
        loss: Compute loss with prompts.   

    Examples:    
        Initialize a YOLOE model
        >>> model = YOLOEModel("yoloe-v8s.yaml", ch=3, nc=80) 
        >>> results = model.predict(image_tensor, tpe=text_embeddings)  
    """ 

    def __init__(self, cfg="yoloe-v8s.yaml", ch=3, nc=None, verbose=True):
        """Initialize YOLOE model with given config and parameters.     
  
        Args:
            cfg (str | dict): Model configuration file path or dictionary. 
            ch (int): Number of input channels.
            nc (int, optional): Number of classes.   
            verbose (bool): Whether to display model information.
        """
        super().__init__(cfg=cfg, ch=ch, nc=nc, verbose=verbose)    
        self.text_model = self.yaml.get("text_model", "mobileclip:blt")

    @smart_inference_mode()
    def get_text_pe(self, text, batch=80, cache_clip_model=False, without_reprta=False):     
        """Get text positional embeddings for offline inference without CLIP model.

        Args:    
            text (list[str]): List of class names.
            batch (int): Batch size for processing text tokens. 
            cache_clip_model (bool): Whether to cache the CLIP model. 
            without_reprta (bool): Whether to return text embeddings without reprta module processing.
   
        Returns:  
            (torch.Tensor): Text positional embeddings.   
        """
        from ultralytics.nn.text_model import build_text_model    

        device = next(self.model.parameters()).device
        if not getattr(self, "clip_model", None) and cache_clip_model:    
            # For backwards compatibility of models lacking clip_model attribute
            self.clip_model = build_text_model(getattr(self, "text_model", "mobileclip:blt"), device=device)

        model = (
            self.clip_model
            if cache_clip_model   
            else build_text_model(getattr(self, "text_model", "mobileclip:blt"), device=device) 
        )   
        text_token = model.tokenize(text)    
        txt_feats = [model.encode_text(token).detach() for token in text_token.split(batch)]
        txt_feats = txt_feats[0] if len(txt_feats) == 1 else torch.cat(txt_feats, dim=0)
        txt_feats = txt_feats.reshape(-1, len(text), txt_feats.shape[-1])    
        if without_reprta:    
            return txt_feats 
   
        head = self.model[-1]  
        assert isinstance(head, YOLOEDetect)     
        return head.get_tpe(txt_feats)  # run auxiliary text head 

    @smart_inference_mode()
    def get_visual_pe(self, img, visual): 
        """Get visual embeddings.

        Args:
            img (torch.Tensor): Input image tensor.
            visual (torch.Tensor): Visual features.
    
        Returns:
            (torch.Tensor): Visual positional embeddings.    
        """
        return self(img, vpe=visual, return_vpe=True)   

    def set_vocab(self, vocab, names):     
        """Set vocabulary for the prompt-free model.
  
        Args:   
            vocab (nn.ModuleList): List of vocabulary items.  
            names (list[str]): List of class names. 
        """
        assert not self.training
        head = self.model[-1]
        assert isinstance(head, YOLOEDetect)
  
        # Cache anchors for head   
        device = next(self.parameters()).device
        self(torch.empty(1, 3, self.args["imgsz"], self.args["imgsz"]).to(device))  # warmup     
     
        cv3 = getattr(head, "one2one_cv3", head.cv3)  
        cv2 = getattr(head, "one2one_cv2", head.cv2)     

        # re-parameterization for prompt-free model 
        self.model[-1].lrpc = nn.ModuleList(
            LRPCHead(cls, pf[-1], loc[-1], enabled=i != 2) for i, (cls, pf, loc) in enumerate(zip(vocab, cv3, cv2))
        )
        for loc_head, cls_head in zip(head.cv2, head.cv3):
            assert isinstance(loc_head, nn.Sequential) 
            assert isinstance(cls_head, nn.Sequential)
            del loc_head[-1]
            del cls_head[-1] 
        self.model[-1].nc = len(names)
        self.names = check_class_names(names)

    def get_vocab(self, names):
        """Get fused vocabulary layer from the model.  

        Args: 
            names (list): List of class names.     

        Returns:
            (nn.ModuleList): List of vocabulary modules.     
        """
        assert not self.training     
        head = self.model[-1]     
        assert isinstance(head, YOLOEDetect)     
        assert not head.is_fused
     
        tpe = self.get_text_pe(names) 
        self.set_classes(names, tpe) 
        device = next(self.model.parameters()).device    
        head.fuse(self.pe.to(device))  # fuse prompt embeddings to classify head
 
        cv3 = getattr(head, "one2one_cv3", head.cv3)
        vocab = nn.ModuleList()    
        for cls_head in cv3: 
            assert isinstance(cls_head, nn.Sequential)   
            vocab.append(cls_head[-1])
        return vocab
  
    def set_classes(self, names, embeddings):   
        """Set classes in advance so that model could do offline-inference without clip model.     

        Args:     
            names (list[str]): List of class names.   
            embeddings (torch.Tensor): Embeddings tensor.  
        """
        assert not hasattr(self.model[-1], "lrpc"), (
            "Prompt-free model does not support setting classes. Please try with Text/Visual prompt models."
        )
        assert embeddings.ndim == 3
        self.pe = embeddings  
        self.model[-1].nc = len(names)   
        self.names = check_class_names(names)   

    def get_cls_pe(self, tpe, vpe):     
        """Get class positional embeddings.  

        Args:  
            tpe (torch.Tensor, optional): Text positional embeddings.
            vpe (torch.Tensor, optional): Visual positional embeddings.

        Returns:    
            (torch.Tensor): Class positional embeddings.
        """
        all_pe = []   
        if tpe is not None:     
            assert tpe.ndim == 3  
            all_pe.append(tpe)
        if vpe is not None: 
            assert vpe.ndim == 3  
            all_pe.append(vpe)
        if not all_pe:
            all_pe.append(getattr(self, "pe", torch.zeros(1, 80, 512)))    
        return torch.cat(all_pe, dim=1)
   
    def predict(    
        self, x, profile=False, visualize=False, tpe=None, augment=False, embed=None, vpe=None, return_vpe=False   
    ):
        """Perform a forward pass through the model.    

        Args:    
            x (torch.Tensor): The input tensor.  
            profile (bool): If True, profile the computation time for each layer.
            visualize (bool): If True, save feature maps for visualization. 
            tpe (torch.Tensor, optional): Text positional embeddings.
            augment (bool): If True, perform data augmentation during inference.
            embed (list, optional): A list of feature vectors/embeddings to return.
            vpe (torch.Tensor, optional): Visual positional embeddings.
            return_vpe (bool): If True, return visual positional embeddings.    
 
        Returns:
            (torch.Tensor): Model's output tensor.
        """    
        y, dt, embeddings = [], [], []  # outputs
        b = x.shape[0]   
        embed = frozenset(embed) if embed is not None else {-1}
        max_idx = max(embed)
        for m in self.model:  # except the head part 
            if m.f != -1:  # if not from previous layer   
                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]  # from earlier layers 
            if profile:
                self._profile_one_layer(m, x, dt)
            if isinstance(m, YOLOEDetect): 
                vpe = m.get_vpe(x, vpe) if vpe is not None else None
                if return_vpe: 
                    assert vpe is not None    
                    assert not self.training   
                    return vpe   
                cls_pe = self.get_cls_pe(m.get_tpe(tpe), vpe).to(device=x[0].device, dtype=x[0].dtype)   
                if cls_pe.shape[0] != b or m.export:
                    cls_pe = cls_pe.expand(b, -1, -1)
                x.append(cls_pe)  # adding cls embedding  
            x = m(x)  # run
  
            y.append(x if m.i in self.save else None)  # save output
            if visualize:
                feature_visualization(x, m.type, m.i, save_dir=visualize)
            if m.i in embed:    
                embeddings.append(torch.nn.functional.adaptive_avg_pool2d(x, (1, 1)).squeeze(-1).squeeze(-1))  # flatten 
                if m.i == max_idx:  
                    return torch.unbind(torch.cat(embeddings, 1), dim=0)    
        return x
   
    def loss(self, batch, preds=None):
        """Compute loss.

        Args:
            batch (dict): Batch to compute loss on.     
            preds (torch.Tensor | list[torch.Tensor], optional): Predictions.
        """   
        if not hasattr(self, "criterion"):
            from ultralytics.utils.loss import TVPDetectLoss

            visual_prompt = batch.get("visuals", None) is not None  # TODO     
            self.criterion = (  
                (E2ELoss(self, TVPDetectLoss) if getattr(self, "end2end", False) else TVPDetectLoss(self))  
                if visual_prompt
                else self.init_criterion()     
            )
        if preds is None: 
            preds = self.forward(
                batch["img"],  
                tpe=None if "visuals" in batch else batch.get("txt_feats", None),    
                vpe=batch.get("visuals", None),   
            )  
        return self.criterion(preds, batch) 
 
     
class YOLOESegModel(YOLOEModel, SegmentationModel):
    """YOLOE segmentation model.
    
    This class extends YOLOEModel to handle instance segmentation tasks with text and visual prompts, providing
    specialized loss computation for pixel-level object detection and segmentation.
    
    Methods:     
        __init__: Initialize YOLOE segmentation model. 
        loss: Compute loss with prompts for segmentation.
    
    Examples:
        Initialize a YOLOE segmentation model   
        >>> model = YOLOESegModel("yoloe-v8s-seg.yaml", ch=3, nc=80)
        >>> results = model.predict(image_tensor, tpe=text_embeddings)  
    """   

    def __init__(self, cfg="yoloe-v8s-seg.yaml", ch=3, nc=None, verbose=True):  
        """Initialize YOLOE segmentation model with given config and parameters. 
     
        Args:
            cfg (str | dict): Model configuration file path or dictionary. 
            ch (int): Number of input channels.
            nc (int, optional): Number of classes.
            verbose (bool): Whether to display model information.   
        """    
        super().__init__(cfg=cfg, ch=ch, nc=nc, verbose=verbose)     
     
    def loss(self, batch, preds=None):  
        """Compute loss.     

        Args:     
            batch (dict): Batch to compute loss on.
            preds (torch.Tensor | list[torch.Tensor], optional): Predictions.
        """ 
        if not hasattr(self, "criterion"):    
            from ultralytics.utils.loss import TVPSegmentLoss
   
            visual_prompt = batch.get("visuals", None) is not None  # TODO
            self.criterion = (
                (E2ELoss(self, TVPSegmentLoss) if getattr(self, "end2end", False) else TVPSegmentLoss(self))  
                if visual_prompt
                else self.init_criterion()    
            )   

        if preds is None:    
            preds = self.forward(batch["img"], tpe=batch.get("txt_feats", None), vpe=batch.get("visuals", None))
        return self.criterion(preds, batch) 
   

class Ensemble(torch.nn.ModuleList):  
    """Ensemble of models.    

    This class allows combining multiple YOLO models into an ensemble for improved performance through model averaging  
    or other ensemble techniques.

    Methods:     
        __init__: Initialize an ensemble of models.
        forward: Generate predictions from all models in the ensemble.

    Examples:     
        Create an ensemble of models 
        >>> ensemble = Ensemble()   
        >>> ensemble.append(model1)    
        >>> ensemble.append(model2)
        >>> results = ensemble(image_tensor)
    """   
     
    def __init__(self):    
        """Initialize an ensemble of models."""    
        super().__init__()

    def forward(self, x, augment=False, profile=False, visualize=False):
        """Generate the YOLO network's final layer.
  
        Args:
            x (torch.Tensor): Input tensor.
            augment (bool): Whether to augment the input.     
            profile (bool): Whether to profile the model.
            visualize (bool): Whether to visualize the features.     
    
        Returns:  
            y (torch.Tensor): Concatenated predictions from all models.  
            train_out (None): Always None for ensemble inference.    
        """
        y = [module(x, augment, profile, visualize)[0] for module in self]  
        # y = torch.stack(y).max(0)[0]  # max ensemble
        # y = torch.stack(y).mean(0)  # mean ensemble
        y = torch.cat(y, 2)  # nms ensemble, y shape(B, HW, C*num_models)     
        return y, None  # inference, train output  
     
  
# Functions ------------------------------------------------------------------------------------------------------------
     

@contextlib.contextmanager     
def temporary_modules(modules=None, attributes=None):   
    """Context manager for temporarily adding or modifying modules in Python's module cache (`sys.modules`).
     
    This function can be used to change the module paths during runtime. It's useful when refactoring code, where you've  
    moved a module from one location to another, but you still want to support the old import paths for backwards
    compatibility.     
     
    Args:
        modules (dict, optional): A dictionary mapping old module paths to new module paths.
        attributes (dict, optional): A dictionary mapping old module attributes to new module attributes.  

    Examples:
        >>> with temporary_modules({"old.module": "new.module"}, {"old.module.attribute": "new.module.attribute"}):
        >>> import old.module  # this will now import new.module    
        >>> from old.module import attribute  # this will now import new.module.attribute  

    Notes:     
        The changes are only in effect inside the context manager and are undone once the context manager exits.     
        Be aware that directly manipulating `sys.modules` can lead to unpredictable results, especially in larger   
        applications or libraries. Use this function with caution. 
    """    
    if modules is None:
        modules = {}
    if attributes is None:
        attributes = {}
    import sys
    from importlib import import_module
  
    try:
        # Set attributes in sys.modules under their old name
        for old, new in attributes.items():
            old_module, old_attr = old.rsplit(".", 1)
            new_module, new_attr = new.rsplit(".", 1)
            setattr(import_module(old_module), old_attr, getattr(import_module(new_module), new_attr))
  
        # Set modules in sys.modules under their old name
        for old, new in modules.items():   
            sys.modules[old] = import_module(new)  
 
        yield
    finally: 
        # Remove the temporary module paths 
        for old in modules:
            if old in sys.modules:
                del sys.modules[old]  


class SafeClass:
    """A placeholder class to replace unknown classes during unpickling."""

    def __init__(self, *args, **kwargs):
        """Initialize SafeClass instance, ignoring all arguments."""
        pass
  
    def __call__(self, *args, **kwargs):
        """Run SafeClass instance, ignoring all arguments."""    
        pass   
 

class SafeUnpickler(pickle.Unpickler):     
    """Custom Unpickler that replaces unknown classes with SafeClass."""
   
    def find_class(self, module, name):
        """Attempt to find a class, returning SafeClass if not among safe modules.
   
        Args:
            module (str): Module name.  
            name (str): Class name.
  
        Returns: 
            (type): Found class or SafeClass.
        """    
        safe_modules = (
            "torch",  
            "collections",
            "collections.abc",
            "builtins",
            "math",
            "numpy",
            # Add other modules considered safe
        ) 
        if module in safe_modules:   
            return super().find_class(module, name)
        else:
            return SafeClass

   
def torch_safe_load(weight, safe_only=False):
    """Attempt to load a PyTorch model with the torch.load() function. If a ModuleNotFoundError is raised, it catches
    the error, logs a warning message, and attempts to install the missing module via the check_requirements()
    function. After installation, the function again attempts to load the model using torch.load().

    Args:
        weight (str): The file path of the PyTorch model.  
        safe_only (bool): If True, replace unknown classes with SafeClass during loading.
   
    Returns:
        ckpt (dict): The loaded model checkpoint.
        file (str): The loaded filename.   
 
    Examples: 
        >>> from ultralytics.nn.tasks import torch_safe_load
        >>> ckpt, file = torch_safe_load("path/to/best.pt", safe_only=True)
    """
    from ultralytics.utils.downloads import attempt_download_asset
   
    check_suffix(file=weight, suffix=".pt")     
    file = attempt_download_asset(weight)  # search online if missing locally
    try: 
        with temporary_modules(  
            modules={
                "ultralytics.yolo.utils": "ultralytics.utils",
                "ultralytics.yolo.v8": "ultralytics.models.yolo",
                "ultralytics.yolo.data": "ultralytics.data",
            },    
            attributes={   
                "ultralytics.nn.modules.block.Silence": "torch.nn.Identity",  # YOLOv9e    
                "ultralytics.nn.tasks.YOLOv10DetectionModel": "ultralytics.nn.tasks.DetectionModel",  # YOLOv10    
                "ultralytics.utils.loss.v10DetectLoss": "ultralytics.utils.loss.E2EDetectLoss",  # YOLOv10
            },    
        ): 
            if safe_only:     
                # Load via custom pickle module
                safe_pickle = types.ModuleType("safe_pickle")     
                safe_pickle.Unpickler = SafeUnpickler
                safe_pickle.load = lambda file_obj: SafeUnpickler(file_obj).load()
                with open(file, "rb") as f:
                    ckpt = torch_load(f, pickle_module=safe_pickle) 
            else:     
                ckpt = torch_load(file, map_location="cpu")

    except ModuleNotFoundError as e:  # e.name is missing module name     
        if e.name == "models": 
            raise TypeError(     
                emojis(
                    f"ERROR ❌️ {weight} appears to be an Ultralytics YOLOv5 model originally trained "     
                    f"with https://github.com/ultralytics/yolov5.\nThis model is NOT forwards compatible with " 
                    f"YOLOv8 at https://github.com/ultralytics/ultralytics."
                    f"\nRecommend fixes are to train a new model using the latest 'ultralytics' package or to "   
                    f"run a command with an official Ultralytics model, i.e. 'yolo predict model=yolo26n.pt'"     
                )
            ) from e    
        elif e.name == "numpy._core":
            raise ModuleNotFoundError(
                emojis(    
                    f"ERROR ❌️ {weight} requires numpy>=1.26.1, however numpy=={__import__('numpy').__version__} is installed."
                )  
            ) from e     
        LOGGER.warning(  
            f"{weight} appears to require '{e.name}', which is not in Ultralytics requirements."   
            f"\nAutoInstall will run now for '{e.name}' but this feature will be removed in the future." 
            f"\nRecommend fixes are to train a new model using the latest 'ultralytics' package or to " 
            f"run a command with an official Ultralytics model, i.e. 'yolo predict model=yolo26n.pt'"  
        )
        check_requirements(e.name)  # install missing module
        ckpt = torch_load(file, map_location="cpu")
  
    if not isinstance(ckpt, dict):
        # File is likely a YOLO instance saved with i.e. torch.save(model, "saved_model.pt")
        LOGGER.warning(     
            f"The file '{weight}' appears to be improperly saved or formatted. "    
            f"For optimal results, use model.save('filename.pt') to correctly save YOLO models."
        )    
        ckpt = {"model": ckpt.model}  

    return ckpt, file   
   

def load_checkpoint(weight, device=None, inplace=True, fuse=False):
    """Load a single model weights.     
 
    Args:     
        weight (str | Path): Model weight path.  
        device (torch.device, optional): Device to load model to.     
        inplace (bool): Whether to do inplace operations.
        fuse (bool): Whether to fuse model.    
   
    Returns: 
        model (torch.nn.Module): Loaded model.
        ckpt (dict): Model checkpoint dictionary. 
    """    
    ckpt, weight = torch_safe_load(weight)  # load ckpt
    args = {**DEFAULT_CFG_DICT, **(ckpt.get("train_args", {}))}  # combine model and default args, preferring model args
    model = (ckpt.get("ema") or ckpt["model"]).float()  # FP32 model
 
    # Model compatibility updates     
    model.args = args  # attach args to model 
    model.pt_path = weight  # attach *.pt file path to model
    model.task = getattr(model, "task", guess_model_task(model))
    if not hasattr(model, "stride"):
        model.stride = torch.tensor([32.0])

    model = (model.fuse() if fuse and hasattr(model, "fuse") else model).eval().to(device)  # model in eval mode
  
    # Module updates
    for m in model.modules():    
        if hasattr(m, "inplace"):    
            m.inplace = inplace
        elif isinstance(m, torch.nn.Upsample) and not hasattr(m, "recompute_scale_factor"):
            m.recompute_scale_factor = None  # torch 1.11.0 compatibility
     
    # Return model and ckpt    
    return model, ckpt
 
     
def parse_model(d, ch, verbose=True, warehouse_manager=None):
    """Parse a YOLO model.yaml dictionary into a PyTorch model.  
    将 YOLO 的 model.yaml 字典解析为 PyTorch 模型。
 
    Args:
        d (dict): Model dictionary (模型配置字典).    
        ch (int): Input channels (输入通道数，通常为3).  
        verbose (bool): Whether to print model details (是否打印模型详细信息).  
 
    Returns:     
        model (torch.nn.Sequential): PyTorch model (构建好的 PyTorch 模型).
        save (list): Sorted list of output layers (需要保存输出以供后续层使用的层索引列表).    
    """
    import ast
    
    # Args / 参数提取
    legacy_from_yaml = "legacy" in d 
    legacy = d["legacy"] if legacy_from_yaml else True  # backward compatibility for v3/v5/v8/v9 models (向后兼容标志)
    max_channels = float("inf")
    # 提取类别数(nc), 激活函数(act), 缩放比例(scales), 端到端模式(end2end)   
    nc, act, scales, end2end = (d.get(x) for x in ("nc", "activation", "scales", "end2end"))   
    # 提取回归最大值(reg_max)，默认为16 (用于DFL)    
    reg_max = d.get("reg_max", 16)
    # 提取深度(depth)、宽度(width)倍率和关键点形状(kpt_shape)   
    depth, width, kpt_shape = (d.get(x, 1.0) for x in ("depth_multiple", "width_multiple", "kpt_shape"))    
    scale = d.get("scale")     
    neck_param_keys = frozenset({"head_channel", "fusion_mode", "node_mode"})
     
    threshold = None

    # 如果定义了 scales (如 n, s, m, l, x)，则覆盖 depth, width, max_channels  
    if scales:     
        if not scale:
            scale = next(iter(scales.keys()))
            LOGGER.warning(f"no model scale passed. Assuming scale='{scale}'.")
        if len(scales[scale]) == 3:  
            depth, width, max_channels = scales[scale]
        elif len(scales[scale]) == 4:   
            depth, width, max_channels, threshold = scales[scale]
        else:
            raise ValueError(f"Unsupported scales entry for scale '{scale}': {scales[scale]}")  

    # 如果定义了激活函数，重定义 Conv 模块的默认激活函数     
    if act:
        Conv.default_act = eval(act)  # redefine default activation, i.e. Conv.default_act = torch.nn.SiLU() 
        if verbose:
            LOGGER.info(f"{colorstr('activation:')} {act}")  # print
  
    if verbose:   
        LOGGER.info( 
            f"\n{BOLD}{ORANGE}{'':>3}{'from':>20}{'n':>3}{'params':>10}  {'module':<60}{'arguments':<30}{RESET}"
        ) 
    
    ch = [ch] # 记录每一层的输出通道数，初始化为输入通道数   
    layers, save, c2 = [], [], ch[-1]  # layers: 层列表, save: 需要保存的层索引, c2: 当前层输出通道   
 
    # 基础模块集合，这些模块通常遵循标准的通道计算逻辑
    base_modules = frozenset(     
        {
            Classify, Conv, ConvTranspose, GhostConv, Bottleneck, GhostBottleneck, SPP, SPPF, C2fPSA, C2PSA,
            DWConv, Focus, BottleneckCSP, C1, C2, C2f, C3k2, RepNCSPELAN4, ELAN1, ADown, torch.nn.Conv2d,
            AConv, SPPELAN, C2fAttn, C3, C3TR, C3Ghost, torch.nn.ConvTranspose2d, DWConvTranspose2d, C3x, RepC3,
            PSA, SCDown, C2fCIB, A2C2f, GSConv, GSBottleneck, GSBottleneckC, VoVGSCSP, CSPStage, RepBlock, CSP_MSCB,   
        }
    )    
    # 重复模块集合，这些模块包含 'n' (重复次数) 参数，受 depth_multiple 影响
    repeat_modules = frozenset(  # modules with 'repeat' arguments
        {
            BottleneckCSP, C1, C2, C2f, C3k2, C2fAttn, C3, C3TR, C3Ghost, C3x,  
            RepC3, C2fPSA, C2fCIB, C2PSA, A2C2f, VoVGSCSP, CSPStage, RepBlock, CSP_MSCB,   
        }     
    )  
  
    # 万物皆可融Block集合，这些模块包含 'n' (重复次数) 参数，受 depth_multiple 影响
    block_repeat_modules = frozenset(  
        {
            C3_Block, C2f_Block, C3k2_Block, MANet   
        }
    )   

    # 万物皆可融Block集合，这些模块不受重复次数的影响     
    block_modules = frozenset(   
        {   
            MetaFormer_Block, MetaFormer_Mona, MetaFormer_SEFN, MetaFormer_Mona_SEFN     
        }
    )

    # attention集合
    attention_modules = frozenset(   
        {  
            ACA, ACAB, CoordAtt, CASAB, ContrastDrivenFeatureAggregation, DeformableLKA, DHPF, EMA, FSDA, FSA, KSFA, LSKBlock,
            MCA, MLCA, MultiSEAM, SimAM    
        }
    )

    # transformer集合 
    transformer_modules = frozenset({CGTA, CirculantAttention, LCGA, WDAM, GLCDM, SAA, SDTA})

    # mamba集合     
    mamba_modules = frozenset({GradMamba, PatchMamba})

    # conv_module集合
    conv_modules = frozenset(    
        {  
            ConvAttn, Converse2D, DiverseBranchBlock, DCNv2, DeformConvV3, DeformConvV4, DEConv, DeepDiverseBranchBlock,  
            DilatedReparamConv, DSA, DySnakeConv, AdaptiveDilatedConv, FDConv, FourierConv, GCConv, gConv, InceptionDWConv2d, 
            A3Conv, Partial_Conv, PSConv, RepMBConv, ScConv, SFS_Conv, ReparamLargeKernelConv, SMPConv, WideDiverseBranchBlock, WTConv2d,
            DEGConv, CKConv, AGIDWC, MSIDWC, PolyConv
        }    
    )

    # stem集合
    stem_modules = frozenset({DPAS, LoGStem, ODALStem, RepStem, SRFD, WGFS})

    # upsample集合
    upsample_modules = frozenset(    
        {  
            CARAFE, Converse2D_Up, DSUB, DySample, EUCB, EUCB_SC, WaveletUnPool 
        }     
    )
  
    # downsample集合
    downsample_modules = frozenset(
        {     
            ADown, DRFD, DMSSP, EdgeLAWDS, FSConv, FSCGD, FreqLAWDS, ContextGuidedBlock_Down, HWD, LAWDS, RouterLAWDS, SPDConv, WaveletPool, V7DownSampling  
        }  
    )    

    # module集合
    module_modules = frozenset(  
        {
            APBottleneck, ARF, BCIM, CAMixer, CFBlock, CNCM, CSSC, DBlock, DGCM, DSEBlock, DRG, DWR, DynamicFilter, EBlock, EFCM, EfficientViMBlock, EfficientViMBlock_CGLU,
            ELGCA_EncoderBlock, ESCBlock, EVA, Faster_Block, Faster_Block_CGLU, Faster_CGA_Block, FAT_Block, FCM, FMA, FrequencyCM, GLGM, HFRB, InceptionDWBlock, IEL, GLSA,
            iRMB, JDPM, LaSEA, LEGBlock, LEGM, LWGA, MAC, MambaOut, MambaOut_DilatedReparamConv, MambaOut_UniRepLKBlock, MFEblock, MSBlock, MSCB,
            PartialNetBlock, PKIBlock, PKINetV2Block, RepConvBlock, RepViTBlock, SPJFrequencyBlock, Star_Block, StripBlock, UniRepLKNetBlock, Heat2D, Wave2D, MyModule,   
            SparseMambaBlock, MSInit, PFG, LFP, AMSI, CMSI, FMSI, HOIE, ADIE, HPFGA, DPFGA, SFEB, FourierSR, RFGM, IRA, RepHMS, CBIMS, PartialBlock, PartialBlock_SFMB
        }
    ) 

    # mlp集合  
    mlp_modules = frozenset({AFFN, MCCG, DFFN, DIFF, DML, DSRG, EDFFN, EFFN, FMFFN, FRFN, GLSFFN, KAN, MSAFFN, PolyMLP})  
   
    # head集合    
    head_modules = frozenset( 
        {    
            Detect_LQE, Segment_LQE, Segment26_LQE, OBB_LQE, OBB26_LQE, Pose_LQE, Pose26_LQE,     
            Detect_DyHead, Segment_DyHead, Segment26_DyHead, OBB_DyHead, OBB26_DyHead, Pose_DyHead, Pose26_DyHead  
        }
    )   
    

    # ==================================================================
    # ===== 0526 版迁移模块注册（本段为新增，未改动上方任何既有代码） =====
    # ==================================================================
    # ① 走 base_modules + repeat_modules：签名 (c1, c2, n, ...)
    legacy0526_csp = frozenset({
        C3k2_EMSC, C3k2_EMSCP, C3k2_DCNv2_Dynamic, C3k2_EIEM, C3k2_DCMB,
        C3k2_MutilScaleEdgeInformationEnhance, C3k2_MutilScaleEdgeInformationSelect,
        C3k2_DCMB_KAN, CSP_PTB, CSP_PMSFA, CSP_FreqSpatial, RGCSPELAN,
        CSP_MSCB_0526, CSP_MSCB_SC,
        C3k2_DWR_DRB, C3k2_iRMB_Cascaded, C3k2_iRMB_DRB, C3k2_iRMB_SWC,
        C3k2_Star_CAA, C3k2_SMPCGLU, C3k2_Faster_KAN, C3k2_GlobalFilter, C3k2_TSSA,
        C3k2_AdditiveBlock_CGLU, C3k2_MSMHSA_CGLU, C3k2_SHSA_CGLU, C3k2_SMAFB_CGLU,
        C3k2_IdentityFormerCGLU, C3k2_RandomMixingCGLU, C3k2_PoolingFormerCGLU,
        C3k2_ConvFormerCGLU, C3k2_CaFormerCGLU, C3k2_ELGCACGLU, C3k2_StripCGLU,
        C3k2_MambaOut_UniRepLK, C3k2_MambaOut_LSConv, C3k2_MambaOut_FDConv,
        C3k2_MambaOut_DSA, C3k2_MambaOut_SFSC,
        C3k2_PFDConv, C3k2_FasterFD, C3k2_PSFSConv, C3k2_FasterSFSC,
        C3k2_DSAN_EDFFN, C3k2_SHSA_EPGO, C3k2_SHSA_EPGO_CGLU,
        C2TSSA_DYT, C2TSSA_DYT_Mona, C2TSSA_DYT_Mona_SEFN,
        C2TSSA_DYT_Mona_SEFFN, C2TSSA_DYT_Mona_EDFFN, C2Pola_DYT,
        # 0526 原始包装类（新版可用 C3k2_Block/MANet 组合语法复现，此处保留原类以保证旧 yaml 逐字可跑）
        C3k2_Faster_CGLU, C3k2_EfficientVIM_CGLU, C3k2_Star,
        MANet_FasterBlock, MANet_FasterCGLU, MANet_Star, MANet_GCConv,
    })
    # ⑦ A2C2f 系（yolo12）：签名 (c1, c2, n, a2, area, ...)，与新版 A2C2f 同形
    legacy0526_a2c2f = frozenset({
        A2C2f_CGLU_DYT, A2C2f_DFFN_DYT, A2C2f_FMFFN_DYT, A2C2f_DFFN_DYT_Mona,
    })
    # ② 只走 base_modules：签名 (c1, c2, ...)，不含重复次数 n
    legacy0526_base = frozenset({
        FeaturePyramidSharedConv, GSConvE,
        DBBNCSPELAN4, OREPANCSPELAN4, DRBNCSPELAN4,
    })
    # ③ 走 attention_modules 形状：c1 = c2 = ch[f]；args = [c2, *args]
    legacy0526_dim = frozenset({
        LAWDS_0526, CSPOmniKernel, ELA_HSFPN, MPCA, EMSConv, EMSConvP,
    })
    # ④ 走 upsample_modules 形状
    legacy0526_upsample = frozenset({EUCB_0526})
    # ⑤ 走 featurefusion 形状：c1 = [ch[fi] for fi in f]；args = [c1, c2, *args[1:]]
    legacy0526_fusion = frozenset({HAFB, ConvEdgeFusion})
    # ⑥ 检测头（本体逐字保留，此处注册的是 head/<名>.py 里的适配层）
    legacy0526_head = frozenset({
        Detect_Efficient, Segment_Efficient,
        Detect_LSCD, Segment_LSCD, Pose_LSCD, OBB_LSCD,
        Detect_TADDH, Segment_TADDH, Pose_TADDH, OBB_TADDH,
        Detect_RSCD, Segment_RSCD, Pose_RSCD, OBB_RSCD,
        Detect_LSCSBD, Segment_LSCSBD, Pose_LSCSBD, OBB_LSCSBD,
        Detect_LSDECD, Segment_LSDECD, Pose_LSDECD, OBB_LSDECD,
        Detect_LSCD_LQE, Segment_LSCD_LQE, Pose_LSCD_LQE, OBB_LSCD_LQE,
    })
    legacy0526_segment_head = frozenset({
        Segment_Efficient, Segment_LSCD, Segment_TADDH,
        Segment_RSCD, Segment_LSCSBD, Segment_LSDECD, Segment_LSCD_LQE,
    })
    # 并入新版既有集合（不修改上方原始定义）
    base_modules       = base_modules       | legacy0526_csp | legacy0526_base | legacy0526_a2c2f
    repeat_modules     = repeat_modules     | legacy0526_csp | legacy0526_a2c2f
    attention_modules  = attention_modules  | legacy0526_dim
    upsample_modules   = upsample_modules   | legacy0526_upsample
    head_modules       = head_modules       | legacy0526_head


    # ================= 第二批：A1/A2/B/C/D 组注册（新增，未改动既有代码） =================
    # A1+A2+B：CSP 包装壳，签名 (c1, c2, n, ...)
    legacy0526_csp2 = frozenset({
        C2ASSA, C2BRA, C2CGA, C2DA, C2DPB, C2MSLA,
        C2Pola, C2TSSA, C3k2_AKConv, C3k2_AP, C3k2_AdditiveBlock, C3k2_AggregatedAtt,
        C3k2_CAMixer, C3k2_CFBlock, C3k2_CKConv, C3k2_CNCM, C3k2_CSI, C3k2_CSSC,
        C3k2_CaFormer, C3k2_CloAtt, C3k2_ContextGuided, C3k2_ConvAttn, C3k2_ConvFormer, C3k2_Converse,
        C3k2_ConverseB, C3k2_DAttention, C3k2_DBB, C3k2_DBlock, C3k2_DCNv2, C3k2_DCNv3,
        C3k2_DCNv4, C3k2_DEConv, C3k2_DEGConv, C3k2_DLKA, C3k2_DRB, C3k2_DRG,
        C3k2_DSA, C3k2_DSAN, C3k2_DSEBlock, C3k2_DTAB, C3k2_DWR, C3k2_DeepDBB,
        C3k2_DySnakeConv, C3k2_DynamicConv, C3k2_DynamicFilter, C3k2_EBlock, C3k2_ELGCA, C3k2_EMA,
        C3k2_EMBC, C3k2_ESC, C3k2_ETB, C3k2_EVA, C3k2_EVS, C3k2_EfficientVIM,
        C3k2_FADC, C3k2_FAT, C3k2_FCA_CTA, C3k2_FCA_SFA, C3k2_FDConv, C3k2_FDT,
        C3k2_FFCM, C3k2_FMABlock, C3k2_FMB, C3k2_Faster, C3k2_Faster_EMA, C3k2_FocusedLinearAttention,
        C3k2_FourierConv, C3k2_FourierSR, C3k2_FrequencyCM, C3k2_GCConv, C3k2_GLCDM, C3k2_GLGM,
        C3k2_GLSS2D, C3k2_GLVSS, C3k2_GhostDynamicConv, C3k2_GradMamba, C3k2_GroupMamba, C3k2_GroupMambaBlock,
        C3k2_HDRAB, C3k2_HFERB, C3k2_HFRB, C3k2_Heat, C3k2_IDWB, C3k2_IDWC,
        C3k2_IEL, C3k2_IRA, C3k2_IdentityFormer, C3k2_JDPM, C3k2_KAN, C3k2_KW,
        C3k2_Kat, C3k2_LEGM, C3k2_LFE, C3k2_LFEM, C3k2_LGLB, C3k2_LSBlock,
        C3k2_LVMB, C3k2_LWGA, C3k2_LaSEA, C3k2_MAC, C3k2_MBRConv3, C3k2_MBRConv5,
        C3k2_MFEB, C3k2_MLCA, C3k2_MSBlock, C3k2_MSInit, C3k2_MSM, C3k2_MambaOut,
        C3k2_MambaVision, C3k2_MobileMamba, C3k2_MogaBlock, C3k2_ODConv, C3k2_OREPA, C3k2_PConv,
        C3k2_PFG, C3k2_PKIModule, C3k2_PPA, C3k2_Parc, C3k2_PartialNetBlock, C3k2_PatchMamba,
        C3k2_PoolingFormer, C3k2_RAB, C3k2_RCB, C3k2_REPVGGOREPA, C3k2_RFAConv, C3k2_RFCAConv,
        C3k2_RFCBAMConv, C3k2_RFGM, C3k2_RMB, C3k2_RMBC, C3k2_RMBC_LA, C3k2_RVB,
        C3k2_RVB_EMA, C3k2_RandomMixing, C3k2_RetBlock, C3k2_SAVSS, C3k2_SBSM, C3k2_SCConv,
        C3k2_SFEB, C3k2_SFHF, C3k2_SFMB, C3k2_SFSConv, C3k2_SHSA, C3k2_SMAFB,
        C3k2_SMB, C3k2_SMS, C3k2_SPJFB, C3k2_SWC, C3k2_ScConv, C3k2_Strip,
        C3k2_TVIM, C3k2_TransMamba, C3k2_TransMixer, C3k2_UniConvBlock, C3k2_UniRepLKNetBlock, C3k2_VSS,
        C3k2_VSSD, C3k2_WDBB, C3k2_WTConv, C3k2_gConv, C3k2_iRMB, C3k2_wConv,
        CSP_EDLAN, RCSOSA,
    })
    # C：签名 (c1, c2, ...)，不含重复次数
    legacy0526_base2 = frozenset({
        DGCST, ESMoE, KWConv, LDConv, RFAConv, RFCAConv,
        RFCBAMConv, RepNCSPELAN4_CAA, SPPF_LSKA, wConv2d,
    })
    # C：AIFI 形 —— args = [ch[f], *args]; c2 = args[0]
    legacy0526_aifi = frozenset({AIFI_RepBN, DTAB, ETB, FDT})
    # C：attention 形 —— c1 = c2 = ch[f]; args = [c2, *args]
    legacy0526_dim2 = frozenset({
        CAA_HSFPN, CA_HSFPN, FocalModulation, AttentiveLayer, RCM, TransNeXt_AggregatedAttention,
    })
    # C：HGStem 形 —— (c1, cm, c2, ...)，含重复次数
    legacy0526_hg = frozenset({Ghost_HGBlock, Rep_HGBlock})
    # D：检测头（本体逐字保留，此处注册的是适配层）
    legacy0526_head2 = frozenset({
        Detect_AFPN_P2345, Detect_AFPN_P2345_Custom, Detect_AFPN_P345, Detect_AFPN_P345_Custom,
        Detect_DyHeadWithDCNV3, Detect_DyHeadWithDCNV4, Detect_LADH, Detect_MultiSEAM,
        Detect_SEAM, OBB_LADH, Pose_LADH, Segment_LADH,
    })
    legacy0526_segment_head2 = frozenset({Segment_LADH})

    # A2C2f / C2PSA 系变体，签名与新版同形
    legacy0526_a2c2f2 = frozenset({
        A2C2f_CGLU, A2C2f_DFFN, A2C2f_DYT, A2C2f_EDFFN, A2C2f_FMFFN,
        A2C2f_FRFN, A2C2f_KAN, A2C2f_Mona, A2C2f_SEFFN, A2C2f_SEFN,
    })
    legacy0526_c2psa2 = frozenset({
        C2PSA_AFFN, C2PSA_BinaryAttn, C2PSA_SDTA, C2PSA_CGLU, C2PSA_CGTA, C2PSA_CirculantAtt,
        C2PSA_DML, C2PSA_DWMMSA, C2PSA_DYT, C2PSA_EDFFN, C2PSA_EGSA,
        C2PSA_EPGO, C2PSA_FMFFN, C2PSA_LCGA, C2PSA_LRSA, C2PSA_MALA,
        C2PSA_Mona, C2PSA_SEFFN, C2PSA_SEFN, C2PSA_SWSA, C2PSA_WCA,
        C2PSA_WDAM, C2PSA_FGSA,
    })

    # YOLOv13 / Mamba-YOLO 架构组件
    legacy0526_v13 = frozenset({DSC3k2, DSConv_YOLO13})
    legacy0526_mambayolo = frozenset({SimpleStem, VisionClueMerge, XSSBlock, VSSBlock_YOLO})

    base_modules       = base_modules       | legacy0526_csp2 | legacy0526_v13 | legacy0526_mambayolo | legacy0526_base2 | legacy0526_a2c2f2 | legacy0526_c2psa2
    repeat_modules     = repeat_modules     | legacy0526_csp2 | frozenset({DSC3k2, XSSBlock, VSSBlock_YOLO}) | legacy0526_a2c2f2 | legacy0526_c2psa2
    attention_modules  = attention_modules  | legacy0526_dim2
    head_modules       = head_modules       | legacy0526_head2

    # ================= 第三批注册（新增，未改动既有代码） =================
    # 0526 版注意力模块（md 第 14 条注意力清单）：c1 = c2 = ch[f]; args = [c2, *args]
    legacy0526_attn3 = frozenset({
        AFGCAttention, BAMBlock, CPCA, ELA, EffectiveSEModule,
        EfficientChannelAttention, SegNext_Attention, SpatialGroupEnhance, TripletAttention,
        SDP_Improved, OREPA_LargeConv, MambaVisionMixer, Mamba2Block, DRMamba,
        SMPBlock, SMPCNN, LeFF, WindowFrequencyModulation_FMFFN, multiRCM,
    })
    # slim-neck 与 RepViT 变体：(c1, c2, n, ...)
    legacy0526_csp3 = frozenset({VoVGSCSPC, VoVGSCSPns, C3k2_RVB_SE, C3k2_KW})
    # (c1, c2, ...) 无重复次数
    legacy0526_base3 = frozenset({GSConvns, RepHDW, ConvUtr, KWConv})
    # 检测头
    legacy0526_head3 = frozenset({Detect_NMSFree, Detect_FGIF, Detect_DyHead_Prune})

    base_modules       = base_modules       | legacy0526_csp3 | legacy0526_base3
    repeat_modules     = repeat_modules     | legacy0526_csp3
    attention_modules  = attention_modules  | legacy0526_attn3
    head_modules       = head_modules       | legacy0526_head3

    # 遍历 backbone 和 head 的每一层配置   
    # f: from (输入来源层索引), n: number (重复次数), m: module (模块名), args: arguments (参数)    
    for i, (f, n, m, args) in enumerate(d["backbone"] + d["head"]):  # from, number, module, args 
        # 动态获取模块类 (例如从字符串 'Conv' 获取 nn.Conv2d 类)
        try: 
            if m == "node_mode" and "node_mode" in d:
                m = d["node_mode"]
            m = (
                getattr(torch.nn, m[3:])  
                if "nn." in m
                else getattr(__import__("torchvision").ops, m[16:]) 
                if "torchvision.ops." in m
                else globals()[m]
            )  # get module
        except Exception as e:   
            pass  
 
        # 解析参数中的字符串 (例如将字符串变量转换为实际值)  
        for j, a in enumerate(args):
            if isinstance(a, str):
                if a in neck_param_keys and a in d:
                    args[j] = d[a]
                    continue    
                with contextlib.suppress(ValueError, SyntaxError):
                    args[j] = locals()[a] if a in locals() else ast.literal_eval(a)  
    
        # 计算该模块的重复次数 (depth gain)，受 depth_multiple 影响
        n = n_ = max(round(n * depth), 1) if n > 1 else n  # depth gain   
        
        # --- 针对不同模块类型的特殊处理 ---     
        if m in (base_modules | conv_modules | module_modules | stem_modules):    
            # 标准模块处理：计算输入通道 c1 和输出通道 c2  
            c1, c2 = ch[f], args[0]
            if c2 != nc:  # if c2 != nc (e.g., Classify() output)
                # 使用 width_multiple 调整通道数，并确保是 8 的倍数     
                c2 = make_divisible(min(c2, max_channels) * width, 8)   
    
            # 特殊模块参数调整   
            if m is C2fAttn:  # set 1) embed channels and 2) num heads   
                args[1] = make_divisible(min(args[1], max_channels // 2) * width, 8)
                args[2] = int(max(round(min(args[2], max_channels // 2 // 32)) * width, 1) if args[2] > 1 else args[2])   

            #以此构建参数列表：[输入通道, 输出通道, *其他参数]
            args = [c1, c2, *args[1:]]     

            # 0526 版 KernelWarehouse：插入仓库管理器与自动生成的层名（照抄 0526 版 tasks.py:1130）
            if m in frozenset({KWConv, C3k2_KW}):
                args.insert(2, f'layer{i}')
                args.insert(2, warehouse_manager)
            
            # 如果是重复模块，插入重复次数 n 到参数列表  
            if m in repeat_modules:    
                args.insert(2, n)  # number of repeats
                n = 1 # 既然参数里传了 n，外层循环就不需要再重复构建了
            
            if m in module_modules:     
                n = 1
            
            # YOLOv10/v11 等新模块的特定逻辑    
            if m in {C3k2, C3k2_Block}:  # for M/L/X sizes
                if not legacy_from_yaml:
                    legacy = False   
                if scale in "mlx":
                    args[3] = True     
            if m in (legacy0526_a2c2f | legacy0526_a2c2f2):
                if scale in "lx":
                    args.extend((True, 1.2))
            if m is A2C2f:    
                if not legacy_from_yaml:     
                    legacy = False
                if scale in "lx":  # for L/X sizes
                    if args[-1] == 'turbo':  
                        args.extend((True, 1.5))
                    else:
                        args.extend((True, 1.2))
            if m is C2fCIB:
                if not legacy_from_yaml:
                    legacy = False 
        
        elif m is AIFI: 
            args = [ch[f], *args]    
        
        elif m in frozenset({HGStem, HGBlock}): # Hourglass 模块处理  
            c1, cm, c2 = ch[f], args[0], args[1]
            args = [c1, cm, c2, *args[2:]]  
            if m is HGBlock:
                args.insert(4, n)  # number of repeats    
                n = 1    
   
        elif m is ResNetLayer:     
            c2 = args[1] if args[3] else args[1] * 4
        
        elif m is torch.nn.BatchNorm2d:
            args = [ch[f]]    
 
        elif m is Concat: # 拼接层：输出通道为所有输入层通道之和
            c2 = sum(ch[x] for x in f)    
        
        # 检测头和分割头处理 (Detect, Segment, Pose, OBB 等)
        elif m in (frozenset({Detect, WorldDetect, YOLOEDetect, Segment, Segment26, YOLOESegment, YOLOESegment26, Pose, Pose26, OBB, OBB26}) | head_modules):     
            # 添加参数: reg_max, end2end, 以及来自 backbone 不同层的通道列表
            args.extend([reg_max, end2end, [ch[x] for x in f]])   
            if m in ({Segment, YOLOESegment, Segment26, YOLOESegment26, Segment_LQE, Segment26_LQE, Segment_DyHead, Segment26_DyHead} | legacy0526_segment_head | legacy0526_segment_head2):     
                # 分割头需要调整 mask 通道数 
                args[2] = make_divisible(min(args[2], max_channels) * width, 8) 
            if m in {  
                Detect,    
                YOLOEDetect,   
                Segment,    
                Segment26,
                YOLOESegment,     
                YOLOESegment26,  
                Pose,
                Pose26,    
                OBB,
                OBB26, 
                Detect_LQE,
                Segment_LQE,
                Segment26_LQE,
                Pose_LQE,
                Pose26_LQE,
                OBB_LQE,
                OBB26_LQE,
                Detect_DyHead,     
                Segment_DyHead,
                Segment26_DyHead,
                Pose_DyHead,
                Pose26_DyHead, 
                OBB_DyHead,
                OBB26_DyHead,
            }:
                m.legacy = legacy
        
        elif m is v10Detect: # YOLOv10 检测头
            args.append([ch[x] for x in f])
   
        elif m is ImagePoolingAttn:    
            args.insert(1, [ch[x] for x in f])  # channels as second arg    
        
        elif m is RTDETRDecoder:  # special case, channels arg must be passed in index 1 (RT-DETR 解码器)    
            args.insert(1, [ch[x] for x in f])    
   
        elif m is CBLinear:
            c2 = args[0]
            c1 = ch[f]   
            args = [c1, c2, *args[1:]] 
        
        elif m is CBFuse:
            c2 = ch[f[-1]]
    
        elif m in frozenset({TorchVision, Index}):
            c2 = args[0] 
            c1 = ch[f] 
            args = [*args[1:]]
     
        elif m in {GetIndexOutput}:
            c2 = ch[f][args[0]]

        elif m is ChannelTransformer:    
            c1 = [ch[x] for x in f] 
            c2 = c1
            args = [c1]
   
        elif m in frozenset({A3Fusion2, A3Fusion3}):
            c1 = [ch[x] for x in f]
            level = args[0]    
            c2 = c1[level]
            args = [level, c1, *args[1:]]

        elif m in (block_repeat_modules | block_modules):   
            # 标准模块处理：计算输入通道 c1 和输出通道 c2     
            c1, c2 = ch[f], args[0]    
            if c2 != nc:  # if c2 != nc (e.g., Classify() output)  
                # 使用 width_multiple 调整通道数，并确保是 8 的倍数
                c2 = make_divisible(min(c2, max_channels) * width, 8)   
            
            selfatt, selfatt_args_index = None, -1    
            if type(args) is list:
                parse_scope_locals = locals().copy()   
 
                def _resolve_module_from_name(module_name):  
                    if isinstance(module_name, str):
                        try:     
                            return getattr(torch.nn, module_name[3:]) if "nn." in module_name else globals()[module_name] 
                        except Exception as e:
                            raise Exception(f"{module_name} is maybe not import in task.py, please check. {e}") 
                    return module_name
 
                def _resolve_module_param_value(value):   
                    if isinstance(value, dict):   
                        if "module" in value: 
                            nested_module = _resolve_module_from_name(value["module"])  
                            nested_param = value.get("param", {})     
                            if isinstance(nested_param, dict):
                                nested_param = {k: _resolve_module_param_value(v) for k, v in nested_param.items()}
                            else:
                                nested_param = _resolve_module_param_value(nested_param)     
                            return partial(nested_module, **nested_param) if isinstance(nested_param, dict) else partial(nested_module, nested_param)
                        return {k: _resolve_module_param_value(v) for k, v in value.items()}
                    if isinstance(value, list):
                        return [_resolve_module_param_value(v) for v in value]     
                    if isinstance(value, tuple):  
                        return tuple(_resolve_module_param_value(v) for v in value)
                    if isinstance(value, str):     
                        try:
                            return (    
                                parse_scope_locals[value]
                                if value in parse_scope_locals
                                else (getattr(torch.nn, value[3:]) if "nn." in value else ast.literal_eval(value))
                            ) 
                        except:  
                            return value  
                    return value

                for j, a in enumerate(args): 
                    if isinstance(a, str):
                        with contextlib.suppress(ValueError):   
                            try:
                                args[j] = locals()[a] if a in locals() else ast.literal_eval(a) 
                            except:
                                args[j] = a 
                    elif type(a) is dict:   
                        if 'module' in a: 
                            module_ = _resolve_module_from_name(a['module'])     
                            module_param = a.get('param', {})
                            if isinstance(module_param, dict):  
                                module_param = {k: _resolve_module_param_value(v) for k, v in module_param.items()}
                            else:
                                module_param = _resolve_module_param_value(module_param)
                            args[j] = partial(module_, **module_param) if isinstance(module_param, dict) else partial(module_, module_param)
                        if 'selfatt' in a:
                            selfatt = a['selfatt']
                            selfatt_args_index = j
            if selfatt_args_index != -1:
                args.pop(j)   
 
            if selfatt is not None:
                args = ([c1, c2, *args[1:]], {'selfatt':selfatt})  
            else:   
                args = [c1, c2, *args[1:]]   
            
            if m in block_repeat_modules: 
                if selfatt is not None:
                    args[0].insert(3, n)
                else:
                    args.insert(3, n)  # number of repeats
                n = 1 # 既然参数里传了 n，外层循环就不需要再重复构建了    
            if m in block_modules:     
                n = 1
 
        elif m is asf_attention_model:  
            c2 = ch[f[0] if isinstance(f, list) else f]    
            args = [c2, *args] 

        elif m in (attention_modules | frozenset({ChannelAttention_HSFPN})):
            c1 = c2 = ch[f]
            args = [c2, *args]

        elif m in (transformer_modules | mamba_modules):
            c2 = ch[f]
            args = [c2, *args]  
   
        elif m in mlp_modules:
            c1 = ch[f]
            c2 = c1   
            if len(args) == 0:     
                args = [c1]
            elif len(args) == 1:
                hidden = args[0]  
                if isinstance(hidden, (int, float)): 
                    hidden = make_divisible(min(hidden, max_channels) * width, 8) 
                args = [c1, hidden]
            else:     
                hidden = args[0]
                out = args[1]     
                if isinstance(hidden, (int, float)):
                    hidden = make_divisible(min(hidden, max_channels) * width, 8)    
                if isinstance(out, (int, float)) and out != nc:
                    out = make_divisible(min(out, max_channels) * width, 8)  
                    c2 = out 
                elif isinstance(out, (int, float)):    
                    c2 = out
                args = [c1, hidden, out, *args[2:]]

        elif m in upsample_modules:

            if m is WaveletUnPool:  
                c1 = ch[f]
                c2 = c1 // 4     
                args = []     
            else:
                c2 = ch[f]
                args = [c2, *args]
    
        elif m in downsample_modules:     

            if m is WaveletPool: 
                c1 = ch[f]   
                c2 = c1 * 4    
                args = []   
            else:
                # 标准模块处理：计算输入通道 c1 和输出通道 c2
                c1, c2 = ch[f], args[0]
                if c2 != nc:  # if c2 != nc (e.g., Classify() output)
                    # 使用 width_multiple 调整通道数，并确保是 8 的倍数     
                    c2 = make_divisible(min(c2, max_channels) * width, 8)     
                args = [c1, c2, *args[1:]]

        elif m is BiFusion:    
            c1 = [ch[x] for x in f]
            c2 = make_divisible(min(args[0], max_channels) * width, 8)
            args = [c1, c2]
    
        elif m in frozenset({SimFusion_4in, AdvPoolFusion}):
            c2 = sum(ch[x] for x in f)   
   
        elif m is SimFusion_3in: 
            c2 = args[0]  
            if c2 != nc: 
                c2 = make_divisible(min(c2, max_channels) * width, 8)  
            args = [[ch[f_] for f_ in f], c2]

        elif m is IFM:
            c1 = ch[f]
            c2 = sum(args[0])     
            args = [c1, *args]
  
        elif m is InjectionMultiSum_Auto_pool:
            c1 = ch[f[0]]    
            c2 = args[0]   
            args = [c1, *args]  

        elif m is PyramidPoolAgg:    
            c2 = args[0] 
            args = [sum(ch[f_] for f_ in f), *args]

        elif m is TopBasicLayer:     
            c2 = sum(args[1])

        elif m is Zoom_cat:   
            c2 = sum(ch[x] for x in f)   

        elif m is Add:
            c2 = ch[f[-1]] 
     
        elif m is Multiply:
            c2 = ch[f[0]]  
     
        elif m is ScalSeq:    
            c1 = [ch[x] for x in f]
            c2 = make_divisible(min(args[0], max_channels) * width, 8)
            args = [c1, c2]  

        elif m is Fusion:     
            mode = d.get(args[0], args[0]) if args else "bifpn"
            c1 = [ch[x] for x in f]
            c2 = sum(c1) if mode == "concat" else ch[f[0]]
            args = [c1, mode]

        elif m is HFP:
            c2 = ch[f]   
            args = [c2, *args]    

        elif m is SDP:  
            c1 = [ch[x] for x in f]   
            c2 = make_divisible(min(args[0], max_channels) * width, 8)
            args = [c1, c2, *args[1:]]
   
        elif m in frozenset({FocusFeature, DynamicFrequencyFocusFeature, AlignmentGuidedFocusFeature}):
            c1 = [ch[i] for i in f]
            c2 = c1[1]
            args = [c1, *args] 

        elif m in frozenset({Detect_LSPCD, Segment_LSPCD, Segment26_LSPCD, OBB_LSPCD, OBB26_LSPCD, Pose_LSPCD, Pose26_LSPCD}):
            # 添加参数: reg_max, end2end, 以及来自 backbone 不同层的通道列表     
            args.extend([reg_max, end2end, [ch[x] for x in f]])
            if m in {Segment_LSPCD, Segment26_LSPCD}:  
                # 分割头需要调整 mask 通道数   
                args[2] = make_divisible(min(args[2], max_channels) * width, 8)    
        
        elif m in frozenset({FAENet}):
            c1 = c2 = ch[f]  
   
        elif m is HyperComputeModule:     
            if threshold is None:
                raise ValueError("HyperComputeModule requires scales entries with a threshold value.") 
            c1, c2 = ch[f], args[0]
            c2 = make_divisible(min(c2, max_channels) * width, 8)
            args = [c1, c2, threshold] 
    
        # ------------------- featurefusion -------------------
        elif m in frozenset({CAFM, CIDAF, ContextGuideFusionModule, DCGRM, DynamicAlignFusion, DPCF, ERM, HAFFormer, HFFE, LCA, MFM, MFPM, MSCRM, MultiScalePCA, MultiScalePCA_Down, MSAM_P5,     
                            GDSAFusion, LowFrequencyFeatureFusion, LPRMAlignUpModule, MAFusion, MultiScaleGatedAttn, WFU, CGAFusion, SDFM, PSFM, WDAF, SFSFusion, FAAFusion}):
            c1, c2 = [ch[fi] for fi in f], args[0] 
            c2 = make_divisible(min(c2, max_channels) * width, 8)
            args = [c1, c2, *args[1:]]
        elif m is CSFCN:   
            c1, c2 = [ch[fi] for fi in f], args[0]     
            c2 = [ch[f[0]], make_divisible(min(c2, max_channels) * width, 8)]
            args = [c1, c2[1], *args[1:]]
        elif m is PST:
            c1, c2 = [ch[fi] for fi in f], args[0] 
            c2 = make_divisible(min(c2, max_channels) * width, 8)
            args = [*c1, c2, *args[1:]]


        elif m in frozenset({CrossLayerChannelAttention, CrossLayerSpatialAttention}):
            # 0526 版 tasks.py:1343，多输出模块，yaml 中配合 GetIndexOutput 取用各尺度
            c2 = [ch[x] for x in f]
            args = [c2[0], *args]

        # ============ 0526 版迁移模块的专属分支（照抄 0526 版 parse_model） ============
        elif m in legacy0526_fusion:
            # 与新版 featurefusion 分支同形，等价于 0526 版 tasks.py:1371
            c1 = [ch[x] for x in f]
            c2 = make_divisible(min(args[0], max_channels) * width, 8)
            args = [c1, c2, *args[1:]]

        elif m is CSMHSA:                              # 0526 版 tasks.py:1297
            c1 = [ch[x] for x in f]
            c2 = ch[f[-1]]
            args = [c1, c2]

        elif m is CFC_CRB_0526:                        # 0526 版 tasks.py:1301
            c1 = ch[f]
            c2 = c1 // 2
            args = [c1, *args]

        elif m is SFC_G2_0526:                         # 0526 版 tasks.py:1305
            c1 = [ch[x] for x in f]
            c2 = c1[0]
            args = [c1]

        elif m in frozenset({EIEStem, Dynamic_HGBlock}):   # 0526 版 tasks.py:1160
            c1, cm, c2 = ch[f], args[0], args[1]
            if c2 != nc:
                c2 = make_divisible(min(c2, max_channels) * width, 8)
                cm = make_divisible(min(cm, max_channels) * width, 8)
            args = [c1, cm, c2, *args[2:]]
            if m is Dynamic_HGBlock:
                args.insert(4, n)
                n = 1

        elif m is MutilScaleEdgeInfoGenetator:         # 0526 版 tasks.py:1355
            # 多输出模块，yaml 中配合 GetIndexOutput 取用各尺度
            c1 = ch[f]
            c2 = [make_divisible(min(i, max_channels) * width, 8) for i in args[0]]
            args = [c1, c2]

        elif m is ContextGuideFusionModule_0526:       # 0526 版 tasks.py:1312
            c1 = [ch[x] for x in f]
            c2 = 2 * c1[1]
            args = [c1]

        elif m is FocusFeature_0526:                   # 0526 版 tasks.py:1290
            c1 = [ch[x] for x in f]
            c2 = int(c1[1] * 0.5 * 3)
            args = [c1, *args]

        elif m is DASI:                                # 0526 版 tasks.py:1294
            c1 = [ch[x] for x in f]
            args = [c1, c2]

        elif m is SBA:                                 # 0526 版 tasks.py:1319
            c1 = [ch[x] for x in f]
            c2 = c1[-1]
            args = [c1, c2]

        elif m is SNI:                                 # 0526 版 tasks.py:1381
            c2, up_f = ch[f], args[0]
            args = [up_f]

        elif m is DynamicScalSeq:                      # 0526 版 tasks.py:1279
            c1 = [ch[x] for x in f]
            c2 = make_divisible(args[0] * width, 8)
            args = [c1, c2]
        # ==========================================================================


        # ============ 第二批 0526 迁移模块的专属分支 ============
        elif m in legacy0526_aifi:                      # 0526 版 tasks.py:1157
            args = [ch[f], *args]
            c2 = args[0]

        elif m in legacy0526_hg:                        # 0526 版 tasks.py:1160
            c1, cm, c2 = ch[f], args[0], args[1]
            if c2 != nc:
                c2 = make_divisible(min(c2, max_channels) * width, 8)
                cm = make_divisible(min(cm, max_channels) * width, 8)
            args = [c1, cm, c2, *args[2:]]
            args.insert(4, n)
            n = 1

        elif m is SDI:                                  # 0526 版 tasks.py:1286
            args = [[ch[x] for x in f]]

        elif m is CAFMFusion:                           # 0526 版 tasks.py:1309
            c2 = ch[f[1]]
            args = [c2, *args]

        elif m in frozenset({PyramidContextExtraction}):  # 0526 版 tasks.py:1330（多输出，配合 GetIndexOutput）
            c1 = [ch[x] for x in f]
            c2 = c1
            args = [c1]

        elif m is DynamicInterpolationFusion:            # 0526 版 tasks.py:1337
            c2 = ch[f[0]]
            args = [[ch[x] for x in f]]

        elif m is FuseBlockMulti:                        # 0526 版 tasks.py:1340
            c2 = ch[f[0]]
            args = [c2]

        elif m is FreqFusion:                            # 0526 版 tasks.py:1346
            c2 = ch[f[0]]
            args = [[ch[x] for x in f], *args]

        elif m is CAFM_Fusion:                           # 0526 版 tasks.py:1371
            c1 = [ch[x] for x in f]
            c2 = make_divisible(min(args[0], max_channels) * width, 8)
            args = [c1, c2, *args[1:]]

        elif m in frozenset({Pzconv, FCM_1, FCM_2, FCM_3}):  # 0526 版 tasks.py:1384
            c2 = ch[f]
            args = [c2]

        elif m is DownsampleConv:                        # 0526 版 tasks.py:1402
            c1 = ch[f]
            c2 = c1 * 2
            args = [c1]
            if scale in "lx":
                args.append(True)

        elif m is FullPAD_Tunnel:                        # 0526 版 tasks.py:1409
            c2 = ch[f[0]]

        elif m is MSAM:                                  # 0526 版 tasks.py:1426
            c1 = [ch[x] for x in f]
            c2 = make_divisible(min(args[0], max_channels) * width, 8)
            args = [c1, c2, *args[1:]]

        elif m is DASI:                                  # 已在第一批注册，此处仅占位说明
            c1 = [ch[x] for x in f]
            args = [c1, c2]
        # =========================================================


        elif m is HyperACE:                              # YOLOv13
            c1 = [ch[x] for x in f]
            c2 = make_divisible(min(args[0], max_channels) * width, 8)
            args = [c1[1], c2, *args[1:]]
        elif isinstance(m, str): 
            t = m
            # if len(args) == 2:        
            #     m = timm.create_model(m, pretrained=args[0], pretrained_cfg_overlay={'file':args[1]}, features_only=True)
            # elif len(args) == 1:
            m = timm.create_model(m, pretrained=args[0], features_only=True)
            c2 = m.feature_info.channels()

        else:     
            c2 = ch[f] # 默认输出通道等于输入通道   
   
        # --- 模块实例化 ---
        if m in (block_repeat_modules | block_modules):
            if type(args) is dict: 
                m_ = m(**args)    
            elif type(args) is list:  
                m_ = m(*args)  # module  
            else:
                m_ = m(*args[0], **args[1])   
            # 获取模块类型字符串，去除 '__main__.' 前缀     
            t = str(m)[8:-2].replace("__main__.", "")  # module type    
        else:
            if isinstance(m, type) == False and type(m) is not str:
                m_ = m    
            else:  
                # 如果 n > 1，创建一个包含 n 个该模块的 Sequential，否则直接实例化该模块     
                m_ = torch.nn.Sequential(*(m(*args) for _ in range(n))) if n > 1 else m(*args)  # module
                # 获取模块类型字符串，去除 '__main__.' 前缀
                t = str(m)[8:-2].replace("__main__.", "")  # module type    
    
        # 计算该层的参数量   
        m_.np = sum(x.numel() for x in m_.parameters())  # number params
        m_.i, m_.f, m_.type = i, f, t  # attach index, 'from' index, type (附加元数据：索引、来源、类型)   
    
        if verbose:
            LOGGER.info(   
                f"{ORANGE}{i:>3}{f!s:>20}{BOLD}{GREEN}{n_:>3}{RESET}{ORANGE}{m_.np:10.0f}  "
                f"{CYAN}{t:<60}{RESET}{YELLOW}{args!s:<30}{RESET}"
            )    # print
        
        # 更新 save 列表：记录哪些层的输出被后续层引用 (Concat, Detect 等需要跨层连接)
        # x % i 处理负索引 (如 -1 代表上一层)     
        save.extend(x % i for x in ([f] if isinstance(f, int) else f) if x != -1)  # append to savelist
        
        layers.append(m_) # 添加到层列表     
        if i == 0:   
            ch = [] # 第0层处理完后清空 ch，后续只记录网络内部的通道变化   
        ch.append(c2) # 记录当前层的输出通道数  
 
    return torch.nn.Sequential(*layers), sorted(save) 
  
    
def yaml_model_load(path): 
    """Load a YOLOv8 model from a YAML file.

    Args:
        path (str | Path): Path to the YAML file.    
 
    Returns:   
        (dict): Model dictionary. 
    """    
    path = Path(path)     
    if path.stem in (f"yolov{d}{x}6" for x in "nsmlx" for d in (5, 8)):
        new_stem = re.sub(r"(\d+)([nslmx])6(.+)?$", r"\1\2-p6\3", path.stem) 
        LOGGER.warning(f"Ultralytics YOLO P6 models now use -p6 suffix. Renaming {path.stem} to {new_stem}.")
        path = path.with_name(new_stem + path.suffix)

    unified_path = re.sub(r"(\d+)([nslmx])(.+)?$", r"\1\3", str(path))  # i.e. yolov8x.yaml -> yolov8.yaml
    yaml_file = check_yaml(unified_path, hard=False) or check_yaml(path)
    d = YAML.load(yaml_file)  # model dict 
    d["scale"] = guess_model_scale(path)
    d["yaml_file"] = str(path)
    return d   

   
def guess_model_scale(model_path):
    """Extract the size character n, s, m, l, or x of the model's scale from the model path. 

    Args:     
        model_path (str | Path): The path to the YOLO model's YAML file.     

    Returns:  
        (str): The size character of the model's scale (n, s, m, l, or x).
    """   
    try:
        return re.search(r"yolo(e-)?[v]?\d+([nslmx])", Path(model_path).stem).group(2)
    except AttributeError:
        return "" 
     

def guess_model_task(model):
    """Guess the task of a PyTorch model from its architecture or configuration.     

    Args: 
        model (torch.nn.Module | dict): PyTorch model or model configuration in YAML format.   
 
    Returns:  
        (str): Task of the model ('detect', 'segment', 'classify', 'pose', 'obb').
    """     
 
    def cfg2task(cfg):
        """Guess from YAML dictionary."""
        m = cfg["head"][-1][-2].lower()  # output module name
        if m in {"classify", "classifier", "cls", "fc"}:
            return "classify"  
        if "detect" in m:
            return "detect"     
        if "segment" in m:
            return "segment"
        if "pose" in m:
            return "pose"
        if "obb" in m:
            return "obb" 

    # Guess from model cfg
    if isinstance(model, dict):
        with contextlib.suppress(Exception): 
            return cfg2task(model)
    # Guess from PyTorch model   
    if isinstance(model, torch.nn.Module):  # PyTorch model  
        for x in "model.args", "model.model.args", "model.model.model.args":
            with contextlib.suppress(Exception): 
                return eval(x)["task"]  # nosec B307: safe eval of known attribute paths 
        for x in "model.yaml", "model.model.yaml", "model.model.model.yaml":  
            with contextlib.suppress(Exception):
                return cfg2task(eval(x))  # nosec B307: safe eval of known attribute paths     
        for m in model.modules():
            if isinstance(m, (Segment, YOLOESegment)):
                return "segment"
            elif isinstance(m, Classify):
                return "classify"     
            elif isinstance(m, Pose):    
                return "pose"
            elif isinstance(m, OBB):
                return "obb"
            elif isinstance(m, (Detect, WorldDetect, YOLOEDetect, v10Detect)):  
                return "detect"

    # Guess from model filename 
    if isinstance(model, (str, Path)):
        model = Path(model)   
        if "-seg" in model.stem or "segment" in model.parts:
            return "segment"   
        elif "-cls" in model.stem or "classify" in model.parts:
            return "classify"
        elif "-pose" in model.stem or "pose" in model.parts:
            return "pose"
        elif "-obb" in model.stem or "obb" in model.parts:  
            return "obb"  
        elif "detect" in model.parts:  
            return "detect"

    # Unable to determine task from model
    LOGGER.warning(
        "Unable to automatically guess model task, assuming 'task=detect'. "
        "Explicitly define task for your model, i.e. 'task=detect', 'segment', 'classify','pose' or 'obb'."  
    ) 
    return "detect"  # assume detect
