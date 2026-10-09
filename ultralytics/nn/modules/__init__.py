# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license  
""" 
Ultralytics neural network modules.
  
This module provides access to various neural network components used in Ultralytics models, including convolution    
blocks, attention mechanisms, transformer components, and detection/segmentation heads.

Examples:
    Visualize a module with Netron    
    >>> from ultralytics.nn.modules import Conv
    >>> import torch
    >>> import subprocess
    >>> x = torch.ones(1, 128, 40, 40)    
    >>> m = Conv(128, 128)
    >>> f = f"{m._get_name()}.onnx"     
    >>> torch.onnx.export(m, x, f) 
    >>> subprocess.run(f"onnxslim {f} {f} && open {f}", shell=True, check=True)  # pip install onnxslim
"""
 
from .block import *
from .conv import *    
from .head import *
from .transformer import * 
 
from . import block, conv, head, transformer

__all__ = (   
    *block.__all__,
    *conv.__all__,
    *head.__all__,    
    *transformer.__all__, 
)