'''   
本文件由BiliBili：魔傀面具整理 
ultralytics/nn/module_images/TPAMI2025-LRPM.png
ultralytics/nn/module_images/TPAMI2025-LRPM.md  
论文链接：https://arxiv.org/pdf/2507.11893
'''    
 
import warnings   
warnings.filterwarnings('ignore')
from calflops import calculate_flops
    
import torch
import torch.nn as nn
import torch.nn.functional as F 
   
try:
    from mmcv.ops.carafe import CARAFE
except ImportError as e:  
    # raise ImportError(    
    #     "无法从 mmcv.ops 导入 CARAFE。"
    #     "请确保已安装 mmcv-full / mmcv>=2.0 且包含 CUDA 算子。"  
    # ) from e
    CARAFE = None  
 
from ultralytics.nn.modules.conv import Conv

class LPRM_efficient(nn.Module):    
    """
    使用 CARAFE 和 PixelShuffle 技巧模拟空洞邻域聚合的 LPRM。
    通过将 scale 维度折叠到 batch 维度来优化内存。     
    """
   
    def __init__(self, channels: int, kernel_size: int = 3, dilation: int = 1, upscale: int = 1):
        """
        初始化模块。
    
        Args:     
            channels (int): 输入和输出 value_feat 的通道数。    
            kernel_size (int): 邻域大小。
            dilation (int): 空洞率。
            upscale (int): CARAFE 的上采样倍数。
        """ 
        super().__init__()
        self.channels = channels
        self.kernel_size = kernel_size     
        self.dilation = dilation
        self.upscale = upscale
  
        self.scale_factor = dilation     
        self.scale_sq = self.scale_factor * self.scale_factor    

        # [0623 对齐] 显式报错，避免 CARAFE 缺失时报难以定位的 NoneType 错误
        if CARAFE is None:
            raise ImportError("LPRM_efficient requires mmcv.ops.carafe.CARAFE.")

        self.carafe_op = CARAFE(
            kernel_size=self.kernel_size,   
            group_size=1,
            scale_factor=upscale    
        )    

        if self.scale_factor > 1:
            self.pixel_unshuffle = nn.PixelUnshuffle(self.scale_factor)
            self.pixel_shuffle = nn.PixelShuffle(self.scale_factor)   

    def forward(self, mask_pred: torch.Tensor, value_feat: torch.Tensor) -> torch.Tensor:
        """
        前向传播。

        Args:
            mask_pred (torch.Tensor): 外部预测的重组权重 (mask)。
                                      形状: [B, k*k, H, W]。 
            value_feat (torch.Tensor): 需要被提炼的值特征。 
                                       形状: [B, C, H, W]。  

        Returns:
            torch.Tensor: 提炼后的特征。形状: [B, C, H*upscale, W*upscale]。     
        """
        if self.scale_factor == 1:   
            # [0623 对齐] softmax 后回落到 value 的 dtype，保证 AMP(fp16) 下不报 dtype mismatch
            mask = mask_pred.softmax(dim=1).to(value_feat.dtype)
            return self.carafe_op(value_feat, mask)   

        # --- 模拟空洞操作 ---     
        B, C, H, W = value_feat.shape
        k_sq = self.kernel_size * self.kernel_size
        s = self.scale_factor
        Hs, Ws = H // s, W // s

        # 1. Unshuffle 并折叠到 Batch 维度   
        unshuffled_value = self.pixel_unshuffle(value_feat)     
        unshuffled_value = unshuffled_value.view(
            B, C, self.scale_sq, Hs, Ws
        ).permute(0, 2, 1, 3, 4).reshape(B * self.scale_sq, C, Hs, Ws)     
  
        unshuffled_mask = self.pixel_unshuffle(mask_pred)
        unshuffled_mask = unshuffled_mask.view(  
            B, k_sq, self.scale_sq, Hs, Ws    
        ).permute(0, 2, 1, 3, 4).reshape(B * self.scale_sq, k_sq, Hs, Ws)    
   
        # 2. 在折叠后的 Batch 上应用 CARAFE  
        # [0623 对齐] 同上，AMP 下 dtype 对齐
        mask = unshuffled_mask.softmax(dim=1).to(unshuffled_value.dtype)
        refined_unshuffled = self.carafe_op(   
            unshuffled_value, mask    
        ) 

        # 3. 逆操作：从 Batch 维度恢复并 PixelShuffle 回原始分辨率
        refined_unshuffled = refined_unshuffled.view(B, self.scale_sq, C, Hs, Ws)    
        refined_shuffled = refined_unshuffled.permute(
            0, 2, 1, 3, 4     
        ).reshape(B, C * self.scale_sq, Hs, Ws)
        refined_feat = self.pixel_shuffle(refined_shuffled)     
   
        return refined_feat 

    
class LPRMAlignUpModule(nn.Module):
    """  
    低分辨率 → 高分辨率的对齐上采样模块。
    使用高分辨率特征作为引导，预测 CARAFE 重组权重，
    以内容感知方式将低分辨率特征上采样，最后与目标尺寸对齐。
    """    
   
    def __init__(
        self,
        in_channels,    
        out_channels,     
        kernel_size: int = 3,   
        align_corners: bool = False,   
        compress_ratio: int = 4,   
    ):
        super().__init__()  
        self.align_corners = align_corners

        low_res_channels, high_res_channels = in_channels   

        predictor_in_channels = low_res_channels + high_res_channels
        compressed_channels = predictor_in_channels // compress_ratio     
   
        self.compress_conv_low = nn.Sequential(     
            nn.Conv2d(low_res_channels, compressed_channels, 1, bias=True),    
        )   
        self.compress_conv_high = nn.Sequential(
            nn.Conv2d(high_res_channels, compressed_channels, 1, bias=True),
        )    
    
        self.lprm = LPRM_efficient(
            channels=compressed_channels,
            kernel_size=kernel_size,
            upscale=2,   
        )
        self.lpr_conv = nn.Conv2d(     
            compressed_channels, kernel_size ** 2, 3, padding=1
        )
 
        self.conv_final = Conv(low_res_channels, out_channels, 1)
    
    def forward(self, inputs) -> torch.Tensor:
        """   
        Args:    
            x_low (torch.Tensor): 低分辨率特征，形状 [B, C_low, H_low, W_low]。     
            guidance_high_aligned (torch.Tensor): 高分辨率引导特征，
                                                   形状 [B, C_high, target_h, target_w]。  

        Returns:
            torch.Tensor: 上采样并对齐到 (target_h, target_w) 的特征，  
                          形状 [B, C_low, target_h, target_w]。   
        """    
        x_low, guidance_high_aligned = inputs  

        B, C, target_h, target_w = guidance_high_aligned.shape   
  
        compressed_low = self.compress_conv_low(x_low)
        compressed_high = self.compress_conv_high(guidance_high_aligned)
   
        # 局部像素关系权重，在目标 2x 尺寸上融合低/高分辨率信息    
        lpr_feat = (
            F.interpolate(
                self.lpr_conv(compressed_low),
                scale_factor=2,     
                mode='bilinear', 
                align_corners=self.align_corners,
            )
            + F.interpolate(    
                self.lpr_conv(compressed_high),
                size=(x_low.size(-2) * 2, x_low.size(-1) * 2),    
                mode='bilinear',
                align_corners=self.align_corners,
            )  
        )   
 
        # CARAFE 内容感知上采样 2 倍
        x_low = self.lprm(lpr_feat, x_low)

        # 最终尺寸对齐 
        x_low_upsampled = F.interpolate(
            x_low,    
            size=(target_h, target_w),     
            mode='bilinear', 
            align_corners=self.align_corners,  
        )   
    
        return self.conv_final(x_low_upsampled) 
   
    
if __name__ == '__main__':   
    RED, GREEN, BLUE, YELLOW, ORANGE, RESET = "\033[91m", "\033[92m", "\033[94m", "\033[93m", "\033[38;5;208m", "\033[0m"
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu') 
    batch_size, channel_1, height_1, width_1 = 1, 64, 20, 20
    batch_size, channel_2, height_2, width_2 = 1, 32, 40, 40 
    ouc_channel = 64
    inputs_1 = torch.randn((batch_size, channel_1, height_1, width_1)).to(device)  
    inputs_2 = torch.randn((batch_size, channel_2, height_2, width_2)).to(device) 

    module = LPRMAlignUpModule([channel_1, channel_2], ouc_channel).to(device)
  
    outputs = module([inputs_1, inputs_2])
    print(GREEN + f'inputs1.size:{inputs_1.size()} inputs2.size:{inputs_2.size()} outputs.size:{outputs.size()}' + RESET)     

    print(ORANGE)     
    flops, macs, _ = calculate_flops(model=module,
                                     args=[[inputs_1, inputs_2]],
                                     output_as_string=True,
                                     output_precision=4,
                                     print_detailed=True)     
    print(RESET) 
