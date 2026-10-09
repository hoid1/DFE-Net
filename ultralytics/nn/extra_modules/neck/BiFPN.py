'''
本文件由BiliBili：魔傀面具整理 
ultralytics/nn/module_images/BIFPN.md   
论文链接：https://arxiv.org/abs/1911.09070 
论文链接：https://openreview.net/pdf?id=q2ZaVU6bEsT
'''
     
import torch     
import torch.nn as nn
import torch.nn.functional as F
   
from ultralytics.nn.modules.conv import Conv  
from ultralytics.nn.extra_modules.featurefusion.legacy_misc_fusion import SDI
   
class Fusion(nn.Module):
    """Multi-branch neck fusion operator with BiFPN-style weighted fusion support."""   

    def __init__(self, inc_list, fusion="bifpn"):
        super().__init__() 
        assert fusion in {"weight", "adaptive", "concat", "bifpn", "SDI"}
        self.fusion = fusion

        if self.fusion == "bifpn":  
            self.fusion_weight = nn.Parameter(torch.ones(len(inc_list), dtype=torch.float32), requires_grad=True)  
            self.epsilon = 1e-4
        elif self.fusion == "SDI":
            self.SDI = SDI(inc_list)
        elif self.fusion in {"weight", "adaptive"}:
            self.fusion_conv = nn.ModuleList([Conv(inc, inc, 1) for inc in inc_list]) 
            if self.fusion == "adaptive":
                self.fusion_adaptive = Conv(sum(inc_list), len(inc_list), 1)     
    
    def forward(self, x):
        if self.fusion in {"weight", "adaptive"}:
            x = [self.fusion_conv[i](feat) for i, feat in enumerate(x)]  

        if self.fusion == "SDI":
            return self.SDI(x)
        if self.fusion == "weight":
            return torch.sum(torch.stack(x, dim=0), dim=0)
        if self.fusion == "adaptive":    
            fusion = torch.softmax(self.fusion_adaptive(torch.cat(x, dim=1)), dim=1)   
            x_weight = torch.split(fusion, [1] * len(x), dim=1)
            return torch.sum(torch.stack([x_weight[i] * x[i] for i in range(len(x))], dim=0), dim=0)  
        if self.fusion == "concat":     
            return torch.cat(x, dim=1) 
  
        fusion_weight = F.relu(self.fusion_weight, inplace=False)
        fusion_weight = fusion_weight / (torch.sum(fusion_weight, dim=0) + self.epsilon)   
        return torch.sum(torch.stack([fusion_weight[i] * x[i] for i in range(len(x))], dim=0), dim=0)
