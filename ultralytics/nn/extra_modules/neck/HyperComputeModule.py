'''
本文件由BiliBili：魔傀面具整理
ultralytics/nn/module_images/TPAMI2025-HyperCompute.png     
ultralytics/nn/module_images/TPAMI2025-HyperCompute.md
论文链接：https://arxiv.org/pdf/2408.04804   
''' 
     
import torch
import torch.nn as nn
  

class MessageAgg(nn.Module):
    def __init__(self, agg_method="mean"):   
        super().__init__()
        self.agg_method = agg_method

    def forward(self, x, path): 
        x = torch.matmul(path, x)   
        if self.agg_method == "mean":
            norm = 1 / torch.sum(path, dim=2, keepdim=True)
            norm[torch.isinf(norm)] = 0    
            return norm * x
        return x 


class HyPConv(nn.Module): 
    def __init__(self, c1, c2):
        super().__init__()     
        self.fc = nn.Linear(c1, c2)    
        self.v2e = MessageAgg(agg_method="mean")    
        self.e2v = MessageAgg(agg_method="mean")

    def forward(self, x, incidence): 
        x = self.fc(x)     
        edge_feat = self.v2e(x, incidence.transpose(1, 2).contiguous())
        return self.e2v(edge_feat, incidence)   
     
   
class HyperComputeModule(nn.Module):
    def __init__(self, c1, c2, threshold):  
        super().__init__()
        self.threshold = threshold    
        self.hgconv = HyPConv(c1, c2)   
        self.bn = nn.BatchNorm2d(c2)    
        self.act = nn.SiLU()     
     
    def forward(self, x):   
        b, c, h, w = x.shape
        x = x.view(b, c, -1).transpose(1, 2).contiguous()
        feature = x.clone()    
        distance = torch.cdist(feature, feature)
        incidence = (distance < self.threshold).to(dtype=x.dtype, device=x.device)     
        x = self.hgconv(x, incidence).to(dtype=x.dtype, device=x.device) + x  
        x = x.transpose(1, 2).contiguous().view(b, c, h, w)   
        return self.act(self.bn(x))    
