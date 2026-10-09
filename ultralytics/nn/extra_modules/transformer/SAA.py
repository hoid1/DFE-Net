'''
本文件由BiliBili：魔傀面具整理   
ultralytics/nn/module_images/CVPR2026F-SAA.png 
ultralytics/nn/module_images/CVPR2026F-SAA.md 
论文链接：https://arxiv.org/pdf/2604.07994   
'''    

import warnings
warnings.filterwarnings('ignore')
from calflops import calculate_flops

import torch
import torch.nn as nn    
import torch.nn.functional as F
    
def cluster_and_merge(x, cluster_num, subsample_factor=4):
    B, N, C = x.shape    
    device = x.device  
    K = cluster_num 

    x_proj = x     

    x_norm = F.normalize(x_proj, dim=-1)  # (B, N, D) where D = proj_dim or C  

    S = min(N, max(2 * K, subsample_factor * K))  # Ensure S >= 2K, cap at N     

    samples_per_region = S // K   
    sub_idx = []  
    for i in range(K):
        start_idx = i * (N // K)
        end_idx = (i + 1) * (N // K) if i < K - 1 else N  
        region_size = end_idx - start_idx  
        n_samples = min(samples_per_region, region_size)  

        if region_size > 0:
            region_perm = torch.randperm(region_size, device=device)[:n_samples]
            sub_idx.append(start_idx + region_perm)
  
    # Add random samples to reach S if needed
    sub_idx = torch.cat(sub_idx) 
    if len(sub_idx) < S:
        remaining = S - len(sub_idx)
        all_idx = torch.arange(N, device=device)
        mask = torch.ones(N, dtype=torch.bool, device=device)
        mask[sub_idx] = False 
        additional = all_idx[mask][torch.randperm((~mask).sum(), device=device)[:remaining]]
        sub_idx = torch.cat([sub_idx, additional])  

    x_norm_sub = x_norm[:, sub_idx]  # (B, S, D)

    # Cosine similarity (normalized dot product)
    sim_sub = x_norm_sub @ x_norm_sub.transpose(1, 2)  # (B, S, S)     
    torch.diagonal(sim_sub, dim1=1, dim2=2).fill_(-1)     
 
    # Mean of top-k similarities 
    k = min(K, S - 1)     
    sim_topk_sub, _ = torch.topk(sim_sub, k=k, dim=-1)  # (B, S, k)     
    density_sub = sim_topk_sub.mean(dim=-1)  # (B, S)     
    density_sub = density_sub + torch.rand_like(density_sub) * 1e-6
    
    # Mask for points with higher density
    mask_higher_density = (density_sub[:, None, :] > density_sub[:, :, None]).float()  # (B, S, S)

    # For points with higher density, keep similarity; otherwise set to very negative
    masked_sim_sub = sim_sub * mask_higher_density - 1e9 * (1.0 - mask_higher_density)  
 
    # Maximum similarity to higher-density points
    max_sim_to_higher, _ = masked_sim_sub.max(dim=-1)  # (B, S)
   
    # Convert to distance: δ = 1 - similarity    
    delta_sub = 1.0 - max_sim_to_higher  # (B, S)

    # Handle points with maximum density (no higher-density neighbors)
    max_density_mask_sub = (mask_higher_density.sum(dim=-1) == 0)  # (B, S)   
     
    # For max density points, use maximum distance in subsample
    min_sim_global = sim_sub.min(dim=-1)[0]  # (B, S)  
    max_dist_global = (1.0 - min_sim_global).to(delta_sub.dtype)
    delta_sub[max_density_mask_sub] = max_dist_global[max_density_mask_sub] 
     
    # Ensure delta is non-negative 
    delta_sub = torch.clamp(delta_sub, min=0.0)     
   
    # Score: γ = ρ × δ
    score_sub = density_sub * delta_sub  # (B, S)
   
    # Select top-K scoring points as cluster centers
    _, center_idx_in_sub = torch.topk(score_sub, k=K, dim=-1)  # (B, K)  
     
    # Map back to original indices     
    center_idx = sub_idx[center_idx_in_sub]  # (B, K)    

    # Get center representations (normalized) 
    centers_norm = torch.gather(
        x_norm,
        1, 
        center_idx[..., None].expand(B, K, x_norm.shape[-1])
    )  # (B, K, D)    

    # Use cosine similarity (consistent with center selection)
    sim_token_center = x_norm @ centers_norm.transpose(1, 2)  # (B, N, K)  

    # Assign to cluster with highest similarity    
    assign_idx = sim_token_center.argmax(dim=-1)  # (B, N)

    # Weighted merging     
    # Merge using original (unprojected) tokens for output quality    
    out = x.new_zeros(B, K, C)    
   
    # One-hot encoding of assignments
    one_hot = F.one_hot(assign_idx, num_classes=K).type_as(x)  # (B, N, K)

    # Count tokens per cluster
    cluster_counts = one_hot.sum(dim=1, keepdim=True).clamp(min=1e-6)  # (B, 1, K) 
  
    # Weighted average: sum tokens per cluster, then normalize  
    out = torch.einsum("bnc,bnk->bkc", x, one_hot) / cluster_counts.transpose(1, 2)
     
    return out 
     
 
class SAA(nn.Module):    
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., c_ratio=0.5, M=0.03):
        super(SAA, self).__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."
        self.dim = dim
        self.num_heads = num_heads   
        self.cr = int(dim * c_ratio)   
        self.scale = qk_scale or (self.cr // num_heads) ** -0.5   
        self.M = M  # Ratio for NF (foreground size)    
 
        # QKV projections   
        self.q = nn.Linear(dim, self.cr, bias=qkv_bias)     
        self.k = nn.Linear(dim, self.cr, bias=qkv_bias)
        self.v = nn.Linear(dim, dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)  
        self.proj = nn.Conv2d(dim, dim, 1)
        self.proj_drop = nn.Dropout(proj_drop)     

    def forward(self, x):
        B, C, H, W = x.shape    
        N = H * W
 
        x = x.flatten(2).transpose(2, 1)
  
        T_unimp = x   
        NF = int(self.M * N)

        # Average and cluster-merge background tokens 
        T_avg = cluster_and_merge(T_unimp, NF)     
        # Norm preservation  
        norms = torch.norm(T_unimp, dim=-1)  # B x num_unimp 
        max_norm = norms.max(dim=-1, keepdim=True)[0].unsqueeze(-1)  # B x 1 x 1
        avg_norm = torch.norm(T_avg, dim=-1, keepdim=True)  # B x 1 x 1    
        epsilon = 1e-6
        mask = avg_norm > epsilon  # B x 1 x 1    
        scaled = (T_avg / (avg_norm + epsilon)) * max_norm   
        T_avg = torch.where(mask, scaled, T_avg)
   
        # Concat for KV_comp    
        KV_comp = T_avg
        K_size = KV_comp.shape[1]
     
        # Cross-Attention
        q = self.q(x).reshape(B, N, self.num_heads, self.cr // self.num_heads).permute(0, 2, 1, 3)
        k = self.k(KV_comp).reshape(B, K_size, self.num_heads, self.cr // self.num_heads).permute(0, 2, 1, 3)    
        v = self.v(KV_comp).reshape(B, K_size, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        attn = (q @ k.transpose(-2, -1)) * self.scale  
        attn = attn.softmax(dim=-1)   
        attn = self.attn_drop(attn)   
        out = (attn @ v).transpose(2, 3).reshape(B, C, H, W)
        out = self.proj(out) 
        out = self.proj_drop(out)
  
        return out

if __name__ == '__main__':
    RED, GREEN, BLUE, YELLOW, ORANGE, RESET = "\033[91m", "\033[92m", "\033[94m", "\033[93m", "\033[38;5;208m", "\033[0m"
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')     
    batch_size, channel, height, width = 1, 16, 20, 20
    inputs = torch.randn((batch_size, channel, height, width)).to(device)
  
    module = SAA(channel, num_heads=8).to(device)
    
    outputs = module(inputs)
    print(GREEN + f'inputs.size:{inputs.size()} outputs.size:{outputs.size()}' + RESET)
     
    print(ORANGE)  
    flops, macs, _ = calculate_flops(model=module,
                                     input_shape=(batch_size, channel, height, width),
                                     output_as_string=True,
                                     output_precision=4,  
                                     print_detailed=True)
    print(RESET)    
