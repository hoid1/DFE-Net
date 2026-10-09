'''
本文件由 0526 版迁移而来（模块代码逐字保留，未作任何修改）
来源：nn/extra_modules/hcfnet.py:116-121 , nn/extra_modules/hcfnet.py:123-171
二次创新(1)：HCFNet 的 DASI 融入自研 FDPN（旧库 md #138）
'''

import os, sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../../../..')

import warnings
warnings.filterwarnings('ignore')

import torch.nn.functional as F
import torch.nn as nn
import torch



# ---- 原样迁移自 nn/extra_modules/hcfnet.py:116-121 ----
class Bag(nn.Module):
    def __init__(self):
        super(Bag, self).__init__()
    def forward(self, p, i, d):
        edge_att = torch.sigmoid(d)
        return edge_att * p + (1 - edge_att) * i

# ---- 原样迁移自 nn/extra_modules/hcfnet.py:123-171 ----
class DASI(nn.Module):
    def __init__(self, in_features, out_features) -> None:
         super().__init__()
         self.bag = Bag()
         self.tail_conv = nn.Conv2d(out_features, out_features, 1)
         self.conv = nn.Conv2d(out_features // 2, out_features // 4, 1)
         self.bns = nn.BatchNorm2d(out_features)

         self.skips = nn.Conv2d(in_features[1], out_features, 1)
         self.skips_2 = nn.Conv2d(in_features[0], out_features, 1)
         self.skips_3 = nn.Conv2d(in_features[2], out_features, kernel_size=3, stride=2, dilation=2, padding=2)
         self.silu = nn.SiLU()

    def forward(self, x_list):
        # x_high, x, x_low = x_list
        x_low, x, x_high = x_list
        if x_high != None:
            x_high = self.skips_3(x_high)
            x_high = torch.chunk(x_high, 4, dim=1)
        if x_low != None:
            x_low = self.skips_2(x_low)
            x_low = F.interpolate(x_low, size=[x.size(2), x.size(3)], mode='bilinear', align_corners=True)
            x_low = torch.chunk(x_low, 4, dim=1)
        x = self.skips(x)
        x_skip = x
        x = torch.chunk(x, 4, dim=1)
        if x_high == None:
            x0 = self.conv(torch.cat((x[0], x_low[0]), dim=1))
            x1 = self.conv(torch.cat((x[1], x_low[1]), dim=1))
            x2 = self.conv(torch.cat((x[2], x_low[2]), dim=1))
            x3 = self.conv(torch.cat((x[3], x_low[3]), dim=1))
        elif x_low == None:
            x0 = self.conv(torch.cat((x[0], x_high[0]), dim=1))
            x1 = self.conv(torch.cat((x[0], x_high[1]), dim=1))
            x2 = self.conv(torch.cat((x[0], x_high[2]), dim=1))
            x3 = self.conv(torch.cat((x[0], x_high[3]), dim=1))
        else:
            x0 = self.bag(x_low[0], x_high[0], x[0])
            x1 = self.bag(x_low[1], x_high[1], x[1])
            x2 = self.bag(x_low[2], x_high[2], x[2])
            x3 = self.bag(x_low[3], x_high[3], x[3])

        x = torch.cat((x0, x1, x2, x3), dim=1)
        x = self.tail_conv(x)
        x += x_skip
        x = self.bns(x)
        x = self.silu(x)

        return x

if __name__ == '__main__':
    RED, GREEN, BLUE, YELLOW, ORANGE, RESET = "\033[91m", "\033[92m", "\033[94m", "\033[93m", "\033[38;5;208m", "\033[0m"
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    batch_size, in_channel, out_channel, height, width = 1, 64, 128, 32, 32
    inputs = torch.randn((batch_size, in_channel, height, width)).to(device)

    # ---- Bag ----
    try:
        module = Bag().to(device)
        outputs = module(inputs)
        print(GREEN + f'Bag  inputs:{tuple(inputs.shape)} -> outputs:{tuple(outputs.shape) if hasattr(outputs, "shape") else type(outputs)}' + RESET)
        try:
            from calflops import calculate_flops
            print(ORANGE, end='')
            calculate_flops(model=module, input_shape=(batch_size, in_channel, height, width),
                            output_as_string=True, output_precision=4, print_detailed=False)
            print(RESET, end='')
        except Exception:
            pass
    except Exception as e:
        print(RED + f'Bag  自测跳过: {e}' + RESET)

