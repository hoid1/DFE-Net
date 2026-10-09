import os, sys    
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../../../..')
    
import warnings    
warnings.filterwarnings('ignore')
from calflops import calculate_flops

import torch
import torch.nn as nn    
from ultralytics.nn.modules.conv import Conv    

class MyModule(nn.Module): 
    def __init__(self, inc, ouc, kernel_size=3):
        super().__init__()

        self.conv1 = Conv(inc, ouc // 2, k=kernel_size)   
        self.conv2 = Conv(ouc // 2, ouc, k=kernel_size)   
    
    def forward(self, x): 
        x = self.conv1(x) 
        x = self.conv2(x)
        return x

if __name__ == '__main__':
    RED, GREEN, BLUE, YELLOW, ORANGE, RESET = "\033[91m", "\033[92m", "\033[94m", "\033[93m", "\033[38;5;208m", "\033[0m"
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    batch_size, in_channel, out_channel, height, width = 1, 16, 32, 64, 64
    inputs = torch.randn((batch_size, in_channel, height, width)).to(device)

    module = MyModule(in_channel, out_channel, kernel_size=3).to(device)    
    module.eval()
    
    outputs = module(inputs)
    print(GREEN + f'inputs.size:{inputs.size()} outputs.size:{outputs.size()}' + RESET)   

    print(ORANGE)   
    flops, macs, _ = calculate_flops(model=module,
                                     input_shape=(batch_size, in_channel, height, width),
                                     output_as_string=True,
                                     output_precision=4,
                                     print_detailed=True)   
    print(RESET)