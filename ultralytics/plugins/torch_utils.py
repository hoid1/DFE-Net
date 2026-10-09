import torch, torchvision
import torch.nn as nn     
import torch.nn.functional as F 
   
import time 
from contextlib import contextmanager

RED, GREEN, BLUE, YELLOW, ORANGE, CYAN, MAGENTA, BOLD, RESET = "\033[91m", "\033[92m", "\033[94m", "\033[93m", "\033[38;5;208m", "\033[96m", "\033[95m", "\033[1m", "\033[0m"    

def check_cuda():  
    print(GREEN + f"PyTorch 版本: {torch.__version__}")
    print(f"Torchvision 版本: {torchvision.__version__}")
    cuda_available = torch.cuda.is_available()     
    print(f"CUDA 是否可用: {cuda_available}")
  
    if cuda_available:    
        device_count = torch.cuda.device_count()
        print(f"GPU 数量: {device_count}")     
    
        for i in range(device_count):     
            print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
            print(f"  显存: {torch.cuda.get_device_properties(i).total_memory / 1024**3:.2f} GB")
            print(f"  计算能力: {torch.cuda.get_device_capability(i)}")   
     
        print(f"当前设备索引: {torch.cuda.current_device()}")
        print(f"当前设备名称: {torch.cuda.get_device_name(torch.cuda.current_device())}" + RESET)
 
class CUDABenchmark:     
    """可重用的 CUDA 基准测试类 - 支持多次测量"""   
   
    def __init__(self, name: str = "Operation", warmup: int = 0, repeat: int = 1):   
        self.name = name
        self.warmup = warmup 
        self.repeat = repeat   
        self.times = []
        self.result = {}
    
    def run(self, func, *args, **kwargs):
        """运行基准测试"""
        import statistics  
 
        # 预热   
        for _ in range(self.warmup):     
            func(*args, **kwargs)  
            if torch.cuda.is_available():
                torch.cuda.synchronize()
        
        # 测量
        times = []
        for _ in range(self.repeat):    
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                start_event = torch.cuda.Event(enable_timing=True)     
                end_event = torch.cuda.Event(enable_timing=True)     
                start_event.record()   
            else:  
                start_time = time.perf_counter()
     
            result = func(*args, **kwargs)  
    
            if torch.cuda.is_available():     
                end_event.record()
                torch.cuda.synchronize()
                elapsed = start_event.elapsed_time(end_event) / 1000.0
            else:  
                elapsed = time.perf_counter() - start_time     
            
            times.append(elapsed)  
        
        self.times = times
        self.result = { 
            'name': self.name,
            'times': times,
            'mean': statistics.mean(times),
            'std': statistics.stdev(times) if len(times) > 1 else 0.0,
            'min': min(times),
            'max': max(times)
        }
  
        return result
    
    def print_stats(self):     
        """打印统计信息"""
        if self.result:
            r = self.result
            print(f"[{r['name']}] Mean: {r['mean']:.4f}s ± {r['std']:.4f}s "    
                  f"(Min: {r['min']:.4f}s, Max: {r['max']:.4f}s)")

def model_fuse_test(model):   
    model.eval()
    for name, m in model.named_modules():
        if hasattr(m, 'convert_to_deploy'):
            print(BLUE + f"Converting module: {m.__class__}" + RESET)     
            m.convert_to_deploy()
    return model    
