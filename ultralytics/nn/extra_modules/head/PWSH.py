'''
自研检测头：Detect_PWSH —— Pinwheel-shaped Weight-Shared Detection Head
（改进点三，在 PSConv / PinwheelHead 思路上重新设计）

设计动机
--------
1. 原版 PinwheelHead（21modules 包）继承 nn.Module 而非 Detect，在本项目
   tasks.py:399 的 `if isinstance(m, Detect)` 判断中命中 else 分支，导致
   m.stride 停留在 __init__ 的 torch.zeros(nl)、bias_init() 从不执行。
   本头直接继承 Detect（与项目内已正确适配的 Detect_LSPCD 同一套写法），
   stride 与 bias_init 均由 DetectionModel 正确注入。

2. 标准 YOLO11 Detect 的参数瓶颈在回归分支 cv2：
   三个尺度各自 Sequential(Conv(x,c2,3), Conv(c2,c2,3), Conv2d(c2,64,1))。
   实测本项目 (n scale, nc=6, ch=[64,128,256])：
       Detect 总参数 431,842，其中 cv2 占 381,888、cv3 仅 49,938。
   所以"降参降算力"的正确切入点是把 cv2 的双 3x3 换成跨尺度共享的轻量茎。

3. NEU-DET 六类缺陷中 scratches（细长划痕）、rolled-in_scale（带状轧制纹）
   具有强方向性；实测这两类 mAP50 尚可（0.943 / 0.772）但 mAP75 很低
   （0.298 / 0.406），说明"能找到、定不准"。PSConv 的四向非对称 padding
   条形卷积提供偏心方向感受野，正对这个短板。

结构
----
    per-level:  reduce_i = Conv_GN(ch_i, hidc, 1)          # 1x1 对齐通道（各尺度独立，很便宜）
    shared   :  stem     = PSConv_GN(hidc, hidc, 3)        # 风车形四向条形卷积（三尺度共享）
                           + DWConv_GN(hidc,hidc,3) + Conv_GN(hidc,hidc,1)
    per-level:  film_i   = 逐通道 (gamma, beta) 仿射        # 层级条件调制，恢复尺度特异性
    shared   :  reg_dw / cls_dw = 深度可分离 3x3           # 双分支轻量解耦
    shared   :  cv2 = Conv2d(hidc, 4*reg_max, 1)
                cv3 = Conv2d(hidc, nc, 1)
    per-level:  scale_i = Scale(1.0)                        # 回归输出逐层缩放（FCOS 式）

三个关键设计点
--------------
* 共享 stem：把 cv2/cv3 三套独立卷积压成一套，参数量主降来源。
* 层级条件仿射 film_i：共享权重最大的风险是三个尺度统计量差异被抹平。
  每层一组 (gamma, beta)（2*hidc 个参数，三层合计仅数百个）让共享茎在不同
  尺度上仍能有各自的响应偏置，是"共享但不完全相同"的极低成本折中。
* GroupNorm 替代 BatchNorm：共享分支要同时吃 80x80 / 40x40 / 20x20 三种
  分布，BN 的 running stats 会被三者互相拉扯；GN 逐样本分组统计，无此问题。
  代价是 GN 无法像 BN 一样在 fuse 时折进卷积，推理速度略有损失。
'''

import os, sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../../../..')

import warnings
warnings.filterwarnings('ignore')

import math, copy

import torch
import torch.nn as nn

from ultralytics.nn.modules.conv import Conv, autopad
from ultralytics.nn.modules.head import Detect
from ultralytics.nn.extra_modules.head.LSPCD import Conv_GN, Scale


class PSConv_GN(nn.Module):
    """Pinwheel-shaped Convolution（AAAI2025）的 GroupNorm 版本。

    与 ultralytics/nn/extra_modules/conv_module/psconv.py 的 PSConv 计算图完全一致，
    仅把内部 Conv(BN) 换成 Conv_GN(GroupNorm)，以适配跨尺度共享的检测头。

    四个 ZeroPad2d 产生四个方向的偏心 padding，配合 (1,k) / (k,1) 条形卷积，
    等效于四个朝向不同的非对称感受野；最后用 2x2 卷积把四路拼接结果融合。
    """

    def __init__(self, c1, c2, k=3, s=1):
        super().__init__()
        assert c2 % 4 == 0, f'PSConv_GN 输出通道需被 4 整除，当前 c2={c2}'
        p = [(k, 0, 1, 0), (0, k, 0, 1), (0, 1, k, 0), (1, 0, 0, k)]
        self.pad = nn.ModuleList(nn.ZeroPad2d(padding=p[g]) for g in range(4))
        self.cw = Conv_GN(c1, c2 // 4, (1, k), s=s, p=0)
        self.ch = Conv_GN(c1, c2 // 4, (k, 1), s=s, p=0)
        self.cat = Conv_GN(c2, c2, 2, s=1, p=0)

    def forward(self, x):
        yw0 = self.cw(self.pad[0](x))
        yw1 = self.cw(self.pad[1](x))
        yh0 = self.ch(self.pad[2](x))
        yh1 = self.ch(self.pad[3](x))
        return self.cat(torch.cat([yw0, yw1, yh0, yh1], dim=1))


class LevelFiLM(nn.Module):
    """层级条件仿射调制（per-level feature-wise linear modulation）。

    共享 stem 让三个检测尺度用同一组卷积权重，代价是抹平了尺度特异性。
    本模块为每个尺度单独学习一组逐通道 (gamma, beta)，在共享特征上做
    y = gamma * x + beta，用 2*C 个参数换回一部分尺度自适应能力。
    初始化为 gamma=1, beta=0，即恒等映射，不干扰训练早期收敛。
    """

    def __init__(self, c):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(1, c, 1, 1))
        self.beta = nn.Parameter(torch.zeros(1, c, 1, 1))

    def forward(self, x):
        return x * self.gamma + self.beta


class DWConv_GN(nn.Module):
    """深度可分离 3x3（GroupNorm 版），用于回归/分类分支的轻量解耦。"""

    def __init__(self, c, k=3):
        super().__init__()
        self.dw = nn.Conv2d(c, c, k, 1, autopad(k, None, 1), groups=c, bias=False)
        self.gn = nn.GroupNorm(16, c)
        self.act = nn.SiLU()

    def forward(self, x):
        return self.act(self.gn(self.dw(x)))


class Detect_PWSH(Detect):
    """Pinwheel-shaped Weight-Shared Detection Head — v2（只共享"提特征"，不共享"出框/出类"）。

    v1 → v2 的变更记录（重要，写清楚是为了消融时能追溯）
    --------------------------------------------------
    v1 把 reg_dw/cls_dw/cv2/cv3 也做成三尺度共享，只留一个逐通道 FiLM 仿射去补偿
    尺度差异。本机实测（NEU-DET, nc=6）：mAP50 0.859→0.805，且 mAP75/mAP50-95 的
    相对降幅（-8.0%/-8.7%）比 mAP50（-6.3%）更大，说明是回归分支的定位精度本身
    变差，而不是分类置信度校准问题——FiLM 的 gamma*x+beta 只能做缩放平移，改变
    不了卷积核学到的空间响应模式，补偿不了三个尺度回归任务在统计上的本质差异。
    逐类看，pitted_surface(-15.3pp)、scratches(-10.5pp) 掉得最狠，且都是小样本
    类别(7-8个实例)，在共享回归头的统计冲突里受伤最重；而实例数最多的 patches
    (17个)几乎没掉，印证了这是"共享容量不足以覆盖尺度差异"而非算子本身的问题。

    v2 保留 v1 里真正省参数、真正让 PSConv 方向性感受野起作用的部分（共享 stem），
    把出框/出类相关的轻量层（reg_dw/cls_dw/cv2/cv3）改回逐尺度独立——这几层每层
    只有几百到几千参数，拆开成本很低，但正是标准 Detect 坚持三尺度独立 cv2/cv3
    的原因（不同尺度目标的回归行为在统计上本就不同）。

    Args:
        nc (int): 类别数，yaml 中唯一需要写的参数。
        reg_max (int): DFL 通道数，由 parse_model 注入。
        end2end (bool): 是否启用 one2one 分支，由 parse_model 注入。
        ch (tuple): 三个尺度的输入通道，由 parse_model 注入。
        hidc (int | None): 共享 stem 隐藏通道。None 时自动取
            max(16, ch[0]//4, reg_max*4) 并向上对齐到 16 的倍数（GroupNorm 需 16 组，
            PSConv 四等分通道也需要），本项目 n scale 下即 64。
    """

    use_ps = True         # 消融开关：共享 stem 首层用 PSConv_GN(True) 还是普通 Conv_GN 3x3(False)
    use_film = True       # 消融开关：是否启用逐尺度条件仿射（v2 里作用变小，仅补充 stem 输出）
    share_tail = False    # 消融开关：True 时退化回 v1 的"尾部也共享"，用于对比验证掉点原因

    def __init__(self, nc: int = 80, reg_max: int = 16, end2end: bool = False, ch: tuple = (), hidc=None):
        super().__init__(nc=nc, reg_max=reg_max, end2end=end2end, ch=ch)

        if hidc is None:
            hidc = max(16, ch[0] // 4, self.reg_max * 4)
        hidc = int(math.ceil(hidc / 16) * 16)  # GroupNorm(16, ·) 与 PSConv 四等分都要求 16 的倍数
        self.hidc = hidc

        # ① 逐尺度 1x1 通道对齐（不共享，参数极少）
        self.reduce = nn.ModuleList(Conv_GN(x, hidc, 1) for x in ch)

        # ② 跨尺度共享的风车形茎 —— 唯一保留共享的部分，参数节省和 PSConv 方向性
        #    感受野都来自这里，负责"提特征"，不直接决定输出的框/类
        self.stem = nn.Sequential(
            PSConv_GN(hidc, hidc, 3) if self.use_ps else Conv_GN(hidc, hidc, 3),
            DWConv_GN(hidc, 3),
            Conv_GN(hidc, hidc, 1),
        )

        # ③ 逐尺度条件仿射，进一步补充 stem 输出的尺度特异性（v2 里是锦上添花，不是主力）
        self.film = nn.ModuleList(LevelFiLM(hidc) if self.use_film else nn.Identity() for _ in ch)

        # ④ 出框/出类的尾部 —— v2 默认逐尺度独立，只有 share_tail=True（消融/对比 v1）时才共享
        # v3 变更：per-scale 尾部不再是三份独立随机初始化，而是"绑定初始化、训练中自由分化"——
        # 建一份原型模块，deepcopy 给另外两个尺度，让 t=0 时刻三个尺度的尾部权重完全相同
        # （数学上等价于 v1 的共享状态），训练过程中梯度决定要不要分化。
        # 目的：消除"3 份独立随机初始化"本身带来的额外优化噪声，把 v1→v2 的差异严格限定在
        # "是否允许分化"这一个变量上，而不是同时引入"初始化点也不同"这个混杂因素。
        if self.share_tail:
            self.reg_dw = DWConv_GN(hidc, 3)
            self.cls_dw = DWConv_GN(hidc, 3)
            self.cv2 = nn.Conv2d(hidc, 4 * self.reg_max, 1)
            self.cv3 = nn.Conv2d(hidc, self.nc, 1)
        else:
            self.reg_dw = self._tied_list(DWConv_GN(hidc, 3), len(ch))
            self.cls_dw = self._tied_list(DWConv_GN(hidc, 3), len(ch))
            self.cv2 = self._tied_list(nn.Conv2d(hidc, 4 * self.reg_max, 1), len(ch))
            self.cv3 = self._tied_list(nn.Conv2d(hidc, self.nc, 1), len(ch))
        self.scale = nn.ModuleList(Scale(1.0) for _ in ch)

        if end2end:
            self.one2one_reg_dw = copy.deepcopy(self.reg_dw)
            self.one2one_cls_dw = copy.deepcopy(self.cls_dw)
            self.one2one_cv2 = copy.deepcopy(self.cv2)
            self.one2one_cv3 = copy.deepcopy(self.cv3)
            self.one2one_scale = copy.deepcopy(self.scale)

    @staticmethod
    def _tied_list(proto: nn.Module, n: int) -> nn.ModuleList:
        """建 n 份"起点完全相同、结构独立"的模块：第一份用传入的 proto 本身，
        其余份 deepcopy 自 proto。t=0 时刻等价于共享（预测完全一致），
        反向传播开始后每份各自更新，是否分化交给训练数据决定。"""
        return nn.ModuleList([proto] + [copy.deepcopy(proto) for _ in range(n - 1)])

    def load_shared_tail_(self, shared_reg_dw: nn.Module, shared_cls_dw: nn.Module,
                           shared_cv2: nn.Module, shared_cv3: nn.Module) -> None:
        """从一个『尾部共享版』（如已训练好的 Detect_PWSH_v1_shareTail）的四个共享子模块，
        warm start 到本实例（v2，尾部逐尺度独立）：把同一份训练好的权重广播复制到三个尺度的
        每一份里，作为比随机初始化更好的起点，再继续微调训练。
        用法见 warm_start_v2_from_v1.py。"""
        assert not self.share_tail, "load_shared_tail_ 只用于 share_tail=False（v2）的实例"
        for i in range(self.nl):
            self.reg_dw[i].load_state_dict(shared_reg_dw.state_dict())
            self.cls_dw[i].load_state_dict(shared_cls_dw.state_dict())
            self.cv2[i].load_state_dict(shared_cv2.state_dict())
            self.cv3[i].load_state_dict(shared_cv3.state_dict())

    def _reg_at(self, reg_dw, cv2, i, feat):
        d = reg_dw if self.share_tail else reg_dw[i]
        c = cv2 if self.share_tail else cv2[i]
        return c(d(feat))

    def _cls_at(self, cls_dw, cv3, i, feat):
        d = cls_dw if self.share_tail else cls_dw[i]
        c = cv3 if self.share_tail else cv3[i]
        return c(d(feat))

    def forward_share_head(self, x, reg_dw=None, cls_dw=None, box_head=None, cls_head=None, scale_head=None):
        if box_head is None or cls_head is None or scale_head is None:  # fused inference
            return dict()
        bs = x[0].shape[0]
        boxes = torch.cat(
            [scale_head[i](self._reg_at(reg_dw, box_head, i, x[i])).view(bs, 4 * self.reg_max, -1)
             for i in range(self.nl)],
            dim=-1,
        )
        scores = torch.cat(
            [self._cls_at(cls_dw, cls_head, i, x[i]).view(bs, self.nc, -1) for i in range(self.nl)],
            dim=-1,
        )
        return dict(boxes=boxes, scores=scores, feats=x)

    def forward(self, x):
        # 共享 stem 只负责提特征；出框/出类在 forward_share_head 内部按 i 走各自的层
        x = [self.film[i](self.stem(self.reduce[i](x[i]))) for i in range(self.nl)]
        preds = self.forward_share_head(
            x, reg_dw=self.reg_dw, cls_dw=self.cls_dw,
            box_head=self.cv2, cls_head=self.cv3, scale_head=self.scale,
        )
        if self.end2end:
            x_detach = [xi.detach() for xi in x]
            one2one = self.forward_share_head(
                x_detach, reg_dw=self.one2one_reg_dw, cls_dw=self.one2one_cls_dw,
                box_head=self.one2one_cv2, cls_head=self.one2one_cv3, scale_head=self.one2one_scale,
            )
            preds = {"one2many": preds, "one2one": one2one}
        if self.training:
            return preds
        y = self._inference(preds["one2one"] if self.end2end else preds)
        if self.end2end:
            y = self.postprocess(y.permute(0, 2, 1))
        return y if self.export else (y, preds)

    def bias_init(self):
        """按每个尺度自己的 stride 初始化分类偏置——v2 里 cv2/cv3 逐尺度独立，
        不再需要像 v1（尾部共享）那样退化成用 stride 均值。"""
        if self.share_tail:
            self.cv2.bias.data[:] = 2.0
            self.cv3.bias.data[: self.nc] = math.log(5 / self.nc / (640 / torch.mean(self.stride)) ** 2)
            if self.end2end:
                self.one2one_cv2.bias.data[:] = 2.0
                self.one2one_cv3.bias.data[: self.nc] = math.log(
                    5 / self.nc / (640 / torch.mean(self.stride)) ** 2
                )
            return
        for i, s in enumerate(self.stride):
            self.cv2[i].bias.data[:] = 2.0
            self.cv3[i].bias.data[: self.nc] = math.log(5 / self.nc / (640 / s) ** 2)
        if self.end2end:
            for i, s in enumerate(self.stride):
                self.one2one_cv2[i].bias.data[:] = 2.0
                self.one2one_cv3[i].bias.data[: self.nc] = math.log(5 / self.nc / (640 / s) ** 2)


class Detect_PWSH_noPS(Detect_PWSH):
    """消融 A：共享茎首层退化为普通 3x3 Conv_GN，用于隔离 PSConv 四向条形卷积的贡献。"""
    use_ps = False


class Detect_PWSH_noFiLM(Detect_PWSH):
    """消融 B：去掉逐尺度条件仿射，用于隔离层级调制的贡献。"""
    use_film = False


class Detect_PWSH_v1_shareTail(Detect_PWSH):
    """对比 v1：尾部（reg_dw/cls_dw/cv2/cv3）也共享，等价于第一版实测掉点(0.805)的结构。
    保留此类仅用于复现/对比，不建议作为正式训练配置使用。"""
    share_tail = True


if __name__ == '__main__':
    GREEN, ORANGE, RESET = "\033[92m", "\033[38;5;208m", "\033[0m"
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    chs = (64, 128, 256)
    feats = [torch.randn(2, c, 80 // (2 ** i), 80 // (2 ** i)).to(device) for i, c in enumerate(chs)]

    head = Detect_PWSH(nc=6, ch=chs).to(device)
    head.stride = torch.tensor([8., 16., 32.])
    head.bias_init()
    print(GREEN + f'isinstance(head, Detect) = {isinstance(head, Detect)}' + RESET)

    head.train()
    out = head([f.clone() for f in feats])
    print(GREEN + f'train forward ok: boxes={out["boxes"].shape} scores={out["scores"].shape}' + RESET)

    head.eval()
    with torch.no_grad():
        y, _ = head([f.clone() for f in feats])
    print(GREEN + f'eval  forward ok: {y.shape}' + RESET)

    for name, cls in [('Detect (官方)', Detect), ('Detect_PWSH (v2)', Detect_PWSH),
                      ('  └ noPS  (消融A)', Detect_PWSH_noPS), ('  └ noFiLM(消融B)', Detect_PWSH_noFiLM),
                      ('  └ v1 shareTail(对比)', Detect_PWSH_v1_shareTail)]:
        h = cls(nc=6, ch=chs)
        h.stride = torch.tensor([8., 16., 32.])
        h.bias_init()
        h.train()
        out = h([f.clone() for f in feats])
        h.eval()
        with torch.no_grad():
            y, _ = h([f.clone() for f in feats])
        print(ORANGE + f'{name:22s} 参数量: {sum(p.numel() for p in h.parameters()):>9,}  eval_out={tuple(y.shape)}' + RESET)
