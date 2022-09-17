import torch.nn as nn
import torch


def IN(A, B, mode='exchange'):
    std_A, mean_A = torch.std_mean(A, [2, 3])
    norm_A = A - mean_A.unsqueeze(-1).unsqueeze(-1)
    norm_A = norm_A/(std_A.unsqueeze(-1).unsqueeze(-1)+1e-5)

    std_B, mean_B = torch.std_mean(B, [2, 3])
    norm_B = B - mean_B.unsqueeze(-1).unsqueeze(-1)
    norm_B = norm_B/(std_B.unsqueeze(-1).unsqueeze(-1)+1e-5)

    out_A = norm_A * (std_B.unsqueeze(-1).unsqueeze(-1)) + (mean_B.unsqueeze(-1).unsqueeze(-1))
    out_B = norm_B * (std_A.unsqueeze(-1).unsqueeze(-1)) + (mean_A.unsqueeze(-1).unsqueeze(-1))
    return out_A, out_B


class ColorAug(nn.Module):
    """对A，B做颜色变换，交换A与B的颜色风格"""
    def __init__(self, norm_type='in', p=1.0):
        super(ColorAug, self).__init__()
        self.norm_type = norm_type  # wct | IN
        self.p = p   # 应用fun的概率

    def forward(self, A, B):
        # torch.rand返回一个张量，包含了从区间[0, 1)的均匀分布中抽取的一组随机数。张量的形状由参数sizes定义。
        if torch.rand(1).item() > self.p:
            return A, B
        if self.norm_type == 'in':
            A, B = IN(A, B)
        else:
            raise NotImplementedError('no such norm type')
        return A, B

