import torch
import torch.nn as nn


class CoralLayer(nn.Module):
    """
    CORAL (Consistent RAnk Logits) 有序回归层
    用于 CVM 骨龄分期等有序分类任务，强制保证类别间的单调性约束。
    """
    def __init__(self, in_features, num_classes):
        super(CoralLayer, self).__init__()
        self.in_features = in_features
        self.weight = nn.Parameter(torch.Tensor(1, in_features))
        self.raw_bias = nn.Parameter(torch.Tensor(num_classes - 1))
        
        nn.init.xavier_uniform_(self.weight)
        nn.init.constant_(self.raw_bias, 0.1)

    def forward(self, x):
        if x.dim() == 4:
            x = x.mean(dim=(1, 2))
        elif x.dim() == 3:
            x = x.mean(dim=1)
            
        # 强制偏置项单调递减，确保 P(y>k) <= P(y>k-1)
        bias_increments = torch.cat([self.raw_bias[:1], -torch.exp(self.raw_bias[1:])])
        sorted_bias = torch.cumsum(bias_increments, dim=0)
        
        logits = torch.matmul(x, self.weight.t()) + sorted_bias
        return logits
