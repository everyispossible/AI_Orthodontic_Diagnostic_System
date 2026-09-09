import torch
import torch.nn.functional as F


def get_windowed_soft_argmax(heatmaps, window_size=7):
    """
    Windowed Soft-Argmax: 先在热力图上用 hard-argmax 定位粗略整数坐标，
    再在局部窗口内做 softmax 加权平均，得到亚像素精度坐标。

    Parameters
    ----------
    heatmaps : torch.Tensor, shape [C, H, W]
        模型输出的 C 通道热力图
    window_size : int
        局部窗口边长（奇数），默认 7

    Returns
    -------
    torch.Tensor, shape [C, 2]
        每个通道的 (x, y) 亚像素坐标
    """
    C, H, W = heatmaps.shape
    device = heatmaps.device
    pad = window_size // 2

    # 用 hard-argmax 找到粗略的整数坐标
    heatmaps_flat = heatmaps.view(C, -1)
    _, max_idx = torch.max(heatmaps_flat, dim=-1)
    hard_x = (max_idx % W)
    hard_y = (max_idx // W)

    # 对热力图进行 padding，防止窗口越界
    heatmaps_padded = F.pad(heatmaps, (pad, pad, pad, pad), mode='constant', value=0)
    subpixel_coords = torch.zeros((C, 2), device=device)

    for c in range(C):
        hx, hy = hard_x[c].item(), hard_y[c].item()

        # 截取局部窗口
        window = heatmaps_padded[c, hy:hy + window_size, hx:hx + window_size]
        probs = F.softmax(window.flatten(), dim=-1).view(window_size, window_size)

        # 构建窗口内局部网格坐标
        grid_y, grid_x = torch.meshgrid(
            torch.arange(window_size, device=device),
            torch.arange(window_size, device=device),
            indexing='ij'
        )

        # 计算窗口内的概率加权重心
        local_cx = torch.sum(probs * grid_x)
        local_cy = torch.sum(probs * grid_y)

        # 还原到全图坐标系：粗坐标 + 局部偏移 - pad 偏移
        subpixel_coords[c, 0] = hx + local_cx - pad
        subpixel_coords[c, 1] = hy + local_cy - pad

    return subpixel_coords
