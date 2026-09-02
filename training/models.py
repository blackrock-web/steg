import torch
import torch.nn as nn
import torch.nn.functional as F

class CostMapCNN(nn.Module):
    """
    Standard 5-layer Convolutional Neural Network for pixel embedding cost estimation.
    Matches the parameter structure of cost_map_cnn.pth.
    """
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, kernel_size=3, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)


class CouplingBlock(nn.Module):
    """
    Invertible affine coupling block for Haar wavelet subbands.
    """
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 2, kernel_size=3, padding=1)
        )

    def forward(self, x1, x2):
        h = self.net(x1)
        s = torch.tanh(h[:, 0:1])
        t = h[:, 1:2]
        y2 = x2 * torch.exp(s) + t
        return x1, y2


class LFRINN(nn.Module):
    """
    Low-Frequency Resonant Invertible Neural Network (LF-RINN).
    Splits spatial domain into Haar DWT wavelet subbands (LL, LH, HL, HH),
    processes multi-scale high-frequency edge residuals, and reconstructs via IDWT.
    """
    def __init__(self):
        super().__init__()
        self.blocks = nn.ModuleList([CouplingBlock() for _ in range(4)])

    def dwt(self, x):
        x00 = x[:, :, 0::2, 0::2]
        x01 = x[:, :, 0::2, 1::2]
        x10 = x[:, :, 1::2, 0::2]
        x11 = x[:, :, 1::2, 1::2]
        ll = (x00 + x01 + x10 + x11) * 0.5
        lh = (x00 - x01 + x10 - x11) * 0.5
        hl = (x00 + x01 - x10 - x11) * 0.5
        hh = (x00 - x01 - x10 + x11) * 0.5
        return ll, lh, hl, hh

    def idwt(self, ll, lh, hl, hh):
        x00 = (ll + lh + hl + hh) * 0.5
        x01 = (ll - lh + hl - hh) * 0.5
        x10 = (ll + lh - hl - hh) * 0.5
        x11 = (ll - lh - hl + hh) * 0.5
        B, C, H2, W2 = ll.shape
        row0 = torch.stack([x00, x01], dim=-1).view(B, C, H2, W2 * 2)
        row1 = torch.stack([x10, x11], dim=-1).view(B, C, H2, W2 * 2)
        out = torch.stack([row0, row1], dim=-2).view(B, C, H2 * 2, W2 * 2)
        return out

    def forward(self, x):
        ll, lh, hl, hh = self.dwt(x)
        ll, lh = self.blocks[0](ll, lh)
        ll, hl = self.blocks[1](ll, hl)
        ll, hh = self.blocks[2](ll, hh)
        h_sum = (lh + hl + hh) * (1.0 / 3.0)
        h_sum, ll = self.blocks[3](h_sum, ll)
        rec = self.idwt(ll, lh, hl, hh)
        return rec


class EdgeBranch(nn.Module):
    """
    High-frequency gradient and spatial edge residual feature extraction branch.
    """
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, kernel_size=3, padding=1)
        )

    def forward(self, x):
        return self.net(x)


class FullLFRINNModel(nn.Module):
    """
    Full Proposed LF-RINN Neural Cost Map Model.
    Fuses Invertible Wavelet representation with High-Frequency Edge branch.
    Outputs cost map in [0, 1] normalized space.
    """
    def __init__(self):
        super().__init__()
        self.lf_rinn = LFRINN()
        self.edge_branch = EdgeBranch()
        self.fuse = nn.Conv2d(2, 1, kernel_size=1)

    def forward(self, cover_patch):
        rinn_out = self.lf_rinn(cover_patch)
        edge_out = self.edge_branch(cover_patch)
        fused = self.fuse(torch.cat([rinn_out, edge_out], dim=1))
        cost_map = torch.sigmoid(fused)
        return cost_map


class SteganalyzerCNN(nn.Module):
    """
    Deep Steganalysis Convolutional Neural Network (SRM / Xu-Net inspired).
    Uses high-pass spatial filtering kernels (KV kernel & 30 SRM residuals)
    followed by convolutional feature extraction and binary classification (Cover vs Stego).
    """
    def __init__(self):
        super().__init__()
        # Pre-initialized SRM High-Pass Filter Kernel (KV filter 5x5)
        kv_kernel = torch.tensor([
            [-1,  2,  -2,  2, -1],
            [ 2, -6,   8, -6,  2],
            [-2,  8, -12,  8, -2],
            [ 2, -6,   8, -6,  2],
            [-1,  2,  -2,  2, -1]
        ], dtype=torch.float32) / 12.0

        # 3x3 minmax / Laplacian high-pass filters
        lap_kernel = torch.tensor([
            [ 0, -1,  0],
            [-1,  4, -1],
            [ 0, -1,  0]
        ], dtype=torch.float32)

        edge_h = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32)
        edge_v = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32)

        self.srm_conv = nn.Conv2d(1, 4, kernel_size=5, padding=2, bias=False)
        with torch.no_grad():
            self.srm_conv.weight.zero_()
            self.srm_conv.weight[0, 0] = kv_kernel
            self.srm_conv.weight[1, 0, 1:4, 1:4] = lap_kernel
            self.srm_conv.weight[2, 0, 1:4, 1:4] = edge_h
            self.srm_conv.weight[3, 0, 1:4, 1:4] = edge_v

        self.features = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.Tanh(),
            nn.AvgPool2d(2, 2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(2, 2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1))
        )

        self.classifier = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # x: [B, 1, H, W] in [0, 1]
        residuals = self.srm_conv(x)
        feat = self.features(residuals)
        feat = feat.view(feat.size(0), -1)
        prob_stego = self.classifier(feat)
        return prob_stego
