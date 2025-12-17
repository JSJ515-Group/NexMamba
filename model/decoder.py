from __future__ import annotations
import torch.nn as nn
from einops import rearrange
from model.vmamba.vmamba import  LayerNorm2d, Linear2d,VSSBlock
from typing import Sequence, Type, Optional
from .head import SFMGHead


class LKPE(nn.Module):
    def __init__(self, dim, dim_scale=2, groups=32):
        super().__init__()
        self.dim_scale = dim_scale

        # 通道压缩 (C → C/2)
        self.reduce = nn.Conv2d(dim, dim // 2, kernel_size=1, bias=False)

        # GroupNorm 在压缩后的通道数上做 (C/2)
        self.norm = nn.GroupNorm(groups, dim // 2)

        # 空间平滑卷积 (保持 C/2)
        self.smooth = nn.Conv2d(dim // 2, dim // 2, kernel_size=3, stride=1, padding=1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 先通道压缩
        # print("reduce input:", type(x), x.shape if x is not None else None)
        # 如果输入是 [B, L, C]，先转回 [B, C, H, W]
        if x.dim() == 3:
            B, L, C = x.shape
            H = W = int(L ** 0.5)  # 假设特征图是方形
            x = x.view(B, H, W, C).permute(0, 3, 1, 2).contiguous()  # [B, C, H, W]

        x = self.reduce(x)   # [B, C/2, H, W]

        # 上采样 (空间扩大 2×)
        B, C, H, W = x.shape
        x = F.interpolate(
            x,
            size=(H * self.dim_scale, W * self.dim_scale),
            mode="bilinear",#mode="bilinear",
            align_corners=False,
        )  # [B, C/2, H*2, W*2]

        # 归一化 + 平滑
        x = self.norm(x)
        x = self.smooth(x)

        return x

# =======================4个尺度的特征图======GSI===============================
class GSIBlock(nn.Module):
    """
    Graph-based Skip Interaction Block (Cyclic-Interleaved Fusion)
    改进点：
    1️ 通道统一到 192，并划分为 4 个 48 通道子块
    2️ 跨尺度循环堆叠交互：1234、2341、3412、4123
    3️ 维持原始输入输出结构（多尺度输入输出）
    4️保留原有稀疏注意力与回写增强机制
    """
    def __init__(self, in_channels_list, embed_dim=192, num_heads=4, attn_drop=0.1):
        super().__init__()
        self.num_scales = len(in_channels_list)
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.split_dim = embed_dim // self.num_scales  # 48 通道一份

        # 通道统一
        self.proj_layers = nn.ModuleList([
            nn.Conv2d(c, self.embed_dim, kernel_size=1, bias=False)
            for c in in_channels_list
        ])

        # 节点压缩增强
        self.node_conv = nn.Conv1d(self.embed_dim, self.embed_dim, kernel_size=3, padding=1, groups=self.embed_dim)

        # 注意力模块
        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.k_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.v_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim)

        self.norm1 = nn.LayerNorm(self.embed_dim)
        self.norm2 = nn.LayerNorm(self.embed_dim)
        self.attn_drop = nn.Dropout(attn_drop)

        # 通道恢复
        self.restore_layers = nn.ModuleList([
            nn.Conv2d(self.embed_dim, c, kernel_size=1, bias=False)
            for c in in_channels_list
        ])

        # 稀疏邻接矩阵（相邻尺度）
        adj = torch.eye(self.num_scales)
        for i in range(self.num_scales - 1):
            adj[i, i + 1] = adj[i + 1, i] = 1
        self.register_buffer("adjacency", adj)

    def forward(self, feats):
        B = feats[0].shape[0]

        # Step 1: 投影 + 全局池化为节点特征
        nodes = []
        proj_feats = []
        for i, f in enumerate(feats):
            f_proj = self.proj_layers[i](f)  # [B, 192, H, W]
            proj_feats.append(f_proj)
            f_pool = F.adaptive_avg_pool2d(f_proj, 1).view(B, self.embed_dim)
            nodes.append(f_pool)

        nodes = torch.stack(nodes, dim=1)  # [B, N=4, C=192]
        nodes = nodes + self.node_conv(nodes.transpose(1, 2)).transpose(1, 2)
        nodes = self.norm1(nodes)

        # Step 2: 稀疏注意力
        Q = self.q_proj(nodes)
        K = self.k_proj(nodes)
        V = self.v_proj(nodes)
        B, N, C = Q.shape
        H = self.num_heads
        head_dim = C // H

        Q = Q.view(B, N, H, head_dim).transpose(1, 2)
        K = K.view(B, N, H, head_dim).transpose(1, 2)
        V = V.view(B, N, H, head_dim).transpose(1, 2)

        attn = (Q @ K.transpose(-2, -1)) / (head_dim ** 0.5)
        mask = (self.adjacency == 0).unsqueeze(0).unsqueeze(0)
        attn = attn.masked_fill(mask, float('-inf'))
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        out = (attn @ V).transpose(1, 2).reshape(B, N, C)
        out = self.out_proj(out)
        out = self.norm2(out + nodes)

        # Step 3: 循环混合堆叠 (Cyclic Interleaved)
        fused_feats = []
        for i in range(self.num_scales):
            # cyclic shift pattern
            pattern = [(i + j) % self.num_scales for j in range(self.num_scales)]
            # 从4个节点中取对应的48通道块
            mixed = torch.cat([
                out[:, p, j*self.split_dim:(j+1)*self.split_dim]
                for j, p in enumerate(pattern)
            ], dim=-1)  # [B, 192]
            fused_feats.append(mixed)

        # Step 4: 回写增强并恢复通道
        updated_feats = []
        for i, f in enumerate(feats):
            u = fused_feats[i].view(B, self.embed_dim, 1, 1)
            f_proj = proj_feats[i]
            f_new = f_proj * (1 + torch.sigmoid(u))
            f_restored = self.restore_layers[i](f_new)
            updated_feats.append(f_restored)

        return updated_feats

class FourierEnhance(nn.Module):
    """Frequency-domain enhancement using learnable spectral gating."""
    def __init__(self, channels, ratio=0.5):
        super().__init__()
        hidden = max(1, int(channels * ratio))
        self.gate = nn.Sequential(
            nn.Linear(channels * 2, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, channels * 2),
            nn.Sigmoid()
        )

    def forward(self, x):
        B, C, H, W = x.shape
        freq = torch.fft.rfft2(x, norm='ortho')  # complex tensor
        freq_real = torch.view_as_real(freq)  # [B, C, H, W/2+1, 2]
        freq_flat = freq_real.view(B, C, -1, 2).mean(dim=2)
        gate = self.gate(freq_flat.view(B, -1)).view(B, C, 1, 1, 2)
        freq_real = freq_real * gate
        freq = torch.view_as_complex(freq_real)
        x_freq = torch.fft.irfft2(freq, s=(H, W), norm='ortho').real
        return x_freq

class MSFA_Block(nn.Module):
    """Multi-Scale Fourier Augmentation Block.
    Input / Output shape: [B, C, H, W].
    Combines pooling-based multi-scale spatial context with frequency enhancement.
    """
    def __init__(self, dim, freq_ratio=0.5, use_residual=True, norm_layer=nn.GroupNorm, act_layer=nn.SiLU):
        super().__init__()
        self.use_residual = use_residual

        # Spatial multi-scale branches
        self.avg_branch = nn.Sequential(
            nn.AvgPool2d(3, stride=1, padding=1),
            nn.Conv2d(dim, dim, 3, padding=1, bias=False)
        )
        self.max_branch = nn.Sequential(
            nn.MaxPool2d(3, stride=1, padding=1),
            nn.Conv2d(dim, dim, 3, padding=1, bias=False)
        )
        self.conv_branch = nn.Conv2d(dim, dim, 3, padding=1, bias=False)

        # Fourier enhancement branch
        self.freq_branch = FourierEnhance(dim, ratio=freq_ratio)
        # self.fct = FCTBlock2D(dim)
        # self.fa = FrequencyAttention(dim)


        # self.ffc = FFC(dim,dim,kernel_size=3, padding=1)

        # Fusion conv
        self.fusion = nn.Conv2d(dim * 4, dim, 1, bias=False)
        # self.fusion = nn.Conv2d(dim * 3, dim, 1, bias=False)
        self.norm = norm_layer(1, dim)
        self.act = act_layer(inplace=True)

    def forward(self, x):
        avg = self.avg_branch(x)
        maxp = self.max_branch(x)
        conv = self.conv_branch(x)
        freq = self.freq_branch(x)
        # ffc = self.ffc(x)
        # fct = self.fct(x)
        # fa = self.fa(x)

        fused = torch.cat([avg, maxp, conv, freq], dim=1)
        # fused = torch.cat([avg, maxp, conv,fa], dim=1)
        # fused = torch.cat([avg, maxp, conv], dim=1)
        fused = self.fusion(fused)
        fused = self.norm(fused)
        fused = self.act(fused)

        if self.use_residual:
            fused = fused + x
        return fused



class Decoder(nn.Module):
    def __init__(
        self,
        dims: Sequence[int],
        num_classes: int,
        depths: Sequence[int] = (2,2,2,2),#原本是2，2，2，qed w2(1, 1, 1, 1)
        drop_path_rate: float = 0.2,
    ) -> None:
        super(Decoder, self).__init__()

        dpr = [x.item() for x in torch.linspace(drop_path_rate, 0, (len(dims) - 1) * 2)]

        # Up1


        self.up1_lkpe = LKPE(dims[0])
        self.concat1 = Linear2d(2 * dims[1], dims[1])

        # Up2
        self.up2_lkpe = LKPE(dims[1])
        self.concat2 = Linear2d(2 * dims[2], dims[2])

        # Up3
        self.up3_lkpe = LKPE(dims[2])
        self.concat3 = Linear2d(2 * dims[3], dims[3])

        #============================GSI==================================
        self.gsi = GSIBlock(in_channels_list=[96,192,384,768], embed_dim=192, num_heads=4, attn_drop=0.1)

        self.msfa1 = MSFA_Block(dim=384, freq_ratio=0.5, use_residual=True)
        self.msfa2 = MSFA_Block(dim=192, freq_ratio=0.5, use_residual=True)
        self.msfa3 = MSFA_Block(dim=96, freq_ratio=0.5, use_residual=True)


        self.seghead = SFMGHead(in_channels=dims[3], num_classes=num_classes)


    # def forward(self, features: Sequence[torch.Tensor]) -> torch.Tensor:
    def forward(self, features: Sequence[torch.Tensor]) -> torch.Tensor:

        x0, x1, x2, x3 = features  # 7x7, 14x14, 28x28, 56x56,通道数，768，384，192，96，
        # print(x0.shape, x1.shape, x2.shape, x3.shape)
        """
                features: [x0, x1, x2, x3]
                mask: [B, 1, 224, 224] (原始掩码)
        ""


        skip_feats = [x3, x2, x1, x0]
        x3_upd, x2_upd, x1_upd,x0_upd = self.gsi(skip_feats)
 


        out = self.up1_lkpe(x0_upd)
        out = torch.cat((out, x1_upd), dim=1)
        out = self.concat1(out)
        out =self.msfa1(out)

        out = self.up2_lkpe(out)
        out = torch.cat((out, x2_upd), dim=1)
        out = self.concat2(out)
        out = self.msfa2(out)

        out = self.up3_lkpe(out)
        out = torch.cat((out, x3_upd), dim=1)
        out = self.concat3(out)
        out = self.msfa3(out)

        return self.seghead(out)
  