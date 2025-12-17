import torch
import torch.nn as nn
import torch.nn.functional as F


# -------------------------
# Helpers
# -------------------------
def collapse_multi_inputs(*inputs):
    """Normalize inputs:
       - single Tensor [B,C,H,W] -> return it
       - stacked Tensor [B,M,C,H,W] -> collapse to [B, M*C, H, W]
       - multiple tensors -> element-wise mean
       - list/tuple input -> take last entry
    """
    if len(inputs) == 1:
        x = inputs[0]
        if isinstance(x, (list, tuple)):
            x = x[-1]
        if x.dim() == 5:
            B, M, C, H, W = x.shape
            x = x.view(B, M * C, H, W)
        return x
    else:
        xs = []
        for t in inputs:
            if isinstance(t, (list, tuple)):
                t = t[-1]
            if t.dim() == 5:
                B, M, C, H, W = t.shape
                t = t.view(B, M * C, H, W)
            xs.append(t)
        return torch.stack(xs, dim=0).mean(dim=0)


# -------------------------
# Spatial Branch
# -------------------------
class SpatialBranch(nn.Module):
    def __init__(self, in_ch, mid_ch):
        super().__init__()
        self.align = nn.Conv2d(in_ch, mid_ch, 1, bias=False)
        self.conv1 = nn.Conv2d(mid_ch, mid_ch, 1, bias=False)
        self.dw3 = nn.Conv2d(mid_ch, mid_ch, 3, padding=1, groups=mid_ch, bias=False)
        self.dw5 = nn.Conv2d(mid_ch, mid_ch, 5, padding=2, groups=mid_ch, bias=False)
        self.merge = nn.Conv2d(mid_ch * 3, mid_ch, 1, bias=False)
        self.bn = nn.BatchNorm2d(mid_ch)
        self.act = nn.ReLU(inplace=True)
        # spatial + channel attention
        self.spatial_att = nn.Sequential(
            nn.Conv2d(mid_ch, mid_ch // 4, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_ch // 4, 1, 1),
            nn.Sigmoid()
        )
        self.channel_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(mid_ch, mid_ch // 4, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_ch // 4, mid_ch, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.align(x)
        x1 = self.conv1(x)
        x3 = self.dw3(x)
        x5 = self.dw5(x)
        out = torch.cat([x1, x3, x5], dim=1)
        out = self.merge(out)
        out = self.bn(out)
        out = self.act(out)
        spat_map = self.spatial_att(out)
        # ch_map = self.channel_att(out)
        return out, spat_map


# -------------------------
# Frequency Branch
# -------------------------
import torch
import torch.nn as nn
import torch.nn.functional as F

class FrequencyBranch(nn.Module):
    """
    Frequency branch with spatial guidance.
    - Input: x [B, C, H, W], spat_map [B,1,H,W]
    - Output: x_freq [B, C, H, W], ch_feedback [B,C,1,1]
    """
    def __init__(self, mid_ch, gate_hidden_div=8):
        super().__init__()
        hidden = max(1, mid_ch // gate_hidden_div)
        self.mid = mid_ch

        # Gate conv: input 2 channels (avg magnitude + spatial map) -> mid_ch channels
        self.gate_conv = nn.Sequential(
            nn.Conv2d(2, hidden, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, mid_ch, 1),
            nn.Sigmoid()
        )

    def forward(self, x, spat_map):
        B, C, H, W = x.shape

        # FFT
        freq = torch.fft.rfft2(x, norm='ortho')           # [B,C,H,W2], complex
        freq_r = torch.view_as_real(freq)                 # [B,C,H,W2,2]
        a = freq_r[..., 0]
        b = freq_r[..., 1]
        mag = torch.sqrt(a**2 + b**2 + 1e-12)            # [B,C,H,W2]
        phase_cos = a / (mag + 1e-12)
        phase_sin = b / (mag + 1e-12)
        W2 = mag.shape[3]

        # Gate input: average mag across channels + spat_map
        mag_avg = mag.mean(dim=1, keepdim=True)          # [B,1,H,W2]
        # 上采样 spat_map 到 W2
        spat_up = F.interpolate(spat_map, size=(H, W2), mode='bilinear', align_corners=False)
        gate_in = torch.cat([mag_avg, spat_up], dim=1)  # [B,2,H,W2]
        gate = self.gate_conv(gate_in)                  # [B,C,H,W2]

        # Apply gate
        mag_enh = mag * (1.0 + gate)                    # [B,C,H,W2]

        # Reconstruct complex tensor
        a_enh = mag_enh * phase_cos
        b_enh = mag_enh * phase_sin
        freq_r_enh = torch.stack([a_enh, b_enh], dim=-1)
        freq_enh = torch.view_as_complex(freq_r_enh)

        # Inverse FFT
        x_freq = torch.fft.irfft2(freq_enh, s=(H, W), norm='ortho').real  # [B,C,H,W]

        # Channel-wise feedback: global pooling of mag_enh
        ch_feedback = mag_enh.mean(dim=[2,3], keepdim=True)   #平均池化            # [B,C,1,1]
        ch_feedback = torch.sigmoid(ch_feedback)

        return x_freq, ch_feedback


# -------------------------
# SFMG Head
# -------------------------
class SFMGHead(nn.Module):
    def __init__(self, in_channels=96, mid_channels=96, num_classes=4, out_size=(224, 224)):
    # def __init__(self, in_channels=96, mid_channels=96, num_classes=4, out_size=(256, 256)):
        super().__init__()
        self.in_ch = in_channels
        self.mid = mid_channels
        self.out_size = out_size

        self.spatial_branch = SpatialBranch(in_channels, mid_channels)
        self.freq_branch = FrequencyBranch(mid_channels)

        self.fuse = nn.Sequential(
            nn.Conv2d(mid_channels * 2, mid_channels, 1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True)
        )

        self.refine = nn.Sequential(
            nn.Conv2d(mid_channels, mid_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, mid_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True)
        )

        self.up_transpose = nn.ConvTranspose2d(mid_channels, mid_channels, kernel_size=4, stride=2, padding=1, bias=False)
        self.final_head = nn.Sequential(
            nn.Conv2d(mid_channels, mid_channels // 2, 3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels // 2, num_classes, 1)
        )

    def forward(self, *inputs):
        x_in = collapse_multi_inputs(*inputs)
        s_feat, spat_map = self.spatial_branch(x_in)
        x_freq, ch_feedback = self.freq_branch(s_feat, spat_map)

        s_mod = s_feat * ch_feedback * (1.0 + spat_map)
        fused = torch.cat([s_mod, x_freq], dim=1)
        fused = self.fuse(fused) + s_feat

        out_feat = fused + self.refine(fused)
        up = self.up_transpose(out_feat)
        up = F.interpolate(up, scale_factor=2, mode='bilinear', align_corners=False)
        pred = self.final_head(up)
        pred = F.interpolate(pred, size=self.out_size, mode='bilinear', align_corners=False)
        return pred#, spat_map, ch_feedback


# -------------------------
# Test
# -------------------------
if __name__ == "__main__":
    B, C, H, W = 2, 96, 56, 56
    x = torch.randn(B, C, H, W).cuda()
    head = SFMGHead(in_channels=96, mid_channels=192, num_classes=9, out_size=(224, 224)).cuda()
    # pred, spat_map, ch_feedback = head(x)
    pred = head(x)

    print(pred.shape)        # [2, 4, 224, 224]
    # print(spat_map.shape)    # [2, 1, 56, 56]
    # print(ch_feedback.shape) # [2, 96, 1, 1]
