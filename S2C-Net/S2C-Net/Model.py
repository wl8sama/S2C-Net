from sam2.build_sam import build_sam2
import torch.fft
from einops import rearrange
import math
import torch
import torch.nn as nn

class LearnableHighPass(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.alpha = nn.Parameter(torch.tensor(0.25))
        self.compensate = nn.Conv2d(dim, dim, 1)

    def forward(self, x):
        B, C, H, W = x.shape
        fft = torch.fft.fft2(x, norm='ortho')
        fft_shift = torch.fft.fftshift(fft)
        u = torch.linspace(-1, 1, H, device=x.device)
        v = torch.linspace(-1, 1, W, device=x.device)
        v_grid, u_grid = torch.meshgrid(v, u, indexing='xy')
        radius = torch.sqrt(u_grid ** 2 + v_grid ** 2)
        mask = (radius > self.alpha.clamp(0.1, 0.99)).float()
        mask = mask[None, None, :, :]
        fft_hp = fft_shift * mask
        fft_hp = torch.fft.ifftshift(fft_hp)
        img_hp = torch.fft.ifft2(fft_hp, norm='ortho').real
        return self.compensate(img_hp)

class DSFFR(nn.Module):
    def __init__(self, blk, reduction=4):
        super().__init__()
        dim = blk.attn.qkv.in_features
        self.block = blk

        self.micro_path = nn.Sequential(
            ChannelSpatialAttention(dim)
        )

        self.freq_path = nn.Sequential(
            LearnableHighPass(dim),
        )

        self.res_weight = nn.Sequential(
            nn.Conv2d(dim * 3, dim, 1),
            nn.ReLU(),
            nn.Sigmoid()
        )

        self.fuse_gate = nn.Sequential(
            nn.Conv2d(dim * 2, dim, 1),
            nn.Sigmoid()
        )

        self.res_scale = 0.1

    def forward(self, x):
        x_c = x.permute(0, 3, 1, 2)  # [B,C,H,W]
        f_f = self.micro_path(x_c)
        f_s = self.freq_path(x_c)

        res = f_f - f_s
        g = self.res_weight(torch.cat([res, f_f, f_s], dim=1))
        f_f_new = f_f + g * res
        f_s_new = f_s - g * res

        alpha = self.fuse_gate(torch.cat([f_f_new, f_s_new], dim=1))
        fused = alpha * f_f_new + (1 - alpha) * f_s_new

        out = fused.permute(0, 2, 3, 1)
        out = x + out
        return self.block(out)

class ChannelSpatialAttention(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.channel_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // 8, 1),
            nn.ReLU(),
            nn.Conv2d(channels // 8, channels, 1),
            nn.Sigmoid()
        )
        self.spatial_att = nn.Sequential(
            nn.Conv2d(channels, 1, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        ca = self.channel_att(x)
        sa = self.spatial_att(x)
        return x * ca * sa

class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x

class GatedFusion(nn.Module):
    def __init__(self, in_channels):
        super(GatedFusion, self).__init__()
        self.gate = nn.Sequential(
            nn.Conv2d(in_channels * 2, in_channels, kernel_size=1),
            nn.Sigmoid()
        )
        self.conv = BasicConv2d(in_channels, in_channels, kernel_size=3, padding=1)

    def forward(self, x1, x2):
        fusion = torch.cat([x1, x2], dim=1)
        gate = self.gate(fusion)
        out = gate * x1 + (1 - gate) * x2
        return self.conv(out)

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=8):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        w = self.fc(self.avg_pool(x))
        return x * w

class ScaleAdaptiveFusion(nn.Module):
    def __init__(self, channel):
        super(ScaleAdaptiveFusion, self).__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.conv_upsample = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample1 = BasicConv2d(2 * channel, 2 * channel, 3, padding=1)

        self.conv_concat2 = BasicConv2d(2 * channel, 2 * channel, 3, padding=1)
        self.conv_concat3 = BasicConv2d(3 * channel, 3 * channel, 3, padding=1)

        self.conv_out = nn.Sequential(
            nn.Conv2d(3 * channel, 256, kernel_size=1),
            nn.ReLU(),
        )
        self.ca = ChannelAttention(channel)
        self.fuse = GatedFusion(channel)

    def forward(self, p3, p4, p5):
        p5_0 = p5

        p5_up = self.conv_upsample(self.upsample(p5))
        p5_att = self.ca(p5_up)
        p5_1 = self.fuse(p5_att, p4)

        p4_up = self.conv_upsample(self.upsample(p4))
        p4_att = self.ca(p4_up)
        p4_1 = self.fuse(p4_att, p3)

        p5_1_up = self.conv_upsample(self.upsample(p5_1))
        p5_1_att = self.ca(p5_1_up)
        p3_1 = self.fuse(p5_1_att, p4_1)

        p5_2 = torch.cat((p5_1, self.conv_upsample(self.upsample(p5_0))), dim=1)
        p5_2 = self.conv_concat2(p5_2)

        p3_2 = torch.cat((p3_1, self.conv_upsample1(self.upsample(p5_2))), dim=1)
        p3_2 = self.conv_concat3(p3_2)

        out = self.conv_out(p3_2)
        return out

class FFNWithLN(nn.Module):
    def __init__(self, channels, expansion=2):
        super().__init__()
        self.ln = nn.LayerNorm(channels)
        self.ffn = nn.Sequential(
            nn.Conv2d(channels, channels * expansion, 1),
            nn.SiLU(inplace=True),
            nn.Conv2d(channels * expansion, channels, 1)
        )

    def forward(self, x):
        identity = x
        b, c, h, w = x.shape
        x_ln = x.permute(0, 2, 3, 1)
        x_ln = self.ln(x_ln).permute(0, 3, 1, 2)
        out = self.ffn(x_ln)
        return out + identity

class Reg(nn.Module):
    def __init__(self,in_channels_list=[144, 288, 576, 1152], out_channels=256):
        super(Reg, self).__init__()
        self.stage1 = nn.Sequential(
            nn.Conv2d(256, 128, 3, padding=1, dilation=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        self.stage2 = nn.Sequential(
            nn.Conv2d(256, 128, 3, padding=2, dilation=2),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        self.stage3 = nn.Sequential(
            nn.Conv2d(256, 128, 3, padding=3, dilation=3),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        self.stage4 = nn.Sequential(
            nn.Conv2d(256, 384, 1),
            nn.BatchNorm2d(384),
            nn.ReLU(inplace=True)
        )
        self.res = nn.Sequential(
            nn.Conv2d(384, 64, 3, padding=1, dilation=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 1, 1),
            nn.ReLU()
        )

        self.SAF = ScaleAdaptiveFusion(256)
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(256 * 4, 256, kernel_size=1),
            nn.ReLU(inplace=True)
        )

        self.lateral_convs = nn.ModuleList([
            SODM_Block(in_ch, out_channels)
            for in_ch in in_channels_list
        ])
        self.ffn = FFNWithLN(256)
        self.init_param()

    def forward(self,x1, x2, x3, x4):
        l3 = self.lateral_convs[1](x2)
        l4 = self.lateral_convs[2](x3)
        l5 = self.lateral_convs[3](x4)

        out = self.SAF(l3, l4, l5)
        out = self.ffn(out)
        y1 = self.stage1(out)
        y2 = self.stage2(out)
        y3 = self.stage3(out)
        y4 = self.stage4(out)
        y = torch.cat((y1, y2, y3), dim=1) + y4
        y = self.res(y)
        return y

    def init_param(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

class SODM_Block(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.ch_conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch // 2, 1),
            nn.BatchNorm2d(out_ch // 2),
            nn.ReLU()
        )

        self.SODM = SODM(out_ch // 2)

        self.proj = nn.Conv2d(out_ch // 2, out_ch, 1)

    def forward(self, x):
        x = self.ch_conv(x)
        x = self.SODM(x)
        return self.proj(x)

class SODM(nn.Module):

    def __init__(self, dim):
        super().__init__()
        self.ff_conv = nn.Sequential(
            nn.Conv2d(dim, dim, 3, padding=1),
            nn.BatchNorm2d(dim),
            nn.ReLU()
        )

        self.bf_conv = nn.Sequential(
            nn.Conv2d(dim, dim, 3, dilation=2, padding=2),
            nn.BatchNorm2d(dim),
            nn.ReLU()
        )

        self.pred_conv = nn.Conv2d(dim, 1, 1)
        self.pred_conv1 = nn.Conv2d(dim, 1, 1)

        self.reduce_conv1 = nn.Conv2d(2, 1, 1)
        self.reduce_conv2 = nn.Conv2d(2, 1, 1)

        self.gate = nn.Parameter(torch.tensor([0.5]))

    def forward(self, x):
        pred = torch.sigmoid(self.pred_conv(x))
        pred1 = torch.sigmoid(self.pred_conv1(x))

        concat1 = torch.cat([1 - pred, pred1], dim=1)

        reduced1 = torch.sigmoid(self.reduce_conv1(concat1))

        concat2 = torch.cat([1 - pred1, pred], dim=1)

        reduced2 = torch.sigmoid(self.reduce_conv2(concat2))

        ff_feat = self.ff_conv(x * reduced1)
        bf_feat = self.bf_conv(x * reduced2)

        fused = torch.sigmoid(self.gate) * ff_feat + \
                (1 - torch.sigmoid(self.gate)) * bf_feat

        return fused

class S2C_Net(nn.Module):
    def __init__(self, checkpoint_path=None) -> None:
        super(S2C_Net, self).__init__()
        super().__init__()
        model_cfg = "configs/sam2/sam2_hiera_l.yaml"
        self.reg = Reg()

        if checkpoint_path:
            model = build_sam2(model_cfg, checkpoint_path)
        else:
            model = build_sam2(model_cfg)
        del model.sam_mask_decoder
        del model.sam_prompt_encoder
        del model.memory_encoder
        del model.memory_attention
        del model.mask_downsample
        del model.obj_ptr_tpos_proj
        del model.obj_ptr_proj
        del model.image_encoder.neck
        self.encoder = model.image_encoder.trunk

        for param in self.encoder.parameters():
            param.requires_grad = False

        blocks = []

        for idx, block in enumerate(self.encoder.blocks):
            if idx in [0, 1]:
                blocks.append(DSFFR(block))
            elif idx in [2, 4, 7]:
                blocks.append(DSFFR(block))
            elif idx in [11, 19, 27, 35, 43]:
                blocks.append(DSFFR(block))
            elif idx in [47]:
                blocks.append(DSFFR(block))
            else:
                blocks.append(block)

        self.encoder.blocks = nn.Sequential(
            *blocks
        )

    def forward(self, x):
        x1, x2, x3, x4 = self.encoder(x)
        mu = self.reg(x1, x2, x3, x4)
        B, C, H, W = mu.size()
        mu_sum = mu.view([B, -1]).sum(1).unsqueeze(1).unsqueeze(2).unsqueeze(3)
        mu_normed = mu / (mu_sum + 1e-6)
        return mu, mu_normed