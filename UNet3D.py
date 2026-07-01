import math
from abc import abstractmethod

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


class TimestepBlock(nn.Module):
    """
    Any module where forward() takes timestep embeddings as a second argument.
    """

    @abstractmethod
    def forward(self, x, emb):
        """
        Apply the module to `x` given `emb` timestep embeddings.
        """


class TimestepEmbedSequential(nn.Sequential, TimestepBlock):
    """
    A sequential module that passes timestep embeddings to the children that
    support it as an extra input.
    """

    def forward(self, x, emb):
        for layer in self:
            if isinstance(layer, TimestepBlock):
                x = layer(x, emb)
            else:
                x = layer(x)
        return x


class PositionalEmbedding(nn.Module):
    # PositionalEmbedding
    """
    Computes Positional Embedding of the timestep
    """

    def __init__(self, dim, scale=1):
        super().__init__()
        assert dim % 2 == 0
        self.dim = dim
        self.scale = scale

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = np.log(10000) / half_dim
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = torch.outer(x * self.scale, emb)
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class Downsample(nn.Module):
    def __init__(self, in_channels, use_conv, out_channels=None):
        super().__init__()
        self.channels = in_channels
        out_channels = out_channels or in_channels
        if use_conv:
            # downsamples by 1/2
            self.downsample = nn.Conv3d(in_channels, out_channels, 3, stride=2, padding=1)
        else:
            assert in_channels == out_channels
            self.downsample = nn.AvgPool3d(kernel_size=2, stride=2)

    def forward(self, x, time_embed=None):
        assert x.shape[1] == self.channels
        return self.downsample(x)


class Upsample(nn.Module):
    def __init__(self, in_channels, use_conv, out_channels=None):
        super().__init__()
        self.channels = in_channels
        self.use_conv = use_conv
        # uses upsample then conv to avoid checkerboard artifacts
        # self.upsample = nn.Upsample(scale_factor=2, mode="nearest")
        if use_conv:
            self.conv = nn.Conv3d(in_channels, out_channels, 3, padding=1)

    def forward(self, x, time_embed=None):
        assert x.shape[1] == self.channels
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        if self.use_conv:
            x = self.conv(x)
        return x


class AttentionBlock(nn.Module):
    """
    An attention block that allows spatial positions to attend to each other.
    Originally ported from here, but adapted to the N-d case.
    https://github.com/hojonathanho/diffusion/blob/1e0dceb3b3495bbe19116a5e1b3596cd0706c543/diffusion_tf/models/unet.py#L66.
    """

    def __init__(self, in_channels, n_heads=1, n_head_channels=-1):
        super().__init__()
        self.in_channels = in_channels
        self.norm = GroupNorm32(32, self.in_channels)
        if n_head_channels in ("", None):
            n_head_channels = -1
        if n_head_channels == -1:
            self.num_heads = n_heads
        else:
            assert (
                    in_channels % n_head_channels == 0
            ), f"q,k,v channels {in_channels} is not divisible by num_head_channels {n_head_channels}"
            self.num_heads = in_channels // n_head_channels

        # query, key, value for attention
        self.to_qkv = nn.Conv1d(in_channels, in_channels * 3, 1)
        self.attention = QKVAttention(self.num_heads)
        self.proj_out = zero_module(nn.Conv1d(in_channels, in_channels, 1))

    def forward(self, x, time=None):
        b, c, *spatial = x.shape
        x = x.reshape(b, c, -1)
        qkv = self.to_qkv(self.norm(x))
        h = self.attention(qkv)
        h = self.proj_out(h)
        return (x + h).reshape(b, c, *spatial)


class QKVAttention(nn.Module):
    """
    A module which performs QKV attention. Matches legacy QKVAttention + input/ouput heads shaping
    """

    def __init__(self, n_heads):
        super().__init__()
        self.n_heads = n_heads

    def forward(self, qkv, time=None):
        """
        Apply QKV attention.
        :param qkv: an [N x (H * 3 * C) x T] tensor of Qs, Ks, and Vs.
        :return: an [N x (H * C) x T] tensor after attention.
        """
        bs, width, length = qkv.shape
        assert width % (3 * self.n_heads) == 0
        ch = width // (3 * self.n_heads)
        q, k, v = qkv.reshape(bs * self.n_heads, ch * 3, length).split(ch, dim=1)
        scale = 1 / math.sqrt(math.sqrt(ch))
        weight = torch.einsum(
                "bct,bcs->bts", q * scale, k * scale
                )  # More stable with f16 than dividing afterwards
        weight = torch.softmax(weight.float(), dim=-1).type(weight.dtype)
        a = torch.einsum("bts,bcs->bct", weight, v)
        return a.reshape(bs, -1, length)


class ResBlock(TimestepBlock):
    def __init__(
            self,
            in_channels,
            time_embed_dim,
            dropout,
            out_channels=None,
            use_conv=False,
            up=False,
            down=False
            ):
        super().__init__()
        out_channels = out_channels or in_channels
        self.in_layers = nn.Sequential(
                GroupNorm32(32, in_channels),
                nn.SiLU(),
                nn.Conv3d(in_channels, out_channels, 3, padding=1)
                )
        self.updown = up or down

        if up:
            self.h_upd = Upsample(in_channels, False)
            self.x_upd = Upsample(in_channels, False)
        elif down:
            self.h_upd = Downsample(in_channels, False)
            self.x_upd = Downsample(in_channels, False)
        else:
            self.h_upd = self.x_upd = nn.Identity()

        self.embed_layers = nn.Sequential(
                nn.SiLU(),
                nn.Linear(time_embed_dim, out_channels)
                )
        self.out_layers = nn.Sequential(
                GroupNorm32(32, out_channels),
                nn.SiLU(),
                nn.Dropout(p=dropout),
                zero_module(nn.Conv3d(out_channels, out_channels, 3, padding=1))
                )
        if out_channels == in_channels:
            self.skip_connection = nn.Identity()
        elif use_conv:
            self.skip_connection = nn.Conv3d(in_channels, out_channels, 3, padding=1)
        else:
            self.skip_connection = nn.Conv3d(in_channels, out_channels, 1)

    def forward(self, x, time_embed):
        if self.updown:
            in_rest, in_conv = self.in_layers[:-1], self.in_layers[-1]
            h = in_rest(x)
            h = self.h_upd(h)
            x = self.x_upd(x)
            h = in_conv(h)
        else:
            h = self.in_layers(x)
        emb_out = self.embed_layers(time_embed).type(h.dtype)
        while len(emb_out.shape) < len(h.shape):
            emb_out = emb_out[..., None]

        h = h + emb_out
        h = self.out_layers(h)
        return self.skip_connection(x) + h


class UNetModel(nn.Module):
    # UNet model
    def __init__(
            self,
            img_size,
            init_channels,
            conv_resample=True,
            n_heads=1,
            n_head_channels=-1,
            channel_mults="",
            num_res_blocks=2,
            dropout=0,
            attention_resolutions="8",
            biggan_updown=True,
            in_channels=1
            ):
        self.dtype = torch.float32
        super().__init__()

        if channel_mults != "" and isinstance(channel_mults, str):
            channel_mults = tuple(map(int, channel_mults.split(",")))

        if channel_mults == "":
            channel_mults = (1, 1)
        attention_ds = []
        for res in attention_resolutions.split(","):
            attention_ds.append(img_size // int(res))

        # --- Determine the channel width at each resolution level (encoder levels) ---
        level_channels = [init_channels * mult for mult in channel_mults]

        # --- Parse n_head_channels into either scalar or per-level list ---
        parsed_head_ch = _parse_int_list(n_head_channels, expected_len=len(level_channels), name="n_head_channels")

        # --- Determine which levels will have attention (based on attention_resolutions) ---
        ds_per_level = []
        ds = 1
        for i in range(len(level_channels)):
            ds_per_level.append(ds)
            if i != len(level_channels) - 1:
                ds *= 2

        attn_level_mask = [ds_val in attention_ds for ds_val in ds_per_level]

        # --- Validate heads if using head-channels mode ---
        if n_head_channels not in (-1, None, ""):
            # Convert scalar head_ch to per-level if needed
            if parsed_head_ch is None:
                raise ValueError("n_head_channels parsing failed unexpectedly")
            if len(parsed_head_ch) == 1:
                head_ch_per_level = parsed_head_ch * len(level_channels)
            else:
                head_ch_per_level = parsed_head_ch

            for i, (C, use_attn, hc) in enumerate(zip(level_channels, attn_level_mask, head_ch_per_level)):
                if not use_attn:
                    continue
                if hc <= 0:
                    raise ValueError(f"Level {i}: n_head_channels must be > 0, got {hc}")
                if C < hc:
                    raise ValueError(
                        f"Level {i}: attention enabled but channels ({C}) < n_head_channels ({hc}). "
                        f"This would create 0 heads."
                    )
                if C % hc != 0:
                    raise ValueError(
                        f"Level {i}: channels ({C}) must be divisible by n_head_channels ({hc}) when attention is enabled."
                    )

        self.image_size = img_size
        self.in_channels = in_channels
        self.model_channels = init_channels
        self.out_channels = in_channels
        self.num_res_blocks = num_res_blocks
        self.attention_resolutions = attention_resolutions
        self.dropout = dropout
        self.channel_mult = channel_mults
        self.conv_resample = conv_resample

        self.dtype = torch.float32
        self.num_heads = n_heads
        self.num_head_channels = n_head_channels

        time_embed_dim = init_channels * 4
        self.time_embedding = nn.Sequential(
                PositionalEmbedding(init_channels, 1),
                nn.Linear(init_channels, time_embed_dim),
                nn.SiLU(),
                nn.Linear(time_embed_dim, time_embed_dim),
                )
        ch = int(channel_mults[0] * init_channels)
        self.down = nn.ModuleList(
                [TimestepEmbedSequential(nn.Conv3d(self.in_channels, init_channels, 3, padding=1))]
                )
        channels = [ch]
        ds = 1
        for i, mult in enumerate(channel_mults):
            # out_channels = init_channels * mult

            for _ in range(num_res_blocks):
                layers = [ResBlock(
                        ch,
                        time_embed_dim=time_embed_dim,
                        out_channels=init_channels * mult,
                        dropout=dropout,
                        )]
                ch = init_channels * mult
                # channels.append(ch)

                if ds in attention_ds:
                    layers.append(
                            AttentionBlock(
                                    ch,
                                    n_heads=n_heads,
                                    n_head_channels=n_head_channels,
                                    )
                            )
                self.down.append(TimestepEmbedSequential(*layers))
                channels.append(ch)
            if i != len(channel_mults) - 1:
                out_channels = ch
                self.down.append(
                        TimestepEmbedSequential(
                                ResBlock(
                                        ch,
                                        time_embed_dim=time_embed_dim,
                                        out_channels=out_channels,
                                        dropout=dropout,
                                        down=True
                                        )
                                if biggan_updown
                                else
                                Downsample(ch, conv_resample, out_channels=out_channels)
                                )
                        )
                ds *= 2
                ch = out_channels
                channels.append(ch)

        self.middle = TimestepEmbedSequential(
                ResBlock(
                        ch,
                        time_embed_dim=time_embed_dim,
                        dropout=dropout
                        ),
                AttentionBlock(
                        ch,
                        n_heads=n_heads,
                        n_head_channels=n_head_channels
                        ),
                ResBlock(
                        ch,
                        time_embed_dim=time_embed_dim,
                        dropout=dropout
                        )
                )
        self.up = nn.ModuleList([])

        for i, mult in reversed(list(enumerate(channel_mults))):
            for j in range(num_res_blocks + 1):
                inp_chs = channels.pop()
                layers = [
                    ResBlock(
                            ch + inp_chs,
                            time_embed_dim=time_embed_dim,
                            out_channels=init_channels * mult,
                            dropout=dropout
                            )
                    ]
                ch = init_channels * mult
                if ds in attention_ds:
                    layers.append(
                            AttentionBlock(
                                    ch,
                                    n_heads=n_heads,
                                    n_head_channels=n_head_channels
                                    ),
                            )

                if i and j == num_res_blocks:
                    out_channels = ch
                    layers.append(
                            ResBlock(
                                    ch,
                                    time_embed_dim=time_embed_dim,
                                    out_channels=out_channels,
                                    dropout=dropout,
                                    up=True
                                    )
                            if biggan_updown
                            else
                            Upsample(ch, conv_resample, out_channels=out_channels)
                            )
                    ds //= 2
                self.up.append(TimestepEmbedSequential(*layers))

        if self.in_channels == 1:
            self.out = nn.Sequential(
                    GroupNorm32(32, ch),
                    nn.SiLU(),
                    zero_module(nn.Conv3d(init_channels * channel_mults[0], 1, 3, padding=1))
                    )
        elif self.in_channels == 2:
            self.out_shared = nn.Sequential(
                GroupNorm32(32, ch),
                nn.SiLU(),
            )    
            self.img_head = zero_module(nn.Conv3d(init_channels * channel_mults[0], 1, 3, padding=1))
            self.seg_head = zero_module(nn.Conv3d(init_channels * channel_mults[0], 1, 3, padding=1))   
            
    def forward(self, x, time):

        time_embed = self.time_embedding(time)

        skips = []

        h = x.type(self.dtype)
        for i, module in enumerate(self.down):
            h = module(h, time_embed)
            skips.append(h)
        h = self.middle(h, time_embed)
        for i, module in enumerate(self.up):
            h = torch.cat([h, skips.pop()], dim=1)
            h = module(h, time_embed)
        h = h.type(x.dtype)

        if self.in_channels == 1:
            h = self.out(h)
            return h
        elif self.in_channels == 2:
            h = self.out_shared(h)
            img_out = self.img_head(h)
            seg_out = self.seg_head(h)
            return img_out, seg_out
        
class ConditioningProjection3D(nn.Module):
    """
    Lightweight multi-scale image conditioning block.

    It resizes the conditioning image to the current denoiser resolution and
    projects it to the denoiser channel width. This gives the mask denoiser
    image information at every encoder scale without changing your existing
    ResBlock/AttentionBlock implementation.
    """
    def __init__(self, cond_channels, out_channels):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv3d(cond_channels, out_channels, 3, padding=1),
            nn.SiLU(),
            nn.Conv3d(out_channels, out_channels, 1),
        )

    def forward(self, cond, target):
        if cond.shape[2:] != target.shape[2:]:
            cond = F.interpolate(cond, size=target.shape[2:], mode="trilinear", align_corners=False)
        return self.net(cond).type(target.dtype)


class ConditionalSegUNetModel(nn.Module):
    """
    Image-conditioned Bernoulli segmentation diffusion U-Net.

    Diffusion variable:
        y_t: noisy binary segmentation mask, shape [B, 1, D, H, W]

    Conditioning signal:
        image: clean image, shape [B, image_channels, D, H, W]

    Output:
        eps_seg_logits: logits for the Bernoulli flip-noise eps_t.

    This implements the "image encoder + mask denoising U-Net" idea by using
    the normal UNetModel(in_channels=1) as the mask denoiser, while injecting
    multi-scale image features after every encoder/down block and at the
    bottleneck.
    """
    is_conditional_seg_model = True

    def __init__(
            self,
            img_size,
            init_channels,
            conv_resample=True,
            n_heads=1,
            n_head_channels=-1,
            channel_mults="",
            num_res_blocks=2,
            dropout=0,
            attention_resolutions="8",
            biggan_updown=True,
            ):
        super().__init__()
        self.dtype = torch.float32
        self.image_channels = 1
        self.mask_channels = 1
        self.cond_scale = 1.0

        # Reuse your existing U-Net as a mask-only denoiser.
        self.denoiser = UNetModel(
            img_size=img_size,
            init_channels=init_channels,
            conv_resample=conv_resample,
            n_heads=n_heads,
            n_head_channels=n_head_channels,
            channel_mults=channel_mults,
            num_res_blocks=num_res_blocks,
            dropout=dropout,
            attention_resolutions=attention_resolutions,
            biggan_updown=biggan_updown,
            in_channels=self.mask_channels,
        )

        # Work out the output channel count of every down block, matching
        # UNetModel.forward(), where one skip is appended after every down module.
        if channel_mults != "" and isinstance(channel_mults, str):
            channel_mults = tuple(map(int, channel_mults.split(",")))
        if channel_mults == "":
            channel_mults = (1, 1)

        down_channels = []
        ch = int(channel_mults[0] * init_channels)
        down_channels.append(init_channels)  # first input conv block
        for i, mult in enumerate(channel_mults):
            for _ in range(num_res_blocks):
                ch = init_channels * mult
                down_channels.append(ch)
            if i != len(channel_mults) - 1:
                down_channels.append(ch)  # downsample block output

        self.cond_down = nn.ModuleList([
            ConditioningProjection3D(self.image_channels, c) for c in down_channels
        ])
        self.cond_middle = ConditioningProjection3D(self.image_channels, ch)

    def forward(self, y_t, time, image):
        """
        y_t:   [B, 1, D, H, W] noisy segmentation mask
        time:  [B] diffusion timestep
        image: [B, 1, D, H, W] clean image condition
        """
        time_embed = self.denoiser.time_embedding(time)

        skips = []
        h = y_t.type(self.denoiser.dtype)
        image = image.type(h.dtype)

        for i, module in enumerate(self.denoiser.down):
            h = module(h, time_embed)
            h = h + self.cond_scale * self.cond_down[i](image, h)
            skips.append(h)

        h = self.denoiser.middle(h, time_embed)
        h = h + self.cond_scale * self.cond_middle(image, h)

        for module in self.denoiser.up:
            h = torch.cat([h, skips.pop()], dim=1)
            h = module(h, time_embed)

        h = h.type(y_t.dtype)
        return self.denoiser.out(h)


class GroupNorm32(nn.GroupNorm):
    def forward(self, x):
        return super().forward(x.float()).type(x.dtype)


def zero_module(module):
    """
    Zero out the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().zero_()
    return module


def update_ema_params(target, source, decay_rate=0.9999):
    targParams = dict(target.named_parameters())
    srcParams = dict(source.named_parameters())
    for k in targParams:
        targParams[k].data.mul_(decay_rate).add_(srcParams[k].data, alpha=1 - decay_rate)

def _parse_int_list(x, expected_len=None, name="value"):
    if x in ("", None):
        return None
    if isinstance(x, int):
        vals = [x]
    elif isinstance(x, (list, tuple)):
        vals = list(map(int, x))
    elif isinstance(x, str):
        vals = list(map(int, x.split(",")))
    else:
        raise TypeError(f"{name} must be int, list/tuple, or comma-separated string")

    if expected_len is not None and len(vals) not in (1, expected_len):
        raise ValueError(f"{name} must have length 1 or {expected_len}, got {len(vals)}")
    return vals


if __name__ == "__main__":
    args = {
        'img_size':          256,
        'init_channels':     64,
        'dropout':           0.3,
        'num_heads':         4,
        'num_head_channels': '32,16,8',
        'lr':                1e-4,
        'Batch_Size':        64
        }
    model = UNetModel(
            args['img_size'], args['init_channels'], dropout=args[
                "dropout"], n_heads=args["num_heads"], attention_resolutions=args["num_head_channels"],
            in_channels=3
            )

    x = torch.randn(1, 3, 512, 512)
    t_batch = torch.tensor([1], device=x.device).repeat(x.shape[0])
    print(model(x, t_batch).shape)
