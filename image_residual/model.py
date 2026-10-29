"""
image_residual/model.py
-----------------------
IR1 — Hard-Anchored Image-Space Residual 3D U-Net (§14-15).
experiment_id: IR1 ; method_name: hard_anchored_image_residual_unet ;
model_family: direct_image_space_reconstruction.

A compact MONAI 3D U-Net predicts a bounded correction to DIRECT image-space linear
interpolation. It never touches the Stable-Diffusion VAE. A zero-initialized 1x1x1
head guarantees the identity:

    delta = 0  =>  x_pred = clip(x_linear) with acquired slices hard-copied
                 = direct linear interpolation (to numerical precision)

Composition (§13):
    delta_raw   = head(unet(x6))                 # zero at init
    delta_pred  = residual_scale * tanh(delta_raw)
    x_preclip   = x_linear + missing_mask * delta_pred
    x_pred      = clip(x_preclip, 0, 1)
    x_pred[known] = x_sparse[known]              # exact hard-copy (inference)
"""
from __future__ import annotations
import torch
import torch.nn as nn
from monai.networks.nets import UNet


DEFAULT_ARCH = dict(
    spatial_dims=3, in_channels=6, out_channels=1,
    channels=(16, 32, 64, 128), strides=(2, 2, 2),
    num_res_units=2, norm="instance", act="PRELU", dropout=0.0,
)


class IR1ResidualUNet(nn.Module):
    def __init__(self, residual_scale: float, arch: dict = None):
        super().__init__()
        self.arch = dict(DEFAULT_ARCH if arch is None else arch)
        self.unet = UNet(**self.arch)
        # zero-initialized final head => delta_raw == 0 at init regardless of UNet internals
        self.head = nn.Conv3d(self.arch["out_channels"], 1, kernel_size=1, bias=True)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)
        # residual scale is a fixed (non-trained) buffer computed on training data only
        self.register_buffer("residual_scale", torch.tensor(float(residual_scale), dtype=torch.float32))

    def raw(self, x6):
        """Unbounded network output (for gradient / finite checks)."""
        return self.head(self.unet(x6))

    def delta(self, x6):
        """Bounded residual = residual_scale * tanh(raw)."""
        return self.residual_scale * torch.tanh(self.raw(x6))

    def forward(self, x6, x_linear, missing_mask, known_mask=None, x_sparse=None,
                hardcopy=False, clip=True):
        """Compose the reconstruction. During TRAINING call with hardcopy=False (the
        loss region excludes acquired slices anyway, and the hard-copy overwrite is
        non-differentiable); during INFERENCE call with hardcopy=True."""
        d = self.delta(x6)
        x = x_linear + missing_mask * d
        if clip:
            x = torch.clamp(x, 0.0, 1.0)
        if hardcopy:
            assert known_mask is not None and x_sparse is not None
            x = torch.where(known_mask > 0.5, x_sparse, x)
        return x, d


def build_ir1(residual_scale: float, arch: dict = None) -> IR1ResidualUNet:
    return IR1ResidualUNet(residual_scale, arch)


def parameter_count(model: nn.Module) -> dict:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return dict(total=int(total), trainable=int(trainable))


@torch.no_grad()
def zero_identity_check(model: IR1ResidualUNet, x6, x_linear, missing_mask, known_mask, x_sparse):
    """Return max|x_pred - direct_linear| with the zero-init head (should be ~0)."""
    model.eval()
    x_pred, d = model(x6, x_linear, missing_mask, known_mask, x_sparse, hardcopy=True, clip=True)
    direct = torch.clamp(x_linear, 0.0, 1.0)
    direct = torch.where(known_mask > 0.5, x_sparse, direct)
    return float((x_pred - direct).abs().max()), float(d.abs().max())
