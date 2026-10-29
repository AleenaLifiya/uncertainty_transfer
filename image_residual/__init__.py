"""
image_residual — IR1 Hard-Anchored Image-Space Residual 3D U-Net.

experiment_id : IR1
method_name   : hard_anchored_image_residual_unet
display_name  : IR1 — Hard-Anchored Image-Space Residual 3D U-Net
model_family  : direct_image_space_reconstruction

A NEW model family (not L3D / L6 / ControlNet). Treats DIRECT image-space linear
interpolation as a hard anchor and predicts only a bounded, mask-conditioned
correction, avoiding the frozen Stable-Diffusion VAE entirely.
"""
from . import masks, inputs, model, residual_scale, losses, trainer, inference, evaluation, reporting

__all__ = ["masks", "inputs", "model", "residual_scale", "losses", "trainer",
           "inference", "evaluation", "reporting"]
