import torch


class Raymarcher():
    """
    Raymarch using the Absorption-Only (AO) algorithm.
    The algorithm independently renders each ray by analyzing density and
    feature values sampled at (typically uniformly) spaced 3D locations along
    each ray. The density values `rays_densities` are of shape
    `(..., n_points_per_ray, 1)`, their values should range between [0, 1], and
    represent the opaqueness of each point (the higher the less transparent).
    The algorithm only measures the total amount of light absorbed along each ray
    and, besides outputting per-ray `opacity` values of shape `(...,)`,
    does not produce any feature renderings.
    The algorithm simply computes `total_transmission = prod(1 - rays_densities)`
    of shape `(..., 1)` which, for each ray, measures the total amount of light
    that passed through the volume.
    It then returns `opacities = 1 - total_transmission`.
    """

    def __init__(self):
        pass

    def __call__(
        self, alpha: torch.Tensor,
        ray_bundle_lenghts,
    ):
        alpha.squeeze_()
        bw = torch.prod(1. - alpha + 1e-10, dim=-1)

        weights = alpha*_shifted_cumprod(1. - alpha)
        
        # Depth estimation
        ray_bundle_lenghts = (ray_bundle_lenghts[..., 1:] + ray_bundle_lenghts[..., :-1]) / 2
        depth = (weights * ray_bundle_lenghts).sum(dim=-1)

        return bw, depth, weights

def _shifted_cumprod(x, shift: int = 1):
    """
    Computes `torch.cumprod(x, dim=-1)` and prepends `shift` number of
    ones and removes `shift` trailing elements to/from the last dimension
    of the result.
    """
    x_cumprod = torch.cumprod(x, dim=-1)
    x_cumprod_shift = torch.cat(
        [torch.ones_like(x_cumprod[..., :shift]), x_cumprod[..., :-shift]], dim=-1
    )
    return x_cumprod_shift
