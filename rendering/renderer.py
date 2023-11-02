import torch
import trimesh
from skimage.measure import marching_cubes
import logging

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Renderer(torch.nn.Module):
    """
    Renderer class
    Englobes the whole rendering process
    """
    def __init__(
        self,
        point_sampler,
        encoder_position,
        density,
        marcher,
    ):
        super().__init__()
        self.point_sampler = point_sampler #
        self.encoder_position = encoder_position #
        self.density = density
        self.marcher = marcher #

    def forward(self, ray_origin, ray_direction):
        samples, depth_values = self.point_sampler(ray_origin, ray_direction)
        
        ## Compute the densities
        densities = self.compute_densities(samples)

        ## Trapezoidal rule for integration
        deltas = depth_values[..., 1:] - depth_values[..., :-1]
        densities = (densities[..., 1:] + densities[..., :-1]) / 2

        alpha = 1 - torch.exp(-densities*deltas)
        # Volume rendering
        bw, depth, _ = self.marcher(alpha, depth_values)
        
        return bw, depth, densities
    

    def forward_batch(self, ray_origin, ray_direction, batchsize):
        ray_origins_batch = [ray_origin[i:i + batchsize]  \
                             for i in range(0, ray_origin.shape[0], batchsize)]
        ray_directions_batch = [ray_direction[i:i + batchsize]  \
                             for i in range(0, ray_direction.shape[0], batchsize)]
        
        bw, depth = [], []

        for ray_o, ray_d in zip(ray_origins_batch, ray_directions_batch):
            bw_, depth_, _ = self.forward(ray_o, ray_d)
            bw.append(bw_)
            depth.append(depth_)

        return torch.cat(bw, dim=0).contiguous(), torch.cat(depth, dim=0).contiguous()

    def compute_densities(self, samples):
        unit_cube_mask = ((samples >= -1) & (samples <= 1)).all(dim=-1)
        unit_cube_samples = samples[unit_cube_mask]
        encoded_samples = self.encoder_position(unit_cube_samples)
        densities = torch.zeros(size=samples.shape[:-1], device=device)
        densities[unit_cube_mask] = self.density(encoded_samples).squeeze()
        return densities

    @torch.no_grad()
    def grid_evaluation(self, resolution):
        n = 64
        x = torch.linspace(-1, 1, resolution).split(n)
        y = torch.linspace(-1, 1, resolution).split(n)
        z = torch.linspace(-1, 1, resolution).split(n)

        field = torch.zeros((resolution, resolution, resolution)).to(device)

        for xi, xs in enumerate(x):
            for yi, ys in enumerate(y):
                for zi, zs in enumerate(z):
                    xx, yy, zz = torch.meshgrid(xs, ys, zs, indexing = "xy")
                    pts = torch.cat([xx.reshape(-1, 1), yy.reshape(-1, 1), zz.reshape(-1, 1)], dim=-1).to(device)
                    densities = self.compute_densities(pts).reshape(len(xs), len(ys), len(zs))
                    field[xi * n: xi * n + len(xs), yi * n: yi * n + len(ys), zi * n: zi * n + len(zs)] = densities.transpose(0,1)
        return field.transpose_(0,1)