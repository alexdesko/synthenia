import torch

class PointSampler():
    """
    Point sampler for the ray marching
    """
    def __init__(self, near, far, n_samples_per_ray, randomized):
        self.near = near
        self.far = far
        self.n_samples_per_ray = n_samples_per_ray
        self.randomized = randomized

    def __call__(self, ray_origins, ray_directions):
        depth_values = torch.linspace(self.near, self.far, self.n_samples_per_ray).to(ray_origins)
        if self.randomized:
            #Pytorch3D implementation
            mids = 0.5 * (depth_values[..., 1:] + depth_values[..., :-1])
            upper = torch.cat((mids, depth_values[..., -1:]), dim=-1)
            lower = torch.cat((depth_values[..., :1], mids), dim=-1)
            # Samples in those intervals.
            depth_values = lower + (upper - lower) * torch.rand_like(lower)
            depth_values.clamp_(self.near, self.far)
        query_points = ray_origins[..., None, :] + ray_directions[..., None, :] * depth_values[..., :, None]
        return query_points, depth_values

    def set_randomized(self, randomized):
        self.randomized = randomized

'''
Old class for the sampling when there is coarse and fine strategy
'''        
class CoarseFinePointSampler():
    def __init__(self, near, far, n_samples_per_ray_coarse, n_samples_per_ray_fine):
        self.near = near
        self.far = far
        self.n_samples_per_ray_coarse = n_samples_per_ray_coarse
        self.n_samples_per_ray_fine = n_samples_per_ray_fine
    
    def sample_coarse(self, ray_origin, ray_direction):
        depth_values = torch.linspace(self.near, self.far, self.n_samples_per_ray_coarse).to(ray_origin)
        query_points = ray_origin[..., None, :] + ray_direction[..., None, :] * depth_values[..., :, None]
        return query_points, depth_values

    def sample_fine(self, ray_origin, ray_direction, densities):
        depth_values = torch.linspace(self.near, self.far, self.n_samples_per_ray_coarse).to(ray_origin)
        densities = densities / torch.sum(densities, 1, keepdim=True)
        densities[torch.isnan(densities)] = 1/self.n_samples_per_ray_coarse
        idx_fine = torch.multinomial(densities, self.n_samples_per_ray_fine, replacement=True)
        depth_fine = depth_values[idx_fine]
        uniform_dist = torch.distributions.uniform.Uniform(torch.tensor([0.0]), torch.tensor([self.far-self.near])/self.n_samples_per_ray_coarse)
        uniform_samples = uniform_dist.rsample(depth_fine.shape).squeeze().to(depth_fine)
        depth_fine = depth_fine + uniform_samples
        depth_values = torch.cat((depth_values.repeat(depth_fine.shape[0],1), depth_fine), dim=-1).clamp(self.near, self.far)
        depth_values, _ = torch.sort(depth_values, dim = -1)
        query_points = ray_origin[..., None, :] + ray_direction[..., None, :] * depth_values[..., :, None]
        return query_points, depth_values

'''
Class to sample random patches of points for 3D NLM regularization
'''
class PatchSampler3D():
    def __init__(self, n_points, patch_size):
        self.n_points = n_points
        self.patch_size = patch_size

    def samples_patches_3D(self):

        ## Careful here with the indexing, maybe this is not the good one
        xi, yi, zi = torch.meshgrid(
            (torch.linspace(-1, 1, self.patch_size),
            torch.linspace(-1, 1, self.patch_size),
            torch.linspace(-1, 1, self.patch_size)),
            indexing="ij"
        )

        # Resolution is universally fixed as being 256
        coordinates_cube = torch.stack((xi, yi, zi), dim=-1).reshape(-1,3)*self.patch_size/(2*256.)
        # sample at random coordinates in the (-1,1)^3 cube
        coordinates = 2*torch.rand(size=(self.n_points, 3)) - 1

        coordinates_patches = coordinates.unsqueeze(1) + coordinates_cube.unsqueeze(0)
        return coordinates_patches




if __name__ == "__main__":
    sampler = PatchSampler3D(100, 3)
    sampler.samples_patches_3D()
    import sys
    sys.exit(-1)
    sampler = PointSampler(0., 1., 10, 20)
    ray_origins = torch.Tensor([0., 0., 0.]).unsqueeze(0).repeat(2,1)
    ray_directions = torch.Tensor([1., 1., 1.]).unsqueeze(0).repeat(2,1)
    print(sampler.sample_coarse(ray_origins, ray_directions))
    print(sampler.sample_fine(ray_origins, ray_directions,1e-3 + torch.arange(0,10.,1).repeat(2,1)))
