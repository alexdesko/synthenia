import torch
import torch.nn as nn
import torch.nn.functional as F

class TensorVolCP(nn.Module):
    def __init__(self, n_components, resolution, n_features):
        super().__init__()
        self.n_components = n_components
        self.n_features = n_features

        density_line = []
        for _ in range(3): # Number of coordinates
            density_line.append(
                torch.nn.Parameter(0.1*torch.rand((1, n_components, resolution, 1)))
            )
        self.density_line = torch.nn.ParameterList(density_line)
        self.basis_mat = nn.Linear(self.n_components, self.n_features, bias = False)


    def forward(self, xyz_sampled):
        coordinate_line = torch.stack((xyz_sampled[..., 0], xyz_sampled[..., 1], xyz_sampled[..., 2]))
        coordinate_line = torch.stack((torch.zeros_like(coordinate_line), coordinate_line), dim=-1).detach().view(3, -1, 1, 2)
        coordinate_line = coordinate_line.expand(1, *coordinate_line.shape)
        line_coef_point = F.grid_sample(self.density_line[0], coordinate_line[:,0,...],
                                            align_corners=True)
        line_coef_point = line_coef_point * F.grid_sample(self.density_line[1], coordinate_line[:,1,...],
                                        align_corners=True)
        line_coef_point = line_coef_point * F.grid_sample(self.density_line[2], coordinate_line[:,2,...],
                                        align_corners=True)
        return self.basis_mat(line_coef_point.squeeze().t())

class TensorVolVM(nn.Module):
    def __init__(self, n_components, resolution, n_features):
        super().__init__()
        self.n_components = n_components
        self.n_features = n_features
        self.resolution = resolution
        plane_coefs = []
        line_coefs = []
        for _ in range(3):
            plane_coefs.append(
                torch.nn.Parameter(0.1*torch.rand((1, n_components, resolution, resolution)))
            )
            line_coefs.append(
                torch.nn.Parameter(0.1*torch.rand((1, n_components, resolution, 1)))
            )
        self.plane_coefs = torch.nn.ParameterList(plane_coefs)
        self.line_coefs = torch.nn.ParameterList(line_coefs)
        self.basis_mat = nn.Linear(3*self.n_components, self.n_features, bias = False)

    def forward(self, xyz_sampled):
        coordinate_plane = torch.stack((xyz_sampled[..., [2,1]], xyz_sampled[..., [2,0]], xyz_sampled[..., [1,0]])).detach().view(3, -1, 1, 2)
        coordinate_line = torch.stack((xyz_sampled[..., 0], xyz_sampled[..., 1], xyz_sampled[..., 2]))
        coordinate_line = torch.stack((torch.zeros_like(coordinate_line), coordinate_line), dim=-1).detach().view(3, -1, 1, 2)
        coordinate_line = coordinate_line.expand(1, *coordinate_line.shape)
        coordinate_plane = coordinate_plane.expand(1, *coordinate_plane.shape)

        plane_components, line_components = [], []
        for i in range(3):
            plane_components.append(
                F.grid_sample(self.plane_coefs[i], coordinate_plane[:,i,...], align_corners=True).squeeze()
            )
            line_components.append(
                F.grid_sample(self.line_coefs[i], coordinate_line[:,i,...], align_corners=True).squeeze()
            )
        plane_components, line_components = torch.cat(plane_components), torch.cat(line_components)
        return self.basis_mat((plane_components*line_components).t())
    

class RegularVol(nn.Module):
    def __init__(self, resolution):
        super().__init__()
        self.resolution = resolution
        self.grid = torch.nn.Parameter(.1*torch.rand((1, resolution, resolution, resolution)))
    
    def forward(self, xyz_sampled):
        return F.grid_sample(self.grid.expand(xyz_sampled.shape[0], 1, *self.grid.shape[1:]), xyz_sampled, align_corners=True).squeeze()
    

if __name__ == "__main__":
    coord = torch.rand((1256, 10, 10, 10, 3))
    vol = RegularVol(128)
    print(vol(coord).shape)