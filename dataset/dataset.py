import os
from PIL import Image
import torch
from torchvision import transforms

from torch.utils.data import Dataset
import numpy as np

class CustomDataset(Dataset):
    """
    Custom dataset for the TEM data
    """
    def __init__(self, path):
        super().__init__()
        images, poses = load_projections(path)
        self.slices = load_slices(path)
        
        idx_train = 2*torch.arange(40)
        idx_test = 25
        self.images, self.poses = images[idx_train], poses[idx_train]
        # Validation projection
        self.test_image, self.test_pose = images[idx_test], poses[idx_test]
        self.W, self.H = self.images[0].shape
        
        # Store already all the origins and the projections
        self.origins, self.directions = get_rays_orthographic(self.H, self.W, self.poses)

        max_angle = idx_train[-1]
        idx_missing = (max_angle + 180) // 2
        self.missing_image, self.missing_pose = images[idx_missing], poses[idx_missing]
        self.missing_origin, self.missing_direction = get_rays_orthographic(self.H, self.W, self.missing_pose)
        

    def __getitem__(self, index):
        o, d = get_rays_orthographic(self.H, self.W, self.poses[index])
        return self.images[index], self.poses[index], o.squeeze(), d.squeeze()
    
    def get_slice(self, index):
        return self.slices[index]
    
    def get_test_data(self):
        o, d = get_rays_orthographic(self.H, self.W, self.test_pose)
        return self.test_image, self.test_pose, o.squeeze(), d.squeeze()
    
    def get_total_nb_pixels(self):
        return torch.numel(self.images)

    def get_random_batch_of_rays_and_pixels(self, n_rays = 4096):
        idx = torch.randperm(torch.numel(self.images))[:n_rays]
        return self.origins.view(-1,3)[idx], self.directions.view(-1,3)[idx], self.images.view(-1)[idx]
    
    def get_random_batch_of_rays_and_pixels_missing(self, n_rays = 1024):
        idx = torch.randperm(torch.numel(self.missing_origin) // 3)[:n_rays]
        return self.missing_origin.view(-1,3)[idx], self.missing_direction.view(-1,3)[idx]

def get_rays_orthographic(height: int, width: int, poses: torch.Tensor):
    if len(poses.shape) == 2:
        poses = poses.unsqueeze(0) 
        batch_size = 1
    else:
        batch_size = poses.shape[0]

    xx, yy = torch.meshgrid(
        torch.linspace(-1, 1, width).to(poses),
        torch.linspace(-1, 1, height).to(poses), 
        indexing="xy"
    )
    directions = torch.stack([torch.zeros_like(xx),
                              torch.zeros_like(xx),
                              -torch.ones_like(xx)], 
                              dim=-1).unsqueeze(-2).expand(width, height, batch_size, 3) #  W H B 3 = X Y B 3
    
    ray_directions = torch.sum(directions[..., None, :] * poses[..., :3, :3], dim=-1)
    origins = torch.stack([xx,
                           yy,
                           torch.zeros_like(xx),
                           torch.ones_like(xx)],
                           dim=-1).unsqueeze(-2).expand(height, width, batch_size, 4)
    ray_origins = torch.sum(origins[..., None, :] * poses[..., :3, :], dim=-1)
    return ray_origins.permute(2,0,1,3).contiguous(), ray_directions.permute(2,0,1,3).contiguous()

def load_image(imagedir):
    # Normalize the images to (0,1) range
    image = Image.open(imagedir).convert('L') 
    image = transforms.ToTensor()(image)
    return image.float()

def load_projections(datadir):
    images = []
    projdir = os.path.join(datadir, 'projections/')
    for file in sorted(os.listdir(projdir)):
        if (file.endswith('.png') or file.endswith('.tif')):
            file_name = os.path.join(projdir, file)
            images.append(load_image(file_name))
    images = torch.cat(images)

    poses = np.load(os.path.join(projdir, 'poses.npz'))['arr_0']
    poses = torch.from_numpy(poses).float()
    return images, poses

def load_slices(datadir):
    slices = []
    slicesdir = os.path.join(datadir, 'slices/')
    for file in sorted(os.listdir(slicesdir)):
        if (file.endswith('.png') or file.endswith('.tif')):
            file_name = os.path.join(slicesdir, file)
            slices.append(load_image(file_name))
    
    return torch.cat(slices)




if __name__ == "__main__":
    datadir = '../data/synthetic_debug/'
    dataset = CustomDataset(datadir)
    print(dataset.images.min())
    print(dataset.images.max())
    print(dataset.slices.min())
    print(dataset.slices.max())
    
    o, d = dataset.get_rays()
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10,10))
    plt.scatter(o[:,0,0,0].cpu().numpy(), o[:,0,0,2].cpu().numpy())
    plt.show()
    print(o[:,0,0,:])

