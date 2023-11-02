import torch
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image
import cv2
import os


# CAUTION: torch.meshgrid has a misleading indexing convention
## Using indexing='xy' is completely different than using 'ij'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def save_video(frames, path):
    fps = int(len(frames)/5) ## 5 second video
    width, height = frames[0].shape
    video = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width,height), 0)
    for frame in frames:
        video.write(frame)
    video.release()

def save_image(image, path):
    img = Image.fromarray(image, 'L')
    img.save(path)

class Scene:
    def __init__(self):
        self.res = 256
        self.n_elements = 50
        self.volume = None
    

    def build(self):
        self.volume = torch.zeros((self.res, self.res, self.res))
        base_density = .1

        for i in tqdm(range(self.n_elements)):
            ## Position and rayon, check if the whole ellipsoid is in, then create an ellipsoid
            pos = 2*torch.rand(3) - 1 ## Coordinates between -1 and 1
            radii = 0.4*torch.rand(3) + 0.1 ## Radii betwween 0.1 and 0.5

            ## Check whether the whole ellipsiod is in the volume
            if (pos + radii > 1).any() | (pos - radii < -1).any():
                print('Element {} goes beyond volume. Skipping'.format(i))
                continue

            ## Thickness of the ellipsoid and densities (attenuations...)
            thickness = .1*torch.rand(1) + .85
            density_out = .2*torch.rand(1) + .75
            density_in = .1*torch.rand(1) + .25

            ## Create meshgrid for the bool
            x, y, z = torch.meshgrid(
                (torch.linspace(-1, 1, self.res),
                torch.linspace(-1, 1, self.res),
                torch.linspace(-1, 1, self.res)),
                indexing="xy"
            )
            ## Create the masks and add to the volume the inner and outer densities
            inner_bool = ((x-pos[0])**2)/(radii[0]*thickness)**2 + ((y-pos[1])**2)/(radii[1]*thickness)**2 + ((z-pos[2])**2)/(radii[2]*thickness)**2 < 1
            outer_bool = ((x-pos[0])**2)/(radii[0]**2) + ((y-pos[1])**2)/(radii[1]**2) + ((z-pos[2])**2)/(radii[2])**2 < 1
            free_bool = self.volume < 1e-5

            inner_bool = torch.logical_and(inner_bool, free_bool)
            outer_bool = torch.logical_and(outer_bool, free_bool)
            self.volume[outer_bool] += density_out - base_density
            self.volume[inner_bool] += density_in - density_out - base_density
        
        self.volume += base_density
        print('Volume min and max densities')
        print(self.volume.min(), self.volume.max())

    def save_volume(self, dir):
        path = os.path.join(dir, 'slices/')
        os.makedirs(path, exist_ok=True)
        if self.volume is None:
            print('Error, the volume should be build before being saved')
        frames = []
        for i in tqdm(range(self.volume.shape[-1])):
            # We don't take the transpose because of the 'xy' meshgrid convention
            frame = (255.*self.volume[i]).cpu().numpy().astype('uint8')
            save_image(frame, os.path.join(path, '{:04d}.png'.format(i)))
            frames.append(frame)
        print('saving video')
        save_video(frames, os.path.join(dir, 'slices.mp4'))

    def save_density_dist(self, dir):
        plt.figure()
        plt.hist(self.volume.reshape(-1).cpu().numpy(), 100, range=[0,1], density=True, label = 'density distribution')
        plt.legend()
        plt.savefig(os.path.join(dir, 'density_dist.png'))

class Projector:
    def __init__(self, scene):
        self.scene = scene
        self.poses = None
        self.projections = None

    def pose(self, alpha, a = 2.):
        # Converting from degrees to radians
        alpha = np.deg2rad(alpha)
        # Rotation matrix around the y axis
        R = torch.tensor([[np.cos(alpha), 0, np.sin(alpha)],
                          [0, 1, 0],
                          [-np.sin(alpha), 0, np.cos(alpha)]], 
                          dtype=torch.float)
        # Translation matrix
        T = torch.tensor([[a*np.sin(alpha), 0, a*np.cos(alpha)]], dtype=torch.float)
        # Pose matrix
        pose = torch.cat((R, T.t()), dim = -1)
        last_line = torch.tensor([[0, 0, 0, 1]], dtype=torch.float)
        pose = torch.cat((pose, last_line), dim = 0)
        return pose

    def generate_projections(self):
        self.projections = []
        self.poses = []
        angles = torch.linspace(0, 359, 360)
        res = self.scene.res
        self.scene.volume = self.scene.volume.to(device)
        for i, alpha in tqdm(enumerate(angles)):
            pose = self.pose(alpha).to(device)
            
            ## First create a meshgrid for XY indexing
            # Note here indexing "xy"
            xx, yy = torch.meshgrid(
                torch.linspace(-1, 1, res).to(device),
                torch.linspace(-1, 1, res).to(device),
                indexing = "xy"
            )
            # At each XY coordinate we'll add z = 0
            zz = torch.zeros_like(xx).to(device)
            # Stack everything as coordinates and perform rotation translation with the pose
            coordinates = torch.cat([xx.unsqueeze(-1), yy.unsqueeze(-1), zz.unsqueeze(-1)], dim = -1).to(device)
            # Note the format of the matrix multiplication to be in accordance with broadcasting
            origins = coordinates@pose[:3,:3].t() + pose[:3,3]
            ## Orthographic setup, so need of only one direction
            ## [0,0,-1] since we are travelling in negative Z direction
            ## And then mutliplying by the rotation matrix aroung the Y axis
            direction = pose[:3, :3] @ torch.tensor([0., 0., -1.], device=device)
            ## Generate depth values along the ROTATED Z direction and recover the query points
            depth_values = torch.linspace(0, 4, 4*res).to(device)
            # Here the shapes are a bit tricky but the reader can figure it out :)
            query_points = origins[..., None, :] + direction[..., None, :] * depth_values[..., :, None]
            ## Careful here
            ## torch.grid_sample has a weird coorindate convention
            ## see https://discuss.pytorch.org/t/surprising-convention-for-grid-sample-coordinates/79997
            samples = torch.nn.functional.grid_sample(self.scene.volume.unsqueeze(0).unsqueeze(0), query_points.unsqueeze(0), align_corners=True, padding_mode='zeros').squeeze()
            dz = 1./res ## 4/(4*res)
            projection = torch.exp(-torch.sum(samples, dim=-1)*dz)
            self.projections.append(projection.cpu())
            self.poses.append(pose.cpu().numpy())

    def save_projections_poses(self, dir):
        path = os.path.join(dir, 'projections/')
        os.makedirs(path, exist_ok=True)
        if self.projections is None or self.poses is None:
            print('Error, the projections should be generated before saving')

        frames = []
        for i, projection in tqdm(enumerate(self.projections)):
            # We don't take the transpose because of the 'xy' meshgrid convention
            frame = (255.*projection).numpy().astype('uint8')
            save_image(frame, os.path.join(path, '{:04d}.png').format(i))
            frames.append(frame)
        save_video(frames, os.path.join(dir, 'projections.mp4'))
        ## saving the poses here
        pose_file = open(os.path.join(path, 'poses.npz'), 'wb')
        np.savez(pose_file, np.array(self.poses))
        


if __name__ == "__main__":
    print('Generating new dataset')
    torch.manual_seed(368)
    scene = Scene()
    scene.build()
    projector = Projector(scene)
    projector.generate_projections()

    ## Saving volume - Projections - Poses in a specified file
    dir = 'data/synthetic/'
    os.makedirs(dir, exist_ok=True)

    scene.save_volume(dir)
    scene.save_density_dist(dir)
    projector.save_projections_poses(dir)
    print('Done')