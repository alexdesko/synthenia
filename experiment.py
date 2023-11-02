import torch
from dataset.dataset import CustomDataset
from rendering.encoder import PositionalEncoder
from rendering.raymarcher import Raymarcher
from rendering.pointsampler import PointSampler, PatchSampler3D
from rendering.renderer import Renderer
from models.fields import LargeDensity
from regularization.regularization import TVdensities, TVfeatures, PatchWiseNLM_1D_PNP, PatchWiseNLM_3D_PnP
import os
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

import logging
import wandb

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Experiment():
    def __init__(self, args):
        """
        Args:
            args: argparse object containing all the parameters for the experiment
        """
        
        ## Loading dataset
        self.datadir = args.datadir
        logging.info('Creating dataset')
        self.dataset = CustomDataset(self.datadir)
        logging.info('Dataset succesfully created')
                    

        ## Misc training parameters
        self.batch_size = args.batch_size
        self.n_iter = args.n_iter
        self.log_every_train = args.log_every_train
        self.log_every_val = args.log_every_val


        ## Instanciating modules for 3D reconstruction
        
        # Ray marcher
        self.ray_marcher = Raymarcher()

        # Point sampler
        self.point_sampler = PointSampler(
            near = args.min_depth,
            far = args.max_depth,
            n_samples_per_ray = args.n_samples_per_ray,
            randomized= args.randomized
        )
        
        # Sinusoidal positional encoder
        self.positional_encoder = PositionalEncoder(
            n_encoding_functions = args.n_encoding_func
        )
        
        # Density network

        self.densitynet = LargeDensity(
            input_dim = 3*2*args.n_encoding_func + 3,
            feature_dim = args.net_feature_dim,
        )

        # Renderer
        self.renderer = Renderer(
            self.point_sampler,
            self.positional_encoder,
            self.densitynet,
            self.ray_marcher,
        ).to(device)
        logging.info('Renderer succesfully created')
        
        # Regularization - Loss function
        if args.reg_densities:
            if args.reg_type_densities == 'TV':
                self.regularizer_densities = TVdensities(args.l_densities_TV)
            elif args.reg_type_densities == 'NLM':
                self.regularizer_densities = PatchWiseNLM_1D_PNP(
                    args.l_densities_NLM,
                    args.sigma,
                    args.patch_size,
                    args.patch_rand,
                    args.threshold,
                    args.alpha
                )
            elif args.reg_type_densities == 'Both':
                self.regularizer_densities= []
                self.regularizer_densities.append(TVdensities(args.l_densities_TV))
                self.regularizer_densities.append(PatchWiseNLM_1D_PNP(
                    args.l_densities_NLM,
                    args.sigma,
                    args.patch_size,
                    args.patch_rand,
                    args.threshold,
                    args.alpha
                ))
            else:
                self.regularizer_densities = None
        else:
            self.regularizer_densities = None

        ## 3D regularization
        if args.reg_densities_3D:
            self.regularizer_densities_3D = PatchWiseNLM_3D_PnP(sigma=.1)
            self.patch_sampler_3D = PatchSampler3D(512, args.patch_size_3D)
            self.l_densities_3D = args.l_densities_3D
        else:
            self.regularizer_densities_3D = None
        # Optimizer and loss function
        self.optimizer = torch.optim.Adam(self.renderer.parameters(), lr = args.lr)

        self.lossstr = args.lossfunc
        if self.lossstr == 'l1':
            self.lossfun = torch.nn.L1Loss()
            self.lossfun_val = torch.nn.functional.l1_loss
        elif self.lossstr == 'mse':
            self.lossfun = torch.nn.MSELoss()
            self.lossfun_val = torch.nn.functional.mse_loss
    
    def run(self):
        logging.info('Begin training')
        test_gt, _, _, _ = self.dataset.get_test_data()
        slice_gt = self.dataset.get_slice(127)
        wandb.log({'test image ground truth': wandb.Image(test_gt),
                   'central slice ground truth': wandb.Image(slice_gt)})
        for iteration in range(self.n_iter):
            loss = self.training_step()
            if (iteration+1) % self.log_every_train == 0:
                logging.info('\nTraining step at iteration ' + str(iteration))
                logging.info('train loss - {} '.format(self.lossstr) + str(loss.item()))
                wandb.log({'train loss - {}'.format(self.lossstr): loss}, step=iteration)

            if (iteration+1) % self.log_every_val == 0:
                logging.info('\nValidation step at iteration ' + str(iteration))
                prediction, depth, image, test_loss, volume, volume_gt = self.validation_step()
                
                slice = volume[..., 127]
                slice_gt = volume_gt[..., 127]
                
                wandb.log({'predicted image': wandb.Image(prediction),
                           'depth map': wandb.Image(depth),
                           'predicted central slice': wandb.Image(slice)}, step=iteration)
                prediction.clamp_(0.0, 1.0)
                depth.clamp_(0.0, 1.0)
                slice.clamp_(0.0, 1.0)
                
                psnr = peak_signal_noise_ratio(prediction.cpu().numpy(), image.cpu().numpy())
                ssim = structural_similarity(prediction.cpu().numpy(), image.cpu().numpy())
                
                psnr_slice = peak_signal_noise_ratio(slice.numpy(), slice_gt.numpy())
                ssim_slice = structural_similarity(slice.numpy(), slice_gt.numpy())

                psnr_volume = peak_signal_noise_ratio(volume.numpy(), volume_gt.numpy())
                ssim_volume = structural_similarity(volume.numpy(), volume_gt.numpy())

                wandb.log({
                    'validation loss - {}'.format(self.lossstr): test_loss,
                    'psnr': psnr,
                    'ssim': ssim,
                    'psnr slice': psnr_slice,
                    'ssim slice': ssim_slice,
                    'psnr volume': psnr_volume,
                    'ssim volume': ssim_volume
                }, step=iteration)
                logging.info('validation loss - {} '.format(self.lossstr) +  str(test_loss.item()))
                logging.info('psnr ' + str(psnr))
                logging.info('ssim ' + str(ssim))

    def training_step(self):
        """
        Training step
        """
        self.point_sampler.set_randomized(True)
        self.renderer.train()
        self.optimizer.zero_grad()
        origins, directions, gt = self.dataset.get_random_batch_of_rays_and_pixels(self.batch_size)
        # missing_origins, missing_directions = self.dataset.get_random_batch_of_rays_and_pixels_missing(self.batch_size)

        # origins_full = torch.cat((origins, missing_origins), dim=0)
        # directions_full = torch.cat((directions, missing_directions), dim=0)

        ## prediction
        prediction, _, densities = self.renderer(origins.to(device), directions.to(device))
        prediction = prediction[:self.batch_size]

        loss = self.lossfun(gt, prediction.cpu())

        ## 3D regularization
        if self.regularizer_densities_3D is not None:
            query_points_3Dreg = self.patch_sampler_3D.samples_patches_3D().to(device)
            n_points, n_patch = query_points_3Dreg.shape[0:2]
            densities_patch, _ = self.renderer.compute_densities(query_points_3Dreg.view(-1,3))
            densities_patch = densities_patch.reshape(n_points, -1)
            denoised_densities_patch = self.regularizer_densities_3D(densities_patch)
            reg = self.l_densities_3D*torch.mean(torch.abs(densities_patch[:, n_patch // 2] - denoised_densities_patch[:, n_patch // 2]))
            loss += reg.cpu()

        if self.regularizer_densities is not None:
            if isinstance(self.regularizer_densities, list):
                reg = 0
                for regularizer in self.regularizer_densities:
                    reg += regularizer(densities)
            else:
                reg = self.regularizer_densities(densities)
            loss += reg.cpu()

        loss.backward()
        self.optimizer.step()
        return loss.detach().cpu().numpy()

    @torch.no_grad()
    def validation_step(self):
        self.point_sampler.set_randomized(False)
        self.renderer.eval()

        ## Projection
        image, pose, origin, direction = self.dataset.get_test_data()
        prediction, depth = self.renderer.forward_batch(
                                    origin.view(-1,3).to(device),
                                    direction.view(-1,3).to(device), 
                                    self.batch_size)
        depth = depth.clamp(self.point_sampler.near, self.point_sampler.far)
        prediction = prediction.reshape(image.shape)
        depth = depth.reshape(image.shape)
        loss = self.lossfun_val(prediction.cpu(), image)

        ## Volume
        density_field = self.renderer.grid_evaluation(256)
        volume_gt = self.dataset.slices.permute(1, 2, 0)

        return  prediction.cpu(), depth.cpu(), image.cpu(), loss.cpu(), density_field.cpu(), volume_gt.cpu()

    def save_checkpoint(self, dir):
        os.makedirs(os.path.join(dir), exist_ok=True)
        checkpoint = self.renderer.state_dict()
        torch.save(checkpoint, os.path.join(dir, 'final_checkpoint.pt'))

    

