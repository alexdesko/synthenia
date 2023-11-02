import torch
import torch.nn.functional as F

import math

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class TVdensities():
    """
    Total variation regularization on the densities
    """
    def __init__(self, l):
        self.l = l

    def __call__(self, densities):
        '''
        Expects a density of shape (N,D,1)
        batch of N rays sampled
        with D depth values
        '''

        reg = torch.mean(torch.abs(densities[:,1:] - densities[:,:-1]))
        return self.l*reg
    

class TVfeatures():
    """
    Total variation regularization on the features
    """
    def __init__(self, l):
        self.l = l

    def __call__(self, features):
        '''
        Expects a density of shape (N,D,nf)
        batch of N rays sampled
        with D depth values
        nf being the number of features
        '''

        reg = torch.mean(torch.abs(features[:,1:] - features[:,:-1]))
        return self.l*reg

class PixelWiseNLM_1D_PNP():
    """
    Pixel wise non local means denoising Plug and Play fashion for 1D densities
    """
    def __init__(self, l, sigma):
        self.l = l
        self.sigma = sigma

    def __call__(self, densities):
        """
        Inputs:
        ------
        densities: torch.tensor
            Tensor of size [B,N] where B is the batch size and N is the
            number of points queried per ray

        Returns:
        ------
        denoised_densities: torch.tensor
            Tensor of same size as the input containing the densities
        """
        densities.unsqueeze_(-1)
        weights = self._weight_tensor(densities)
        denoised_densities = weights@densities
        return self.l*torch.mean(torch.abs((densities.squeeze_() - denoised_densities.squeeze())))

    def _weight_tensor(self, densities):
        """
        Internal function to compute the weight tensor for the denoising
        """
        weight_tensor = torch.exp( - (densities - densities.transpose(1,2))**2 / self.sigma**2)
        return weight_tensor / torch.sum(weight_tensor, dim=-1, keepdim=True)
    

class PatchWiseNLM_1D_PNP():
    """
    Patch wise non local means denoising Plug and Play fashion for 1D densities
    """
    def __init__(self, l, sigma, patchsize, patch_rand = False, threshold = 0., alpha = -1):
        self.l = l
        self.sigma = sigma
        self.alpha = alpha
        self.patchsize = patchsize
        self.patch_rand = patch_rand
        self.threshold = threshold
        self.gaussian_weights = self._gaussian_kernel(
            torch.arange(-self.patchsize//2, self.patchsize//2) + self.patchsize % 2
        ).to(device).unsqueeze(-1)

    def _gaussian_kernel(self, delta):
        """
        Internal function to compute the gaussian kernel
        """
        if self.alpha == -1:
            return torch.ones_like(delta)
        return torch.exp( - (delta)**2 / (2*self.alpha**2)) / (math.sqrt(2*math.pi*self.alpha))

    def __call__(self, densities):
        """
        Inputs:
        ------
        densities: torch.tensor
            Tensor of size [B,N] where B is the batch size and N is the
            number of points queried per ray

        Returns:
        ------
        denoised_densities: torch.tensor
            Tensor of same size as the input containing the densities
        """
        if self.patch_rand:
            rand_idx = torch.randperm(densities.nelement())
            weights = self._weight_tensor(densities, rand_idx)
            densities = densities.view(-1)[rand_idx].view(densities.size())
        else:
            rand_idx = None
            weights = self._weight_tensor(densities)
        denoised_densities = weights@densities.unsqueeze(-1)
        return self.l*torch.mean(torch.abs(densities - denoised_densities.squeeze()))

    def _weight_tensor(self, densities, rand_idx = None):
        """
        Internal function to compute the weight tensor for the denoising
        """
        padded_densities = F.pad(densities, (self.patchsize //2, self.patchsize - self.patchsize//2-1), mode='constant', value=0)
        unfolded_densities = padded_densities.unfold(-1, self.patchsize, 1)

        if rand_idx is not None:
            unfolded_densities = unfolded_densities.reshape(-1, self.patchsize)[rand_idx,:].reshape(unfolded_densities.size())
        ## CAREFULL, the factor 2 in the normalization is purposly missing
        weight_tensor = torch.exp(
            - ((unfolded_densities.unsqueeze(-1) - unfolded_densities.transpose(1,2).unsqueeze(1)).pow(2)*self.gaussian_weights).sum(dim=2) / (self.sigma**2)
        )
        weight_tensor = torch.relu(weight_tensor - self.threshold)
        return weight_tensor / torch.sum(weight_tensor, dim=-1, keepdim=True)
    
class PatchWiseNLM_3D_PnP():
    """
    Patch wise non local means denoising Plug and Play fashion for 3D densities
    """
    def __init__(self, sigma = .1):
        self.sigma = sigma
    
    def __call__(self, densities_in_patches):
        densities_in_patches.unsqueeze_(-1)
        weight_tensor = torch.exp( - (densities_in_patches - densities_in_patches.transpose(0,2)).norm(dim=1)**2 / self.sigma**2)
        weight_tensor = weight_tensor / torch.sum(weight_tensor, dim=-1, keepdim=True)
        denoised_densities = weight_tensor@densities_in_patches.squeeze_()
        return denoised_densities.squeeze()
 


if __name__ == "__main__":
    torch.manual_seed(0)
    #test = torch.ra,nd((10,1000))
    #reg = PatchWiseNLM_3D_PnP()
    #reg(test)
    #import sys
    #sys.exit(-1)
    test = torch.randint(2,size=(10,1000)) + .01*torch.rand(10,1000)
    reg = PatchWiseNLM_1D_PNP(10,.1, 6)
    reg1 = PixelWiseNLM_1D_PNP(10, .1)
    denoised_densities = reg(test)
    denoised_densities1 = reg1(test)
    import matplotlib.pyplot as plt
    plt.figure()
    import numpy as np
    y = np.arange(1000)
    plt.scatter(y, test.numpy()[0])
    plt.scatter(y, denoised_densities.numpy()[0], alpha=.5)
    plt.scatter(y, denoised_densities1.numpy()[0], alpha=.5)
    plt.show()