from pathlib import Path
from typing import Callable
from abc import abstractmethod

import torchvision
from torchvision.transforms import v2
import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() \
                      else "mps" if torch.backends.mps.is_available() \
                      else "cpu")


"""
Base Loss Class
"""
class Loss(nn.Module):
    ...

class MSELoss(Loss):
    def __init__(self, target: torch.Tensor):
        super(MSELoss, self).__init__()
        # detach target feature map content from gradient computation tree at a
        # particular layer.
        self.target = target.detach()

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        # distance of the input feature map from the target (L2 Norm)
        self.loss = F.mse_loss(input, self.target)
        return input


"""
Standard Gram Matrix Loss

Reference:
Gatys, Leon & Ecker, Alexander & Bethge, Matthias. (2015). A Neural Algorithm of
Artistic Style. arXiv. 10.1167/16.12.326.
"""
class GramMatrixLoss(Loss):

    def __init__(self, target: torch.Tensor):
        super(GramMatrixLoss, self).__init__()
        self.target = self.__loss(target).detach()

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        # distance of the input feature map from the target (L2 Norm)
        self.loss = F.mse_loss(self.__loss(input), self.target)
        return input
    
    def __loss(self, input: torch.Tensor) -> torch.Tensor:
        # batches, channels, height, width = input.shape
        channels, height, width = input.shape

        # reshape feature map to 2-d matrix
        # feature_map = input.view(batches * channels, height * width)  
        feature_map = input.view(channels, height * width)  

        # compute gram product
        gram_product = torch.mm(feature_map, feature_map.t())

        # must divide the gram product by the number of features to normalize the
        # values since a large number of layers since the values scale by the
        # number of dimensions
        # return gram_product.div(batches * channels * height * width)
        return gram_product.div(channels * height * width)


"""
Sliced Wasserstein Loss

Reference:
E. Heitz, K. Vanhoey, T. Chambon and L. Belcour, "A Sliced Wasserstein Loss
    for Neural Texture Synthesis," 2021 IEEE/CVF Conference on Computer
    Vision and Pattern Recognition (CVPR), Nashville, TN, USA, 2021, pp.
    9407-9415, doi: 10.1109/CVPR46437.2021.00929. keywords: {Training;Neural
    networks;Computer architecture;Production;Tools;Feature
    extraction;Complexity theory},
"""  
class SlicedWassersteinLoss(Loss):

    def __init__(self, target: torch.Tensor, scalar: float = 2e-6, proj_n: float = 32):
        super(SlicedWassersteinLoss, self).__init__()
        self.target = target.detach()
        self.scalar = scalar
        self.proj_n = proj_n

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        self.loss = self.__loss(input, self.target) * self.scalar
        return input

    def __loss(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        height, width = input.shape[-2:]
        # proj_n number of random projections
        projection = F.normalize(torch.randn(height, self.proj_n).to(device), dim = 0)
        # Project and then sort the input and target to the projection shape
        input_proj = self.__sort_projections(input, projection)
        target_proj = self.__sort_projections(target, projection)
        # target_interpolated = F.interpolate(target_proj, width, mode = "nearest")
        return (input_proj - target_proj).square().sum()

    def __sort_projections(self, source, projection) -> torch.Tensor:
        return torch.einsum('bcn,cp->bpn', source, projection).sort()[0]


"""
L2 Wasserstein Gaussian Loss (Optimal Transport)

Reference:
https://github.com/VinceMarron/style_transfer/blob/master/style-transfer-theory.pdf
"""
class L2WassersteinGaussianLoss(Loss):

    def __init__(self, target: torch.Tensor, scalar: float = 2e-6):
        super(L2WassersteinGaussianLoss, self).__init__()
        self.target = target.detach()
        self.scalar = scalar

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        mu_target, cov_target = self.__calc_mean_cov(self.target)
        mu_input, cov_input = self.__calc_mean_cov(input)
        self.loss = self.__loss(mu_target, cov_target, mu_input, cov_input) * self.scalar
        return input

    def __loss(self, mu_target: torch.Tensor, cov_target: torch.Tensor, 
               mu_input: torch.Tensor, cov_input: torch.Tensor) -> torch.Tensor:
        
        eig_vals_t, eig_vecs_t = torch.linalg.eigh(cov_target.cpu())
        eig_vals_t = eig_vals_t.to(device)
        cov_sqrt_eig_main_diag = torch.diag(torch.sqrt(eig_vals_t.clamp(0)))
        a_cov = torch.sum(eig_vals_t.clamp(0))

        a_cov_in = torch.sum(torch.linalg.eigvalsh(cov_input.cpu()).clamp(0))
        delta_mu_sq = torch.mean((mu_input - mu_target)**2)
        cov_prod = torch.matmul(torch.matmul(cov_sqrt_eig_main_diag,cov_input),cov_sqrt_eig_main_diag)
        var_overlap = torch.sum(torch.sqrt(torch.linalg.eigvalsh(cov_prod.cpu()).clamp(0.1))) # Need nonzero to retrieve eigenvalues
        distance = delta_mu_sq+a_cov+a_cov_in.to(device)-2*var_overlap.to(device)
        return distance
    
    def __calc_mean_cov(self, t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        channel, height, width = t.shape
        # reshape to 2d tensor
        t = t.view(channel, height*width)
        # calculate mean across each channel (row)
        mu = t.mean(dim = -1, keepdim = True)
        # subtract the channel means from tensor, then multiply with its transpose
        cov = torch.matmul(t - mu, torch.transpose(t - mu, -1, -2))
        return mu, cov

if __name__ == "__main__":
    from imagehandler import *

    script_dir = Path(__file__).resolve().parent
    style_image_dir = Path(script_dir, 'images/styles')
    content_image_dir = Path(script_dir, 'images/content')
    style_img_orig = load_image(Path(style_image_dir, "style.jpg"))
    style_img_512x250 = load_image(Path(style_image_dir, "style.jpg"), (512, 250))

    imshow(style_img_orig)
    imshow(style_img_512x250)