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

# def extract_model_layers(model: torch.nn.Module) -> torch.nn.Module:
#     return model.features.to(device).eval()

def extract_features(img: torch.Tensor, model: torch.nn.Module, model_layers: list[int] = [18, 25]):
    normalize = v2.Normalize(mean=[0.485, 0.456, 0.406], 
                             std=[0.229, 0.224, 0.225]
                             ).to(device)
    feature_map = normalize(img)
    feature_maps = []
    for i, nn_layer in enumerate(model[:max(model_layers) + 1]):
        feature_map = nn_layer(feature_map)
        if i in model_layers:
            feature_maps.append(feature_map.clone())
    return feature_maps

# Base Loss Class
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
    
class SlicedWassersteinLoss(Loss):
    def __init__(self, target: torch.Tensor, scalar: float = 2e-6, proj_n: float = 32):
        super().__init__()
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

class VincentLoss(Loss):
    ...

if __name__ == "__main__":
    from imagehandlerV2 import *

    script_dir = Path(__file__).resolve().parent
    style_image_dir = Path(script_dir, 'images/styles')
    content_image_dir = Path(script_dir, 'images/content')
    style_img_orig = load_image(Path(style_image_dir, "style.jpg"))
    style_img_512x250 = load_image(Path(style_image_dir, "style.jpg"), (512, 250))

    imshow(style_img_orig)
    imshow(style_img_512x250)