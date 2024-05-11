from pathlib import Path
from typing import Callable

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

# class ContentLoss():
#     def __init__(self, target_img: torch.Tensor, target_layers: list[int] = [18, 25]):
#         self.target_layers = target_layers
#         with torch.no_grad():
#             self.target_features = extract_features(target_img, target_layers)
#     def __call__(self, content_img):
#         features_layer_pairs = zip(extract_features(content_img, self.target_layers), self.target_features)
#         layers_mse = [(content_features - target_features).pow(2).mean() for content_features, target_features in features_layer_pairs]
#         return sum(layers_mse)

class Loss(nn.Module):
    def __init__(self, target: torch.Tensor):
        super(nn.Module, self).__init__()

class ContentLoss(Loss):
    def __init__(self, target: torch.Tensor):
        super(ContentLoss, self).__init__()
        self.target = target.detach()

    def forward(self, input: torch.Tensor):
        self.loss = F.mse_loss(input, self.target)
        return input
    
class GramLoss(Loss):
    def __init__(self, target: torch.Tensor):
        super(GramLoss, self).__init__()
        self.target = self.__gram_loss(target).detach()

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        self.loss = F.mse_loss(self.__gram_loss(input), self.target)
        return input
    
    def __gram_loss(self, input: torch.Tensor) -> torch.Tensor:
        channels, height, width = input.size()
        features = input.view(channels, height * width)  
        G = torch.mm(features, features.t())  
        return G.div(channels * height * width)

if __name__ == "__main__":
    from imagehandler import *

    script_dir = Path(__file__).resolve().parent
    style_image_dir = Path(script_dir, 'images/styles')
    content_image_dir = Path(script_dir, 'images/content')
    style_img_orig = load_image(Path(style_image_dir, "style.jpg"))
    style_img_512x250 = load_image(Path(style_image_dir, "style.jpg"), (512, 250))

    imshow(style_img_orig)
    imshow(style_img_512x250)