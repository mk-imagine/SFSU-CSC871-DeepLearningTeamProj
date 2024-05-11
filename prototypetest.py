import copy
from pathlib import Path
from typing import Callable
from tqdm.auto import tqdm, trange

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.models as models

from imagehandler import *
from lossfunctions import *

device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

# desired size of the output image
imsize = 512 if torch.cuda.is_available() else 256 if torch.backends.mps.is_available() else 128

# Get the directory of the script file
script_dir = Path(__file__).resolve().parent

# Construct the paths to the image files
style_image_dir = Path(script_dir, 'images/styles')
content_image_dir = Path(script_dir, 'images/content')

# Load the images
style_img = load_image(Path(style_image_dir, "style.jpg"), (imsize, imsize))
content_img = load_image(Path(content_image_dir, "content.jpg"), (imsize, imsize))
assert style_img.size() == content_img.size(), "style and content images must be the same size"

# Importing the VGG 16 model
vgg16 = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1).features.to(device).eval()

content_layers_default = ['conv_4']
style_layers_default = ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']

def get_style_model_and_losses(pretrained_cnn: nn.Sequential,
                               content_img: torch.Tensor,
                               style_img: torch.Tensor,
                               C_loss: Loss,
                               S_loss: Loss,
                               content_layers: list[str] = content_layers_default,
                               style_layers: list[str] = style_layers_default
                               ) -> tuple[nn.Sequential, list[torch.Tensor], list[torch.Tensor]]:
    
    pretrained_cnn = copy.deepcopy(pretrained_cnn)

    normalize = v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]).to(device)

    content_losses = []
    style_losses = []

    model = nn.Sequential(normalize)

    i = 0
    for layer in pretrained_cnn.children():
        if isinstance(layer, nn.Conv2d):
            i += 1
            name = 'conv_{}'.format(i)
        elif isinstance(layer, nn.ReLU):
            name = 'relu_{}'.format(i)
            layer = nn.ReLU(inplace=False)
        elif isinstance(layer, nn.MaxPool2d):
            name = 'pool_{}'.format(i)
        elif isinstance(layer, nn.BatchNorm2d):
            name = 'bn_{}'.format(i)
        else:
            raise RuntimeError('Unrecognized layer: {}'.format(layer.__class__.__name__))

        model.add_module(name, layer)

        if name in content_layers:
            target = model(content_img).detach()
            content_loss = C_loss(target)
            model.add_module("content_loss_{}".format(i), content_loss)
            content_losses.append(content_loss)

        if name in style_layers:
            target_feature = model(style_img).detach()
            style_loss = S_loss(target_feature)
            model.add_module("style_loss_{}".format(i), style_loss)
            style_losses.append(style_loss)

    for i in range(len(model) - 1, -1, -1):
        if isinstance(model[i], ContentLoss) or isinstance(model[i], GramLoss):
            break

    model = model[:(i + 1)]

    return model, style_losses, content_losses

def get_input_optimizer(input_img):
    optimizer = optim.LBFGS([input_img.requires_grad_()])
    return optimizer

def run_style_transfer(pretrained_cnn, content_img, style_img, input_img, num_steps=300,
                       style_weight=1000000, content_weight=1):
    
    print('Building the style transfer model..')

    model, style_losses, content_losses = get_style_model_and_losses(pretrained_cnn,
                                                                     content_img,
                                                                     style_img,
                                                                     ContentLoss,
                                                                     GramLoss,)
    
    optimizer = get_input_optimizer(input_img)

    print('Optimizing..')
    run = [0]

    def closure():
        input_img.data.clamp_(0, 1)

        optimizer.zero_grad()
        model(input_img)
        style_score = 0
        content_score = 0

        for sl in style_losses:
            style_score += sl.loss
        for cl in content_losses:
            content_score += cl.loss

        style_score *= style_weight
        content_score *= content_weight

        loss = style_score + content_score
        loss.backward()

        run[0] += 1
        run.append((run[0], 
                    content_score.item(), 
                    style_score.item(), 
                    input_img.data.clamp_(0, 1) if run[0] % 5 == 0 else None))

        return style_score + content_score

    pbar = tqdm(total = num_steps / optimizer.state_dict()["param_groups"][0]["max_iter"])
    while run[0] < num_steps:
        optimizer.step(closure)
        pbar.set_description(f"Content Loss: {run[run[0]][1]:4f}  Style Loss: {run[run[0]][1]:4f}")
        pbar.update()

    input_img.data.clamp_(0, 1)

    return input_img, run

output, run = run_style_transfer(vgg16, content_img, style_img, content_img, num_steps=60)

plt.figure()
imshow(output, title='Output Image')

plt.ioff()
plt.show()
