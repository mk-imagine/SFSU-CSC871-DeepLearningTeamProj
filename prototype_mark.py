from pathlib import Path
from tqdm.auto import tqdm

import torch
import torch.nn as nn
import torch.optim as optim

from torchvision.models import vgg19, VGG19_Weights
from torchvision.utils import save_image

from imagehandlerV2 import *
from lossfunctionsV2 import *

CONTENT_LAYERS_DEFAULT = ['conv_4']
STYLE_LAYERS_DEFAULT = ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']

def get_conv_layers(model):
    conv2d_count = 0
    layer_names = []
    for layer in model.children():
        if isinstance(layer, nn.Conv2d):
            conv2d_count += 1
            name = 'conv_{}'.format(conv2d_count)
            layer_names.append(name)
        elif isinstance(layer, nn.ReLU):
            name = 'relu_{}'.format(conv2d_count)
            # in-place does not work nice with our loss functions
            layer = nn.ReLU(inplace=False)
        elif isinstance(layer, nn.MaxPool2d):
            name = 'pool_{}'.format(conv2d_count)
        elif isinstance(layer, nn.BatchNorm2d):
            name = 'bn_{}'.format(conv2d_count)
        else:
            raise RuntimeError('Unrecognized layer: {}'.format(layer.__class__.__name__))
    return layer_names

def get_style_model_with_losses(model: nn.Sequential,
                                content_img: torch.Tensor,
                                style_img: torch.Tensor,
                                c_loss_func: Loss = MSELoss,
                                s_loss_func: Loss = GramMatrixLoss,
                                content_layers: list[str] = CONTENT_LAYERS_DEFAULT,
                                style_layers: list[str] = STYLE_LAYERS_DEFAULT,
                                model_normalization: \
                                    tuple[list[float], list[float]] \
                                        = ([0.485, 0.456, 0.406],[0.229, 0.224, 0.225]),
                                **kwargs
                                ) -> tuple[nn.Sequential, list[float], list[float]]:

    content_losses = []
    style_losses = []

    model_mean, model_std = model_normalization
    model = nn.Sequential(
        v2.Normalize(mean = model_mean, std = model_std).to(device)
    )

    conv2d_count = 0
    for layer in pretrained_cnn.children():
        if isinstance(layer, nn.Conv2d):
            conv2d_count += 1
            name = 'conv_{}'.format(conv2d_count)
        elif isinstance(layer, nn.ReLU):
            name = 'relu_{}'.format(conv2d_count)
            # in-place does not work nice with our loss functions
            layer = nn.ReLU(inplace=False)
        elif isinstance(layer, nn.MaxPool2d):
            name = 'pool_{}'.format(conv2d_count)
        elif isinstance(layer, nn.BatchNorm2d):
            name = 'bn_{}'.format(conv2d_count)
        else:
            raise RuntimeError('Unrecognized layer: {}'.format(layer.__class__.__name__))
        
        model.add_module(name, layer)

        # add content loss layers specified
        if name in content_layers:
            # run the content image through the layers up to this point in the model
            target_c = model(content_img).detach()
            # calculate loss between the content image and the target image and
            # add the module
            content_loss = c_loss_func(target_c)
            model.add_module(f"c_loss_{conv2d_count}", content_loss)
            content_losses.append(content_loss)

        # add style loss layers specified (same format at content_layers)
        if name in style_layers:
            target_s = model(style_img).detach()
            if isinstance(s_loss_func, SlicedWassersteinLoss):
                style_loss = s_loss_func(target_s, kwargs["scalar"], kwargs["proj_n"])
            else:
                style_loss = s_loss_func(target_s)
            model.add_module("s_loss_{}".format(conv2d_count), style_loss)
            style_losses.append(style_loss)

    # truncate layers after last content or style layers
    for i in range(len(model) - 1, -1, -1):
        if isinstance(model[i], Loss):
            break
    
    model = model[: (i + 1)]

    return model, content_losses, style_losses

def get_input_optimizer(input_img: torch.Tensor) -> optim.Optimizer:
    # input image requires a gradient because we running our model "backwards",
    # so we are training the input image to minimize the content and style losses.
    optimizer = optim.LBFGS([input_img])
    return optimizer

def run_style_transfer(model: nn.Sequential,
                       content_img: torch.Tensor,
                       style_img: torch.Tensor,
                       output_img: torch.Tensor,
                       c_loss_func: Loss,
                       s_loss_func: Loss,
                       epochs: int = 300,
                       content_weight: int = 1,
                       style_weight: int = 1000000,
                       content_layers: list[str] = CONTENT_LAYERS_DEFAULT,
                       style_layers: list[str] = STYLE_LAYERS_DEFAULT,
                       model_normalization: \
                        tuple[list[float], list[float]] \
                            = ([0.485, 0.456, 0.406],[0.229, 0.224, 0.225]),
                       **kwargs):
    
    print("Building Style Transfer Model")
    model, content_losses, style_losses = get_style_model_with_losses(
        model, content_img, style_img, c_loss_func, s_loss_func, content_layers, 
        style_layers, model_normalization, **kwargs
    )

    # Optimize the output img (e.g. mixed style image) and not the model parameters
    output_img.requires_grad_(True)

    # Set model into evaluation mode, so that dropout and batch normalization
    # behave correctly.
    model.eval()
    model.requires_grad_(False)

    optimizer = get_input_optimizer(output_img)

    print("Begin Optimization")

    run = [0]

    pbar = tqdm(total = epochs / optimizer.state_dict()["param_groups"][0]["max_iter"])
    while run[0] <= epochs:

        def closure():
            # clip values of tensor to remain between (0, 1)
            with torch.no_grad():
                output_img.clamp_(0, 1)

            optimizer.zero_grad()
            model(output_img)
            content_score = 0
            style_score = 0

            for cl in content_losses:
                content_score += cl.loss
            for sl in style_losses:
                style_score += sl.loss

            style_score *= style_weight
            content_score *= content_weight

            loss = style_score + content_score
            loss.backward()

            run[0] += 1

            run.append((run[0], 
                    content_score.item(), 
                    style_score.item(), 
                    output_img.clone() if run[0] % 20 == 0 else None))

            if run[0] % optimizer.state_dict()["param_groups"][0]["max_iter"] == 0:
                print(f"Run: {run[0]}\nStyle Loss: {style_score.item()}   Content Loss: {content_score.item()}\n")

            return style_score + content_score
        
        if not isinstance(run[run[0]], int) and abs(run[run[0]][2] - run[run[0]][2]) > 5000:
                break
        
        optimizer.step(closure)
        pbar.update()

    with torch.no_grad():
        output_img.clamp_(0, 1)

    return output_img, run

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() \
                        else "mps" if torch.backends.mps.is_available() \
                            else "cpu")

    torch.set_default_device(device)

    img_size = 512 if torch.cuda.is_available() \
        else 256 if torch.backends.mps.is_available() \
            else 128


    # Get the directory of the script file
    script_dir = Path(__file__).resolve().parent

    # Construct the paths to the image files
    content_image_dir = Path(script_dir, 'images/content')
    style_image_dir = Path(script_dir, 'images/style')
    output_image_dir = Path(script_dir, 'images/combined')
    style_img = "picasso.jpg"
    content_img = "dancing.jpg"

    # Load the images
    content_img_t = load_image(Path(content_image_dir, content_img), (img_size, img_size))
    style_img_t = load_image(Path(style_image_dir, style_img), (img_size, img_size))
    output_img_t = content_img_t.clone()
    assert style_img_t.size() == content_img_t.size(), "style and content images must be the same size"

    # Load Pretrained Model
    pretrained_cnn = vgg19(weights=VGG19_Weights.DEFAULT).features.eval()

    vgg19_imgnet_norms = ([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])

    vgg19_conv2d_layer_names = get_conv_layers(pretrained_cnn)

    # GramMatrixLoss: content_weight = 1, style_weight = 1000000

    # SlicedWasserSteinLoss: content_weight = 1, style_weight = 100, scalar =
    # 2e-6, proj_n = 32, epochs = 100
    
    c_layers = ['conv_4']
    s_layers = ['conv_4', 'conv_8', 'conv_12', 'conv_14', 'conv_16']

    

    epochs = 200
    output_img_t = content_img_t.clone()
    gm_output, gm_metadata = run_style_transfer(pretrained_cnn, content_img_t, style_img_t, output_img_t,
                                              MSELoss, GramMatrixLoss,
                                              epochs, content_weight=1, style_weight = 1000000,
                                              content_layers = c_layers,
                                              style_layers = s_layers
                                              )
    
    epochs = 400
    output_img_t = content_img_t.clone()
    swl_output, swl_metadata = run_style_transfer(pretrained_cnn, content_img_t, style_img_t, output_img_t,
                                              MSELoss, SlicedWassersteinLoss,
                                              epochs, content_weight=1, style_weight = 100,
                                              content_layers = c_layers,
                                              style_layers = s_layers,
                                              scalar = 2e-5, proj_n = 32
                                              )
    
    imshow(content_img_t)
    imshow(style_img_t)
    imshow(gm_output)
    imshow(swl_output)

    save_image(gm_output, Path(output_image_dir, "gm_01.jpg"))
    save_image(swl_output, Path(output_image_dir, "swl_01.jpg"))