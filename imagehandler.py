from PIL import Image
import matplotlib.pyplot as plt
import torchvision.transforms as v2
import torch

device = torch.device("cuda" if torch.cuda.is_available() \
                      else "mps" if torch.backends.mps.is_available() \
                      else "cpu")

def load_image(image_name: str, size: tuple[int, int] = None) -> torch.Tensor:
    # normalize = v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]).to(device)
    image = Image.open(image_name)
    if size is None:
        transform = v2.Compose([v2.ToTensor()])
    else:
        height, width = size
        transform = v2.Compose([v2.Resize((height, width)), v2.ToTensor()])
    image = transform(image)  # adds a batch dimension to fit network's expectations
    return image.to(device, torch.float)

def imshow(tensor, title=None):
    image_t = tensor.cpu().clone()  # clone tensor so we can't change it inadvertently
                                               # & removes the batch dimension
    transform_to_image = v2.ToPILImage()
    plt.imshow(transform_to_image(image_t))
    if title is not None:
        plt.title(title)
    plt.pause(0.001)

if __name__ == "__main__":
    from pathlib import Path

    style_img = "picasso.jpg"
    content_img = "dancing.jpg"

    # Get the directory of the script file
    script_dir = Path(__file__).resolve().parent

    # Construct the paths to the image files
    style_image_dir = Path(script_dir, 'images/style')
    content_image_dir = Path(script_dir, 'images/content')

    content_img_orig = load_image(Path(content_image_dir, content_img))
    content_img_100x300 = load_image(Path(content_image_dir, content_img), (100, 300))

    style_img_orig = load_image(Path(style_image_dir, style_img))
    style_img_512x250 = load_image(Path(style_image_dir, style_img), (512, 250))

    imshow(content_img_orig)
    imshow(content_img_100x300)
    imshow(style_img_orig)
    imshow(style_img_512x250)

    assert content_img_orig.size() == style_img_orig.size(), \
        "Images must be the same size"