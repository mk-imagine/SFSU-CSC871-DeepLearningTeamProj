
def sort_projections(source, projections) -> torch.Tensor:
    return torch.einsum('bcn,cp->bpn', source, projections).sort()[0]

proj_n = 32
height, width = content_img.shape[-2:]
# Random matrix with image height and proj_n width for the images to be
# projected onto
projection = F.normalize(torch.randn(height, proj_n).to(device), dim = 0)
input_proj = sort_projections(content_img, projection)
target_proj = sort_projections(style_img, projection)
target_interpolated = F.interpolate(target_proj, height, mode = "nearest")
input_proj - target_interpolated

