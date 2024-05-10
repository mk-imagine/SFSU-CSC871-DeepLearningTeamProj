from torchvision import models, transforms
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

def extract_model_layers(model: torch.nn.Module) -> torch.nn.Module:
    return model.features.to(device).eval()

def extract_features(image, model, model_layers = (18, 25)):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                     std=[0.229, 0.224, 0.225]
                                     ).to(device)
    feature = normalize(image)
    features = []
    for i, nnlayer in enumerate(model[:max(model_layers) + 1]):
        feature = nnlayer(feature)
        if i in model_layers:
            features.append(feature.clone())
    return features

def gram_matrix(input):
    a, channels, height, width = input.size()  
    features = input.view(a * channels, height * width)  
    G = torch.mm(features, features.t())  
    return G.div(a * channels * height * width)

if __name__ == "__main__":
    pass