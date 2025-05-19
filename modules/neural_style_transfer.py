# modules/neural_style_transfer.py
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, models
from PIL import Image
import numpy as np
from utils.image_processing import load_image, save_image
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def perform_style_transfer(content_path, style_path, output_path, 
                         num_steps=300, style_weight=1e6, content_weight=1,
                         image_size=512):
    """
    Performs neural style transfer with proper image resizing and error handling.
    
    Args:
        content_path: Path to content image
        style_path: Path to style image
        output_path: Path to save output image
        num_steps: Number of optimization steps
        style_weight: Weight for style loss
        content_weight: Weight for content loss
        image_size: Size to resize both images to
        
    Returns:
        tuple: (success, output_path_or_error_message)
    """
    try:
        # 1. Load and resize images to same dimensions
        print("Loading and preprocessing images...")
        content_img = load_and_preprocess(content_path, image_size).to(device)
        style_img = load_and_preprocess(style_path, image_size).to(device)
        
        # Verify image dimensions match
        if content_img.shape != style_img.shape:
            return False, "Image dimensions mismatch after preprocessing"
        
        # 2. Initialize with content image
        input_img = content_img.clone().requires_grad_(True)
        
        # 3. Load VGG19 model
        print("Loading VGG19 model...")
        cnn = models.vgg19(pretrained=True).features.to(device).eval()
        
        # 4. Normalization parameters
        cnn_normalization_mean = torch.tensor([0.485, 0.456, 0.406]).to(device)
        cnn_normalization_std = torch.tensor([0.229, 0.224, 0.225]).to(device)
        
        # 5. Layer selection
        content_layers = ['conv_4']
        style_layers = ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']
        
        # 6. Build model with loss layers
        model, style_losses, content_losses = build_model_with_losses(
            cnn, cnn_normalization_mean, cnn_normalization_std,
            style_img, content_img, style_layers, content_layers
        )
        
        # 7. Set up optimizer
        optimizer = optim.LBFGS([input_img.requires_grad_()], lr=0.8)
        
        # 8. Run style transfer
        print("Starting style transfer...")
        run = [0]
        while run[0] <= num_steps:
            def closure():
                # Clamp pixel values
                input_img.data.clamp_(0, 1)
                
                optimizer.zero_grad()
                model(input_img)
                
                style_score = sum(sl.loss for sl in style_losses) * style_weight
                content_score = sum(cl.loss for cl in content_losses) * content_weight
                total_loss = style_score + content_score
                
                total_loss.backward()
                
                run[0] += 1
                if run[0] % 50 == 0:
                    print(f"Step {run[0]}: Style={style_score.item():.2f} "
                          f"Content={content_score.item():.2f}")
                
                return total_loss
            
            optimizer.step(closure)
        
        # Final processing
        with torch.no_grad():
            input_img.data.clamp_(0, 1)
        
        print(f"Saving result to {output_path}")
        save_image(input_img, output_path)
        return True, output_path
        
    except Exception as e:
        return False, f"Style transfer failed: {str(e)}"

def load_and_preprocess(image_path, size=512):
    """Load and preprocess image with consistent resizing"""
    transform = transforms.Compose([
        transforms.Resize(size),
        transforms.CenterCrop(size),  # Ensures exact size
        transforms.ToTensor()
    ])
    image = Image.open(image_path).convert('RGB')
    return transform(image).unsqueeze(0)  # Add batch dimension

def build_model_with_losses(cnn, mean, std, style_img, content_img, 
                          style_layers, content_layers):
    """Build model with content and style loss layers"""
    normalization = Normalization(mean, std).to(device)
    content_losses = []
    style_losses = []
    
    model = nn.Sequential(normalization)
    
    i = 0
    for layer in cnn.children():
        if isinstance(layer, nn.Conv2d):
            i += 1
            name = f'conv_{i}'
        elif isinstance(layer, nn.ReLU):
            name = f'relu_{i}'
            layer = nn.ReLU(inplace=False)
        elif isinstance(layer, nn.MaxPool2d):
            name = f'pool_{i}'
        elif isinstance(layer, nn.BatchNorm2d):
            name = f'bn_{i}'
        else:
            raise RuntimeError(f"Unrecognized layer: {layer.__class__.__name__}")
        
        model.add_module(name, layer)
        
        if name in content_layers:
            target = model(content_img).detach()
            content_loss = ContentLoss(target)
            model.add_module(f"content_loss_{i}", content_loss)
            content_losses.append(content_loss)
        
        if name in style_layers:
            target = model(style_img).detach()
            style_loss = StyleLoss(target)
            model.add_module(f"style_loss_{i}", style_loss)
            style_losses.append(style_loss)
    
    # Trim after last loss layer
    for i in range(len(model)-1, -1, -1):
        if isinstance(model[i], (ContentLoss, StyleLoss)):
            break
    return model[:i+1], style_losses, content_losses

class ContentLoss(nn.Module):
    def __init__(self, target):
        super().__init__()
        self.target = target.detach()
    
    def forward(self, input):
        self.loss = torch.mean((input - self.target)**2)
        return input

class StyleLoss(nn.Module):
    def __init__(self, target):
        super().__init__()
        self.target = self.gram_matrix(target).detach()
    
    def forward(self, input):
        G = self.gram_matrix(input)
        self.loss = torch.mean((G - self.target)**2)
        return input
    
    @staticmethod
    def gram_matrix(input):
        batch, channel, h, w = input.size()
        features = input.view(batch * channel, h * w)
        G = torch.mm(features, features.t())
        return G.div(batch * channel * h * w)

class Normalization(nn.Module):
    def __init__(self, mean, std):
        super().__init__()
        self.register_buffer('mean', mean.view(-1, 1, 1))
        self.register_buffer('std', std.view(-1, 1, 1))
    
    def forward(self, img):
        return (img - self.mean) / self.std