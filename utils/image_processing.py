from PIL import Image
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt

def load_image(image_path, max_size=512, shape=None):
    """Load and preprocess image"""
    image = Image.open(image_path).convert('RGB')
    
    if max(image.size) > max_size:
        size = max_size
    else:
        size = max(image.size)
    
    if shape is not None:
        size = shape
    
    in_transform = transforms.Compose([
        transforms.Resize(size),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])
    
    image = in_transform(image).unsqueeze(0)
    return image

def save_image(tensor, filename):
    """Save tensor as image"""
    image = tensor.cpu().clone()
    image = image.squeeze(0)
    image = transforms.ToPILImage()(image)
    image.save(filename)

def imshow(tensor, title=None):
    """Display image from tensor"""
    image = tensor.cpu().clone()
    image = image.squeeze(0)
    image = transforms.ToPILImage()(image)
    
    plt.imshow(image)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)