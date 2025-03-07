import cv2
import torch
import imageio
import numpy as np
from PIL import Image
import torch.nn as nn
import tifffile as tiff
from torchvision import models



IMG_HEIGHT = 128
IMG_WIDTH = 128

def load_image(image_path):
    img = imageio.imread(image_path)
    img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))
    return img

def normalize_image(image):
     image = image.astype(np.float32)
     image[:, :, [2, 3, 4]] /= 255.0
     other_bands = [i for i in range(12) if i not in [2, 3, 4]]
     image[:,:,other_bands] /= 10000.0
     return image
     


def visualize_image(image_path, target_size=(128, 128)):
     image = tiff.imread(image_path)
     image_rgb = image[:, :, :3]
     image_rgb = (image_rgb / image_rgb.max() * 255).astype(np.uint8)
     image_rgb_resized = cv2.resize(image_rgb, target_size, interpolation=cv2.INTER_LINEAR)

     return image_rgb_resized

def prediction(model, X, device):
    model.eval()
    with torch.no_grad():
         X_tensor = torch.tensor(X, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0).to(device)
         pred = torch.sigmoid(model(X_tensor)).cpu().numpy()[0, 0, :, :]
    

    return pred



def visualize_prediction(prediction, target_size=(128, 128)):
    
    # If the prediction is a PyTorch tensor, convert it to a NumPy array
    if isinstance(prediction, torch.Tensor):
        prediction = prediction.squeeze().cpu().numpy()

    # Normalize the prediction to the range [0, 255]
    if prediction.max() > 1.0:
        prediction = (prediction / prediction.max() * 255).astype(np.uint8)
    else:
        prediction = (prediction * 255).astype(np.uint8)

    # Resize the prediction to the target size
    prediction_resized = cv2.resize(prediction, target_size, interpolation=cv2.INTER_LINEAR)

    # Convert the NumPy array to a PIL image
    if prediction_resized.ndim == 2:  # Grayscale image
        pil_image = Image.fromarray(prediction_resized, mode='L')
    elif prediction_resized.ndim == 3:  # RGB or multi-channel image
        pil_image = Image.fromarray(prediction_resized, mode='RGB')
    else:
        raise ValueError("Unsupported prediction format. Expected 2D (grayscale) or 3D (RGB) array.")


    return pil_image