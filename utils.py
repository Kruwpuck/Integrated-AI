import os
import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import models, transforms
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.preprocessing import image
from scipy.fftpack import dct
from scipy.fft import fft2
from PIL import Image

# ==========================================
# 1. Grad-CAM Class (PyTorch)
# ==========================================
class GradCAM:
    """Class to produce Grad-CAM heatmap for PyTorch models"""
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activation = None
        
        # Register hooks
        self.target_layer.register_forward_hook(self.save_activation)
        self.target_layer.register_full_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output):
        self.activation = output

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def __call__(self, x):
        # Forward pass
        output = self.model(x)
        class_idx = torch.argmax(output)
        
        # Backward pass
        self.model.zero_grad()
        output[0, class_idx].backward()
        
        # Generate Heatmap
        pooled_gradients = torch.mean(self.gradients, dim=[0, 2, 3])
        if self.activation is None:
             raise ValueError("Activation not captured. Ensure forward pass worked.")
             
        activation = self.activation[0]
        
        for i in range(activation.shape[0]):
            activation[i, :, :] *= pooled_gradients[i]
            
        heatmap = torch.mean(activation, dim=0).cpu().detach().numpy()
        heatmap = np.maximum(heatmap, 0) # ReLU
        heatmap /= np.max(heatmap) if np.max(heatmap) != 0 else 1 # Normalize
        
        return heatmap, output

# ==========================================
# 2. Frequency Feature Extraction
# ==========================================
def extract_frequency_features(image_array, mask_array):
    """
    Extracts 12 frequency-based features (DCT & FFT) from the masked region.
    Input:
        image_array: Grayscale image (2D numpy array)
        mask_array: Binary mask (2D numpy array, 0 or 1)
    Output:
        Dictionary of features
    """
    # Ensure mask is binary 0/1
    mask_array = (mask_array > 0).astype(np.uint8)
    
    # Get pixels in mask
    pixels = image_array[mask_array == 1]

    if len(pixels) == 0:
        # Fallback if mask is empty
        return {f'{name}_{stat}': 0.0 for name in ['DCT', 'FFT'] 
                for stat in ['Mean', 'Variance', 'Energy', 'Entropy', 'Max', 'Min']}

    # Bounding rect around mask
    x, y, w, h = cv2.boundingRect(mask_array)
    roi_crop = image_array[y:y+h, x:x+w]
    roi_resized = cv2.resize(roi_crop, (64, 64))

    # DCT
    dct_coeffs = dct(dct(roi_resized.T, norm='ortho').T, norm='ortho')
    
    # FFT
    fft_coeffs = np.abs(fft2(roi_resized))

    feats = {}
    for name, coeffs in [('DCT', dct_coeffs), ('FFT', fft_coeffs)]:
        flat = coeffs.flatten()
        feats[f'{name}_Mean'] = np.mean(flat)
        feats[f'{name}_Variance'] = np.var(flat)
        feats[f'{name}_Energy'] = np.sum(flat**2)
        # Add epsilon to log2 to avoid log(0)
        prob = flat / (np.sum(flat) + 1e-10)
        feats[f'{name}_Entropy'] = -np.sum(prob * np.log2(prob + 1e-10))
        feats[f'{name}_Max'] = np.max(flat)
        feats[f'{name}_Min'] = np.min(flat)

    return feats

# ==========================================
# 3. Preprocessing Functions
# ==========================================
def preprocess_resnet(img_path):
    """
    Loads image, applies Median Filter (k=3), resizes to 224x224, 
    and preprocesses for ResNet50 (Keras).
    """
    # 1. Load with OpenCV for Median Filter
    img = cv2.imread(img_path)
    if img is None:
        raise ValueError(f"Could not load image at {img_path}")
    
    # Median Filter
    img = cv2.medianBlur(img, 3)
    
    # Resize manually or via keras
    img = cv2.resize(img, (224, 224))
    
    # Convert to RGB (OpenCV is BGR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Convert to array and expand dims
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    
    # ResNet50 preprocessing (likely subtract mean/scaling)
    # Using generic keras preprocess_input matching the model type usually
    # But since we use load_model, we should use the one corresponding to ResNet50
    from tensorflow.keras.applications.resnet50 import preprocess_input
    x = preprocess_input(x)
    return x

def preprocess_densenet201_keras(img_path):
    """
    Loads image, resizes to 224x224, no specific filters mention (Standard),
    and preprocesses for DenseNet201 (Keras).
    """
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    
    from tensorflow.keras.applications.densenet import preprocess_input
    x = preprocess_input(x)
    return x

def preprocess_densenet121_pytorch(img_path, device):
    """
    Loads image, applies Median Filter + CLAHE, 
    and transforms to Tensor for PyTorch.
    Returns: Tensor and Original Grayscale (for Radiomics)
    """
    # Load gambar original
    img_pil = Image.open(img_path).convert('RGB')
    
    # 1. Alur UltrasoundEnhancement (Sesuai Notebook)
    img_np = np.array(img_pil.convert('L')) # Ke Grayscale
    
    # 2. Denoising (Kernel 5 sesuai notebook)
    img_np = cv2.medianBlur(img_np, 5) #
    
    # 3. CLAHE (ClipLimit 2.0, Grid 8x8)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)) #
    img_np = clahe.apply(img_np)
    
    # 4. Kembalikan ke RGB agar bisa diterima DenseNet
    img_processed_np = cv2.cvtColor(img_np, cv2.COLOR_GRAY2RGB) #
    img_processed_pil = Image.fromarray(img_processed_np)

    # 5. Transformasi Standar PyTorch
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), #
    ])
    
    tensor = transform(img_processed_pil).unsqueeze(0).to(device)
    return tensor, img_processed_np

# ==========================================
# 4. Model Loaders
# ==========================================
def load_pytorch_densenet121(path, device):
    model = models.densenet121(weights=None)
    # Check classifier size. Usually 3 classes if trained by Humic
    # We'll check the checkpoint
    try:
        checkpoint = torch.load(path, map_location=device)
        
        # Infer num classes from checkpoint if possible, else default to 3
        # But we must initialize the model first.
        # Assuming 3 classes based on context
        model.classifier = nn.Linear(model.classifier.in_features, 3)
        
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        else:
            model.load_state_dict(checkpoint, strict=False)

        model.to(device)
        model.eval()
        return model
    except Exception as e:
        print(f"Error loading PyTorch model: {e}")
        return None

def load_keras_model_feature_extractor(path, layer_name=None):
    """
    Loads a Keras model and creates a sub-model that outputs features 
    from Global Average Pooling or the layer before the final Dense.
    """
    model = load_model(path, compile=False)
    layer_name = None
    # Mencari layer Dense terakhir sebelum output (untuk mendapatkan 256/128 fitur)
    for layer in reversed(model.layers):
        if 'dense' in layer.name.lower() and layer != model.layers[-1]:
            layer_name = layer.name
            break
    if not layer_name: layer_name = model.layers[-2].name
    return Model(inputs=model.input, outputs=model.get_layer(layer_name).output)