import os
import sys
import json
import base64
import torch
import torch.nn as nn
import joblib
import numpy as np
import cv2
import SimpleITK as sitk
from radiomics import featureextractor
from torchvision import models, transforms
from PIL import Image

# --- 1. SETUP ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OUTPUT_JSON_DIR = 'output'
OUTPUT_IMG_DIR = 'gambar'

# Ensure output directories exist
os.makedirs(OUTPUT_JSON_DIR, exist_ok=True)
os.makedirs(OUTPUT_IMG_DIR, exist_ok=True)

# --- 2. CLASS & UTILITY FUNCTIONS ---
class GradCAM:
    """Class to generate Grad-CAM Heatmap"""
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
        activation = self.activation[0]
        
        for i in range(activation.shape[0]):
            activation[i, :, :] *= pooled_gradients[i]
            
        heatmap = torch.mean(activation, dim=0).cpu().detach().numpy()
        heatmap = np.maximum(heatmap, 0) # ReLU
        heatmap /= np.max(heatmap) if np.max(heatmap) != 0 else 1 # Normalize
        
        return heatmap, output

def image_to_base64(img_cv):
    """Convert OpenCV Image to Base64 String"""
    _, buffer = cv2.imencode('.jpg', img_cv)
    return base64.b64encode(buffer).decode('utf-8')

def load_dl_model(path):
    print(f"[INFO] Loading DL Model from {path}...")
    model = models.densenet121(weights=None)
    num_ftrs = model.classifier.in_features
    # Model has 3 output classes
    model.classifier = nn.Linear(num_ftrs, 3) 
    
    try:
        state_dict = torch.load(path, map_location=device)
        model.load_state_dict(state_dict, strict=False)
    except Exception as e:
        print(f"[ERROR] Failed to load DL weights: {e}")
        sys.exit(1)
        
    model.to(device)
    model.eval()
    return model

def extract_deep_features(model, img_tensor):
    """Extract features from the layer before classification"""
    with torch.no_grad():
        # For DenseNet, features are in model.features
        features = model.features(img_tensor)
        out = nn.functional.relu(features, inplace=True)
        out = nn.functional.adaptive_avg_pool2d(out, (1, 1))
        out = torch.flatten(out, 1)
    return out.cpu().numpy().flatten()

def extract_radiomics(image_path, mask_img):
    """Extract Radiomics features using PyRadiomics"""
    # Load Original Image (SimpleITK format)
    img_sitk = sitk.ReadImage(image_path)
    # If RGB (Vector), get the first channel (Grayscale)
    if img_sitk.GetNumberOfComponentsPerPixel() > 1:
        img_sitk = sitk.VectorIndexSelectionCast(img_sitk, 0)
    
    # Convert Mask (Grad-CAM threshold) to SimpleITK
    mask_sitk = sitk.GetImageFromArray(mask_img)
    mask_sitk.CopyInformation(img_sitk) # Sync metadata
    
    # Setup Extractor
    extractor = featureextractor.RadiomicsFeatureExtractor()
    extractor.disableAllFeatures()
    extractor.enableAllFeatures() 
    extractor.settings['label'] = 255
    extractor.settings['force2D'] = True 
    
    try:
        result = extractor.execute(img_sitk, mask_sitk)
        # Get values only, sorted by key for consistency
        keys = sorted([k for k in result.keys() if "original_" in k])
        values = [result[k] for k in keys]
        return np.array(values)
    except Exception as e:
        # print(f"[WARN] Radiomics extraction failed: {e}")
        # Expected total around 1419 features (Deep=1024, Radiomics=~395)
        return np.zeros(395)

# --- GLOBAL MODEL LOADING ---
# Pastikan file model ada di folder yang sama atau sesuaikan path-nya
PATH_DL = "model_densenet121_deeplearning.pth"
PATH_ML = "model_lgbm_machinelearning.joblib"

if not os.path.exists(PATH_DL) or not os.path.exists(PATH_ML):
    print(f"[ERROR] Model files not found. Please check {PATH_DL} and {PATH_ML}")
    sys.exit(1)

dl_model = load_dl_model(PATH_DL)
bundle = joblib.load(PATH_ML)

scaler = bundle['scaler']
anova_selector = bundle['reducer']
lgbm_model = bundle['classifier']
label_encoder = bundle['label_encoder']

# --- 3. PREPROCESSING (REVISI: STANDARD / NO CLAHE) ---
def preprocess_image_standard(image_bytes):
    """
    Preprocessing standar (Resize + Normalize) tanpa CLAHE/Median Blur.
    Wajib digunakan jika model training LGBM dilatih dengan fitur dari raw image.
    """
    # 1. Decode Image (BGR default OpenCV)
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # 2. Convert BGR to RGB (PENTING! DenseNet butuh RGB)
    # HAPUS Median Filter & CLAHE
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # 3. Convert to PIL
    img_pil = Image.fromarray(img_rgb)
    
    # 4. Standard Transforms (Harus sama persis dengan training notebook)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    return transform(img_pil).unsqueeze(0).to(device)

# --- 4. PREDICT FUNCTION (CLI) ---
def run_cli_inference(image_path):
    if not os.path.exists(image_path):
        print(json.dumps({"status": "error", "message": f"File not found: {image_path}"}))
        return

    # 1. Process Image
    with open(image_path, "rb") as f:
        contents = f.read()
    
    # GUNAKAN FUNGSI PREPROCESSING STANDAR (REVISI)
    img_tensor = preprocess_image_standard(contents)
    
    # Load gambar untuk visualisasi OpenCV
    img_cv = cv2.imread(image_path)
    filename = os.path.basename(image_path)

    # 2. Generate Grad-CAM (for Radiomics Mask & Visualization)
    # Target Layer: features.denseblock4.denselayer16.conv2 (Standard for DenseNet121)
    target_layer = dl_model.features.denseblock4.denselayer16.conv2
    gradcam = GradCAM(dl_model, target_layer)
    
    heatmap_raw, _ = gradcam(img_tensor)
    
    # Create Visualization (Overlay)
    heatmap_color = cv2.applyColorMap(np.uint8(255 * heatmap_raw), cv2.COLORMAP_JET)
    heatmap_color = cv2.resize(heatmap_color, (img_cv.shape[1], img_cv.shape[0])) # Resize to original
    overlay_img = cv2.addWeighted(img_cv, 0.6, heatmap_color, 0.4, 0)

    # Save Grad-CAM Image
    output_img_path = os.path.join(OUTPUT_IMG_DIR, f"gradcam_{filename}")
    cv2.imwrite(output_img_path, overlay_img)
    
    # 3. Create Binary Mask from Heatmap (for Radiomics)
    heatmap_resized = cv2.resize(heatmap_raw, (img_cv.shape[1], img_cv.shape[0]))
    _, binary_mask = cv2.threshold(np.uint8(255 * heatmap_resized), 127, 255, cv2.THRESH_BINARY)
    
    # 4. Feature Extraction & Fusion
    # A. Deep Features (1024)
    deep_features = extract_deep_features(dl_model, img_tensor)
    
    # B. Radiomics Features (~395)
    radiomics_features = extract_radiomics(image_path, binary_mask)
    
    # C. Fusion
    input_features = np.concatenate([deep_features, radiomics_features]).reshape(1, -1)

    # 5. Prediction with ML Pipeline
    try:
        # Scaling -> Feature Selection (ANOVA) -> Prediction
        features_scaled = scaler.transform(input_features)
        features_reduced = anova_selector.transform(features_scaled)
        
        prediction = lgbm_model.predict(features_reduced)
        probability = lgbm_model.predict_proba(features_reduced)

        if hasattr(label_encoder, 'inverse_transform'):
             class_name = label_encoder.inverse_transform(prediction)[0]
        else:
             class_name = str(prediction[0])

        # Generate Base64 for JSON
        gradcam_b64 = image_to_base64(overlay_img)

        result = {
            "status": "success",
            "filename": filename,
            "prediction": {
                "class": class_name,
                "probability": float(np.max(probability)),
                "model_used": "DenseNet121 + LGBM Hybrid (Standard Preprocessing)"
            },
            "gradcam_image_base64": f"data:image/jpeg;base64,{gradcam_b64}",
            "gradcam_file_path": output_img_path
        }
        
        # Save JSON Output
        json_filename = os.path.splitext(filename)[0] + ".json"
        output_json_path = os.path.join(OUTPUT_JSON_DIR, json_filename)
        
        with open(output_json_path, 'w') as f:
            json.dump(result, f, indent=4)

        # Print output to console for backend capture
        print(json.dumps(result, indent=4))
    
    except ValueError as e:
        print(json.dumps({
            "status": "error", 
            "message": f"Feature mismatch: {str(e)}",
            "hint": f"Model expects {scaler.n_features_in_} features, input was {input_features.shape[1]}."
        }, indent=4))
    except Exception as e:
        print(json.dumps({"status": "error", "message": str(e)}))

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python inference_fixed.py <image_path>")
        sys.exit(1)
    
    img_path = sys.argv[1]
    run_cli_inference(img_path)