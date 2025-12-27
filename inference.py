import argparse
import os
import json
import base64
import cv2
import numpy as np
import joblib
import io
import sys
from PIL import Image

# Deep Learning Imports
import torch
import torch.nn as nn
from torchvision import models, transforms
import SimpleITK as sitk
from radiomics import featureextractor

# ==========================================
# 1. KONFIGURASI PATH MODEL (SESUAIKAN INI)
# ==========================================
PATH_MODEL_DL = 'model_densenet121_deeplearning.pth'           # Path ke model .pth (DenseNet121)
PATH_PIPELINE_ML = 'model_lgbm_machinelearning.joblib' # Path ke pipeline .joblib (Hadiyan)
OUTPUT_JSON_DIR = 'output'
OUTPUT_IMG_DIR = 'gambar'

# Setup Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==========================================
# 2. CLASS & FUNGSI UTILITY
# ==========================================

class GradCAM:
    """Kelas untuk menghasilkan Heatmap Grad-CAM"""
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

def load_dl_model(path):
    """Load Model DenseNet121 (.pth)"""
    print(f"[INFO] Loading Deep Learning Model: {path}")
    # Definisi Arsitektur (Harus sama persis dengan saat training)
    model = models.densenet121(weights=None)
    
    # Sesuaikan classifier (Dense layer terakhir) agar state_dict cocok
    # Jika model Fazari dilatih untuk 3 kelas, ubah out_features jadi 3
    model.classifier = nn.Linear(model.classifier.in_features, 3) 
    
    try:
        state_dict = torch.load(path, map_location=device)
        model.load_state_dict(state_dict, strict=False)
    except Exception as e:
        print(f"[ERROR] Gagal load model DL: {e}")
        print("Pastikan file .pth benar. Menggunakan bobot random untuk testing struktur code...")
    
    model.to(device)
    model.eval()
    return model

def extract_deep_features(model, img_tensor):
    """Ekstrak fitur dari layer sebelum klasifikasi"""
    with torch.no_grad():
        # Untuk DenseNet, fitur ada di model.features
        features = model.features(img_tensor)
        out = nn.functional.relu(features, inplace=True)
        out = nn.functional.adaptive_avg_pool2d(out, (1, 1))
        out = torch.flatten(out, 1)
    return out.cpu().numpy().flatten()

def extract_radiomics(image_path, mask_img):
    """Ekstrak fitur Radiomics menggunakan PyRadiomics"""
    # Load Image Original
    img_sitk = sitk.ReadImage(image_path)
    # Jika RGB (Vector), ambil channel pertama (Grayscale)
    if img_sitk.GetNumberOfComponentsPerPixel() > 1:
        img_sitk = sitk.VectorIndexSelectionCast(img_sitk, 0)
    
    # Konversi Mask (Grad-CAM threshold) ke SimpleITK
    mask_sitk = sitk.GetImageFromArray(mask_img)
    mask_sitk.CopyInformation(img_sitk) # Samakan metadata
    
    # Setup Extractor
    extractor = featureextractor.RadiomicsFeatureExtractor()
    extractor.disableAllFeatures()
    # extractor.enableFeatureClassByName('glcm') 
    # extractor.enableFeatureClassByName('shape')
    # extractor.enableFeatureClassByName('firstorder')
    extractor.enableAllFeatures()  # Enable all to match training likelyhood
    extractor.settings['label'] = 255
    extractor.settings['force2D'] = True # Handle 2D images properly
    
    try:
        result = extractor.execute(img_sitk, mask_sitk)
        # Ambil values saja, urutkan berdasarkan key agar konsisten
        keys = sorted([k for k in result.keys() if "original_" in k])
        values = [result[k] for k in keys]
        return np.array(values)
    except Exception as e:
        print(f"[WARN] Radiomics extraction failed: {e}")
        # Return array nol (agar pipeline tidak crash total)
        # Expected total 1419 - 1024 (Deep) = 395 Radiomics Features
        return np.zeros(395)

def image_to_base64(img_cv):
    """Convert OpenCV Image ke Base64 String"""
    _, buffer = cv2.imencode('.jpg', img_cv)
    return base64.b64encode(buffer).decode('utf-8')

# Preprocessing Pipeline untuk Torch
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# ==========================================
# 3. MAIN EXECUTION
# ==========================================
def main():
    # 1. Parsing Arguments
    parser = argparse.ArgumentParser(description='AI Inference Pipeline: DL + Radiomics + Hybrid')
    parser.add_argument('image_path', type=str, help='Path ke file gambar input')
    args = parser.parse_args()

    img_path = args.image_path
    if not os.path.exists(img_path):
        print(f"[ERROR] File gambar tidak ditemukan: {img_path}")
        sys.exit(1)

    # Buat direktori output jika belum ada
    os.makedirs(OUTPUT_JSON_DIR, exist_ok=True)
    os.makedirs(OUTPUT_IMG_DIR, exist_ok=True)

    try:
        # 2. Load Models
        dl_model = load_dl_model(PATH_MODEL_DL)
        
        print(f"[INFO] Loading ML Pipeline: {PATH_PIPELINE_ML}")
        ml_pipeline = joblib.load(PATH_PIPELINE_ML)

        # 3. Process Image
        img_pil = Image.open(img_path).convert('RGB')
        img_tensor = preprocess(img_pil).unsqueeze(0).to(device)
        img_cv = cv2.imread(img_path)
        img_cv_resized = cv2.resize(img_cv, (224, 224))

        # 4. Generate Grad-CAM (Target Layer DenseNet121)
        # Sesuaikan 'features.denseblock4...' jika arsitektur beda
        target_layer = dl_model.features.denseblock4.denselayer16.conv2
        gradcam = GradCAM(dl_model, target_layer)
        
        heatmap_raw, _ = gradcam(img_tensor)
        
        # Overlay Heatmap
        heatmap_color = cv2.applyColorMap(np.uint8(255 * heatmap_raw), cv2.COLORMAP_JET)
        heatmap_color = cv2.resize(heatmap_color, (img_cv.shape[1], img_cv.shape[0])) # Resize ke ori
        overlay_img = cv2.addWeighted(img_cv, 0.6, heatmap_color, 0.4, 0)

        # Simpan Gambar Grad-CAM ke folder /gambar
        filename = os.path.basename(img_path)
        output_img_path = os.path.join(OUTPUT_IMG_DIR, f"gradcam_{filename}")
        cv2.imwrite(output_img_path, overlay_img)
        print(f"[INFO] Grad-CAM disimpan di: {output_img_path}")

        # 5. Feature Extraction & Fusion
        # A. Deep Features
        deep_features = extract_deep_features(dl_model, img_tensor)
        
        # B. Radiomics (Masking dari GradCAM)
        # Bikin binary mask dari heatmap
        _, binary_mask = cv2.threshold(np.uint8(255 * heatmap_raw), 127, 255, cv2.THRESH_BINARY)
        binary_mask = cv2.resize(binary_mask, (img_cv.shape[1], img_cv.shape[0]), interpolation=cv2.INTER_NEAREST)
        
        radiomics_features = extract_radiomics(img_path, binary_mask)
        
        # C. Fusion (Concatenate)
        # Pastikan dimensi radiomics sesuai dengan yang diharapkan ML Pipeline
        # Jika ML pipeline hanya mengharapkan Deep Feature, comment bagian radiomics
        # Di sini kita gabung: [Deep (1024) + Radiomics (N)]
        # WARNING: Jumlah fitur harus sama persis dengan saat training Hadiyan!
        # Jika pipeline Hadiyan menghandle seleksi fitur, kita cukup masukkan raw features.
        
        # Skenario: Gabung deep + radiomics
        input_features = np.concatenate([deep_features, radiomics_features]).reshape(1, -1)
        
        # Cek dimensi fitur (Debugging)
        # print(f"Input Feature Shape: {input_features.shape}")

        # 6. Predict (ML Pipeline: Scaler -> ANOVA -> LGBM)
        # Handle dictionary pipeline
        if isinstance(ml_pipeline, dict):
             scaler = ml_pipeline['scaler']
             reducer = ml_pipeline['reducer']
             classifier = ml_pipeline['classifier']
             
             X_scaled = scaler.transform(input_features)
             X_reduced = reducer.transform(X_scaled)
             
             prediction_class_idx = classifier.predict(X_reduced)[0]
             prediction_prob = classifier.predict_proba(X_reduced)[0].max()
             
             # Decode Class Label (0,1,2 -> Benign, Malignant, Normal)
             if 'label_encoder' in ml_pipeline:
                 le = ml_pipeline['label_encoder']
                 prediction_class = le.inverse_transform([int(prediction_class_idx)])[0]
             else:
                 prediction_class = str(prediction_class_idx)

        else:
             # Fallback if it is a pipeline object
             prediction_class_idx = ml_pipeline.predict(input_features)[0]
             prediction_prob = ml_pipeline.predict_proba(input_features)[0].max()
             prediction_class = str(prediction_class_idx)

        # 7. Generate Output JSON
        gradcam_b64 = image_to_base64(overlay_img)
        
        result_data = {
            "status": "success",
            "filename": filename,
            "prediction": {
                "class": str(prediction_class),
                "probability": float(f"{prediction_prob:.4f}"),
                "model_used": "DenseNet121 + LGBM Hybrid"
            },
            "gradcam_image_base64": f"data:image/jpeg;base64,{gradcam_b64}",
            "gradcam_file_path": output_img_path
        }

        # Simpan JSON ke folder /output
        json_filename = os.path.splitext(filename)[0] + ".json"
        output_json_path = os.path.join(OUTPUT_JSON_DIR, json_filename)
        
        with open(output_json_path, 'w') as f:
            json.dump(result_data, f, indent=4)
            
        print(f"[INFO] JSON Output disimpan di: {output_json_path}")
        print(json.dumps(result_data, indent=2)) # Print ke console juga

    except Exception as e:
        print(f"[ERROR] Pipeline Failed: {e}")
        error_data = {"status": "error", "message": str(e)}
        with open(os.path.join(OUTPUT_JSON_DIR, "error.json"), 'w') as f:
            json.dump(error_data, f)

if __name__ == "__main__":
    main()