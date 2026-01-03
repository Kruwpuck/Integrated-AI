import os
import sys
import json
import shutil
import warnings
import numpy as np
import joblib
import cv2
import torch
import uvicorn
import pandas as pd
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager

# Suppress warnings
warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from utils import (
    GradCAM, 
    extract_frequency_features, 
    preprocess_resnet,
    preprocess_densenet201_keras,
    preprocess_densenet121_pytorch,
    load_pytorch_densenet121,
    load_keras_model_feature_extractor
)

# Constants
PATH_MODEL_DN121 = 'Best_Model/model_densenet121_humic.pth'
PATH_MODEL_DN201 = 'Best_Model/DenseNet201_2.keras'
PATH_MODEL_RESNET = 'Best_Model/best_resnet50_medianfilter_fixed.keras'
PATH_ML_PIPELINE = 'Best_Model/best_lightgbm_pipeline.joblib'

OUTPUT_JSON_DIR = 'output_api'
OUTPUT_IMG_DIR = 'gambar_api'
OUTPUT_CSV_DIR = 'tabel_fitur_api'

CLASSES = ['Benign', 'Malignant', 'Normal'] 

models_dict = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("[INFO] Loading Models...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    models_dict['device'] = device
    
    # Load DL Models
    models_dict['dn121'] = load_pytorch_densenet121(PATH_MODEL_DN121, device)
    if models_dict['dn121'] is None:
        print("[ERROR] Failed to load DenseNet121")
        sys.exit(1)
        
    models_dict['dn201'] = load_keras_model_feature_extractor(PATH_MODEL_DN201)
    models_dict['resnet'] = load_keras_model_feature_extractor(PATH_MODEL_RESNET)
    
    # Load ML Pipeline
    try:
        models_dict['ml_pipeline'] = joblib.load(PATH_ML_PIPELINE)
        print("[INFO] ML Pipeline loaded.")
    except Exception as e:
        print(f"[ERROR] Failed to load ML Pipeline: {e}")
        sys.exit(1)
    
    # Create Directories
    os.makedirs(OUTPUT_JSON_DIR, exist_ok=True)
    os.makedirs(OUTPUT_IMG_DIR, exist_ok=True)
    os.makedirs(OUTPUT_CSV_DIR, exist_ok=True)
    
    yield
    models_dict.clear()

app = FastAPI(title="Integrated AI Inference API", lifespan=lifespan)

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    temp_filename = f"temp_{file.filename}"
    with open(temp_filename, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
        
    try:
        img_path = temp_filename
        device = models_dict['device']
        model_dn121 = models_dict['dn121']
        model_dn201 = models_dict['dn201']
        model_resnet = models_dict['resnet']
        ml_pipeline = models_dict['ml_pipeline']

        # --- 1. DenseNet121 Features ---
        tensor_dn121, img_processed_dn121 = preprocess_densenet121_pytorch(img_path, device)
        with torch.no_grad():
            features_map = model_dn121.features(tensor_dn121)
            feat_dn121 = torch.nn.functional.adaptive_avg_pool2d(features_map, (1, 1))
            feat_dn121 = torch.flatten(feat_dn121, 1).cpu().numpy().flatten()
        
        # --- 2. Grad-CAM & Frequency Features ---
        try:
            target_layer = model_dn121.features.denseblock4.denselayer16.conv2
            gradcam = GradCAM(model_dn121, target_layer)
            heatmap, _ = gradcam(tensor_dn121)
            
            img_bgr = cv2.imread(img_path)
            img_bgr_resized = cv2.resize(img_bgr, (224, 224))
            heatmap_resized = cv2.resize(heatmap, (img_bgr_resized.shape[1], img_bgr_resized.shape[0]))
            
            # Save Grad-CAM
            heatmap_color = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)
            overlay = cv2.addWeighted(img_bgr_resized, 0.6, heatmap_color, 0.4, 0)
            cam_path = os.path.join(OUTPUT_IMG_DIR, f"gradcam_{file.filename}")
            cv2.imwrite(cam_path, overlay)
            
            # Masking for Frequency
            _, mask = cv2.threshold(np.uint8(255 * heatmap_resized), 127, 255, cv2.THRESH_BINARY)
            gray_image = cv2.cvtColor(img_bgr_resized, cv2.COLOR_BGR2GRAY)
            freq_feats_dict = extract_frequency_features(gray_image, mask)
            freq_feats = np.array([freq_feats_dict[k] for k in sorted(freq_feats_dict.keys())])
            
        except Exception as e:
            print(f"[WARN] Grad-CAM/Frequency failed: {e}")
            freq_feats = np.zeros(12) 
            freq_feats_dict = {f"Error_{i}": 0 for i in range(12)}
            cam_path = None

        # --- 3. Keras Features ---
        x_dn201 = preprocess_densenet201_keras(img_path)
        feat_dn201 = model_dn201.predict(x_dn201, verbose=0).flatten()

        x_resnet = preprocess_resnet(img_path)
        feat_resnet = model_resnet.predict(x_resnet, verbose=0).flatten()

        # --- 4. Fusion & Table Generation ---
        # URUTAN: Freq (12), DN121 (1024), DN201 (256), ResNet (128) -> Total 1420
        fused_features = np.concatenate([freq_feats, feat_dn121, feat_dn201, feat_resnet])
        
        # Buat Header Kolom untuk Tabel
        header_freq = sorted(freq_feats_dict.keys())
        header_dn121 = [f"DN121_{i}" for i in range(len(feat_dn121))]
        header_dn201 = [f"DN201_{i}" for i in range(len(feat_dn201))]
        header_resnet = [f"ResNet_{i}" for i in range(len(feat_resnet))]
        all_headers = header_freq + header_dn121 + header_dn201 + header_resnet
        
        # Simpan Tabel Full Features
        df_full = pd.DataFrame([fused_features], columns=all_headers)
        df_full.to_csv(os.path.join(OUTPUT_CSV_DIR, f"full_features_{file.filename}.csv"), index=False)

        # --- 5. Prediction & ANOVA ---
        X_input = fused_features.reshape(1, -1)
        X_input = np.nan_to_num(X_input, nan=0.0)

        if isinstance(ml_pipeline, dict):
            scaler = ml_pipeline.get('scaler')
            reducer = ml_pipeline.get('reducer')
            clf = ml_pipeline.get('classifier') or ml_pipeline.get('model')
            le = ml_pipeline.get('label_encoder')
            
            if X_input.shape[1] == 1420:
                # Kita potong 1 fitur terakhir supaya jadi 1419
                X_input = X_input[:, :1419]
            # 1. Scaling
            if scaler: 
                X_input = scaler.transform(X_input)
            
            # 2. ANOVA Selection & Tabel ANOVA
            if reducer:
                if hasattr(reducer, 'get_support'):
                    selected_indices = reducer.get_support(indices=True)
                    # Ambil kolom yang terpilih (ANOVA 100)
                    df_anova = df_full.iloc[:, selected_indices]
                    df_anova.to_csv(os.path.join(OUTPUT_CSV_DIR, f"anova_100_{file.filename}.csv"), index=False)
                
                X_input = reducer.transform(X_input)
            
            # 3. Predict
            prediction = clf.predict(X_input)[0]
            proba = clf.predict_proba(X_input)[0]
            current_classes = le.classes_.tolist() if le else CLASSES
        else:
            # Fallback jika bukan bundle
            prediction = ml_pipeline.predict(X_input)[0]
            proba = ml_pipeline.predict_proba(X_input)[0]
            current_classes = CLASSES

        class_name = current_classes[int(prediction)]
        confidence = float(np.max(proba))
        
        result_data = {
            "filename": file.filename,
            "prediction": int(prediction),
            "class": class_name,
            "confidence": confidence,
            "probabilities": proba.tolist(),
            "gradcam_path": cam_path,
            "tables_created": [
                f"full_features_{file.filename}.csv",
                f"anova_100_{file.filename}.csv"
            ]
        }
        
        with open(os.path.join(OUTPUT_JSON_DIR, f"{file.filename}.json"), 'w') as f:
            json.dump(result_data, f, indent=4)
            
        return result_data

    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
        
    finally:
        if os.path.exists(temp_filename):
            os.remove(temp_filename)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)