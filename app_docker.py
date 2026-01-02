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
from fastapi import FastAPI, File, UploadFile, HTTPException
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

# --- Configuration via Environment Variables ---
BASE_DIR = os.getenv("APP_HOME", ".")
MODEL_DIR = os.path.join(BASE_DIR, "model DL")
OUTPUT_DIR = os.getenv("OUTPUT_DIR", "output_data")
IMG_OUTPUT_DIR = os.path.join(OUTPUT_DIR, "images")
JSON_OUTPUT_DIR = os.path.join(OUTPUT_DIR, "json")

# Model Paths
# Note: Docker will likely copy models to a flat structure or keep original.
# We'll check for existence.
PATH_MODEL_DN121 = os.getenv("MODEL_DN121", os.path.join(MODEL_DIR, 'model_densenet121_humic.pth'))
PATH_MODEL_DN201 = os.getenv("MODEL_DN201", os.path.join(MODEL_DIR, 'DenseNet201_2.keras'))
PATH_MODEL_RESNET = os.getenv("MODEL_RESNET", os.path.join(MODEL_DIR, 'best_resnet50_medianfilter_fixed.keras'))
PATH_ML_PIPELINE = os.getenv("MODEL_ML", os.path.join(BASE_DIR, 'model_lgbm_machinelearning.joblib'))

CLASSES = ['Benign', 'Malignant', 'Normal'] 

models_dict = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    # --- Startup ---
    print(f"[INFO] Starting up... Base Dir: {BASE_DIR}")
    
    # Ensure dirs exist
    os.makedirs(IMG_OUTPUT_DIR, exist_ok=True)
    os.makedirs(JSON_OUTPUT_DIR, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Device: {device}")
    models_dict['device'] = device
    
    # 1. Load DenseNet121 (PyTorch)
    if os.path.exists(PATH_MODEL_DN121):
        print(f"[INFO] Loading DN121 from {PATH_MODEL_DN121}")
        models_dict['dn121'] = load_pytorch_densenet121(PATH_MODEL_DN121, device)
    else:
        # Fallback to root if not in subfolder (common in Docker copies)
        alt_path = 'model_densenet121_humic.pth'
        if os.path.exists(alt_path):
             print(f"[INFO] Loading DN121 from {alt_path}")
             models_dict['dn121'] = load_pytorch_densenet121(alt_path, device)
        else:
             print(f"[ERROR] DN121 model not found at {PATH_MODEL_DN121}")
             sys.exit(1)

    # 2. Load DenseNet201 (Keras)
    if os.path.exists(PATH_MODEL_DN201):
        print(f"[INFO] Loading DN201 from {PATH_MODEL_DN201}")
        models_dict['dn201'] = load_keras_model_feature_extractor(PATH_MODEL_DN201)
    else:
        print(f"[ERROR] DN201 model not found at {PATH_MODEL_DN201}")
        sys.exit(1)

    # 3. Load ResNet50 (Keras)
    if os.path.exists(PATH_MODEL_RESNET):
        print(f"[INFO] Loading ResNet50 from {PATH_MODEL_RESNET}")
        models_dict['resnet'] = load_keras_model_feature_extractor(PATH_MODEL_RESNET)
    else:
        print(f"[ERROR] ResNet model not found at {PATH_MODEL_RESNET}")
        sys.exit(1)

    # 4. Load ML Pipeline
    if os.path.exists(PATH_ML_PIPELINE):
        print(f"[INFO] Loading ML Pipeline from {PATH_ML_PIPELINE}")
        models_dict['ml_pipeline'] = joblib.load(PATH_ML_PIPELINE)
    else:
        print(f"[ERROR] ML Pipeline not found at {PATH_ML_PIPELINE}")
        sys.exit(1)
        
    print("[INFO] All models loaded successfully.")
    
    yield
    
    # --- Shutdown ---
    models_dict.clear()

app = FastAPI(title="Integrated AI Inference API (Docker)", lifespan=lifespan)

@app.get("/health")
def health_check():
    return {"status": "ok", "models_loaded": len(models_dict) >= 5} # device + 4 models

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    temp_filename = f"temp_{file.filename}"
    try:
        with open(temp_filename, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
            
        img_path = temp_filename
        device = models_dict['device']
        model_dn121 = models_dict['dn121']
        model_dn201 = models_dict['dn201']
        model_resnet = models_dict['resnet']
        ml_pipeline = models_dict['ml_pipeline']

        # --- DenseNet121 Features & GradCAM ---
        tensor_dn121, img_processed_dn121 = preprocess_densenet121_pytorch(img_path, device)
        
        with torch.no_grad():
            features_map = model_dn121.features(tensor_dn121)
            feat_dn121 = torch.nn.functional.adaptive_avg_pool2d(features_map, (1, 1))
            feat_dn121 = torch.flatten(feat_dn121, 1).cpu().numpy().flatten()
        
        # --- Grad-CAM ---
        cam_path = ""
        freq_feats = np.zeros(12)
        try:
            target_layer = model_dn121.features.denseblock4.denselayer16.conv2
            gradcam = GradCAM(model_dn121, target_layer)
            heatmap, _ = gradcam(tensor_dn121)
            
            # Save Image
            img_bgr = cv2.imread(img_path)
            img_bgr_resized = cv2.resize(img_bgr, (224, 224))
            
            heatmap_resized = cv2.resize(heatmap, (img_bgr_resized.shape[1], img_bgr_resized.shape[0]))
            heatmap_color = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)
            overlay = cv2.addWeighted(img_bgr_resized, 0.6, heatmap_color, 0.4, 0)
            
            cam_filename = f"gradcam_{file.filename}"
            cam_path = os.path.join(IMG_OUTPUT_DIR, cam_filename)
            cv2.imwrite(cam_path, overlay)
            
            # Freq Feats
            _, mask = cv2.threshold(np.uint8(255 * heatmap_resized), 127, 255, cv2.THRESH_BINARY)
            gray_image = cv2.cvtColor(img_bgr_resized, cv2.COLOR_BGR2GRAY)
            freq_feats_dict = extract_frequency_features(gray_image, mask)
            
            freq_feats = []
            for key in sorted(freq_feats_dict.keys()):
                 freq_feats.append(freq_feats_dict[key])
            freq_feats = np.array(freq_feats)
            
        except Exception as e:
            print(f"[WARN] Grad-CAM Error: {e}")

        # --- Other Models ---
        x_dn201 = preprocess_densenet201_keras(img_path)
        feat_dn201 = model_dn201.predict(x_dn201, verbose=0).flatten()

        x_resnet = preprocess_resnet(img_path)
        feat_resnet = model_resnet.predict(x_resnet, verbose=0).flatten()

        # --- Fusion ---
        fused_features = np.concatenate([freq_feats, feat_dn121, feat_dn201, feat_resnet])
        
        EXPECTED_DIM = 1419
        if fused_features.shape[0] > EXPECTED_DIM:
            fused_features = fused_features[:EXPECTED_DIM]
        elif fused_features.shape[0] < EXPECTED_DIM:
            padding = np.zeros(EXPECTED_DIM - fused_features.shape[0])
            fused_features = np.concatenate([fused_features, padding])
            
        X_input = fused_features.reshape(1, -1)

        # --- Prediction ---
        if isinstance(ml_pipeline, dict):
            scaler = ml_pipeline.get('scaler')
            reducer = ml_pipeline.get('reducer')
            clf = ml_pipeline.get('classifier') or ml_pipeline.get('model')
            
            if np.isnan(X_input).any():
                X_input = np.nan_to_num(X_input, nan=0.0)

            if scaler: X_input = scaler.transform(X_input)
            if reducer: X_input = reducer.transform(X_input)
            
            prediction = clf.predict(X_input)[0]
            proba = clf.predict_proba(X_input)[0] 
        else:
             X_input = np.nan_to_num(X_input, nan=0.0)
             prediction = ml_pipeline.predict(X_input)[0]
             proba = ml_pipeline.predict_proba(X_input)[0]

        result_data = {
            "filename": file.filename,
            "prediction": int(prediction),
            "class_name": CLASSES[prediction],
            "confidence": float(np.max(proba)),
            "probabilities": proba.tolist(),
            "gradcam_local_path": cam_path
        }
        
        json_path = os.path.join(JSON_OUTPUT_DIR, f"{file.filename}.json")
        with open(json_path, 'w') as f:
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
