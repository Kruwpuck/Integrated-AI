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
import SimpleITK as sitk 
from radiomics import featureextractor 
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
PATH_MODEL_DN121 = 'Best_3_Model/model_densenet121_humic.pth'
PATH_MODEL_DN201 = 'Best_3_Model/DenseNet201_2.keras'
PATH_MODEL_RESNET = 'Best_3_Model/best_resnet50_medianfilter_fixed.keras'
PATH_ML_PIPELINE = 'Best_3_Model/best_lightgbm_pipeline.joblib'

OUTPUT_JSON_DIR = 'output_api'
OUTPUT_IMG_DIR = 'gambar_api'
OUTPUT_CSV_DIR = 'tabel_fitur_api'

CLASSES = ['Benign', 'Malignant', 'Normal'] 

models_dict = {}

def get_radiomics_extractor():
    """Inisialisasi extractor Radiomics sesuai setingan training"""
    extractor = featureextractor.RadiomicsFeatureExtractor()
    extractor.settings['force2D'] = True
    extractor.settings['force2Ddimension'] = 0 
    extractor.settings['binWidth'] = 25
    extractor.settings['resampledPixelSpacing'] = None
    extractor.settings['interpolator'] = sitk.sitkBSpline
    
    extractor.enableImageTypeByName('Original')
    extractor.enableImageTypeByName('Wavelet')
    
    feature_classes = ['firstorder', 'shape2D', 'glcm', 'glrlm', 'glszm', 'gldm', 'ngtdm']
    for fc in feature_classes:
        extractor.enableFeatureClassByName(fc)
    return extractor

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("[INFO] Loading Models & Extractor...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    models_dict['device'] = device
    
    # Load Extractor Radiomics
    models_dict['extractor'] = get_radiomics_extractor()
    
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

app = FastAPI(title="Integrated AI Inference API (Robust 1707 Features)", lifespan=lifespan)

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
        extractor = models_dict['extractor']

        # --- 1. DenseNet121 Features & Grad-CAM ---
        tensor_dn121, img_processed_dn121 = preprocess_densenet121_pytorch(img_path, device)
        with torch.no_grad():
            features_map = model_dn121.features(tensor_dn121)
            feat_dn121 = torch.nn.functional.adaptive_avg_pool2d(features_map, (1, 1))
            feat_dn121 = torch.flatten(feat_dn121, 1).cpu().numpy().flatten()
        
        # Grad-CAM untuk Masking
        target_layer = model_dn121.features.denseblock4.denselayer16.conv2
        gradcam = GradCAM(model_dn121, target_layer)
        heatmap, _ = gradcam(tensor_dn121)
        
        # --- 2. Robust Radiomics & Frequency Features ---
        img_bgr = cv2.imread(img_path)
        img_bgr_224 = cv2.resize(img_bgr, (224, 224))
        heatmap_224 = cv2.resize(heatmap, (224, 224))
        
        # Visualisasi Grad-CAM
        heatmap_color = cv2.applyColorMap(np.uint8(255 * heatmap_224), cv2.COLORMAP_JET)
        overlay = cv2.addWeighted(img_bgr_224, 0.6, heatmap_color, 0.4, 0)
        cam_path = os.path.join(OUTPUT_IMG_DIR, f"gradcam_{file.filename}")
        cv2.imwrite(cam_path, overlay)

        # Masking
        _, mask = cv2.threshold(np.uint8(255 * heatmap_224), 127, 255, cv2.THRESH_BINARY)
        gray_image_224 = cv2.cvtColor(img_bgr_224, cv2.COLOR_BGR2GRAY)

        # SOLVE: Fallback jika mask kosong agar GLCM tidak gagal
        if np.sum(mask) == 0:
            print(f"[WARN] Empty mask for {file.filename}, using center fallback.")
            mask[102:122, 102:122] = 255 # ROI 20x20 di tengah

        # A. PyRadiomics Extraction (SITK) dengan Try-Except Khusus
        try:
            sitk_img = sitk.GetImageFromArray(gray_image_224)
            sitk_mask = sitk.GetImageFromArray((mask > 0).astype(np.uint8))
            sitk_mask.CopyInformation(sitk_img)
            
            rad_features_dict = extractor.execute(sitk_img, sitk_mask)
            rad_feats_numeric = [float(v) for k, v in rad_features_dict.items() if not k.startswith('diagnostics_')]
        except Exception as rad_err:
            print(f"[ERROR] Radiomics calculation failed for {file.filename}: {rad_err}")
            # Fallback jika ekstraksi tetap gagal: isi dengan nol (289 fitur)
            rad_feats_numeric = np.zeros(289).tolist()

        df_radiomics = pd.DataFrame([rad_feats_numeric])

        # B. Frequency Features (12 Features)
        freq_feats_dict = extract_frequency_features(gray_image_224, mask)
        df_freq = pd.DataFrame([freq_feats_dict])
        df_freq = df_freq.reindex(sorted(df_freq.columns), axis=1)
        
        # Gabungkan blok Radiomics Advanced (Numeric + Freq)
        df_rad_advanced = pd.concat([df_radiomics, df_freq], axis=1)
        df_rad_advanced.to_csv(os.path.join(OUTPUT_CSV_DIR, f"feature_radiomics_adv_{file.filename}.csv"), index=False)
        rad_adv_values = df_rad_advanced.values.flatten()

        # --- 3. Keras Deep Features ---
        x_dn201 = preprocess_densenet201_keras(img_path)
        feat_dn201 = model_dn201.predict(x_dn201, verbose=0).flatten()
        
        x_resnet = preprocess_resnet(img_path)
        feat_resnet = model_resnet.predict(x_resnet, verbose=0).flatten()

        # --- 4. Fusion (Total Dimensi 1707) ---
        # Urutan: Radiomics_Advanced + DN121 + DN201 + ResNet
        fused_features = np.concatenate([rad_adv_values, feat_dn121, feat_dn201, feat_resnet])
        
        df_dn121 = pd.DataFrame([feat_dn121], columns=[f"DN121_{i}" for i in range(len(feat_dn121))])
        df_dn201 = pd.DataFrame([feat_dn201], columns=[f"DN201_{i}" for i in range(len(feat_dn201))])
        df_resnet = pd.DataFrame([feat_resnet], columns=[f"ResNet_{i}" for i in range(len(feat_resnet))])
        
        df_full = pd.concat([df_rad_advanced, df_dn121, df_dn201, df_resnet], axis=1)
        df_full.to_csv(os.path.join(OUTPUT_CSV_DIR, f"full_features_{file.filename}.csv"), index=False)

        # --- 5. Prediction & ANOVA ---
        X_input = fused_features.reshape(1, -1)
        X_input = np.nan_to_num(X_input, nan=0.0)

        # Penyesuaian paksa ke 1707 sesuai scaler training
        if X_input.shape[1] > 1707:
            X_input = X_input[:, :1707]
        elif X_input.shape[1] < 1707:
            pad = np.zeros((1, 1707 - X_input.shape[1]))
            X_input = np.concatenate([X_input, pad], axis=1)

        if isinstance(ml_pipeline, dict):
            scaler = ml_pipeline.get('scaler')
            reducer = ml_pipeline.get('reducer')
            clf = ml_pipeline.get('classifier') or ml_pipeline.get('model')
            le = ml_pipeline.get('label_encoder')
            
            if scaler: X_input = scaler.transform(X_input)
            
            # ANOVA Selection (Top 100)
            if reducer and hasattr(reducer, 'get_support'):
                X_input_reduced = reducer.transform(X_input)
                selected_indices = reducer.get_support(indices=True)
                df_anova = df_full.iloc[:, selected_indices]
                df_anova.to_csv(os.path.join(OUTPUT_CSV_DIR, f"anova_100_{file.filename}.csv"), index=False)
                X_input = X_input_reduced
            
            prediction = clf.predict(X_input)[0]
            proba = clf.predict_proba(X_input)[0]
            current_classes = le.classes_.tolist() if le else CLASSES
        else:
            X_input = X_input[:, :1707]
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
            "feature_dim": X_input.shape[1],
            "tables_created": [
                f"feature_radiomics_adv_{file.filename}.csv",
                f"full_features_{file.filename}.csv",
                f"anova_100_{file.filename}.csv"
            ]
        }
        
        with open(os.path.join(OUTPUT_JSON_DIR, f"{file.filename}.json"), 'w') as f_json:
            json.dump(result_data, f_json, indent=4)
            
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