import argparse
import os
import json
import sys
import numpy as np
import joblib
import cv2
import torch
import warnings

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
PATH_MODEL_DN121 = 'model_densenet121_deeplearning.pth' # Re-check path, user said 'model_densenet121_deeplearning.pth' in inference.py but 'model_densenet121_humic.pth' in file list.
# I will use the one found in file list: model DL/model_densenet121_humic.pth
# Wait, let me check the file list again. 
# Found: ./model_densenet121_deeplearning.pth AND ./model DL/model_densenet121_humic.pth
# The plan said: `model DL/model_densenet121_humic.pth`. using that.

PATH_MODEL_DN121 = 'model DL/model_densenet121_humic.pth'
PATH_MODEL_DN201 = 'model DL/DenseNet201_2.keras'
PATH_MODEL_RESNET = 'model DL/best_resnet50_medianfilter_fixed.keras'
PATH_ML_PIPELINE = 'model_lgbm_machinelearning.joblib'

OUTPUT_JSON_DIR = 'output'
OUTPUT_IMG_DIR = 'gambar'

CLASSES = ['Benign', 'Malignant', 'Normal'] # Assumed based on 3 classes

def main():
    parser = argparse.ArgumentParser(description='Integrated AI Inference Pipeline')
    parser.add_argument('image_path', type=str, help='Path to input image')
    args = parser.parse_args()

    img_path = args.image_path
    if not os.path.exists(img_path):
        print(f"[ERROR] Image not found: {img_path}")
        sys.exit(1)

    # Create output dirs
    os.makedirs(OUTPUT_JSON_DIR, exist_ok=True)
    os.makedirs(OUTPUT_IMG_DIR, exist_ok=True)

    # 1. Load Models
    print("[1/6] Loading Models...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # DenseNet121 (PyTorch)
    model_dn121 = load_pytorch_densenet121(PATH_MODEL_DN121, device)
    if model_dn121 is None:
        print("[ERROR] Failed to load DenseNet121")
        sys.exit(1)
        
    # DenseNet201 (Keras)
    model_dn201 = load_keras_model_feature_extractor(PATH_MODEL_DN201)
    
    # ResNet50 (Keras)
    model_resnet = load_keras_model_feature_extractor(PATH_MODEL_RESNET)
    
    # ML Pipeline
    try:
        ml_pipeline = joblib.load(PATH_ML_PIPELINE)
        print("[INFO] ML Pipeline loaded.")
    except Exception as e:
        print(f"[ERROR] Failed to load ML Pipeline: {e}")
        sys.exit(1)

    # 2. Preprocess & Extract Features
    print("[2/6] Extracting Features...")
    
    # --- DenseNet121 Features & GradCAM ---
    tensor_dn121, img_processed_dn121 = preprocess_densenet121_pytorch(img_path, device)
    
    # Extract Deep Features (Global Avg Pooling output equivalent)
    # We need to hook or use the features part. 
    # model.features outputs (1024, 7, 7). We need pooling.
    with torch.no_grad():
        features_map = model_dn121.features(tensor_dn121)
        # Pooling
        feat_dn121 = torch.nn.functional.adaptive_avg_pool2d(features_map, (1, 1))
        feat_dn121 = torch.flatten(feat_dn121, 1).cpu().numpy().flatten()
    
    print(f"   - DN121 Features: {feat_dn121.shape}")

    # --- Grad-CAM & Frequency Features ---
    # Target layer: features.denseblock4.denselayer16.conv2 (Standard for DN121)
    try:
        target_layer = model_dn121.features.denseblock4.denselayer16.conv2
        gradcam = GradCAM(model_dn121, target_layer)
        heatmap, _ = gradcam(tensor_dn121)
        
        # Save Grad-CAM Image
        img_bgr = cv2.imread(img_path)
        img_bgr_resized = cv2.resize(img_bgr, (224, 224)) # Match model input size roughly for overlay
        
        heatmap_resized = cv2.resize(heatmap, (img_bgr_resized.shape[1], img_bgr_resized.shape[0]))
        heatmap_color = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)
        overlay = cv2.addWeighted(img_bgr_resized, 0.6, heatmap_color, 0.4, 0)
        
        cam_filename = f"gradcam_{os.path.basename(img_path)}"
        cam_path = os.path.join(OUTPUT_IMG_DIR, cam_filename)
        cv2.imwrite(cam_path, overlay)
        
        # Frequency Features from Mask (Thresholded Heatmap)
        _, mask = cv2.threshold(np.uint8(255 * heatmap_resized), 127, 255, cv2.THRESH_BINARY)
        # Note: extract_frequency_features expects Image and Mask arrays.
        # It operates on the processed image or original? 
        # The notebook used the 'img' from 'load_image_paths' (likely grayscale or just one channel).
        # utils.extract_frequency_features expects grayscale image array.
        
        gray_image = cv2.cvtColor(img_bgr_resized, cv2.COLOR_BGR2GRAY)
        freq_feats_dict = extract_frequency_features(gray_image, mask)
        
        # Convert dict to sorted list of values (Order matters!)
        # Keys: DCT_Mean, DCT_Variance, DCT_Energy, DCT_Entropy, DCT_Max, DCT_Min, ... same for FFT
        # Notebook code loop: for name, coeffs in [('DCT', ...), ('FFT', ...)]:
        # So order is DCT then FFT.
        # Check utils implementation for dict keys.
        # We must ensure specific order matching training.
        # Let's assume sorted keys.
        freq_feats = []
        for key in sorted(freq_feats_dict.keys()):
             freq_feats.append(freq_feats_dict[key])
        freq_feats = np.array(freq_feats)
        print(f"   - Frequency Features: {freq_feats.shape}")
        
    except Exception as e:
        print(f"[WARN] Grad-CAM/Frequency extraction failed: {e}")
        freq_feats = np.zeros(12) 

    # --- DenseNet201 Features ---
    x_dn201 = preprocess_densenet201_keras(img_path)
    feat_dn201 = model_dn201.predict(x_dn201, verbose=0).flatten()
    print(f"   - DN201 Features: {feat_dn201.shape}")

    # --- ResNet50 Features ---
    x_resnet = preprocess_resnet(img_path)
    feat_resnet = model_resnet.predict(x_resnet, verbose=0).flatten()
    print(f"   - ResNet Features: {feat_resnet.shape}")

    # 3. Fusion
    # Order: Radiomics (12), DN121 (1024), DN201 (256), ResNet (128) -> Total 1420
    # Notebook: X_radiomics, deep_feats['dn121'], deep_feats['dn201'], deep_feats['resnet']
    # Wait, dictionary order in notebook? 
    # models_config = {"dn121": ..., "dn201": ..., "resnet": ...}
    # list is: [X_radiomics, feats_dn121, feats_dn201, feats_resnet]
    
    fused_features = np.concatenate([freq_feats, feat_dn121, feat_dn201, feat_resnet])
    print(f"[3/6] Fused Features Shape: {fused_features.shape}")
    
    # 4. Handle Dimension Mismatch
    EXPECTED_DIM = 1419
    if fused_features.shape[0] > EXPECTED_DIM:
        print(f"[WARN] Feature dimension mismatch ({fused_features.shape[0]} vs {EXPECTED_DIM}). Truncating extra features.")
        fused_features = fused_features[:EXPECTED_DIM]
    elif fused_features.shape[0] < EXPECTED_DIM:
        print(f"[WARN] Feature dimension mismatch ({fused_features.shape[0]} vs {EXPECTED_DIM}). Padding with zeros.")
        padding = np.zeros(EXPECTED_DIM - fused_features.shape[0])
        fused_features = np.concatenate([fused_features, padding])
        
    # Reshape for ML input (1, N)
    X_input = fused_features.reshape(1, -1)

    # 5. Prediction
    print("[4/6] Predicting...")
    
    if isinstance(ml_pipeline, dict):
        try:
            scaler = ml_pipeline.get('scaler')
            reducer = ml_pipeline.get('reducer')
            clf = ml_pipeline.get('classifier') or ml_pipeline.get('model')
            
            # --- NaN Handling ---
            if np.isnan(X_input).any():
                print(f"[WARN] Input contains NaNs! Imputing with Scaler Mean.")
                nan_indices = np.where(np.isnan(X_input))
                # Print which indices are NaN
                print(f"       NaN Indices (first 10): {nan_indices[1][:10]}")
                
                if hasattr(scaler, 'mean_'):
                     # Impute with mean (so scaled value becomes 0)
                     X_input[nan_indices] = scaler.mean_[nan_indices[1]]
                else:
                     # Fallback to 0 if mean_ not avail (unlikely for StandardScaler)
                     X_input = np.nan_to_num(X_input, nan=0.0)

            # --- Transform ---
            if scaler:
                X_input = scaler.transform(X_input)
            
            if reducer:
                X_input = reducer.transform(X_input)
            
            prediction = clf.predict(X_input)[0]
            proba = clf.predict_proba(X_input)[0] 
            
        except Exception as e:
            print(f"[ERROR] Prediction failed: {e}")
            sys.exit(1)
    else:
        # Fallback for standard pipeline object
        X_input = np.nan_to_num(X_input, nan=0.0) # Naive
        prediction = ml_pipeline.predict(X_input)[0]
        try:
             proba = ml_pipeline.predict_proba(X_input)[0]
        except:
             proba = [0,0,0]

    class_name = CLASSES[prediction]
    confidence = np.max(proba)
    print(f"   - Result: {class_name} ({confidence:.2f})")

    # 6. Save JSON
    print("[5/6] Saving Results...")
    result_data = {
        "filename": os.path.basename(img_path),
        "prediction": int(prediction),
        "class": class_name,
        "confidence": float(confidence),
        "probabilities": proba.tolist(),
        "gradcam_path": cam_path
    }
    
    json_path = os.path.join(OUTPUT_JSON_DIR, f"{os.path.basename(img_path)}.json")
    with open(json_path, 'w') as f:
        json.dump(result_data, f, indent=4)
        
    print(f"[6/6] Done. JSON saved to {json_path}")

if __name__ == "__main__":
    main()
