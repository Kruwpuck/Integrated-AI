import json

nb_path = '/home/habb/Kuliah/HUMIC/Integrated-AI/CONFERENCE/densenet.ipynb'

new_code = r"""# Grad-CAM Visualization with Manual Path Support
import PIL.Image
import cv2
import numpy as np
import torch
from torchvision import transforms
import matplotlib.pyplot as plt
import os
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

# --- PREPARE VARIABLES ---
try:
    model = model_dense
    class_names = train_set_dense.classes
    # Try to find a test dataset. If dataset_test exists (from previous cells?), use it.
    # Otherwise use test_set_dense
    if 'test_set_dense' in locals():
        dataset_test = test_set_dense
    elif 'loader_d' in locals():
        dataset_test = loader_d['test'].dataset
    else:
        # Fallback if neither exists (unlikely if run linearly)
        print("Warning: Could not automatically determine test dataset. visualize_gradcam_fixed might fail.")
        dataset_test = [] 
except NameError:
    print("Warning: model_dense or train_set_dense not defined. Ensure previous cells are run.")
    model = None
    class_names = []
    dataset_test = []

# --- USER CONFIGURATION ---
# Enter paths to your test images here.
test_image_paths = [
    '../Dataset/benign/benign (2).png',
    '../Dataset/malignant/malignant (4).png',
    '../Dataset/normal/normal (4).png'
]
# --------------------------

def get_target_layers(model):
    if hasattr(model, 'features'): return [model.features[-1]]
    if hasattr(model, 'blocks'): return [model.blocks[-1]]
    if hasattr(model, 'conv_head'): return [model.conv_head]
    # Fallback search
    last_conv = None
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Conv2d):
            last_conv = module
    if last_conv: return [last_conv]
    raise AttributeError("Could not find a suitable target layer for Grad-CAM")

def predict_single_image_gradcam(model, image_path, class_names):
    print(f"Processing manual image: {image_path}")
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    try:
        if not os.path.exists(image_path):
            print(f"Error: Image not found at {image_path}")
            return
            
        image = PIL.Image.open(image_path).convert('RGB')
        input_tensor = transform(image).unsqueeze(0).to(device)
        
        model.eval()
        target_layers = get_target_layers(model)
        cam = GradCAM(model=model, target_layers=target_layers)
        grayscale_cam = cam(input_tensor=input_tensor, targets=None)[0, :]
        
        # Prediction
        with torch.no_grad():
            outputs = model(input_tensor)
            prob = torch.nn.functional.softmax(outputs, dim=1)
            score, preds = torch.max(prob, 1)
            predicted_label = class_names[preds[0]]
            confidence = score.item()
            
        # Visualization
        img_display = np.array(image.resize((224, 224))) / 255.0
        visualization = show_cam_on_image(img_display, grayscale_cam, use_rgb=True)
        
        fig, ax = plt.subplots(1, 2, figsize=(10, 5))
        ax[0].imshow(img_display)
        ax[0].axis('off')
        ax[0].set_title(f"Pred: {predicted_label} ({confidence:.2f})")
        
        ax[1].imshow(visualization)
        ax[1].set_title("Grad-CAM - DenseNet121")
        ax[1].axis('off')
        plt.show()
        
    except Exception as e:
        print(f"Error processing manual image: {e}")
        import traceback
        traceback.print_exc()

def visualize_gradcam_fixed(model, test_dataset, class_names):
    model.eval()
    try:
       target_layers = get_target_layers(model)
    except AttributeError as e:
       print(e); return

    class_indices = {}
    # Try to find one example per class
    for idx in range(len(test_dataset)):
        try:
            _, label = test_dataset[idx]
            class_name = class_names[label]
            if class_name not in class_indices:
                class_indices[class_name] = idx
            if len(class_indices) == len(class_names):
                break
        except:
             continue
            
    count = len(class_indices)
    if count == 0:
        print("No examples found in test dataset.")
        return

    fig = plt.figure(figsize=(15, 5 * count))
    
    for i, (class_name, idx) in enumerate(class_indices.items()):
        image, label = test_dataset[idx]
        input_tensor = image.unsqueeze(0).to(device)
        cam = GradCAM(model=model, target_layers=target_layers)
        grayscale_cam = cam(input_tensor=input_tensor, targets=None)[0, :]
        
        # Denormalize
        img_display = image.permute(1, 2, 0).cpu().numpy()
        img_display = (img_display * np.array([0.229, 0.224, 0.225])) + np.array([0.485, 0.456, 0.406])
        img_display = np.clip(img_display, 0, 1)
        visualization = show_cam_on_image(img_display, grayscale_cam, use_rgb=True)

        ax = fig.add_subplot(count, 2, 2*i + 1)
        ax.imshow(img_display)
        ax.set_title(f"Original: {class_name}")
        ax.axis('off')
        ax = fig.add_subplot(count, 2, 2*i + 2)
        ax.imshow(visualization)
        ax.set_title(f"Grad-CAM: {class_name}")
        ax.axis('off')
    plt.tight_layout()
    plt.show()

# --- Execution Logic ---
manual_images_processed = False
if 'test_image_paths' in locals() and test_image_paths:
    print(f"Found {len(test_image_paths)} manual images to process.")
    for img_path in test_image_paths:
        if img_path: # Skip empty strings
            predict_single_image_gradcam(model, img_path, class_names)
            manual_images_processed = True

if not manual_images_processed:
    print("No manual images found/processed. Generating Grad-CAM for fixed examples from dataset...")
    if 'dataset_test' in locals() and dataset_test:
        visualize_gradcam_fixed(model, dataset_test, class_names)
    else:
        print("Error: Test dataset not found.")
"""

try:
    with open(nb_path, 'r') as f:
        nb = json.load(f)

    # Convert the raw string to a list of lines for the JSON
    source_lines = [line + '\n' for line in new_code.split('\n')]
    # Remove the extra newline at the end if it exists
    if source_lines[-1] == '\n':
        source_lines.pop()
        
    # Replace the last cell
    nb['cells'][-1]['source'] = source_lines
    
    with open(nb_path, 'w') as f:
        json.dump(nb, f, indent=2)
        
    print(f"Successfully updated Grad-CAM code in {nb_path}.")

except Exception as e:
    print(f"Error: {e}")
