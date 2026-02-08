import json

nb_path = 'vggaja.ipynb'

with open(nb_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

# The new code to inject
new_gradcam_code = [
    "# Grad-CAM Visualization with Manual Paths\\n",
    "import PIL.Image\\n",
    "import cv2\\n",
    "import numpy as np\\n",
    "import torch\\n",
    "from torchvision import transforms\\n",
    "import matplotlib.pyplot as plt\\n",
    "from pytorch_grad_cam import GradCAM\\n",
    "from pytorch_grad_cam.utils.image import show_cam_on_image\\n",
    "import os\\n",
    "\\n",
    "# --- USER CONFIGURATION ---\\n",
    "test_image_paths = [\\n",
    "    '../Dataset/benign/benign (2).png',\\n",
    "    '../Dataset/malignant/malignant (4).png',\\n",
    "    '../Dataset/normal/normal (4).png'\\n",
    "]\\n",
    "# --------------------------\\n",
    "\\n",
    "def start_gradcam_manual(model, model_name, target_layer, image_paths, class_names):\\n",
    "    model.eval()\\n",
    "    transform = transforms.Compose([\\n",
    "        transforms.Resize((224, 224)),\\n",
    "        transforms.ToTensor(),\\n",
    "        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])\\n",
    "    ])\\n",
    "    \\n",
    "    cam = GradCAM(model=model, target_layers=target_layer)\\n",
    "    \\n",
    "    for img_path in image_paths:\\n",
    "        if not os.path.exists(img_path):\\n",
    "            print(f\"Image not found: {img_path}\")\\n",
    "            continue\\n",
    "            \\n",
    "        print(f\"Processing: {img_path}\")\\n",
    "        try:\\n",
    "            image = PIL.Image.open(img_path).convert('RGB')\\n",
    "            input_tensor = transform(image).unsqueeze(0).to(device)\\n",
    "            \\n",
    "            grayscale_cam = cam(input_tensor=input_tensor, targets=None)[0, :]\\n",
    "            \\n",
    "            # Get prediction\\n",
    "            with torch.no_grad():\\n",
    "                outputs = model(input_tensor)\\n",
    "                probs = torch.nn.functional.softmax(outputs, dim=1)\\n",
    "                confidence, preds = torch.max(probs, 1)\\n",
    "                predicted_class = class_names[preds[0]]\\n",
    "            \\n",
    "            # Visualization\\n",
    "            img_np = np.array(image.resize((224, 224))) / 255.0\\n",
    "            visualization = show_cam_on_image(img_np, grayscale_cam, use_rgb=True)\\n",
    "            \\n",
    "            plt.figure(figsize=(10, 4))\\n",
    "            plt.subplot(1, 2, 1)\\n",
    "            plt.imshow(img_np)\\n",
    "            plt.axis('off')\\n",
    "            plt.title(f\"Original: {predicted_class} ({confidence.item():.2f})\")\\n",
    "            \\n",
    "            plt.subplot(1, 2, 2)\\n",
    "            plt.imshow(visualization)\\n",
    "            plt.axis('off')\\n",
    "            plt.title(f\"Grad-CAM {model_name}\")\\n",
    "            plt.show()\\n",
    "            \\n",
    "        except Exception as e:\\n",
    "            print(f\"Error processing {img_path}: {e}\")\\n",
    "\\n",
    "# Call the function\\n",
    "if 'model_vgg16' in locals():\\n",
    "    target_layer_vgg16 = [model_vgg16.features[-1]]\\n",
    "    # Ensure class_names is available (it should be from earlier cells)\\n",
    "    if 'class_names' not in locals():\\n",
    "        class_names = ['Benign', 'Malignant', 'Normal'] # Fallback default\\n",
    "    start_gradcam_manual(model_vgg16, \"VGG16\", target_layer_vgg16, test_image_paths, class_names)\\n"
]

# Find the cell that calls `start_gradcam` and replace it
found = False
for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        source = "".join(cell['source'])
        if "start_gradcam" in source and "VGG16" in source:
            cell['source'] = new_gradcam_code
            found = True
            break

if not found:
    print("Could not find the target cell to replace. Appending new cell.")
    nb['cells'].append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": new_gradcam_code
    })

with open(nb_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=2)

print(f"Updated Grad-CAM code in {nb_path}")
