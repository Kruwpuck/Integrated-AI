import json
import os

notebook_path = '/home/habb/Kuliah/HUMIC/Integrated-AI/CONFERENCE/DenseNet121_Model_Humic.ipynb'

with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Find the first code cell
for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        # Check if imports are already there
        source = cell['source']
        has_gradcam = any('pytorch_grad_cam' in line for line in source)
        
        if not has_gradcam:
            # Find insertion point (after standard imports)
            insert_idx = 0
            for i, line in enumerate(source):
                if 'import torch' in line:
                    insert_idx = i + 1
            
            # Insert lines
            new_lines = [
                "from pytorch_grad_cam import GradCAM\n",
                "from pytorch_grad_cam.utils.image import show_cam_on_image\n",
                "from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget\n"
            ]
            
            # Insert after the found index
            source[insert_idx:insert_idx] = new_lines
            cell['source'] = source
            print("Imports added successfully.")
        else:
            print("Imports already present.")
        
        break

with open(notebook_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=2, ensure_ascii=False)
