import json
import os

notebook_path = '/home/habb/Kuliah/HUMIC/Integrated-AI/CONFERENCE/DenseNet121_Model_Humic.ipynb'

with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

for i, cell in enumerate(nb['cells']):
    if cell['cell_type'] == 'code':
        source = cell['source']
        print(f"Checking Cell {i} content:")
        # Normalize source to list of strings
        if isinstance(source, str):
            source = source.splitlines(keepends=True)
        
        # Check carefully
        has_gradcam_import = any('from pytorch_grad_cam import' in line for line in source)
        
        if not has_gradcam_import:
            print("Imports NOT found. Adding them...")
            # Find insertion point
            insert_idx = 0
            for j, line in enumerate(source):
                if 'import torch' in line:
                    insert_idx = j + 1
            
            new_lines = [
                "from pytorch_grad_cam import GradCAM\n",
                "from pytorch_grad_cam.utils.image import show_cam_on_image\n",
                "from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget\n"
            ]
            
            source[insert_idx:insert_idx] = new_lines
            cell['source'] = source
            print("Imports added.")
        else:
            print("Imports ALREADY FOUND.")
            for line in source:
                if 'pytorch_grad_cam' in line:
                    print(f"Found line: {line.strip()}")
        
        # Only check the first code cell
        break

with open(notebook_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=2, ensure_ascii=False)
