import json

nb_path = '/home/habb/Kuliah/HUMIC/Integrated-AI/CONFERENCE/DenseNet121_Model_Humic.ipynb'

try:
    with open(nb_path, 'r') as f:
        nb = json.load(f)

    for i, cell in enumerate(nb['cells']):
        if cell['cell_type'] == 'code':
            source = "".join(cell['source'])
            print(f"--- Cell {i} ---")
            print(source[:200].replace('\n', ' '))
            print("...")
except Exception as e:
    print(f"Error: {e}")
