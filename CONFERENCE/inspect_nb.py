import json

nb_path = '/home/habb/Kuliah/HUMIC/Integrated-AI/CONFERENCE/DenseNet121_Model_Humic.ipynb'

try:
    with open(nb_path, 'r') as f:
        nb = json.load(f)

    print(f"Total cells: {len(nb['cells'])}")
    for i, cell in enumerate(nb['cells']):
        if cell['cell_type'] == 'code':
            source = "".join(cell['source'])
            print(f"Cell {i} Source Length: {len(source)}")
            if "def get_model" in source:
                print(f"FOUND definition in Cell {i}!")
                print(source[:200]) # Print beginning
            if "loader_d" in source and "DataLoader" in source:
                 print(f"FOUND DataLoader in Cell {i}!")
except Exception as e:
    print(f"Error: {e}")
