import json

nb_path = '/home/habb/Kuliah/HUMIC/Integrated-AI/CONFERENCE/densenet.ipynb'

try:
    with open(nb_path, 'r') as f:
        nb = json.load(f)

    # The last cell is the one with the error
    cell = nb['cells'][-1]
    source = cell['source']
    new_source = []
    
    for line in source:
        if line.strip() == "try:":
            new_source.append("# " + line)
        elif line.strip() == "except:":
            new_source.append("# " + line)
        else:
            new_source.append(line)
            
    cell['source'] = new_source
    
    with open(nb_path, 'w') as f:
        json.dump(nb, f, indent=2)
        
    print(f"Successfully fixed {nb_path}.")

except Exception as e:
    print(f"Error: {e}")
