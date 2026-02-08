import json
import re

src_path = '/home/habb/Kuliah/HUMIC/Integrated-AI/CONFERENCE/DenseNet121_Model_Humic.ipynb'
dst_path = '/home/habb/Kuliah/HUMIC/Integrated-AI/CONFERENCE/densenet.ipynb'

def filter_training_cell(source):
    new_source = []
    skip = False
    for line in source:
        if "TRAIN XCEPTION" in line:
            skip = True
        if not skip:
            new_source.append(line)
        # If we were skipping but found something that looks like end of block? 
        # Actually simplest is to just truncate at TRAIN XCEPTION if it's the second block.
    return new_source

def filter_eval_cell(source):
    # This might be tricky if it's mixed. 
    # Strategy: Comment out or remove lines with 'xcp' or 'xception' in variable names/strings
    # ONLY if they are not part of the definition of get_model (which is Cell 11, handled separately)
    # Cell 14 is evaluation.
    new_source = []
    for line in source:
        if 'xcp' in line.lower() or 'xception' in line.lower():
            # Check if this is a definition line we want to keep?
            # Cell 14 defines evaluate_and_get_f1. It shouldn't have hardcoded 'xcp' inside the function body 
            # unless it's for comparison.
            # If it's a function call like evaluate(model_xcp...), remove it.
            if "def " not in line: 
                # It's likely a call or specific variable usage
                # We comment it out for safety
                new_source.append("# " + line)
            else:
                new_source.append(line)
        else:
            new_source.append(line)
    return new_source

try:
    with open(src_path, 'r') as f:
        nb = json.load(f)

    new_cells = []
    
    # Cells 0-11: Keep as is
    for i in range(12):
        new_cells.append(nb['cells'][i])
        
    # Cell 12: Training - Filter
    train_cell = nb['cells'][12]
    train_cell['source'] = filter_training_cell(train_cell['source'])
    new_cells.append(train_cell)
    
    # Cell 13: Plotting Def - Keep
    new_cells.append(nb['cells'][13])
    
    # Cell 14: Eval - Filter calls
    eval_cell = nb['cells'][14]
    eval_cell['source'] = filter_eval_cell(eval_cell['source'])
    new_cells.append(eval_cell)

    # Create new notebook object
    new_nb = nb.copy()
    new_nb['cells'] = new_cells
    
    with open(dst_path, 'w') as f:
        json.dump(new_nb, f, indent=2)
        
    print(f"Successfully created {dst_path} with {len(new_cells)} cells.")

except Exception as e:
    print(f"Error: {e}")
