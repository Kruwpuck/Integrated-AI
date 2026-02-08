import json

nb_path = '/home/habb/Kuliah/HUMIC/Integrated-AI/CONFERENCE/densenet.ipynb'

try:
    with open(nb_path, 'r') as f:
        nb = json.load(f)

    # The last cell is the one with the error
    cell = nb['cells'][-1]
    source = cell['source']
    new_source = []
    
    # We expect two try/except blocks. 
    # The first one is for DenseNet (should be active).
    # The second one is for Xception (should be commented out).
    
    # Let's count try/except blocks to be safe.
    tc = 0
    ec = 0
    
    for i, line in enumerate(source):
        l = line.strip()
        
        # Check if this line is part of the DenseNet block (first one)
        # We can heuristic: if we haven't seen "model_xcp" or "Xception" comments yet, it's likely DenseNet.
        # OR just count indices.
        
        # A better approach: 
        # If line is "# try:" (commented by previous script), uncomment it IF the next line is NOT commented.
        # But wait, I commented out ALL try/except.
        
        # Let's look at the window around the line.
        
        is_try = l == "try:" or l == "# try:"
        is_except = l == "except:" or l == "# except:"
        
        if is_try:
            # Look ahead to see if body is commented out
            # The body of Xception block (lines 999-1000) was commented out by filter_eval_cell.
            # The body of DenseNet block (lines 989-990) was NOT commented out.
            
            # Find next non-empty line
            is_body_commented = False
            for j in range(i+1, len(source)):
                if source[j].strip():
                    if source[j].strip().startswith("#"):
                         is_body_commented = True
                    break
            
            if is_body_commented:
                # This is likely Xception block. Make sure it is commented.
                new_source.append("# try:\n" if not l.startswith("#") else line)
            else:
                # This is likely DenseNet block. Ensure it is UNCOMMENTED.
                new_source.append("try:\n")

        elif is_except:
             # Look ahead
            is_body_commented = False
            for j in range(i+1, len(source)):
                if source[j].strip():
                    if source[j].strip().startswith("#"):
                         is_body_commented = True
                    break
            
            if is_body_commented:
                # This is likely Xception block. Make sure it is commented.
                new_source.append("# except:\n" if not l.startswith("#") else line)
            else:
                # This is likely DenseNet block. Ensure it is UNCOMMENTED.
                new_source.append("except:\n")
        else:
            new_source.append(line)

    cell['source'] = new_source
    
    with open(nb_path, 'w') as f:
        json.dump(nb, f, indent=2)
        
    print(f"Successfully fixed {nb_path} with v2 logic.")

except Exception as e:
    print(f"Error: {e}")
