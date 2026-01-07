
import json

file_path = r"music classification\audio_feature_scale.json"

try:
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    new_lines = []
    in_head = False
    found_head = False
    
    if lines[0].startswith("<<<<<<<"):
        in_head = True
        found_head = True
    else:
        # If no marker at start, maybe it's valid or marker is lower
        # But we saw it at line 1.
        pass

    for line in lines:
        if line.startswith("<<<<<<<"):
            in_head = True
            found_head = True
            continue
        if line.startswith("======="):
            in_head = False # Stop taking lines
            continue
        if line.startswith(">>>>>>>"):
            # End of conflict block
            continue
            
        if in_head:
            new_lines.append(line)
        elif not found_head:
            # If we haven't seen markers yet, keep lines (if any)
            new_lines.append(line)
            
    # Check if the last collected line closes the JSON
    content = "".join(new_lines).strip()
    if not content.endswith("}"):
        print("Closing brace missing. Adding it.")
        content += "\n}"
        
    # Verify
    try:
        data = json.loads(content)
        print("JSON parsed successfully.")
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"Fixed {file_path} (Kept HEAD version)")
    except json.JSONDecodeError as e:
        print(f"Still invalid JSON: {e}")
        # Make a backup plan: Take the BOTTOM version?
        # But assume HEAD is recoverable.

except Exception as e:
    print(f"Error: {e}")
