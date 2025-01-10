import re

def delete_special(pre_text, character_list):
    for c in character_list:
        pre_text = pre_text.replace(c, "")
    return pre_text

def break_down2scenes(text: str):
    # Separate text based on S#
    scenes = re.split(r'(s#\d+)', text)
    
    # Remove empty elements from the split result
    scenes = [scene for scene in scenes if scene.strip()]
    
    scenes_list = []
    current_scene_number = None
    for i in range(0, len(scenes), 2):  # Process scene number and corresponding text as pairs
        scene_marker = scenes[i].strip()
        try:scene_number = int(scene_marker.split('#')[1])  # Extract only the number
        except:
            if len(scenes) % 2 == 1:
                return [scenes[0]]
        scene_text = scenes[i+1].strip() if i+1 < len(scenes) else ""

        # Verify if scene numbers are in correct sequence
        if current_scene_number is not None:
            expected_scene_number = current_scene_number + 1
            if scene_number != expected_scene_number:
                raise ValueError(f"Unexpected scene number: {scene_number}, expected {expected_scene_number}")

        # Store scene number and text together
        scenes_list.append({
            'detected_scene_number': scene_number,
            'text': f"{scene_marker}\n{scene_text}".strip()
        })
    filtered_scene_list = []
    scene_number = 0
    for scene_dict in scenes_list:
        detected_scene_number = int(scene_dict['detected_scene_number'])
        filtered_scene_list.append(scene_dict['text'])
        scene_number = detected_scene_number

    return filtered_scene_list