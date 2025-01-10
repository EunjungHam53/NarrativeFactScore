import re
from typing import List

from transformers import AutoTokenizer
from tiktoken import encoding_for_model

SCENE_INDICATORS = ['SS##','S#','s#','S','s','#\d+.','\d+.']

def delete_special(pre_text, character_list):
        for c in character_list:
            pre_text = pre_text.replace(c, "")
        return pre_text

def preprocess_script(script:str) -> str:

    lines = script.split("\n")
    
    new_text = ""
    for line in lines:
        line = delete_special(line, ["\n", "\t", "\xa0",'၀','ᝰ','ศ','ನ','tุ','\x00Ā\x00\x00\x00'])
        cleaned = re.sub('[^가-힣a-zA-Z0-9\s,.!?/#]',' ', line).strip()
        cleaned = delete_special(cleaned, ["  "]).strip()
        cleaned = cleaned.replace("<|start|>", "").replace("<|end|>","")
        if len(cleaned)>0:
            new_text += f"{line}\n"
    new_text = new_text.strip()
    
    return new_text


def preprocess_scripts(scripts:List[str]) -> List[str]:
    scripts = [preprocess_script(s) for s in scripts]

    return scripts

def break_down2scenes(text: str):
    # Separate text based on S#
    scenes = re.split(r'(s#\d+)', text)
    
    # Remove empty elements from the split result
    scenes = [scene for scene in scenes if scene.strip()]
    
    scenes_list = []
    current_scene_number = None

    for i in range(0, len(scenes), 2): # Process scene number and corresponding text as pairs
        scene_marker = scenes[i].strip()
        scene_number = int(scene_marker.split('#')[1])  # Extract only the number
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
    return scenes_list

def chunk_script_gpt(script:str,
                    model:str,
                    chunk_size:int=-1) -> List[str]:
    if chunk_size == -1:
        chunks = [script]
        print("Single Inference Mode")
        return chunks

    encoding = encoding_for_model(model)
    
    scenes = break_down2scenes(script)
    
    len_scenes = len(scenes)

    chunks = []
    if len_scenes > 10:
        print(f"Num of detected scenes : {len_scenes}")

        chunk = ""
        token_len_chunk = 0
        for i, scene_data in enumerate(scenes):
            scene = scene_data["text"].strip()
            token_len_scene = len(encoding.encode_ordinary(scene))
            if token_len_chunk + token_len_scene > chunk_size:
                if token_len_chunk == 0:
                    chunk += scene
                    token_len_chunk += token_len_scene
                else:
                    chunks.append(chunk)
                    chunk = scene
                    token_len_chunk = token_len_scene
            else:
                chunk += scene
                token_len_chunk += token_len_scene

            if i == len_scenes-1:
                chunks.append(chunk)
    else:
        print(f"No Detected Scenes ({len_scenes})")
        tokenized_script = encoding.encode_ordinary(script)
        token_len_script = len(tokenized_script)

        for start in range(0,token_len_script,chunk_size):
            if start + chunk_size >= token_len_script:
                end = token_len_script+1
            else:
                end = start+chunk_size
            
            chunk = encoding.decode(tokenized_script[start:end])
            chunks.append(chunk)
    print(f"Num of chunks : {len(chunks)}")
    return chunks

def chunk_script(script:str,
                 tokenizer:AutoTokenizer,
                 chunk_size:int=-1) -> List[str]:

    if chunk_size == -1:
        chunks = [script]
        print("Single Inference Mode")
        return chunks

    scenes = break_down2scenes(script)

    len_scenes = len(scenes)

    chunks = []
    if len_scenes > 10:
        print(f"Num of detected scenes : {len_scenes}")

        chunk = ""
        token_len_chunk = 0
        for i, scene_data in enumerate(scenes):
            scene = scene_data["text"].strip()
            token_len_scene = len(tokenizer.encode(scene))
            if token_len_chunk + token_len_scene > chunk_size:
                if token_len_chunk == 0:
                    chunk += scene
                    token_len_chunk += token_len_scene
                else:
                    chunks.append(chunk)
                    chunk = scene
                    token_len_chunk = token_len_scene
            else:
                chunk += scene
                token_len_chunk += token_len_scene

            if i == len_scenes-1:
                chunks.append(chunk)

    else:
        print("No Detected Scenes")
        tokenized_script = tokenizer.encode(script)
        token_len_script = len(tokenized_script)
        for start in range(0,token_len_script,chunk_size):
            if start + chunk_size >= token_len_script:
                end = token_len_script+1
            else:
                end = start+chunk_size
            
            chunk = tokenizer.decode(tokenized_script[start:end])
        chunks.append(chunk)
    print(f"Num of chunks : {len(chunks)}")

    return chunks

def chunk_script_split(
    script:str,
    tokenizer:AutoTokenizer,
    split_size:int=-1) -> List[str]:


    scenes = break_down2scenes(script)

    len_scenes = len(scenes)
    assert len_scenes >= split_size 
    
    chunks = []
    chunk = ""
    
    concat_size = int(len_scenes/split_size)+1
    
    print(concat_size)
    
    for i in range(len_scenes):
        chunk += f"\n\n{scenes[i]}"
        
        if (i+1)%concat_size == 0 and chunk:
            chunks.append(chunk)
            chunk = ""
        
        if i == len_scenes-1:
            chunks.append(chunk)
            chunk = ""

    print(f"Num of chunks : {len(chunks)}")

    return chunks