import argparse
import json
import os
import re
from tqdm import tqdm
from datasets import load_dataset

def split_and_clean_script(script):
    # Step 1: Split the script into scenes based on <scene> tags
    scenes = re.split(r'<scene>', script)

    # Step 2: Remove other tags like <character>, <dialogue>, etc.
    cleaned_scenes = []
    for scene in scenes:
        cleaned_scene = re.sub(r'<.*?>', '', scene)  # Remove all tags
        cleaned_scene = cleaned_scene.strip()  # Remove leading/trailing whitespace
        if cleaned_scene:  # Only add non-empty scenes
            cleaned_scenes.append(cleaned_scene)
    
    return cleaned_scenes


def main():
    parser = argparse.ArgumentParser()
    # choose moviesum or mensa
    parser.add_argument('--dataset', type=str, required=True, choices=['MovieSum', 'MENSA'], help="Specify the dataset name.")
    parser.add_argument('--data_type', type=str, default='test', choices=['train', 'validation', 'test'], help="Specify the dataset type.")
    args = parser.parse_args()
    
    
    # 0) make data path
    os.makedirs(f'./dataset/{args.dataset}/0_raw/', exist_ok=True)

    # 1) load data
    if args.dataset == 'MovieSum':
        data_key = 'rohitsaxena/MovieSum'
    elif args.dataset == 'MENSA':
        data_key = 'rohitsaxena/MENSA'
    else:
        raise ValueError(f"Invalid data name: {args.dataset}")
    dataset = load_dataset(data_key)

    # 2) preprocess data
    if args.dataset == 'MovieSum':
        with open(f'./dataset/{args.dataset}/0_raw/{args.data_type}.jsonl', 'w', encoding='utf8') as f:
            for item in tqdm(dataset[args.data_type]):
                scenes = item['script']
                scenes_list = split_and_clean_script(scenes)
                new_elem = {'scenes': scenes_list,'summary': item['summary'], 'name': item['movie_name'], 'imdb_id': item['imdb_id']}
                json_line = json.dumps(new_elem)
                f.write(json_line + '\n')
    elif args.dataset == 'MENSA':
        with open(f'./dataset/{args.dataset}/0_raw/{args.data_type}.jsonl', 'w', encoding='utf8') as f:
            for item in tqdm(dataset[args.data_type]):
                json_line = json.dumps(item)
                f.write(json_line + '\n')

if __name__ == "__main__":
    main()