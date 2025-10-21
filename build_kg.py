import argparse
import json
import os
from tqdm import tqdm

from src.kg.main import script2kg

import logging
# Configure logging
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, required=True, help='Path to your data')
    parser.add_argument('--kg_path', type=str, required=True, help='Path to save your knowledge graph')
    parser.add_argument('--response_exist', action='store_true')
    parser.add_argument('--refine', type=str, required=True, default='ner')
    args = parser.parse_args()
    
    # 1) load data
    with open(args.data_path, 'r', encoding='utf8') as f:
        data = [json.loads(line) for line in f]

    # 2) build kg
    kg_list = []
    for idx, elem in enumerate(tqdm(data)):
        try:
            kg = script2kg(elem['scenes'], idx, args, elem['name'])
            kg_list.append(kg)
            logger.info(f"✓ Successfully processed {idx}: {elem['name']}")
        except Exception as e:
            logger.error(f"✗ Failed to process {idx} ({elem['name']}): {str(e)}")
            kg_list.append(None)  # Thêm None để giữ index đúng

if __name__ == "__main__":
    main()