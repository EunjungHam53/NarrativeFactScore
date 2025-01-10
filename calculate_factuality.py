import argparse
import json
import os
import numpy as np
import random
from tqdm import tqdm
import torch

from src.fact.narrativefactscore import NarrativeFactScore

def _set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main():
    # 0) initial settings
    ## 1> arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, required=True, help='Path to your data')
    parser.add_argument('--summary_path', type=str, required=True, help='Path to the summary')
    parser.add_argument('--output_path', type=str, required=True, default='./dataset/MENSA/3_factscore/iter_0/gpt-4o-mini-2024-07-18/test', help='Path to save summaries')
    parser.add_argument('--model', type=str, required=True, default='gpt-4o-mini-2024-07-18', help='Path to save summaries')
    parser.add_argument('--start', type=int, default=0)
    parser.add_argument('--end', type=int, default=-1)
    parser.add_argument('--kg_path', type=str, default='dataset/MENSA/1_kg/test')
    args = parser.parse_args()
    ## 2> seed
    _set_seed(42)
    
    # 1) load data
    with open(args.data_path, 'r', encoding='utf8') as f:
        data = [json.loads(line) for line in f]
    with open(args.summary_path, 'r', encoding='utf8') as f:
        summary_data = json.load(f)
    kg_list = []
    kg_dirs = os.listdir(args.kg_path)
    for kg_dir in kg_dirs:
        with open(f'{args.kg_path}/{kg_dir}/3_knowledge_graphs/final_kg.jsonl', 'r', encoding='utf8') as f:
            kg_data = [json.loads(line) for line in f]
        kg_elem = []
        for elem in kg_data:
            if elem['subject'] == elem['object']:
                kg_elem.append(f"{elem['subject']} {elem['predicate']}")
            else:
                kg_elem.append(f"{elem['subject']} {elem['predicate']} {elem['object']}")
        kg_list.append(kg_elem)

    # 2) load scorer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    factscorer = NarrativeFactScore(device, model=args.model)

    # 3) calculate fact_score
    if args.end == -1: args.end = len(summary_data)
    for idx, elem in enumerate(tqdm(summary_data)):
        if args.start > idx or args.end <= idx:
            continue
        print(idx)
        ## 1> load data
        chunks, summaries, kgs = elem['script_chunks'], elem['script_summaries'], kg_list[idx]#;import ipdb;ipdb.set_trace(context=10)
        total_output = {'fact_score': 0, 'output_list': []}
        total_score = 0
        ## 2> calculate score
        scores, scores_per_sent, relevant_scenes, summary_chunks, feedbacks = factscorer.score_src_hyp_long(chunks, summaries, kgs)
        for i, score in enumerate(scores):
            output_elem = {}
            output_elem['src'] = chunks[i]
            output_elem['summary'] = summaries[i]
            output_elem['score'] = score
            output_elem['scores_per_sent'] = scores_per_sent[i]
            output_elem['relevant_scenes'] = relevant_scenes[i]
            output_elem['summary_chunks'] = summary_chunks[i]
            output_elem['feedbacks'] = feedbacks[i]
            total_output['output_list'].append(output_elem)
            total_score += score
        total_output['fact_score'] = total_score / len(scores)
        # 3> save summary
        save_path = f'{args.output_path}/{idx}_{data[idx]["name"]}'
        os.makedirs(save_path, exist_ok=True)
        with open(f'{save_path}/factscore.json', 'w', encoding='utf8') as f:
            json.dump(total_output, f, indent=4, ensure_ascii=False)


    
if __name__ == "__main__":
    main()