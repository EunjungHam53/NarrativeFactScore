import argparse
import json
import os
import numpy as np
import random
from tqdm import tqdm
import torch

from src.summary.scripty_summarizer import ScriptySummarizer
from src.summary.utils import preprocess_script, chunk_script_gpt
from src.summary.prompt import build_summarizer_prompt

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
    parser.add_argument('--model', type=str, required=True, help='Path to your model')
    parser.add_argument('--chunk_size', type=int, default=2048, help='Chunk size for summarization')
    parser.add_argument('--output_path', type=str, required=True, default='./dataset/3_summary', help='Path to save summaries')
    args = parser.parse_args()
    ## 2> seed
    _set_seed(42)
    
    # 1) load data
    with open(args.data_path, 'r', encoding='utf8') as f:
        data = [json.loads(line) for line in f]

    # 2) load model
    scripty_summarizer = ScriptySummarizer(
        model=args.model,
        seed=42,
    )


    # 3) initial_summarization
    final_datasets = []
    for idx, elem in enumerate(tqdm(data)):
        # 1> split chunk
        scenes = []
        scenes = [f"s#{i}\n{s}" for i, s in enumerate(elem['scenes'])]
        script = "\n\n".join(scenes)
        script_chunks = chunk_script_gpt(script=script, model=args.model, chunk_size=args.chunk_size)
        # 2> summarize chunk
        script_summaries = []
        for chunk in tqdm(script_chunks):
            chunk = preprocess_script(chunk)
            prompt = build_summarizer_prompt(
                prompt_template="./templates/external_summary.txt",
                input_text_list=[chunk]
            )
            script_summ = scripty_summarizer.inference_with_gpt(prompt=prompt)
            script_summaries.append(script_summ.strip())
        # 4) sanity check
        is_blank = False
        for s in script_summaries:
            if s == "":
                is_blank = True
        if is_blank:
            continue

        # 5) save summaries per script
        elem_dict_list = []
        agg_dict = {}
        for i, (chunk, summary) in enumerate(zip(script_chunks, script_summaries)):
            elem_dict = {
                "chunk_index": i,
                "chunk": chunk.strip(),
                "summary": summary.strip(),
            }
            elem_dict_list.append(elem_dict)
        agg_dict['script'] = ' '.join(script_chunks)
        agg_dict['summaries'] = ' '.join(script_summaries)
        save_path = f'{args.output_path}/{idx}_{elem["name"]}'
        os.makedirs(save_path, exist_ok=True)
        with open(f'{save_path}/summary_sep.json', 'w') as f:
            json.dump(elem_dict_list, f, ensure_ascii=False, indent=4)
        with open(f'{save_path}/summary_agg.json', 'w') as f:
            json.dump(agg_dict, f, ensure_ascii=False, indent=4)
        
        # 6) save total summaries 
        processed_dataset = {
            "script": script,
            "scenes": scenes,
            "script_chunks": script_chunks,
            "script_summaries": script_summaries,
        }
        final_datasets.append(processed_dataset)

    with open(f'{args.output_path}/summary.jsonl', 'w', encoding='utf-8') as f:
        for p_d in final_datasets:
            jsonline = json.dumps(p_d, ensure_ascii=False)
            f.write(f"{jsonline}\n")

    with open(f'{args.output_path}/summary.json', 'w') as f:
        json.dump(final_datasets, f, ensure_ascii=False, indent=4)


    
if __name__ == "__main__":
    main()