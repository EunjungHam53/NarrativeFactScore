import argparse
import json
import os
import numpy as np
import random
from tqdm import tqdm
import re
import torch

from src.summary.scripty_summarizer import ScriptySummarizer
from src.summary.utils import chunk_script, preprocess_script
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
    parser.add_argument('--init_summary_path', type=str, required=True, help='Path to your initial summary')
    parser.add_argument('--fact_path', type=str, required=True, help='Path to your fact_score')
    parser.add_argument('--output_path', type=str, required=True, default='dataset/4_corrected_summary', help='Path to save summaries')
    parser.add_argument('--num_gpu', type=int, default=1)
    parser.add_argument('--threshold', type=float, default=0.95)
    args = parser.parse_args()
    ## 2> seed
    _set_seed(42)
    
    # 1) load data
    ## 1> load raw script
    with open(args.data_path, 'r', encoding='utf8') as f:
        data = [json.loads(line) for line in f]
    ## 2> load initial summary
    with open(f'{args.init_summary_path}/summary.jsonl', 'r', encoding='utf8') as f:
        init_summary_data = [json.loads(line) for line in f]
    ## 3> load factscore
    factscore_list = []
    for idx in range(len(data)):
        with open(f'{args.fact_path}/{idx}_{data[idx]["name"]}/factscore.json', 'r', encoding='utf8') as f:
            factscore_elem = json.load(f)
        factscore_list.append(factscore_elem['output_list'])
    
    # 2) load model
    scripty_summarizer = ScriptySummarizer(
        model=args.model,
        seed=42,
    )
        
    # 3) self_correction
    final_datasets = []
    for idx, factscore_elem in enumerate(tqdm(factscore_list)):
        script_summaries = []
        elem = data[idx]
        scenes = [f"s#{i}\n{s}" for i, s in enumerate(elem['scenes'])]
        script = "\n\n".join(scenes)
        script_chunks = []
        hallu_part_list, hallu_scene_list, org_summ_list, feedback_list = [], [], [], []
        for factscore_chunk in tqdm(factscore_elem):
            total_hallu_summ = ''
            ## 1> load data from factscore
            src_chunk = factscore_chunk['src']
            summary = factscore_chunk['summary']
            hallu_idxs = np.where(np.array(factscore_chunk['scores_per_sent']) == 0)[0]
            hallu_scenes = np.array(factscore_chunk['relevant_scenes'])[hallu_idxs]
            hallu_summary_parts = np.array(factscore_chunk['summary_chunks'])[hallu_idxs]
            feedbacks = np.array(factscore_chunk['feedbacks'])[hallu_idxs]

            ## 2> make prompt
            prompt = build_summarizer_prompt(
                prompt_template="./templates/self_correction.txt",
                input_text_list=[src_chunk, summary],
            )
            for j, (hallu_summ, feedback) in enumerate(zip(hallu_summary_parts, feedbacks)):
                prompt += f"\n- Statement to Revise {j+1}: {hallu_summ} (Reason for Revision: {feedback})"
            prompt += f"\n- Revised Summary: "

            org_summ_list.append(summary)
            hallu_part_list.append(hallu_summary_parts)
            hallu_scene_list.append(hallu_scenes)
            feedback_list.append(feedbacks)

            ## 3> self-correction or not
            if factscore_chunk['score'] >= args.threshold:
                script_summaries.append(summary.strip())
                script_chunks.append(src_chunk)
            else:
                script_summ = scripty_summarizer.inference_with_gpt(prompt=prompt)

                if len(script_summ.strip()) == 0:
                    script_summ = summary
                script_summaries.append(script_summ.strip())
                script_chunks.append(src_chunk)
                    
        # 4) save summaries per script
        elem_dict_list = []
        agg_dict = {}
        for i, (chunk, summary) in enumerate(zip(script_chunks, script_summaries)):
            
            elem_dict = {
                "chunk_index": i,
                "chunk": chunk.strip(),
                "summary": summary.strip(),
                "org_summary": org_summ_list[i].strip(),
                "hallu_in_summary": list(hallu_part_list[i]),
                "hallu_scene": list(hallu_scene_list[i]),
                "feedbacks": list(feedback_list[i]),
            }
            elem_dict_list.append(elem_dict)
        agg_dict['script'] = script
        agg_dict['summaries'] = ' '.join(script_summaries)#;import ipdb;ipdb.set_trace(context=10)
        save_path = f'{args.output_path}/{idx}_{data[idx]["name"]}'
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
            "script_summaries": script_summaries
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