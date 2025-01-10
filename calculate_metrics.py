import argparse
import json
import os
import numpy as np
import random
from tqdm import tqdm
from typing import List
import traceback
import torch
import torch.nn as nn
from transformers import BartTokenizer, BartForConditionalGeneration

import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
from bert_score.scorer import BERTScorer
nltk.download('punkt')

class BARTScorer:
    def __init__(self, device='cuda:0', max_length=1024, checkpoint='facebook/bart-large-cnn'):
        # Set up model
        self.device = device
        self.max_length = max_length
        self.tokenizer = BartTokenizer.from_pretrained(checkpoint)
        self.model = BartForConditionalGeneration.from_pretrained(checkpoint)
        self.model.eval()
        self.model.to(device)

        # Set up loss
        self.loss_fct = nn.NLLLoss(reduction='none', ignore_index=self.model.config.pad_token_id)
        self.lsm = nn.LogSoftmax(dim=1)

    def load(self, path=None):
        """ Load model from paraphrase finetuning """
        if path is None:
            path = 'models/bart.pth'
        self.model.load_state_dict(torch.load(path, map_location=self.device))

    def score(self, srcs, tgts, batch_size=4):
        """ Score a batch of examples """
        score_list = []
        for i in range(0, len(srcs), batch_size):
            src_list = srcs[i: i + batch_size]
            tgt_list = tgts[i: i + batch_size]
            try:
                with torch.no_grad():
                    encoded_src = self.tokenizer(
                        src_list,
                        max_length=self.max_length,
                        truncation=True,
                        padding=True,
                        return_tensors='pt'
                    )
                    encoded_tgt = self.tokenizer(
                        tgt_list,
                        max_length=self.max_length,
                        truncation=True,
                        padding=True,
                        return_tensors='pt'
                    )
                    src_tokens = encoded_src['input_ids'].to(self.device)
                    src_mask = encoded_src['attention_mask'].to(self.device)

                    tgt_tokens = encoded_tgt['input_ids'].to(self.device)
                    tgt_mask = encoded_tgt['attention_mask']
                    tgt_len = tgt_mask.sum(dim=1).to(self.device)

                    output = self.model(
                        input_ids=src_tokens,
                        attention_mask=src_mask,
                        labels=tgt_tokens
                    )
                    logits = output.logits.view(-1, self.model.config.vocab_size)
                    loss = self.loss_fct(self.lsm(logits), tgt_tokens.view(-1))
                    loss = loss.view(tgt_tokens.shape[0], -1)
                    loss = loss.sum(dim=1) / tgt_len
                    curr_score_list = [-x.item() for x in loss]
                    score_list += curr_score_list

            except RuntimeError:
                traceback.print_exc()
                print(f'source: {src_list}')
                print(f'target: {tgt_list}')
                exit(0)
        return score_list

    def multi_ref_score(self, srcs, tgts: List[List[str]], agg="mean", batch_size=4):
        # Assert we have the same number of references
        ref_nums = [len(x) for x in tgts]
        if len(set(ref_nums)) > 1:
            raise Exception("You have different number of references per test sample.")

        ref_num = len(tgts[0])
        score_matrix = []
        for i in range(ref_num):
            curr_tgts = [x[i] for x in tgts]
            scores = self.score(srcs, curr_tgts, batch_size)
            score_matrix.append(scores)
        if agg == "mean":
            score_list = np.mean(score_matrix, axis=0)
        elif agg == "max":
            score_list = np.max(score_matrix, axis=0)
        else:
            raise NotImplementedError
        return list(score_list)

    def test(self, batch_size=3):
        """ Test """
        src_list = [
            'This is a very good idea. Although simple, but very insightful.',
            'Can I take a look?',
            'Do not trust him, he is a liar.'
        ]

        tgt_list = [
            "That's stupid.",
            "What's the problem?",
            'He is trustworthy.'
        ]

        print(self.score(src_list, tgt_list, batch_size))


def calculate_bleu(reference_summary, generated_summary):
    # Tokenize the reference and generated summaries
    reference_tokens = [nltk.word_tokenize(reference_summary.lower())]
    generated_tokens = nltk.word_tokenize(generated_summary.lower())
    
    # Use SmoothingFunction to avoid zero scores in case of small text sizes
    smoothing = SmoothingFunction().method4
    
    # Calculate BLEU score
    bleu_score = sentence_bleu(reference_tokens, generated_tokens, smoothing_function=smoothing)
    
    return bleu_score

def calculate_rouge(reference_summary, generated_summary):
    # Create a ROUGE scorer
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    
    # Compute ROUGE scores
    scores = scorer.score(reference_summary, generated_summary)
    
    # Extract the F1 scores for ROUGE-1, ROUGE-2, and ROUGE-L
    rouge_1 = scores['rouge1'].fmeasure
    rouge_2 = scores['rouge2'].fmeasure
    rouge_l = scores['rougeL'].fmeasure
    
    return rouge_1, rouge_2, rouge_l

def calculate_bert_score(bert_scorer, reference_summary, generated_summary):
    p_sci, r_sci, f1_sci = bert_scorer.score([generated_summary], [reference_summary])
    return p_sci[0].item(), r_sci[0].item(), f1_sci[0].item()

def calculate_bart_score(bart_scorer, reference_summary, generated_summary):
    scores = bart_scorer.score([generated_summary], [reference_summary])
    return scores[0]



def main():
    # 0) initial settings
    ## 1> arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, required=True, help='Path to your data')
    parser.add_argument('--summary_path', type=str, required=True, help='Path to the summary')
    parser.add_argument('--output_path', type=str, required=True, default='./dataset/4_factscore', help='Path to save summaries')
    args = parser.parse_args()

    # 0) settings
    if torch.cuda.is_available():
        device = "cuda:0"
    else:
        device = "cpu"

    bert_scorer = BERTScorer("bert-base-uncased", device=device)
    bart_scorer = BARTScorer(checkpoint="facebook/bart-large", device=device)
    
    # 1) load data
    with open(args.data_path, 'r', encoding='utf8') as f:
        data = [json.loads(line) for line in f]
    with open(args.summary_path, 'r', encoding='utf8') as f:
        summary_data = json.load(f)

    # 2) calculate metrics
    bleu_list, rouge_1_list, rouge_2_list, rouge_L_list = [], [], [], []
    bert_score_p_list, bert_score_r_list, bert_score_f1_list = [], [], []
    bart_score_list = []
    result_list = []
    for elem, summary_elem in zip(tqdm(data), summary_data):
        generated_summary = ''
        for summary in summary_elem['script_summaries']:
            generated_summary += summary + ' '
        reference_summary = elem['summary']
        script_org = ' '.join(elem['scenes'])
        
        #reference_summary = script_org
        bleu = calculate_bleu(reference_summary, generated_summary)
        rouge_1, rouge_2, rouge_L = calculate_rouge(reference_summary, generated_summary)
        bert_score_p, bert_score_r, bert_score_f1 = calculate_bert_score(bert_scorer, reference_summary, generated_summary)
        bart_score = calculate_bart_score(bart_scorer, reference_summary, generated_summary)

        bleu_list.append(bleu)
        rouge_1_list.append(rouge_1)
        rouge_2_list.append(rouge_2)
        rouge_L_list.append(rouge_L)
        bert_score_p_list.append(bert_score_p)
        bert_score_r_list.append(bert_score_r)
        bert_score_f1_list.append(bert_score_f1)
        bart_score_list.append(bart_score)

        result_elem = {'bleu': bleu, 'rouge_1': rouge_1, 'rouge_2': rouge_2, 'rouge_L': rouge_L,
        'bert_score_p': bert_score_p, 'bert_score_r': bert_score_r, 'bert_score_f1': bert_score_f1, 'bart_score': bart_score}
        result_list.append(result_elem)
    
    final_result = {
        'total_bleu': np.mean(bleu_list), 
        'total_rouge_1': np.mean(rouge_1_list), 
        'total_rouge_2': np.mean(rouge_2_list), 
        'total_rouge_L': np.mean(rouge_L), 
        'total_bert_score_p': np.mean(bert_score_p_list), 
        'total_bert_score_r': np.mean(bert_score_r_list), 
        'total_bert_score_f1': np.mean(bert_score_f1_list), 
        'total_bart_score': np.mean(bart_score_list),
        'result_list': result_list
    }
    os.makedirs(args.output_path, exist_ok=True)
    with open(f'{args.output_path}/metrics.json', 'w', encoding='utf8') as f:
        json.dump(final_result, f, ensure_ascii=False, indent=4)
    
if __name__ == "__main__":
    main()