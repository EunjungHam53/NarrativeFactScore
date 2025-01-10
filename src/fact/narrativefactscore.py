from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")

import os
import re
import torch
import torch.nn as nn
import traceback
import numpy as np
from nltk import sent_tokenize
from sentence_transformers import util
from FlagEmbedding import BGEM3FlagModel
import logging
import openai
from openai.error import (APIError, RateLimitError, ServiceUnavailableError,
                          Timeout, APIConnectionError, InvalidRequestError)
from tenacity import (before_sleep_log, retry, retry_if_exception_type,
                      stop_after_attempt, wait_random_exponential)
from .utils import break_down2scenes
from .prompt import build_fact_prompt
from .openai_api import openai_api_response

openai.api_key = os.getenv('OPENAI_API_KEY')
logger = logging.getLogger(__name__)

class NarrativeFactScore():
    def __init__(self, device="cuda:0", model="gpt-4o-mini-2024-07-18", split_type="gpt", checkpoint=None):
        self.sent_model = BGEM3FlagModel('BAAI/bge-m3', use_fp16=True)
        self.model = model
        self.device = device
        self.split_type = split_type
        self.checkpoint = checkpoint
        self.metric = GPTScore(model=self.model)
        self.metric_function = self.metric.gpt_score


    def get_surrounding_sentences(self, sentence_array, ii):
        if ii > 0 and ii < len(sentence_array) - 1:
            sents = " ".join(np.array(sentence_array)[ii - 1 : ii + 1])
        elif ii == 0:
            sents = " ".join(np.array(sentence_array)[:2])
        elif ii == len(sentence_array) - 1:
            sents = " ".join(np.array(sentence_array)[ii - 1 :])
        return sents

    def group_into_sections(self, sentence_array, num_sent):
        sectioned_sents = []
        for ii in range(0, len(sentence_array), num_sent):
            sectioned_sents.append(" ".join(sentence_array)[ii : ii + num_sent])
        return sectioned_sents
    
    def split_sent(self, text):
        text_list = []
        if self.split_type == "fast":
            for t in text.split('.'):
                if len(t) == 0:
                    continue
                text_list.append(t)
            return text_list
        elif self.split_type == "gpt":
            prompt = build_fact_prompt(
                prompt_template = './templates/atomic_fact.txt',
                input_text_list=[text],
            )
            response = openai_api_response(prompt, model=self.model)
            text_list = []
            for res in response.split('\n'):
                text_list.append(res.strip())
            return text_list
        else:
            return None

    def score_src_hyp_long(self, srcs, hyps, kgs):
        all_scores = []
        all_scores_per_sent = []
        all_relevant_scenes = []
        all_summary_chunks = []
        all_feedback_list = []
        # src is a list containing source documents.
        # hyps is a list containing predicted documents
        total_score = 0
        for global_idx, (src, hyp) in enumerate(zip(tqdm(srcs), hyps)):
            src_sents = break_down2scenes(src)
            sentence_embeddings_src = self.sent_model.encode(src_sents, batch_size=12, max_length=8192)['dense_vecs']
            sentence_embeddings_kg = self.sent_model.encode(kgs, batch_size=12, max_length=8192)['dense_vecs']
            # [S, D (768)]
            doc_scores = []
            relevant_scenes = []
            feedbacks = []
            hyp_array = self.split_sent(hyp)
            for idx, hyp_sentence in enumerate(hyp_array):
                # for each sentence in summary, calculate the most similar sentence in the source article
                sentence_embeddings_hyp = self.sent_model.encode(hyp_sentence, max_length=8192)['dense_vecs']
                # [D (768) ]
                scores = util.cos_sim(sentence_embeddings_hyp, sentence_embeddings_src)[0] 
                scores_kg = util.cos_sim(sentence_embeddings_hyp, sentence_embeddings_kg)[0]
                # [1, S] -> [S]
                sorted_idxs = np.argsort(-1 * scores) # descending order
                sorted_idxs_kg = np.argsort(-1 * scores_kg) # descending order
                similar_src_sentences = []
                similar_src_sentences_kg = []
                triple = ''
                #  get sentences surrounding the most similar sentences in the source article'
                for sorted_idx, ii in enumerate(sorted_idxs_kg[0:1]):
                    if sorted_idx == 0:
                        triple += f'{kgs[ii]}'
                    else:
                        triple += f', {kgs[ii]}'
                for ii in sorted_idxs[0:1]:
                    similar_sents = src_sents[ii]
                    similar_src_sentences.append(similar_sents)
                # calculate metric for 3 most similar sections of source article
                scores, feedback_list = self.metric_function(similar_src_sentences, [hyp_sentence for i in range(0, len(similar_src_sentences))], triple)
                # Take the max scoring section to use
                score = np.max(scores)
                max_scene_idx = np.argmax(scores)
                max_scene = similar_src_sentences[max_scene_idx]
                feedback = feedback_list[max_scene_idx]
                
                doc_scores.append(score)
                relevant_scenes.append(max_scene)
                feedbacks.append(feedback)

            # calculate average score over whole doc
            doc_score = np.mean(doc_scores)
            # Append scores to the list
            all_scores_per_sent.append(doc_scores)
            all_scores.append(doc_score)
            all_relevant_scenes.append(relevant_scenes)
            all_summary_chunks.append(hyp_array)
            all_feedback_list.append(feedbacks)
            total_score += doc_score
            if global_idx % 100 == 99:
                print(f"Document mean {global_idx+1} Score: {total_score/(global_idx+1)} Score")
        return all_scores, all_scores_per_sent, all_relevant_scenes, all_summary_chunks, all_feedback_list

class GPTScore():
    def __init__(self, model="gpt-4o-mini-2024-07-18", prompt='./templates/fact_score.txt'):
        # Set up model
        self.max_length = 1024
        self.model = model
        self.prompt = prompt
    
    @retry(retry=retry_if_exception_type((APIError, Timeout, RateLimitError,
                                        ServiceUnavailableError, APIConnectionError, InvalidRequestError)),
        wait=wait_random_exponential(max=60), stop=stop_after_attempt(10),
        before_sleep=before_sleep_log(logger, logging.WARNING))
    def gpt_inference(self, prompt):
        prompt_messages = [{"role": "user", "content": prompt}]
        try:
            response = openai.ChatCompletion.create(model = self.model, messages = prompt_messages, temperature = 0)
            response = response.choices[0].message.content
        except InvalidRequestError:
            response = 1
        return response

    def gpt_score(self, srcs, tgts, kgs, batch_size=4):
        ### Taken from 
        """Score a batch of examples"""
        score_list = []
        feedback_list = []

        for i in range(len(srcs)):
            src = srcs[i]
            tgt = tgts[i]
            

            prompt = build_fact_prompt(
                prompt_template = self.prompt,
                input_text_list=[src, kgs, tgt],
            )
            try:
                score = self.gpt_inference(prompt)
                if '1' in score:
                    score_list.append(float(1))
                    feedback_list.append('')
                else:
                    score_list.append(float(0))
                    feedback_list.append(score)
            
            except RuntimeError:
                traceback.print_exc()
                print(f"source: {src_list}")
                print(f"target: {tgt_list}")
                exit(0)
        return score_list, feedback_list