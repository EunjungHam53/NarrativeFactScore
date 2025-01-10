import json
import logging
import os
from pathlib import Path

import openai
from dotenv import load_dotenv
from openai.error import (APIError, RateLimitError, ServiceUnavailableError,
                          Timeout, APIConnectionError, InvalidRequestError)
from tenacity import (before_sleep_log, retry, retry_if_exception_type,
                      stop_after_delay, wait_random_exponential, stop_after_attempt)
from tiktoken import Encoding, encoding_for_model

logger = logging.getLogger(__name__)

load_dotenv()
openai.api_key = os.getenv('OPENAI_API_KEY')

OPENAI_MODEL_ID = 'gpt-4o-mini'
# This value is set by OpenAI for the selected model and cannot be changed.
MAX_MODEL_TOKEN_COUNT = 4096
# This value can be changed.
MAX_RESPONSE_TOKEN_COUNT = 512



@retry(retry=retry_if_exception_type((APIError, Timeout, RateLimitError,
                                      ServiceUnavailableError, APIConnectionError, InvalidRequestError)),
       wait=wait_random_exponential(max=60), stop=stop_after_attempt(10),
       before_sleep=before_sleep_log(logger, logging.WARNING))
def openai_api_response(prompt, model,
                             save_path=None):
    """
    Use a prompt to make a request to the OpenAI API and save the response to a
    JSON file.
    """
    try:
        prompt_messages = [{"role": "user", "content": prompt}]
        #import ipdb;ipdb.set_trace(context=10)
        logger.info(f'Calling OpenAI API and saving response to {save_path}...')
        response = openai.ChatCompletion.create(
            model=OPENAI_MODEL_ID, messages=prompt_messages, temperature=0)
        finish_reason = response.choices[0].finish_reason
        if finish_reason != 'stop':
            logger.error(f'`finish_reason` is `{finish_reason}` for {save_path}.')
        save_data = {'model': response.model, 'usage': response.usage,
                     'finish_reason': finish_reason,
                     'prompt_messages': prompt_messages,
                     'response': response.choices[0].message.content}
    except InvalidRequestError:
        logger.error(f'InvalidRequestError encountered 10 times. Returning empty string for {save_path}.')
        save_data = {'model': None, 'usage': None,
                     'finish_reason': 'invalid_request',
                     'prompt_messages': prompt_messages,
                     'response': ' '}
    return save_data['response']
