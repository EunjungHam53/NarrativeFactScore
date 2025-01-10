
import os
import math
from typing import Union, Optional
import logging

import openai
from openai.error import (APIError, RateLimitError, ServiceUnavailableError,
                          Timeout, APIConnectionError, InvalidRequestError)
from tenacity import (before_sleep_log, retry, retry_if_exception_type,
                      stop_after_delay, wait_random_exponential, stop_after_attempt)

openai.api_key = os.getenv('OPENAI_API_KEY')
logger = logging.getLogger(__name__)

class ScriptySummarizer:
    def __init__(self,
                 model:str,
                 seed=42,
                 ) -> None:
        self.model = model
        self.seed = seed
    
    @retry(retry=retry_if_exception_type((APIError, Timeout, RateLimitError,
                                        ServiceUnavailableError, APIConnectionError, InvalidRequestError)),
        wait=wait_random_exponential(max=60), stop=stop_after_attempt(10),
        before_sleep=before_sleep_log(logger, logging.WARNING))
    def inference_with_gpt(self, prompt):
        prompt_messages = [{"role": "user", "content": prompt}]
        try:
            response = openai.ChatCompletion.create(model = self.model, messages = prompt_messages, temperature = 0.01)
            response = response.choices[0].message.content
        except InvalidRequestError:
            response = ''
        
        return response
    