
import os
import math
from typing import Union, Optional
import logging
from openai.error import (APIError, RateLimitError, ServiceUnavailableError,
                          Timeout, APIConnectionError, InvalidRequestError)
from tenacity import (before_sleep_log, retry, retry_if_exception_type,
                      stop_after_delay, wait_random_exponential, stop_after_attempt)

import google.generativeai as genai

genai.configure(api_key=os.getenv('GEMINI_API_KEY'))

logger = logging.getLogger(__name__)

class ScriptySummarizer:
    def __init__(self, model:str = "gemini-1.5-flash", seed=42) -> None:
        self.model = model
        self.seed = seed
        self.gemini_model = genai.GenerativeModel(model)
    
    @retry(retry=retry_if_exception_type((google_exceptions.ResourceExhausted,)),
           wait=wait_random_exponential(max=60), stop=stop_after_attempt(10),
           before_sleep=before_sleep_log(logger, logging.WARNING))
    def inference_with_gpt(self, prompt):
        try:
            response = self.gemini_model.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(temperature=0.01)
            )
            return response.text
        except Exception as e:
            print(f"Gemini API error: {e}")
            return ''
    