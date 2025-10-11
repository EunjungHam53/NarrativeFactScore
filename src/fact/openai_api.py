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

import google.generativeai as genai

@retry(retry=retry_if_exception_type((google_exceptions.ResourceExhausted,
                                      google_exceptions.ServiceUnavailable)),
       wait=wait_random_exponential(max=60), stop=stop_after_attempt(10),
       before_sleep=before_sleep_log(logger, logging.WARNING))
def openai_api_response(prompt, model, save_path=None):
    """Use Gemini API"""
    try:
        gemini_model = genai.GenerativeModel(model)
        response = gemini_model.generate_content(
            prompt,
            generation_config=genai.types.GenerationConfig(temperature=0.0)
        )
        return response.text
    except Exception as e:
        logger.error(f'Error: {e}. Returning empty string.')
        return ' '
