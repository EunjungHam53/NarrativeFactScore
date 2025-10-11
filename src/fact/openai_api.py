import json
import logging
import os
from pathlib import Path

from dotenv import load_dotenv

logger = logging.getLogger(__name__)

load_dotenv()

import google.generativeai as genai
from config import GEMINI_API_KEY, GEMINI_MODEL

load_dotenv()
genai.configure(api_key=GEMINI_API_KEY)

OPENAI_MODEL_ID = GEMINI_MODEL

def openai_api_response(prompt, model, save_path=None):
    try:
        logger.info(f'Calling Gemini API...')
        model_instance = genai.GenerativeModel(model)
        response = model_instance.generate_content(prompt)
        
        save_data = {
            'model': model,
            'usage': None,
            'finish_reason': 'stop',
            'prompt_messages': [{"role": "user", "content": prompt}],
            'response': response.text
        }
    except Exception as e:
        logger.error(f'Error calling Gemini API: {str(e)}')
        save_data = {
            'model': model,
            'usage': None,
            'finish_reason': 'error',
            'prompt_messages': [{"role": "user", "content": prompt}],
            'response': ' '
        }
    return save_data['response']
