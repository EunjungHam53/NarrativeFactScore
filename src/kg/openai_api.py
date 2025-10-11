import os
import logging
from pathlib import Path

import google.generativeai as genai
from config import GEMINI_API_KEY, GEMINI_MODEL

logger = logging.getLogger(__name__)

genai.configure(api_key=GEMINI_API_KEY)

def save_openai_api_response(prompt_messages, save_path):
    """Use a prompt to make a request to the Gemini API and save the response to a JSON file."""
    # Gemini không hỗ trợ system messages như OpenAI
    # Kết hợp messages thành một prompt
    prompt = ""
    for msg in prompt_messages:
        if msg["role"] == "user":
            prompt = msg["content"]
    
    try:
        logger.info(f'Calling Gemini API and saving response to {save_path}...')
        model = genai.GenerativeModel(GEMINI_MODEL)
        response = model.generate_content(prompt)
        
        save_data = {
            'model': GEMINI_MODEL,
            'usage': None,
            'finish_reason': 'stop',
            'prompt_messages': prompt_messages,
            'response': response.text
        }
    except Exception as e:
        logger.error(f'Error calling Gemini API: {str(e)}')
        save_data = {
            'model': GEMINI_MODEL,
            'usage': None,
            'finish_reason': 'error',
            'prompt_messages': prompt_messages,
            'response': ' '
        }
    
    save_path.parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, 'w') as save_file:
        import json
        json.dump(save_data, save_file, indent=4, ensure_ascii=False)


def load_response_text(save_path):
    """Load the response text from a JSON file containing response data from the Gemini API."""
    import json
    with open(save_path, 'r') as save_file:
        save_data = json.load(save_file)
    return save_data['response']