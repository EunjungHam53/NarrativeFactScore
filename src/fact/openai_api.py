import json
import logging
import os
from pathlib import Path
from dotenv import load_dotenv

logger = logging.getLogger(__name__)
load_dotenv()

import google.generativeai as genai
from config import GEMINI_MODEL, GEMINI_API_KEYS, GEMINI_RPM_LIST, GEMINI_RPD_LIST

try:
    from api_key_manager import APIKeyManager
    api_manager = APIKeyManager(GEMINI_API_KEYS, GEMINI_RPM_LIST, GEMINI_RPD_LIST)
    USE_API_MANAGER = True
    logger.info(f"✓ src/fact/openai_api.py: Khởi tạo APIKeyManager với {len(GEMINI_API_KEYS)} keys")
except Exception as e:
    logger.warning(f"✗ src/fact/openai_api.py: Không thể khởi tạo APIKeyManager: {e}")
    USE_API_MANAGER = False
    api_manager = None

OPENAI_MODEL_ID = GEMINI_MODEL

def openai_api_response(prompt, model, save_path=None):
    try:
        # Lấy API key động
        if USE_API_MANAGER:
            current_api_key = api_manager.get_available_key(max_wait_time=300)
            if not current_api_key:
                raise RuntimeError("Không có API key khả dụng")
            logger.info(f'Calling Gemini API with available key (fact)...')
        else:
            from config import GEMINI_API_KEY
            current_api_key = GEMINI_API_KEY
            logger.info(f'Calling Gemini API with default key (fact)...')
        
        # ========== MỚI: Cấu hình genai động ==========
        genai.configure(api_key=current_api_key)
        
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
        logger.error(f'Error calling Gemini API (fact): {str(e)}')
        save_data = {
            'model': model,
            'usage': None,
            'finish_reason': 'error',
            'prompt_messages': [{"role": "user", "content": prompt}],
            'response': ' '
        }
    
    # ========== MỚI: Lưu state ==========
    if USE_API_MANAGER:
        try:
            api_manager.save_state('./logs/api_manager_state_fact.json')
        except Exception as e:
            logger.warning(f"Không thể lưu API manager state: {e}")
    
    return save_data['response']

# ========== MỚI: Thêm helper functions ==========
def get_api_status():
    if USE_API_MANAGER:
        return api_manager.get_status()
    else:
        return {'status': 'API Manager không được kích hoạt (fact)'}

def reset_api_quotas():
    if USE_API_MANAGER:
        api_manager.reset_all()
        logger.info("✓ Reset API key quotas (fact)")