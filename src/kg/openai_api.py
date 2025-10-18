import os
import logging
from pathlib import Path

import google.generativeai as genai
from config import GEMINI_API_KEY, GEMINI_MODEL, GEMINI_API_KEYS, GEMINI_RPM_LIST, GEMINI_RPD_LIST

logger = logging.getLogger(__name__)

# ========== MỚI: Import API Key Manager ==========
try:
    from api_key_manager import APIKeyManager
    api_manager = APIKeyManager(GEMINI_API_KEYS, GEMINI_RPM_LIST, GEMINI_RPD_LIST)
    USE_API_MANAGER = True
    logger.info(f"✓ src/kg/openai_api.py: Khởi tạo APIKeyManager với {len(GEMINI_API_KEYS)} keys")
except Exception as e:
    logger.warning(f"✗ src/kg/openai_api.py: Không thể khởi tạo APIKeyManager: {e}")
    USE_API_MANAGER = False
    api_manager = None

def save_openai_api_response(prompt_messages, save_path):
    prompt = ""
    for msg in prompt_messages:
        if msg["role"] == "user":
            prompt = msg["content"]
    
    try:
        # ========== MỚI: Lấy API key động ==========
        if USE_API_MANAGER:
            current_api_key = api_manager.get_available_key(max_wait_time=300)
            if not current_api_key:
                raise RuntimeError("Không có API key khả dụng")
            logger.info(f'Calling Gemini API and saving response to {save_path}... (kg)')
        else:
            current_api_key = GEMINI_API_KEY
            logger.info(f'Calling Gemini API and saving response to {save_path}... (kg)')
        
        # ========== MỚI: Cấu hình genai động ==========
        genai.configure(api_key=current_api_key)
        
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
        logger.error(f'Error calling Gemini API (kg): {str(e)}')
        save_data = {
            'model': GEMINI_MODEL,
            'usage': None,
            'finish_reason': 'error',
            'prompt_messages': prompt_messages,
            'response': ' '
        }
    
    # ========== MỚI: Lưu state ==========
    if USE_API_MANAGER:
        try:
            api_manager.save_state('./logs/api_manager_state_kg.json')
        except Exception as e:
            logger.warning(f"Không thể lưu API manager state: {e}")
    
    save_path.parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, 'w') as save_file:
        import json
        json.dump(save_data, save_file, indent=4, ensure_ascii=False)

def load_response_text(save_path):
    import json
    with open(save_path, 'r') as save_file:
        save_data = json.load(save_file)
    return save_data['response']

# ========== MỚI: Thêm helper functions ==========
def get_api_status():
    if USE_API_MANAGER:
        return api_manager.get_status()
    else:
        return {'status': 'API Manager không được kích hoạt (kg)'}

def reset_api_quotas():
    if USE_API_MANAGER:
        api_manager.reset_all()
        logger.info("✓ Reset API key quotas (kg)")