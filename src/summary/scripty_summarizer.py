import os
import math
from typing import Union, Optional
import logging
import time

import google.generativeai as genai
from config import GEMINI_API_KEY, GEMINI_MODEL, GEMINI_API_KEYS, GEMINI_RPM_LIST, GEMINI_RPD_LIST

# genai.configure(api_key=GEMINI_API_KEY)

logger = logging.getLogger(__name__)

try:
    from api_key_manager import APIKeyManager
    api_manager = APIKeyManager(GEMINI_API_KEYS, GEMINI_RPM_LIST, GEMINI_RPD_LIST)
    USE_API_MANAGER = True
    logger.info(f"✓ src/summary/scripty_summarizer.py: Khởi tạo APIKeyManager với {len(GEMINI_API_KEYS)} keys")
except Exception as e:
    logger.warning(f"✗ src/summary/scripty_summarizer.py: Không thể khởi tạo APIKeyManager: {e}")
    USE_API_MANAGER = False
    api_manager = None

class ScriptySummarizer:
    def __init__(self,
                 model: str = GEMINI_MODEL,
                 seed=42,
                 ) -> None:
        self.model = model
        self.seed = seed
    
    def inference_with_gpt(self, prompt):
        """
        ========== SỬA: Thêm API Key Manager ==========
        Sửa từ: Sử dụng API key cố định
        Thành: Lấy API key động từ manager
        """
        try:
            # ========== MỚI: Lấy API key ==========
            if USE_API_MANAGER:
                current_api_key = api_manager.get_available_key(max_wait_time=300)
                if not current_api_key:
                    raise RuntimeError("Không có API key khả dụng")
                logger.info(f'Calling Gemini API for summarization...')
            else:
                from config import GEMINI_API_KEY
                current_api_key = GEMINI_API_KEY
                logger.info(f'Calling Gemini API with default key for summarization...')
            
            # ========== MỚI: Cấu hình genai động ==========
            genai.configure(api_key=current_api_key)
            
            model = genai.GenerativeModel(self.model)
            response = model.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=0.01,
                    max_output_tokens=2048
                )
            )
            response = response.text.strip()
            if not response:
                logger.warning("API returned empty response")
                return ""
            return response
        except Exception as e:
            logger.error(f"Error in inference_with_gpt: {str(e)}")
            return ""
    
    # ========== MỚI: Thêm helper functions ==========
    def get_api_status(self):
        """Lấy trạng thái API keys"""
        if USE_API_MANAGER:
            return api_manager.get_status()
        else:
            return {'status': 'API Manager không được kích hoạt (summary)'}
    
    def reset_api_quotas(self):
        """Reset quota"""
        if USE_API_MANAGER:
            api_manager.reset_all()
            logger.info("✓ Reset API key quotas (summary)")