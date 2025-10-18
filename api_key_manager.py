import time
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Optional
from dataclasses import dataclass, field
import json
from pathlib import Path

logger = logging.getLogger(__name__)

@dataclass
class APIKeyQuota:
    """Quản lý quota cho từng API key"""
    api_key: str
    rpm: int  # Request per minute
    rpd: int  # Request per day
    requests_this_minute: int = 0
    requests_this_day: int = 0
    minute_start_time: float = field(default_factory=time.time)
    day_start_time: float = field(default_factory=time.time)
    is_exhausted: bool = False
    exhausted_at: Optional[float] = None
    
    def reset_minute_counter(self):
        """Reset bộ đếm phút"""
        self.requests_this_minute = 0
        self.minute_start_time = time.time()
    
    def reset_day_counter(self):
        """Reset bộ đếm ngày"""
        self.requests_this_day = 0
        self.day_start_time = time.time()
    
    def can_make_request(self) -> bool:
        """Kiểm tra xem có thể gửi request không"""
        current_time = time.time()
        
        # Kiểm tra nếu đã qua 1 phút
        if current_time - self.minute_start_time >= 60:
            self.reset_minute_counter()
        
        # Kiểm tra nếu đã qua 1 ngày (86400 giây)
        if current_time - self.day_start_time >= 86400:
            self.reset_day_counter()
        
        # Kiểm tra quota
        if self.requests_this_minute >= self.rpm:
            logger.warning(f"API key {self.api_key[:10]}... đã hết RPM quota ({self.rpm})")
            return False
        
        if self.requests_this_day >= self.rpd:
            logger.warning(f"API key {self.api_key[:10]}... đã hết RPD quota ({self.rpd})")
            self.is_exhausted = True
            self.exhausted_at = current_time
            return False
        
        return True
    
    def increment_request(self):
        """Tăng số request"""
        self.requests_this_minute += 1
        self.requests_this_day += 1
    
    def get_wait_time(self) -> float:
        """Tính thời gian cần chờ (giây)"""
        current_time = time.time()
        wait_minute = max(0, 60 - (current_time - self.minute_start_time))
        return wait_minute
    
    def get_status(self) -> Dict:
        """Lấy trạng thái hiện tại"""
        return {
            'api_key': self.api_key[:10] + '...',
            'rpm_usage': f"{self.requests_this_minute}/{self.rpm}",
            'rpd_usage': f"{self.requests_this_day}/{self.rpd}",
            'is_exhausted': self.is_exhausted,
            'can_use': self.can_make_request()
        }


class APIKeyManager:
    """Quản lý nhiều API keys với auto-switching"""
    
    def __init__(self, api_keys: List[str], rpm_list: List[int], rpd_list: List[int]):
        """
        Args:
            api_keys: Danh sách API keys
            rpm_list: Danh sách RPM tương ứng
            rpd_list: Danh sách RPD tương ứng
        """
        if not (len(api_keys) == len(rpm_list) == len(rpd_list)):
            raise ValueError("Số lượng API keys, RPM, và RPD phải bằng nhau")
        
        self.quota_list: List[APIKeyQuota] = [
            APIKeyQuota(api_key=key, rpm=rpm, rpd=rpd)
            for key, rpm, rpd in zip(api_keys, rpm_list, rpd_list)
        ]
        self.current_index = 0
        self.total_requests = 0
        logger.info(f"Khởi tạo APIKeyManager với {len(api_keys)} keys")
    
    def get_available_key(self, max_wait_time: int = 60) -> Optional[str]:
        """
        Lấy API key khả dụng, chờ nếu cần
        
        Args:
            max_wait_time: Thời gian chờ tối đa (giây)
        
        Returns:
            API key hoặc None nếu hết tất cả quota
        """
        start_time = time.time()
        attempts = 0
        max_attempts = len(self.quota_list) * 3
        
        while attempts < max_attempts:
            current_quota = self.quota_list[self.current_index]
            
            if current_quota.can_make_request():
                current_quota.increment_request()
                self.total_requests += 1
                logger.info(
                    f"Sử dụng API key #{self.current_index} "
                    f"({current_quota.api_key[:10]}...) - "
                    f"Total requests: {self.total_requests}"
                )
                return current_quota.api_key
            
            # Chuyển sang key tiếp theo
            self.current_index = (self.current_index + 1) % len(self.quota_list)
            
            # Nếu đã duyệt hết tất cả keys, chờ một chút
            if self.current_index == 0:
                elapsed = time.time() - start_time
                if elapsed > max_wait_time:
                    logger.error("Hết tất cả API key quota và hết thời gian chờ")
                    self._log_all_status()
                    return None
                
                wait_time = 5
                logger.warning(
                    f"Tất cả keys bận, chờ {wait_time}s trước khi thử lại "
                    f"(elapsed: {elapsed:.1f}s / max: {max_wait_time}s)"
                )
                time.sleep(wait_time)
            
            attempts += 1
        
        logger.error(f"Không tìm được API key khả dụng sau {max_attempts} lần thử")
        self._log_all_status()
        return None
    
    def _log_all_status(self):
        """In trạng thái tất cả API keys"""
        logger.info("=== Trạng thái tất cả API keys ===")
        for idx, quota in enumerate(self.quota_list):
            status = quota.get_status()
            logger.info(f"Key #{idx}: {status}")
    
    def get_status(self) -> Dict:
        """Lấy trạng thái của tất cả keys"""
        return {
            'current_index': self.current_index,
            'total_requests': self.total_requests,
            'keys_status': [q.get_status() for q in self.quota_list]
        }
    
    def reset_all(self):
        """Reset tất cả counters"""
        for quota in self.quota_list:
            quota.reset_minute_counter()
            quota.reset_day_counter()
            quota.is_exhausted = False
        logger.info("Reset tất cả API key counters")
    
    def save_state(self, filepath: str):
        """Lưu trạng thái hiện tại"""
        state = {
            'timestamp': datetime.now().isoformat(),
            'total_requests': self.total_requests,
            'current_index': self.current_index,
            'quota_status': [
                {
                    'api_key': q.api_key[:10] + '...',
                    'requests_this_minute': q.requests_this_minute,
                    'requests_this_day': q.requests_this_day,
                    'is_exhausted': q.is_exhausted,
                }
                for q in self.quota_list
            ]
        }
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, 'w') as f:
            json.dump(state, f, indent=2)
        logger.info(f"Đã lưu state tại {filepath}")


# ============ HƯỚNG DẪN SỬ DỤNG ============

"""
# 1. Cấu hình trong config.py:
GEMINI_API_KEYS = [
    "sk-key1-xxx",
    "sk-key2-yyy",
    "sk-key3-zzz",
]
GEMINI_RPM_LIST = [100, 100, 150]  # Request per minute
GEMINI_RPD_LIST = [10000, 10000, 15000]  # Request per day

# 2. Sử dụng trong openai_api.py:
from config import GEMINI_API_KEYS, GEMINI_RPM_LIST, GEMINI_RPD_LIST
from api_key_manager import APIKeyManager

api_manager = APIKeyManager(GEMINI_API_KEYS, GEMINI_RPM_LIST, GEMINI_RPD_LIST)

def openai_api_response(prompt, model, save_path=None):
    try:
        # Lấy API key khả dụng
        api_key = api_manager.get_available_key(max_wait_time=300)
        if not api_key:
            raise RuntimeError("Không có API key khả dụng")
        
        logger.info(f'Calling Gemini API with key...')
        genai.configure(api_key=api_key)  # ← Sửa từ cấu hình tĩnh
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
    
    # Lưu state định kỳ
    api_manager.save_state('./logs/api_manager_state.json')
    return save_data['response']
"""