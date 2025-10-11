# config.py
import os
from dotenv import load_dotenv

load_dotenv()

# Gemini API Config
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
GEMINI_MODEL = 'gemini-1.5-flash'  # hoặc 'gemini-1.5-pro'

# Token limits
MAX_INPUT_TOKENS = 30000
MAX_OUTPUT_TOKENS = 1000