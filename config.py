# config.py
import os
from dotenv import load_dotenv

load_dotenv()

# Gemini API Config
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
GEMINI_MODEL = os.getenv('GEMINI_MODEL')

# Token limits
MAX_INPUT_TOKENS = os.getenv('MAX_INPUT_TOKENS')
MAX_OUTPUT_TOKENS = os.getenv('MAX_OUTPUT_TOKENS')