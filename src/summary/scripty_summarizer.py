
import os
import math
from typing import Union, Optional
import logging

import google.generativeai as genai
from config import GEMINI_API_KEY, GEMINI_MODEL

genai.configure(api_key=GEMINI_API_KEY)

logger = logging.getLogger(__name__)

class ScriptySummarizer:
    def __init__(self,
                 model:str = GEMINI_MODEL,
                 seed=42,
                 ) -> None:
        self.model = model
        self.seed = seed
    
    def inference_with_gpt(self, prompt):
        try:
            model = genai.GenerativeModel(self.model)
            response = model.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(temperature=0.01)
            )
            response = response.text
        except Exception:
            response = ''
        
        return response
    