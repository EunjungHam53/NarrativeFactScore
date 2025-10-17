import os
import logging
from typing import List, Optional

# Configure logging
logger = logging.getLogger(__name__)

def build_summarizer_prompt(
    prompt_template: str,
    input_text_list: List[str]) -> str:
    
    if os.path.isfile(prompt_template):
        with open(prompt_template, 'r', encoding='utf-8') as f:
            prompt_template = f.read()
    
    assert isinstance(prompt_template, str)
    
    try:
        # Đếm số {} trong template
        placeholder_count = prompt_template.count('{}')
        if placeholder_count != len(input_text_list):
            logger.warning(
                f"Placeholder mismatch: template có {placeholder_count}, "
                f"nhưng nhận {len(input_text_list)} input"
            )
        
        try:
            prompt = prompt_template.format(*input_text_list)
        except (IndexError, KeyError) as e:
            logger.error(f"Format error: {e}. Template placeholders không match input.")
            prompt = prompt_template
    except IndexError as e:
        logger.error(f"Format error: {str(e)}")
        prompt = prompt_template
    
    return prompt

def get_prompt_template_input_length(prompt_template:str) -> int:
    
    if os.path.isfile(prompt_template):
        with open(prompt_template,'r') as f:
            prompt_template = f.read() 
    else:
        pass

    prompt_template_input_len = prompt_template.count("{}")

    return prompt_template_input_len