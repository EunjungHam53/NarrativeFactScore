import os
from typing import List, Optional


def build_summarizer_prompt(
    prompt_template:str,
    input_text_list:List[str]) -> str:
    
    """_summary_
    chat_mode(str) : 'hf-chat', 'kullm', 'None'

    Returns:
        _type_: _description_
    """
    
    if os.path.isfile(prompt_template):
        with open(prompt_template,'r') as f:
            prompt_template = f.read() 
    else:
        pass
    
    assert isinstance(prompt_template, str)
    
    prompt = prompt_template.format(*input_text_list)
    
    return prompt

def get_prompt_template_input_length(prompt_template:str) -> int:
    
    if os.path.isfile(prompt_template):
        with open(prompt_template,'r') as f:
            prompt_template = f.read() 
    else:
        pass

    prompt_template_input_len = prompt_template.count("{}")

    return prompt_template_input_len