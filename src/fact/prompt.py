import os
from typing import List, Optional


def build_fact_prompt(
    prompt_template:str,
    input_text_list:List[str],
    chat_mode:Optional[str] = None) -> str:
    
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
    
    # 예외처리 필요
    assert isinstance(prompt_template, str)
    
    prompt = prompt_template.format(*input_text_list)
    
    if chat_mode == "hf-chat":
        prompt = _get_hf_chat_template().format(prompt)
    elif chat_mode == "kullm":
        prompt = _get_kullm_template().format(prompt)
    
    return prompt