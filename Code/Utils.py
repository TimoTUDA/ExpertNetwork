import Agent, DatabaseConnector
from my_openai_utils import openai_execute
import json
import os
import DatabaseConnector

def construct_request_dummy(model, system_prompt, first_message, output_tokens=3500):
    if model == "o3-mini-2025-01-31":
        request_dummy = [{
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": first_message},
        ],
        "max_completion_tokens": output_tokens,
        }]
        print("o3-mini used")
    else:
         request_dummy = [{
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": first_message},
        ],
        "max_tokens": output_tokens,
        }]
    
    return request_dummy