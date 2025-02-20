import Agent, DatabaseConnector
from my_openai_utils import openai_execute
import json
import os
import copy

def prepare_for_ollama(request: dict) -> dict:
    """Prepare Ollama API request.

    Args:
        request: The API request.

    Returns:
        The API request prepared for the Ollama API.
    """
    request = copy.deepcopy(request)

    # move `seed` to options
    if "seed" in request.keys():
        if "options" not in request.keys():
            request["options"] = {}
        request["options"]["seed"] = request["seed"]
        del request["seed"]

    # move `temperature` to options
    if "temperature" in request.keys():
        if "options" not in request.keys():
            request["options"] = {}
        request["options"]["temperature"] = request["temperature"]
        del request["temperature"]

    # move `max_tokens` to options as `num_predict`
    max_tokens = None
    if "max_completion_tokens" in request.keys():
        max_tokens = request["max_completion_tokens"]
        del request["max_completion_tokens"]
    elif "max_tokens" in request.keys():
        max_tokens = request["max_tokens"]
        del request["max_tokens"]

    if max_tokens is not None:
        if "options" not in request.keys():
            request["options"] = {}
        request["options"]["num_predict"] = max_tokens

    if "options" not in request.keys():
        request["options"] = {}
        request["options"]["num_ctx"] = 4096
        request["options"]["seed"] = 1712
        request["options"]["temperature"] = 0
    else:
        request["options"]["num_ctx"] = 126000

    # set `stream` to False
    request["stream"] = False
    
    return request

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
    elif model == "deepseek-r1:70b":
        request_dummy = [{
        "model": model,
        "messages": [
            #{"role": "system", "content": system_prompt},
            {"role": "user", "content": first_message},
        ],
        "response_format":{
        'type': 'json_object'
        },
        "max_tokens": output_tokens,
        }]
        return prepare_for_ollama(request_dummy)
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
