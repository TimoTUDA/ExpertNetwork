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
    """
    request["response_format"] = {
        'type': 'json_object'
    }
    if "model" in request:
        if request["model"] == "deepseek-r1:70b":
            request["format"] = {
            "type": "object",
            "properties": {
                "Option": {
                    "type": "string",
                    "enum": ["A", "B", "C", "Z"],
                    "description": "Specifies which option is chosen, as defined in the instructions."
                },
                "thinking": {
                    "type": "string",
                    "description": "A chain-of-thought or explanation detailing your reasoning."
                },
                "Subquestion": {
                    "type": "string",
                    "description": "The subquestion generated. Required for Option A and Option B."
                },
                "Reasoning": {
                    "type": "string",
                    "description": "A concise reasoning for the chosen subquestion. Required for Option A."
                },
                "SubSQL": {
                    "type": "string",
                    "description": "The SQL query corresponding to the subquestion. Required for Option B."
                },
                "finalGeneratedQuery": {
                    "type": "string",
                    "description": "The final SQL query that answers the original question. Required for Option C."
                }
            },
            "required": ["Option", "thinking"],
            "additionalProperties": False,
            "description": (
                "This JSON object is the output structure the model should generate. Depending on the "
                "option chosen (A, B, C, or Z), the response should include the appropriate additional keys. "
                "The 'thinking' property can be used to provide a chain-of-thought."
            )
            }
    """
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
            {"role": "system", "content": system_prompt},
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

