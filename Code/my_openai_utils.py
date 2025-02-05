########################################################################################################################
# OpenAI API helpers version: 2024-07-22
#
# use the following methods:
# openai_model(...)        ==> get information about the models
# openai_execute(...)      ==> execute API requests
# openai_cost_for_cache()  ==> compute the total dollar cost incurred by all cached requests/responses
#
# You must store your OpenAI API key in an environment variable, for example using:
# export OPENAI_API_KEY="<your-key>"
#
# To call openai_execute(...) from multiple processes, you must use a global context:
# with multiprocessing.Manager() as manager:
#     context = manager.dict()
#     semaphore = manager.Semaphore()
#     # every call now requires the context and semaphore:
#     responses = openai_execute(requests, global_context=context, global_context_semaphore=semaphore)
#
# next steps: shared limits, local and global context in the same session
########################################################################################################################
import collections
import dataclasses
import datetime
import functools
import hashlib
import json
import logging
import os
import pathlib
import threading
import time

import requests
import tiktoken
import tqdm

logger = logging.getLogger(__name__)

_completion_url = "https://api.openai.com/v1/completions"
_chat_url = "https://api.openai.com/v1/chat/completions"
_additional_tokens_per_message = 10
_cost_for_failed_requests = 0.0
_usage_for_failed_requests = 0
_wait_before_try = 0.001
_wait_before_retry = 0.1
_api_share = 0.45
_cache_path = pathlib.Path("openai_cache")  # TODO: set path
_cache_size = 100_000
_local_context = {}

# pricing: https://openai.com/pricing
# context: https://platform.openai.com/docs/models
# limits: https://platform.openai.com/account/limits
_model_parameters = {  # TODO: update parameters according to the OpenAI project
    "gpt-3.5-turbo-1106": {
        "chat_or_completion": "chat",
        "max_rpm": 10_000,
        "max_tpm": 50_000_000,
        "cost_per_1k_input_tokens": 0.001,
        "cost_per_1k_output_tokens": 0.002,
        "max_context": 16_385,
        "max_output_tokens": 4_096
    },
    "gpt-3.5-turbo-0125": {
        "chat_or_completion": "chat",
        "max_rpm": 10_000,
        "max_tpm": 50_000_000,
        "cost_per_1k_input_tokens": 0.0005,
        "cost_per_1k_output_tokens": 0.0015,
        "max_context": 16_385,
        "max_output_tokens": 4_096
    },
    "gpt-3.5-turbo-instruct-0914": {
        "chat_or_completion": "completion",
        "max_rpm": 3_500,
        "max_tpm": 90_000,
        "cost_per_1k_input_tokens": 0.0015,
        "cost_per_1k_output_tokens": 0.002,
        "max_context": 4_096,
        "max_output_tokens": None
    },
    "gpt-4-1106-preview": {
        "chat_or_completion": "chat",
        "max_rpm": 10_000,
        "max_tpm": 2_000_000,
        "cost_per_1k_input_tokens": 0.01,
        "cost_per_1k_output_tokens": 0.03,
        "max_context": 128_000,
        "max_output_tokens": 4_096
    },
    "gpt-4-0125-preview": {
        "chat_or_completion": "chat",
        "max_rpm": 10_000,
        "max_tpm": 2_000_000,
        "cost_per_1k_input_tokens": 0.01,
        "cost_per_1k_output_tokens": 0.03,
        "max_context": 128_000,
        "max_output_tokens": 4_096
    },
    "gpt-4-turbo-2024-04-09": {
        "chat_or_completion": "chat",
        "max_rpm": 10_000,
        "max_tpm": 2_000_000,
        "cost_per_1k_input_tokens": 0.01,
        "cost_per_1k_output_tokens": 0.03,
        "max_context": 128_000,
        "max_output_tokens": 4_096  # ?
    },
    "gpt-4o-2024-05-13": {
        "chat_or_completion": "chat",
        "max_rpm": 10_000,
        "max_tpm": 30_000_000,
        "cost_per_1k_input_tokens": 0.005,
        "cost_per_1k_output_tokens": 0.015,
        "max_context": 128_000,
        "max_output_tokens": 4_096  # ?
    },
    "gpt-4o-mini-2024-07-18": {
        "chat_or_completion": "chat",
        "max_rpm": 10_000,
        "max_tpm": 15_000_000,
        "cost_per_1k_input_tokens": 0.00015,
        "cost_per_1k_output_tokens": 0.0006,
        "max_context": 128_000,
        "max_output_tokens": 4_096  # ?
    }
}


@functools.cache
def _get_model_params(model: str) -> dict:
    if model not in _model_parameters.keys():
        raise AssertionError(f"Unknown model '{model}'!")
    else:
        return _model_parameters[model]


@functools.cache
def _get_encoding_cached(model: str) -> tiktoken.Encoding:
    return tiktoken.encoding_for_model(model)


class _Request:
    request: dict

    def __init__(self, request: dict) -> None:
        self.request = request

    @functools.cached_property
    def model(self) -> str:
        if "model" not in self.request.keys():
            raise AttributeError("The request is missing the required field `model`!")
        return self.request["model"]

    @functools.cached_property
    def messages(self) -> list[dict]:
        if "messages" not in self.request.keys():
            raise AttributeError("The request is missing the field `messages`, which is required for this model!")
        return self.request["messages"]

    @functools.cached_property
    def prompt(self) -> str:
        if "prompt" not in self.request.keys():
            raise AttributeError("The request is missing the field `prompt`, which is required for this model!")
        return self.request["prompt"]

    @functools.cache
    def is_chat_or_completion(self) -> str:  # use `model` to determine if request is for chat or completion
        return _get_model_params(self.model)["chat_or_completion"]

    @functools.cache
    def estimate_input_tokens(self) -> int:
        #encoding = _get_encoding_cached(self.model)
        encoding = tiktoken.get_encoding("cl100k_base")
        
        if self.is_chat_or_completion() == "chat":
            extra_tokens = _additional_tokens_per_message
            return sum(len(encoding.encode(message["content"])) + extra_tokens for message in self.messages)
        elif self.is_chat_or_completion() == "completion":
            return len(encoding.encode(self.prompt))
        else:
            raise AssertionError(f"Invalid parameter `chat_or_completion` for model '{self.model}'!")

    @functools.cache
    def estimate_max_output_tokens(self) -> int:
        if "max_tokens" in self.request.keys() and self.request["max_tokens"] is not None:
            return self.request["max_tokens"]
        else:
            model_params = _get_model_params(self.model)
            left_for_output = max(0, model_params["max_context"] - self.estimate_input_tokens())
            if model_params["max_output_tokens"] is not None and model_params["max_output_tokens"] < left_for_output:
                return model_params["max_output_tokens"]
            else:
                return left_for_output

    @functools.cache
    def estimate_max_total_tokens(self) -> int:
        return self.estimate_input_tokens() + self.estimate_max_output_tokens()

    @functools.cache
    def estimate_input_usage(self) -> int:
        if "best_of" in self.request.keys():
            n = self.request["best_of"]
        elif "n" in self.request.keys():
            n = self.request["n"]
        else:
            n = 1
        return n * self.estimate_input_tokens()

    @functools.cache
    def estimate_max_output_usage(self) -> int:
        if "best_of" in self.request.keys():
            n = self.request["best_of"]
        elif "n" in self.request.keys():
            n = self.request["n"]
        else:
            n = 1
        return n * self.estimate_max_output_tokens()

    @functools.cache
    def estimate_max_total_usage(self) -> int:
        return self.estimate_input_usage() + self.estimate_max_output_usage()

    @functools.cache
    def estimate_max_cost(self) -> float:
        model_params = _get_model_params(self.model)

        input_cost = self.estimate_input_usage() * (model_params["cost_per_1k_input_tokens"] / 1000)
        output_cost = self.estimate_max_output_usage() * (model_params["cost_per_1k_output_tokens"] / 1000)
        return input_cost + output_cost

    @functools.cache
    def compute_hash(self) -> str:
        return hashlib.sha256(bytes(json.dumps(self.request), "utf-8")).hexdigest()

    def check(self) -> None:
        model_params = _get_model_params(self.model)
        estimated_input_tokens = self.estimate_input_tokens()
        if model_params["max_context"] < estimated_input_tokens:
            logger.warning("Unable to process the input tokens due to the model's `max_context` parameter!")

        if model_params["max_context"] == estimated_input_tokens:
            logger.warning("Unable to generate any output tokens due to the model's `max_context` parameter!")

        if "max_tokens" in self.request.keys() and self.request["max_tokens"] is not None:
            if model_params["max_output_tokens"] is not None \
                    and model_params["max_output_tokens"] < self.request["max_tokens"]:
                logger.warning(
                    "Unable to generate `max_tokens` output tokens due to the model's `max_output_tokens` parameter!")

            if model_params["max_context"] < estimated_input_tokens + self.request["max_tokens"]:
                logger.warning(
                    "Unable to generate `max_tokens` output tokens due to the model's `max_context` parameter!")

        #if "seed" not in self.request.keys():
            #logger.warning("The request is missing the optional field `seed`, which is required for reproducibility!")

        #if "temperature" not in self.request.keys() or self.request["temperature"] != 0:
            #logger.warning("The request's field `temperature` is not set to 0, which is required for reproducibility!")

    def load_cached_response(self, cache_file_paths: collections.defaultdict | None = None):  # -> Response | None
        if cache_file_paths is not None:
            matching_file_paths = list(sorted(cache_file_paths[self.compute_hash()]))
        else:
            matching_file_paths = list(sorted(_cache_path.glob(f"*-{self.compute_hash()}.json")))
        if len(matching_file_paths) > 0:
            matching_file_path = matching_file_paths[0]
            with open(matching_file_path, "r", encoding="utf-8") as file:
                cached_pair = json.load(file)
                cached_request = _Request(cached_pair["request"])
                cached_response = _Response(cached_pair["response"])
                if self.request == cached_request.request:
                    return cached_response
        return None

    def execute(self) -> tuple["_Response", bool]:
        response = self.load_cached_response()
        if response is not None:
            return response, True

        if "OPENAI_API_KEY" not in os.environ.keys():
            raise AssertionError(f"Missing `OPENAI_API_KEY` in environment variables!")

        if self.is_chat_or_completion() == "chat":
            url = _chat_url
        elif self.is_chat_or_completion() == "completion":
            url = _completion_url
        else:
            raise AssertionError(f"Invalid parameter `chat_or_completion` for model '{self.model}'!")

        http_response = requests.post(
            url=url,
            json=self.request,
            headers={"Content-Type": "application/json", "Authorization": f"Bearer {os.environ['OPENAI_API_KEY']}"}
        )
        response = _Response(http_response.json())

        if http_response.status_code != 200:
            logger.warning(f"Request failed: {http_response.content}")
        else:
            request_hash = self.compute_hash()
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S-%f")
            cache_file_path = _cache_path / f"{timestamp}-{request_hash}.json"
            with open(cache_file_path, "w", encoding="utf-8") as cache_file:
                json.dump({"request": self.request, "response": response.response}, cache_file)

        return response, False


class _Response:
    response: dict

    def __init__(self, response: dict) -> None:
        self.response = response

    @functools.cached_property
    def model(self) -> str:
        if "model" not in self.response.keys():
            raise AttributeError("The response is missing the field `model` which is required for successful requests!")
        return self.response["model"]

    @functools.cached_property
    def usage(self) -> dict:
        if "usage" not in self.response.keys():
            raise AttributeError("The response is missing the field `usage` which is required for successful requests!")
        return self.response["usage"]

    @functools.cache
    def was_successful(self) -> bool:
        return "choices" in self.response.keys()  # use entry `choices` to determine if request was successful

    @functools.cache
    def compute_total_usage(self) -> int:
        if self.was_successful():
            return self.usage["total_tokens"]
        else:
            return _usage_for_failed_requests

    @functools.cache
    def compute_total_cost(self) -> float:
        if self.was_successful():
            model_params = _get_model_params(self.model)
            total_cost = 0
            if "prompt_tokens" in self.usage.keys():
                total_cost += self.usage["prompt_tokens"] * (model_params["cost_per_1k_input_tokens"] / 1000)
            if "completion_tokens" in self.usage.keys():
                total_cost += self.usage["completion_tokens"] * (model_params["cost_per_1k_output_tokens"] / 1000)
            return total_cost
        else:
            return _cost_for_failed_requests


def openai_model(
        model: str
) -> dict:
    """Get information about the OpenAI model.

    Args:
        model: The name of the model.

    Returns:
        A dictionary with information about the OpenAI model.
    """
    return _get_model_params(model)


@dataclasses.dataclass
class _Budget:
    requests: int
    tokens: int
    last_update: float


@dataclasses.dataclass
class _Pair:
    request: _Request
    response: _Response | None = None
    was_cached: bool = False
    usage: int | None = None
    thread: threading.Thread | None = None


def openai_execute(
        requests: list[dict],
        *,
        force: float | None = None,
        silent: bool = False,
        global_context: dict | None = None,
        global_context_semaphore: "multiprocessing.Semaphore | None" = None
) -> list[dict]:
    """Execute a list of requests against the OpenAI API.

    This method also computes the maximum cost incurred by the requests, caches requests and responses, and waits
    between requests to abide the API limits.

    Args:
        requests: A list of API requests.
        force: An optional float specifying the cost below or equal to which no confirmation should be required.
        silent: Whether to display log messages and progress bars.
        global_context: Optional global context to manage API limits.
        global_context_semaphore: Optional semaphore to guard the global context.

    Returns:
        A list of API responses.
    """
    global _local_context

    if (global_context is None) != (global_context_semaphore is None):
        raise AssertionError("You must provide either both the global context and its semaphore or neither.")

    if global_context is not None:
        context = global_context
    else:
        context = _local_context

    pairs = [_Pair(_Request(request)) for request in requests]

    # check requests
    for pair in pairs:
        pair.request.check()

    # create cache directory
    os.makedirs(_cache_path, exist_ok=True)

    # load cached pairs
    cache_file_paths = collections.defaultdict(list)
    for cache_file_path in list(sorted(_cache_path.glob("*.json"))):
        cache_file_paths[cache_file_path.name[27:-5]].append(cache_file_path)

    for pair in pairs:
        pair.response = pair.request.load_cached_response(cache_file_paths=cache_file_paths)
        if pair.response is not None:
            pair.was_cached = True

    pairs_to_execute = [pair for pair in pairs if not pair.was_cached]

    # compute maximum cost
    total_max_cost = sum(pair.request.estimate_max_cost() for pair in pairs_to_execute)
    if force is None or total_max_cost > force:
        logger.info(f"Press enter to continue and spend up to around ${total_max_cost:.2f}.")
        input(f"Press enter to continue and spend up to around ${total_max_cost:.2f}.")
        if not silent:
            logger.info("Begin execution.")
    elif not silent:
        logger.info(f"Spending up to around ${total_max_cost:.2f}.")

    # sort to execute longest requests first
    for pair in pairs_to_execute:
        pair.usage = pair.request.estimate_max_total_usage()
    pairs_to_execute.sort(key=lambda p: p.usage, reverse=True)

    # execute requests
    with tqdm.tqdm(total=len(pairs), desc="execute requests", disable=silent) as progress_bar:
        progress_bar.update(len(pairs) - len(pairs_to_execute))
        for pair in pairs_to_execute:
            model_params = _get_model_params(pair.request.model)

            while True:
                if global_context_semaphore is not None:
                    global_context_semaphore.acquire()

                # update API budget for the model
                if pair.request.model in context.keys():
                    budget = context[pair.request.model]
                    del context[pair.request.model]
                    now = time.time()
                    delta = now - budget.last_update
                    budget.requests = min(model_params["max_rpm"] * _api_share,
                                          budget.requests + model_params["max_rpm"] * _api_share * delta / 60)
                    budget.tokens = min(model_params["max_tpm"] * _api_share,
                                        budget.tokens + model_params["max_tpm"] * _api_share * delta / 60)
                    budget.last_update = now
                    context[pair.request.model] = budget
                else:
                    # initialize budget based on cache
                    now = datetime.datetime.now()
                    budget = _Budget(model_params["max_rpm"] * _api_share, model_params["max_tpm"] * _api_share,
                                     time.time())
                    for cache_file_path in list(sorted(_cache_path.glob("*.json"))):
                        cache_time = datetime.datetime.strptime(cache_file_path.name[:19], "%Y-%m-%d-%H-%M-%S")
                        if now - cache_time < datetime.timedelta(minutes=1):
                            with open(cache_file_path, "r", encoding="utf-8") as file:
                                cached_pair = json.load(file)
                                request = _Request(cached_pair["request"])
                                if request.model == pair.request.model:
                                    response = _Response(cached_pair["response"])
                                    budget.requests -= 1
                                    budget.tokens -= response.compute_total_usage()
                    context[pair.request.model] = budget

                if context[pair.request.model].requests >= 1 and context[pair.request.model].tokens >= pair.usage:
                    do = True
                    sleep = _wait_before_try
                    budget = context[pair.request.model]
                    del context[pair.request.model]
                    budget.requests -= 1
                    budget.tokens -= pair.usage
                    context[pair.request.model] = budget
                else:
                    do = False
                    sleep = _wait_before_retry

                if global_context_semaphore is not None:
                    global_context_semaphore.release()

                time.sleep(sleep)
                if do:
                    def execute(p, pb, c, gcs, mp):
                        p.response, p.was_cached = p.request.execute()
                        if p.was_cached:
                            actual_usage = 0
                        else:
                            actual_usage = p.response.compute_total_usage()
                        if actual_usage < p.usage:
                            if gcs is not None:
                                gcs.acquire()
                            budget = c[p.request.model]
                            del c[p.request.model]
                            budget.tokens = min(mp["max_tpm"] * _api_share, budget.tokens + p.usage - actual_usage)
                            c[p.request.model] = budget
                            if gcs is not None:
                                gcs.release()
                        p.usage = actual_usage
                        pb.update()

                    pair.thread = threading.Thread(target=execute, args=(
                        pair, progress_bar, context, global_context_semaphore, model_params))
                    pair.thread.start()
                    break

        for pair in pairs_to_execute:
            if pair.thread is not None:
                pair.thread.join()

    # shrink cache
    cache_file_paths = list(sorted(_cache_path.glob("*.json")))
    if len(cache_file_paths) > _cache_size:
        logger.warning(f"OpenAI cache is too large ({len(cache_file_paths)} > {_cache_size}) and will be shrunk!")
        for cache_file_path in cache_file_paths[:-_cache_size]:
            os.remove(cache_file_path)

    # describe output
    num_failed_requests = sum(not pair.response.was_successful() for pair in pairs)
    if num_failed_requests > 0:
        logger.warning(f"{num_failed_requests} requests failed!")

    total_cost = sum(pair.response.compute_total_cost() for pair in pairs_to_execute if not pair.was_cached)
    if not silent:
        message = f"Spent ${total_cost:.2f}."
        was_cached = sum(pair.was_cached for pair in pairs)
        if was_cached > 0:
            message += f" ({was_cached} responses were already cached)"
        logger.info(message)

    return [pair.response.response for pair in pairs]


def openai_cost_for_cache() -> float:
    """Compute the total dollar cost incurred by executing all cached requests/responses.

    Returns:
        The total dollar cost incurred by executing all cached requests/responses.
    """
    file_paths = list(sorted(_cache_path.glob(f"*.json")))
    total_cost = 0
    for file_path in file_paths:
        with open(file_path, "r", encoding="utf-8") as file:
            cached_pair = json.load(file)
            cached_response = _Response(cached_pair["response"])
            total_cost += cached_response.compute_total_cost()

    return total_cost
