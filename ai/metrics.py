import time
from prometheus_client import Counter, Histogram

# Metrics for LLM API
LLM_REQUESTS_TOTAL = Counter(
    'llm_requests_total', 
    'Total number of requests made to LLM API',
    ['model', 'status']
)

LLM_REQUEST_LATENCY = Histogram(
    'llm_request_latency_seconds', 
    'Latency of LLM API requests',
    ['model']
)

LLM_TOKENS_TOTAL = Counter(
    'llm_tokens_total', 
    'Total tokens used by LLM API',
    ['model', 'token_type']
)

# Application metrics
BOT_MESSAGES_SENT_TOTAL = Counter(
    'bot_messages_sent_total',
    'Total number of messages sent by the bot',
    ['type'] # e.g. 'reply', 'proactive'
)

class Timer:
    def __init__(self, histogram, **labels):
        self.histogram = histogram
        self.labels = labels
        self.start = None

    def __enter__(self):
        self.start = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        duration = time.perf_counter() - self.start
        if self.labels:
            self.histogram.labels(**self.labels).observe(duration)
        else:
            self.histogram.observe(duration)

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.outputs import LLMResult
from typing import Any, Dict, List

class PrometheusCallbackHandler(BaseCallbackHandler):
    """
    LangChain callback handler to automatically track LLM metrics in Prometheus.
    """
    def __init__(self):
        super().__init__()
        self.start_times = {}

    def on_llm_start(self, serialized: Dict[str, Any], prompts: List[str], **kwargs: Any) -> None:
        run_id = kwargs.get("run_id")
        if run_id:
            self.start_times[run_id] = time.perf_counter()

    def _get_model_name(self, kwargs: Any) -> str:
        model_name = "unknown"
        if "invocation_params" in kwargs:
            params = kwargs["invocation_params"]
            model_name = params.get("model_name") or params.get("model") or "unknown"
        return model_name

    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
        run_id = kwargs.get("run_id")
        model_name = self._get_model_name(kwargs)
            
        if run_id and run_id in self.start_times:
            duration = time.perf_counter() - self.start_times.pop(run_id)
            LLM_REQUEST_LATENCY.labels(model=model_name).observe(duration)
            
        LLM_REQUESTS_TOTAL.labels(model=model_name, status="success").inc()
        
        if response.llm_output and "token_usage" in response.llm_output:
            token_usage = response.llm_output["token_usage"]
            prompt_tokens = token_usage.get("prompt_tokens", 0)
            completion_tokens = token_usage.get("completion_tokens", 0)
            
            if prompt_tokens:
                LLM_TOKENS_TOTAL.labels(model=model_name, token_type="prompt").inc(prompt_tokens)
            if completion_tokens:
                LLM_TOKENS_TOTAL.labels(model=model_name, token_type="completion").inc(completion_tokens)

    def on_llm_error(self, error: Exception, **kwargs: Any) -> None:
        run_id = kwargs.get("run_id")
        if run_id and run_id in self.start_times:
            self.start_times.pop(run_id)
            
        model_name = self._get_model_name(kwargs)
        LLM_REQUESTS_TOTAL.labels(model=model_name, status="error").inc()
