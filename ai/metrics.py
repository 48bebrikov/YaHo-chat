import time
from prometheus_client import Counter, Histogram

# Metrics for Gemini API
GEMINI_REQUESTS_TOTAL = Counter(
    'gemini_requests_total', 
    'Total number of requests made to Gemini API',
    ['model', 'status']
)

GEMINI_REQUEST_LATENCY = Histogram(
    'gemini_request_latency_seconds', 
    'Latency of Gemini API requests',
    ['model']
)

GEMINI_TOKENS_TOTAL = Counter(
    'gemini_tokens_total', 
    'Total tokens used by Gemini API',
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
