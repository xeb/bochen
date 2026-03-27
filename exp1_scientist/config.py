"""Configuration for Experiment 1: LLM Scientist Agent."""

# LLM backend
LOCAL_MODEL = "Qwen/Qwen2.5-32B-Instruct-AWQ"
LOCAL_URL = "http://localhost:8000/v1"
API_KEY = "not-needed"  # vLLM doesn't need a real key

# Agent behavior
MAX_PROBE_STEPS = 30
MAX_EXPLOIT_STEPS = 50
MAX_TOTAL_STEPS = 80
CONFIDENCE_THRESHOLD = 0.7
TEMPERATURE = 0.3
MAX_TOKENS = 1024

# Episode timeout in seconds
EPISODE_TIMEOUT = 300
