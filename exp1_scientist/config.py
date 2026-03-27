"""Configuration for Experiment 1: LLM Scientist Agent."""

import os

# LLM backend — vLLM serving Qwen2.5-7B-Instruct-AWQ on local GPU
LOCAL_MODEL = os.environ.get("EXP1_MODEL", "Qwen/Qwen2.5-7B-Instruct-AWQ")
LOCAL_URL = os.environ.get("EXP1_URL", "http://localhost:8000/v1")
API_KEY = os.environ.get("EXP1_API_KEY", "not-needed")

# Agent behavior
MAX_PROBE_STEPS = 8
MAX_EXPLOIT_STEPS = 12
MAX_TOTAL_STEPS = 20
CONFIDENCE_THRESHOLD = 0.7
TEMPERATURE = 0.3
MAX_TOKENS = 300

# Episode timeout in seconds
EPISODE_TIMEOUT = 600
