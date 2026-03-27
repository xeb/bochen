"""Configuration for Experiment 3: Two-Phase Probe-Solve with Causal Memory."""

# Probe phase
MAX_PROBE_STEPS = 25
PROBE_RESET_BETWEEN = True  # reset env between individual probes for clean state

# Solve phase
MAX_SOLVE_STEPS = 60

# Memory
MEMORY_EMBED_DIM = 128
MEMORY_TOP_K = 10

# Episode timeout in seconds
EPISODE_TIMEOUT = 60
