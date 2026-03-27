# JAXIFY: Migrate from PyTorch to JAX

**Goal**: Remove NVIDIA/CUDA lock-in and enable running massive experiments on GCP TPUs. Replace all PyTorch usage with JAX + Flax/Linen + Optax.

**Status**: Plan

---

## Why JAX

- **TPU-native**: JAX was built for TPUs. No shim layers, no degraded performance. PyTorch XLA exists but is a second-class citizen with constant breakage.
- **Hardware portability**: Same code runs on CPU, GPU (NVIDIA or AMD via ROCm), and TPU with zero changes. Just swap the runtime.
- **`jit` + `vmap`**: Automatic vectorization (`vmap` over batch dims) with trivial code changes. The info-gain search in exp2 and the memory retrieval in exp3 both benefit. Multi-device `pmap` is available but requires fixed shard shapes — add it later if single-device throughput is actually the bottleneck.
- **Functional purity**: Stateless transforms make reproducibility trivial. Every random op takes an explicit key. No hidden global state.
- **XLA compilation**: `jax.jit` compiles entire forward/backward passes into fused XLA ops. On TPUs, this means bfloat16 matrix multiplies at hardware speed.
- **Ecosystem**: Flax/Linen for modules, Optax for optimizers, Orbax for checkpointing — all Google-maintained, all TPU-first.

---

## Current PyTorch Footprint

Seven files use torch. Zero files use it for anything exotic.

| File | What torch does | Migration complexity |
|------|----------------|---------------------|
| `exp2_worldmodel/models.py` | 3 `nn.Module` classes: ObjectSegmenter (CNN+BatchNorm), TransitionPredictor (MLP+Embedding), GridEncoder (CNN) | **Medium** — rewrite as Flax `nn.Module`. ObjectSegmenter's BatchNorm adds Flax mutable-state bookkeeping (see note below) |
| `exp2_worldmodel/train.py` | Training loop: Adam, MSE loss, backward, checkpoint save/load. Imports `TensorDataset`/`DataLoader` but doesn't use them. Uses `.detach()` for stop-gradient on target encoder | **Medium** — Optax optimizer, `jax.value_and_grad`, explicit `jax.lax.stop_gradient` |
| `exp2_worldmodel/agent.py` | Loads checkpoint, batched inference in `no_grad`, argmax over scores | **Easy** — `jax.jit` the forward pass, `jnp.argmax` |
| `exp3_probe_solve/encoder.py` | StateEncoder CNN (`nn.Module`), `encode_grid()` inference | **Easy** — rewrite as Flax module |
| `exp3_probe_solve/memory.py` | GPU tensor storage, `torch.mm` for cosine similarity, `torch.topk`. Persistence is **pickle** (not torch save) with mixed Python dicts/strings + numpy arrays | **Easy** for compute, **Medium** for persistence (not a clean Orbax swap) |
| `exp3_probe_solve/agent.py` | Creates encoder/memory, loads `memory.pkl` checkpoint | **Easy** — keep pickle for metadata, use Orbax only for encoder params |
| `harness.py` | `torch.cuda.empty_cache()` (1 line) | **Trivial** — delete (no JAX equivalent; see note below) |

**exp1_scientist** has zero torch usage (pure LLM via OpenAI client). No changes needed.

**shared/** has zero torch imports, but `one_hot_grid()` in `perception.py` defines the channel-first `(C, H, W)` padded layout that all models consume. Flax expects channel-last `(H, W, C)`. Decision: keep `one_hot_grid()` as-is (stable API boundary) and transpose NCHW→NHWC once inside each Flax module.

### Pre-Migration Bug: ACTION6 Representation

**This should be fixed before or during the migration.** The TransitionPredictor only receives `action_idx` — it has no access to ACTION6 click coordinates `(x, y)`. All ACTION6 candidates score identically in the info-gain search because `collect.py` stores only integer actions (not coordinates) and the predictor's embedding layer maps all ACTION6 calls to the same vector. The `vmap`/`pmap` speedup story for candidate search is moot until coordinates are part of the model input.

---

## Migration Plan

### Phase 0: Dependency Setup

```bash
# Remove torch from the venv entirely
# Install JAX with TPU support (or GPU for local dev)

# Local dev (NVIDIA GPU):
uv pip install "jax[cuda12]"

# GCP TPU VM:
uv pip install "jax[tpu]" -f https://storage.googleapis.com/jax-releases/libtpu_releases.html

# Shared deps:
uv pip install flax optax orbax-checkpoint
```

Update `pyproject.toml`:
```toml
dependencies = [
    "arc-agi>=0.0.7",
    "numpy",
    "openai",
    "jax[cuda12]",
    "flax",
    "optax",
    "orbax-checkpoint",
]
```

For TPU deployment, use an extras group or environment marker:
```toml
[project.optional-dependencies]
tpu = ["jax[tpu]"]
gpu = ["jax[cuda12]"]
```

### Phase 1: exp2_worldmodel/models.py (core models)

Two models need Flax rewrites: GridEncoder and TransitionPredictor. ObjectSegmenter uses BatchNorm which requires Flax mutable-state bookkeeping (`batch_stats` collection, `use_running_average` flag, `mutable=['batch_stats']` in `.apply()` calls). Since ObjectSegmenter is **unused in the runtime path** (neither agent imports it), defer or drop it. If you do port it, strip BatchNorm first — it's not worth the complexity for this model size.

**Before (PyTorch)**:
```python
class GridEncoder(nn.Module):
    def __init__(self, num_colors=16, embed_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(num_colors, 32, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(16),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(4),
            nn.Flatten(),
            nn.Linear(64 * 4 * 4, embed_dim),
        )
    def forward(self, x):
        return self.net(x)
```

**After (JAX/Flax)**:
```python
import jax
import jax.numpy as jnp
import flax.linen as nn

class GridEncoder(nn.Module):
    num_colors: int = 16
    embed_dim: int = 128

    @nn.compact
    def __call__(self, x):
        # x: (B, num_colors, H, W) -> transpose to (B, H, W, C) for Flax conv
        x = jnp.transpose(x, (0, 2, 3, 1))
        x = nn.Conv(features=32, kernel_size=(3, 3), padding='SAME')(x)
        x = nn.relu(x)
        # AdaptiveAvgPool equivalent: just use avg_pool with computed window
        x = _adaptive_avg_pool(x, output_size=(16, 16))
        x = nn.Conv(features=64, kernel_size=(3, 3), padding='SAME')(x)
        x = nn.relu(x)
        x = _adaptive_avg_pool(x, output_size=(4, 4))
        x = x.reshape((x.shape[0], -1))  # flatten
        x = nn.Dense(features=self.embed_dim)(x)
        return x

def _adaptive_avg_pool(x, output_size):
    """Replicate PyTorch's AdaptiveAvgPool2d using jax.image.resize + mean."""
    B, H, W, C = x.shape
    oh, ow = output_size
    # Compute window and stride sizes
    wh, ww = H // oh, W // ow
    x = x.reshape(B, oh, wh, ow, ww, C)
    return x.mean(axis=(2, 4))
```

Key pattern differences:
- Flax uses `(B, H, W, C)` channel-last format (matches TPU layout). Transpose on entry.
- `nn.compact` eliminates `__init__` — submodules are created inline on first call.
- Parameters are external (passed as `variables`), not stored on `self`.
- `AdaptiveAvgPool2d` doesn't exist in JAX — use reshape+mean or `jax.lax.reduce_window`.

Apply the same pattern to `TransitionPredictor`. The `nn.Embedding` becomes `nn.Embed(num_embeddings=8, features=32)`. Defer `ObjectSegmenter` (unused at runtime; BatchNorm makes the port disproportionately complex).

### Phase 2: exp2_worldmodel/train.py (training loop)

**Before (PyTorch)**:
```python
optimizer = torch.optim.Adam(list(encoder.parameters()) + list(predictor.parameters()), lr=lr)
for epoch in range(epochs):
    for batch in dataloader:
        loss = compute_loss(encoder, predictor, batch)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
```

**After (JAX/Optax)**:
```python
import optax
import jax
import jax.numpy as jnp

tx = optax.adam(learning_rate=lr)

# params is a nested dict, not an object with .parameters()
params = {
    'encoder': encoder.init(rng, dummy_input),
    'predictor': predictor.init(rng, dummy_state, dummy_action),
}
opt_state = tx.init(params)

@jax.jit
def train_step(params, opt_state, batch):
    def loss_fn(params):
        state_emb = encoder.apply(params['encoder'], batch['grids'])
        # CRITICAL: current code uses .detach() on target encoder output.
        # Without stop_gradient, JAX will backprop through both encoder
        # passes and silently change the optimization problem.
        target_emb = jax.lax.stop_gradient(
            encoder.apply(params['encoder'], batch['next_grids'])
        )
        preds = predictor.apply(params['predictor'], state_emb, batch['actions'])
        return jnp.mean((preds['next_state'] - target_emb) ** 2)

    loss, grads = jax.value_and_grad(loss_fn)(params)
    updates, opt_state = tx.update(grads, opt_state, params)
    params = optax.apply_updates(params, updates)
    return params, opt_state, loss

for epoch in range(epochs):
    for batch in data_iter:
        params, opt_state, loss = train_step(params, opt_state, batch)
```

Key differences:
- No `.backward()` — `jax.grad` computes gradients as a function transform.
- No `.zero_grad()` — gradients are returned fresh each call.
- **`jax.lax.stop_gradient`** replaces `.detach()` — must be explicit or you break the loss.
- No `DataLoader` — just iterate over numpy arrays (or use `tf.data` on TPU for prefetching).
- `@jax.jit` compiles the entire step to XLA — huge speedup, especially on TPU.
- Params are pytrees (nested dicts), not module attributes.

### Data Pipeline Warning

The current `train.py` materializes the full dataset as one-hot tensors in memory. At `50k transitions/game * 3 games`, that's two `(150k, 16, 64, 64)` float32 arrays = **~75 GB**. The plan to "just load into host memory and `jax.device_put`" will OOM.

**Fix**: One-hot encode on-the-fly per batch, not upfront. Store raw integer grids in memory, encode inside the jitted `train_step`:

```python
def one_hot_batch(grids_int, num_colors=16):
    """grids_int: (B, H, W) int8 -> (B, H, W, num_colors) float32"""
    return jax.nn.one_hot(grids_int, num_colors)

@jax.jit
def train_step(params, opt_state, raw_grids, raw_next_grids, actions):
    grids = one_hot_batch(raw_grids)      # on-the-fly, inside XLA
    next_grids = one_hot_batch(raw_next_grids)
    # ... rest of training step
```

This reduces host memory from ~75 GB to ~1.2 GB (raw int8 grids).

### Phase 3: exp2_worldmodel/agent.py (inference)

Replace:
```python
# Before
state_tensor = torch.from_numpy(grid).unsqueeze(0).to(self.device)
with torch.no_grad():
    emb = self.encoder(state_tensor)
    preds = self.predictor(emb_batch, action_batch)
scores = preds['info_gain'].argmax().item()
```

With:
```python
# After
state_array = jnp.array(grid)[None, ...]

@jax.jit
def predict_batch(params, grids, actions):
    emb = encoder.apply(params['encoder'], grids)
    return predictor.apply(params['predictor'], emb, actions)

preds = predict_batch(self.params, state_batch, action_batch)
best = jnp.argmax(scores).item()
```

No `no_grad` context — JAX only computes gradients when you explicitly call `jax.grad`. Inference is the default.

### Phase 4: exp3_probe_solve/encoder.py + memory.py

**encoder.py**: Same pattern as GridEncoder above. Rewrite `StateEncoder` as a Flax module. The `encode_grid()` method becomes a `jax.jit`-compiled function that takes params + grid.

**memory.py**: The GPU-accelerated retrieval is the easiest migration:

```python
# Before (PyTorch)
sims = torch.mm(query, self.embeddings.T)
top_k = torch.topk(sims, k)

# After (JAX)
sims = jnp.dot(query, self.embeddings.T)
top_vals, top_idx = jax.lax.top_k(sims[0], k)
```

For the embedding storage, **don't try to make the whole memory object JAX-native**. The `store()` and `retrieve()` calls live inside Python-heavy probe/solve loops with dict filtering and per-step appends. Using `.at[i].set()` outside `jit` still creates a new array each call — no XLA optimization, just copies.

Better approach: store embeddings as a plain numpy array, only move to JAX for the retrieval matmul:

```python
class CausalMemory:
    def __init__(self, max_entries=10000, embed_dim=128):
        self.embeddings_np = np.zeros((max_entries, embed_dim), dtype=np.float32)
        self.count = 0
        self.entries = []  # Python dicts with metadata

    def store(self, emb_np, action, effect, game_id):
        self.embeddings_np[self.count] = emb_np
        self.count += 1
        self.entries.append({"action": action, "effect": effect, "game_id": game_id})

    def retrieve(self, query_np, k=10):
        """Move to device only for the matmul, then back."""
        emb = jnp.array(self.embeddings_np[:self.count])
        query = jnp.array(query_np)
        sims = jnp.dot(query, emb.T)
        top_vals, top_idx = jax.lax.top_k(sims[0], min(k, self.count))
        indices = top_idx.tolist()
        return [(self.entries[i], float(top_vals[j])) for j, i in enumerate(indices)]
```

**Persistence**: Keep pickle. The memory stores arbitrary Python dicts with strings, nested dicts, and numpy arrays — Orbax handles pytrees, not this. Use Orbax only for model params (encoder).

### Phase 5: exp3_probe_solve/agent.py + harness.py

**agent.py**: Use Orbax for encoder params, keep pickle for memory:
```python
import orbax.checkpoint as ocp

# Encoder params (pytree of arrays — Orbax works great)
checkpointer = ocp.StandardCheckpointer()
checkpointer.save(path / "encoder", params)
params = checkpointer.restore(path / "encoder")

# Memory (mixed Python + numpy — keep pickle)
memory.save(path / "memory.pkl")  # existing pickle serialization
```

**harness.py**: Delete the `torch.cuda.empty_cache()` call. There is no JAX equivalent — `jax.clear_caches()` clears *compilation* caches, not device memory. JAX's XLA allocator manages device memory automatically. If memory pressure becomes an issue, the fix is managing long-lived array references, not calling a cleanup function.

### Phase 6: Checkpoint Format Migration

Old `.pt` checkpoints won't load in JAX. The current `.pt` file stores nested dicts under `"encoder"` and `"predictor"` keys, but Flax parameters have different names (e.g., `Conv_0` vs `conv1`) and conv kernels use a different layout (HWIO vs PyTorch's OIHW).

**Recommended: clean break.** Delete old checkpoints and retrain. Models are small and train in minutes. A conversion script would need a manual name mapping table and kernel transposes — more effort than retraining.

If you must convert (e.g., preserving a tuned model):
```python
# convert_checkpoints.py — requires both torch and jax installed temporarily
import torch, numpy as np

pt = torch.load("world_model_best.pt", map_location="cpu", weights_only=True)

# Manual mapping: PyTorch name -> Flax name + optional transpose
# Conv2d weights: PyTorch (O,I,H,W) -> Flax (H,W,I,O)
for key in pt["encoder"]:
    arr = pt["encoder"][key].numpy()
    if arr.ndim == 4:  # conv kernel
        arr = arr.transpose(2, 3, 1, 0)  # OIHW -> HWIO
    # ... save with correct Flax param tree structure
```

This is fragile and model-specific. Retraining is strongly preferred.

---

## TPU Deployment Notes

### GCP TPU VM Setup

```bash
# Create TPU VM (v4-8 = 4 chips, good for experimentation)
gcloud compute tpus tpu-vm create bochen-tpu \
    --zone=us-central2-b \
    --accelerator-type=v4-8 \
    --version=tpu-vm-v4-base

# SSH in
gcloud compute tpus tpu-vm ssh bochen-tpu --zone=us-central2-b

# Install JAX for TPU
pip install "jax[tpu]" -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
pip install flax optax orbax-checkpoint

# Verify
python -c "import jax; print(jax.devices())"
# [TpuDevice(id=0, ...), TpuDevice(id=1, ...), ...]
```

### Multi-Device Parallelism

The info-gain search in exp2 evaluates ~100 candidate actions per step. `jit` + `vmap` on a dense action tensor is the right approach:

```python
# Vectorize across candidates on a single device — this is the easy win
@jax.jit
def evaluate_candidates(params, state, action_indices):
    # action_indices: (num_candidates,) int array
    return jax.vmap(lambda a: predict(params, state, a))(action_indices)
```

**`pmap` is not trivial here.** Candidate generation is Python-side, variable-length, and includes non-array metadata. `pmap` requires fixed per-device shard shapes — 100 candidates must divide evenly across device count or be padded. For this codebase, stick with `jit` + `vmap` first. Add `pmap` only if single-device throughput is actually the bottleneck (unlikely for 100 candidates with tiny models).

The memory retrieval in exp3 (cosine similarity over all stored embeddings) is a single matrix multiply — TPUs eat this for breakfast.

### TPU-Specific Considerations

- **bfloat16 by default**: TPU v4 runs matmuls in bfloat16. Our small models (32-64 channels, 128-dim embeddings) won't lose meaningful precision. No code changes needed — JAX promotes automatically.
- **Static shapes**: `jax.jit` requires static array shapes. For variable-length batches, pad to a fixed size.
- **No CUDA calls**: Obviously. But also no `torch.cuda.synchronize()`, no `pin_memory`, no `non_blocking` transfers. JAX handles all device placement via its runtime.
- **Data pipeline**: At the configured scale (50k transitions/game x 3 games), raw int8 grids fit in host memory (~1.2 GB). One-hot encode on-the-fly inside `jit` (see Phase 2). For significantly larger datasets, use `tf.data` for TPU-optimized prefetching with grain.

---

## Scaling Experiments on TPU

With JAX on TPU, several things become cheap that were expensive on a single RTX 4090:

| Experiment | RTX 4090 (current) | TPU v4-8 (projected) |
|------------|--------------------|-----------------------|
| exp2 world model training (50 epochs, 50k transitions) | ~2 min | ~15 sec (XLA fusion + bfloat16) |
| exp2 info-gain search (100 candidates/step) | ~50ms/step | ~5ms/step (vmap across cores) |
| exp3 memory retrieval (10k embeddings) | ~1ms | ~0.1ms (TPU matmul unit) |
| Full harness sweep (3 games x 3 experiments) | ~30 min | ~5 min |
| Hyperparameter sweep (100 configs) | ~50 hours | ~8 hours |

The real win is hyperparameter sweeps and ablation studies — the kind of "massive experiments" that justify TPU cost.

---

## Migration Order (Recommended)

Migrate the numeric core first. Don't try to JIT whole agents — the agent loops are Python-heavy with environment interaction and dict manipulation that doesn't belong inside XLA.

1. **exp3_probe_solve/memory.py** — smallest file, simplest torch usage (just `mm` and `topk`). Switch to numpy storage + JAX retrieval kernel. Keep pickle persistence. Good warmup.
2. **exp3_probe_solve/encoder.py** — single small CNN module. Rewrite as Flax, add `encode_grid()` as a jitted standalone function.
3. **exp3_probe_solve/agent.py** — swap encoder checkpoint to Orbax, keep memory as pickle.
4. **exp2_worldmodel/models.py** — GridEncoder + TransitionPredictor as Flax modules. **Defer ObjectSegmenter** (unused at runtime, BatchNorm adds disproportionate complexity). Also fix ACTION6 coordinate representation while you're here.
5. **exp2_worldmodel/train.py** — Optax training loop. Add `jax.lax.stop_gradient` for target encoder. Switch to on-the-fly one-hot encoding to avoid 75 GB memory blowup.
6. **exp2_worldmodel/agent.py** — inference, depends on models.py.
7. **harness.py** — delete the `torch.cuda.empty_cache()` call (no JAX equivalent needed).

Test each file after migration. Run the existing game suite (`--games ls20`) to verify identical behavior before moving to the next file.

---

## Gotchas

- **JAX arrays are immutable**. No in-place ops like `tensor[i] = val`. Use `.at[i].set(val)` which returns a new array — but only inside `jit` does XLA optimize the copy away. Outside `jit` (e.g., the memory store loop), `.at[].set()` is a full copy every time. Use numpy for mutable hot paths.
- **Random numbers require explicit keys**. No `torch.randn(shape)`. Instead: `jax.random.normal(key, shape)`. Split keys for each random call.
- **`jax.jit` traces Python control flow**. `if x > 0` inside a jitted function uses the *traced* value, not the runtime value. Use `jax.lax.cond` for dynamic branches, or keep the condition outside the jit boundary.
- **No dynamic shapes inside `jit`**. Batch size, sequence length, etc. must be static or padded. Our models use fixed 64x64 grids so this isn't an issue.
- **Channel order**: Flax convolutions expect `(B, H, W, C)`, not `(B, C, H, W)`. Keep `one_hot_grid()` as-is for API stability; transpose NCHW→NHWC once inside each Flax module's `__call__`.
- **Conv kernel layout**: PyTorch uses `(O, I, H, W)`, Flax/JAX uses `(H, W, I, O)`. This matters for checkpoint conversion — don't assume `v.numpy()` produces loadable Flax params.
- **`.detach()` → `jax.lax.stop_gradient()`**: JAX has no implicit gradient tracking. Gradients only flow through `jax.grad` calls. But if you compute a target inside a `grad`-wrapped function without `stop_gradient`, JAX *will* differentiate through it. The `.detach()` in `train.py` must become an explicit `stop_gradient` call.
- **`jax.clear_caches()` ≠ `torch.cuda.empty_cache()`**: The former clears compilation/staging caches, not device memory. There is no JAX equivalent for freeing device memory on demand.
- **Flax modules are dataclasses**. Don't put mutable state in them. All state (params, optimizer state, RNG keys) lives outside the module as pytrees.
- **Flax BatchNorm is fiddly**. It requires separate `params` and `batch_stats` variable collections, `use_running_average` toggling between train/eval, and `mutable=['batch_stats']` in `.apply()`. Avoid unless necessary (ObjectSegmenter).
- **`jax.debug.print`** instead of `print()` inside jitted functions.
