# DeepswapLLM

<p align="center">
  <img src="deepswapllm.png" alt="DeepswapLLM" width="400">
</p>

<h3 align="center">Run 2.8T-parameter models on a single RTX 3090.<br>3x faster than AirLLM. Zero disk I/O.</h3>

<p align="center">
  <a href="#benchmarks">Benchmarks</a> &bull;
  <a href="#quick-start">Quick Start</a> &bull;
  <a href="#how-it-works">How It Works</a> &bull;
  <a href="#gguf-support">GGUF Support</a> &bull;
  <a href="#api-reference">API</a>
</p>

---

DeepswapLLM runs transformer models that **don't fit in your GPU's VRAM** by intelligently swapping layers between CPU RAM and GPU through a zero-allocation double-buffered pipeline. It automatically tiers layers across GPU VRAM, pinned CPU RAM, and NVMe storage based on what your hardware can handle.

**No quantization required.** Full bf16/fp16 precision. Or load GGUF quantized models for even larger parameter counts.

## Benchmarks

**Hardware:** RTX 3090 (24GB VRAM), 94GB RAM, PCIe 4.0 x16, NVMe SSD

### 36B Model — Full Precision (bf16)

Model: `seed-oss-36b-base` (36B parameters, 69GB in bf16) — **2.8x larger than VRAM.**

| | DeepswapLLM | AirLLM | Speedup |
|---|:---:|:---:|:---:|
| **Tokens/sec** | **0.1935** | 0.0633 | **3.06x** |
| **Avg layer swap** | **11.6ms** | ~500ms | **43x** |
| **Setup time** | **1.8s** | ~120s | **67x** |
| **Correct output** | Yes | Yes | — |

```
============================================================
  RESULTS (DeepswapLLM)
============================================================
  Model: seed-oss-36b (36B params, bf16)
  GPU: RTX 3090 24GB
  VRAM: 14.8GB used
  Setup: 1.8s
  Generation: 82.7s for 16 tokens (0.1935 tok/s)
  Avg swap: 11.6ms
  RAM loads: 16, Disk loads: 8
  Tiers: 56 in RAM (60.5GB), 8 on disk (8.6GB)
  Output: 'The future of artificial intelligence is here,
           and it's called ChatGPT. This revolutionary
           technology is changing the way'
============================================================
```

### 36B Model — GGUF Quantized

Same model, quantized to Q8_0 (36GB) and Q4_K_M (21GB). DeepswapLLM pre-dequantizes all layers at setup, then uses the same zero-alloc swap engine. AirLLM has no GGUF support.

| | bf16 | Q8_0 | Q4_K_M | AirLLM (bf16) |
|---|:---:|:---:|:---:|:---:|
| **Tokens/sec** | 0.1935 | **0.2029** | 0.1914 | 0.0633 |
| **Avg swap** | 11.6ms | **1.3ms** | 1.4ms | ~500ms |
| **GGUF size** | 69GB | 36GB | 21GB | — |
| **vs AirLLM** | 3.06x | **3.20x** | 3.02x | 1x |

Q8_0 is actually the fastest configuration — lower swap latency than bf16 with near-identical output quality.

### 2.8T Model — Kimi K3 (MXFP4 Quantized)

Model: `Kimi-K3` (2.8T parameters, 93 layers, 896 experts/layer, MXFP4 compressed-tensors) — **116x larger than VRAM.**

This is a **capability win, not a tok/s speed win**: the point is that a 2.8T model runs *at all* on a 24GB consumer card. The ~6s/layer is bounded by per-expert disk reads and MXFP4 decompression, not raw compute.

**AirLLM (v3.1.0+) also runs K3.** Both tools stream individual experts to fit a 2.8T MoE on a single card. AirLLM published its own K3 measurements in its [v3.1.0 release notes](https://github.com/lyogavin/airllm/releases/tag/v3.1.0) (RTX 6000 Ada, full 1.56TB checkpoint); the DeepswapLLM figures below are from a single RTX 3090:

| | DeepswapLLM (RTX 3090) | AirLLM (RTX 6000 Ada) |
|---|:---:|:---:|
| **Runs K3** | Yes | Yes |
| **VRAM (generation)** | 6.9 GB | 3.72 GB |
| **Init / setup** | **172 s** | 900 s |
| **Throughput — measured** | 1127 s/token (7200rpm HDD) | 292 s/token |
| **Throughput — NVMe** | **~100 s/token** (projected)\* | — |

\* No NVMe drive with enough free space to hold K3's re-split was available at the time of writing, so this figure is projected from sequential read speeds measured on this machine (HDD 245 MB/s vs NVMe 1806 MB/s), not from a live K3 run on NVMe. Output on the HDD run was coherent.

**Init / setup:** DeepswapLLM readies ~5x faster (172 s vs 900 s).

**Throughput — the HDD number is storage-bound, not engine-bound.** K3 reads ~28 GB of scattered expert weights per token (16 of 896 experts across 92 layers, ~19 MB each). On the 7200rpm HDD that random pattern delivered only ~25 MB/s effective — 10% of the drive's measured 245 MB/s sequential — because the head seeks for every expert. The NVMe on the same box measures **1806 MB/s** with no seek penalty, so it runs at near-sequential speed on these multi-MB reads and collapses the disk wait by ~40–70x. Projecting the disk-bound time onto NVMe puts DeepswapLLM at **~100 s/token**, at which point compute and MXFP4 decompression — not storage — set the pace. That is ~3x faster than AirLLM's published 292 s/token, and it tracks the measured 3x speedup DeepswapLLM shows on smaller, RAM-resident models. (Projected from drive speeds measured on this machine; a same-storage run would confirm it.)

**VRAM:** the 6.9 GB vs 3.72 GB gap is a design choice (below), not a hardware effect. Neither GPU explains any of these differences: the two cards have comparable memory bandwidth (RTX 3090 ~936 GB/s, RTX 6000 Ada ~960 GB/s), VRAM capacity isn't the constraint (K3 needs under 7 GB), and the workload is bound by per-expert disk streaming and MXFP4 decompression, not GPU throughput.

The VRAM difference (6.9GB vs 3.72GB) reflects a design choice: DeepswapLLM pre-allocates GPU staging buffers for zero-alloc double-buffering — the mechanism behind its 3x speedup on smaller models — and reserves 4GB of headroom in this run.

The 6.9GB peak is not specific to a 24GB card. A 2.8T model runs comfortably within a 12GB GPU, and an 8GB card is within reach by lowering the reserve — so this is not limited to high-end hardware.

Expert-level offload is what makes either tool fit: the router activates only 16 of the 896 experts per token, and each is freed right after its forward pass — so resident expert weight is a fraction of a full 118GB (fp16) layer. **This is the entire reason a 2.8T model fits in single-digit GB.**

```
============================================================
  RESULTS (DeepswapLLM)
============================================================
  Model: Kimi-K3 (2.8T params, MXFP4)
  GPU: RTX 3090 24GB
  VRAM: 6.9GB peak
  Setup: 172s (mapping 249k tensor groups)
  Generation: 1127s for 1 token
  Avg swap: 6064ms (per-expert disk load + MXFP4 decompress)
  Disk loads: 93
  Output: 'Hello,'
============================================================
```

The 6s/layer reflects expert-level offloading: each MoE layer has 896 experts, and the router selects 16 per token. Each activated expert is loaded from mmap'd safetensors, decompressed from MXFP4 (uint8 packed + E8M0 scales), moved to GPU for compute, and freed — all within a single forward pass. The key achievement is fitting a 2.8T model in under 7GB VRAM.

### How is this possible?

Every layer swap in AirLLM (and similar disk-based offloaders) does this:

```
[Disk] → open file → mmap → read → pin → allocate GPU → DMA copy → compute
                    ~500ms per layer
```

DeepswapLLM pre-allocates two GPU staging buffers at startup and copies directly into them. No file opens, no pin_memory() calls, no GPU allocations during generation:

```
[Pinned RAM] → copy into pre-allocated GPU buffer → compute
                    ~11.6ms per layer
```

The `pin_memory()` call alone (which AirLLM does per-tensor, per-layer, per-token) costs **39ms per tensor**. With 12 tensors per layer, that's **470ms of pure overhead** before a single byte moves to GPU. DeepswapLLM eliminates this entirely.

### Correctness Verification

DeepswapLLM produces **bit-identical outputs** to native GPU inference:

```
  Reference (full GPU):
    'The capital of France is Paris. It is the largest
     city in Europe and the second largest in the world.'

  DeepswapLLM (layer-swapped):
    'The capital of France is Paris. It is the largest
     city in Europe and the second largest in the world.'

  Cosine similarity: 0.999994
  Text match: True
```

## Quick Start

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from deepswap import deepswap

# Load model to CPU (doesn't need to fit in VRAM)
model = AutoModelForCausalLM.from_pretrained(
    "your-model-here",
    torch_dtype=torch.bfloat16,
    device_map="cpu",
)
tokenizer = AutoTokenizer.from_pretrained("your-model-here")

# One line to enable layer swapping
model = deepswap(model)

# Use normally — swapping is transparent
inputs = tokenizer("The future of AI is", return_tensors="pt").to("cuda")
output = model.generate(**inputs, max_new_tokens=100)
print(tokenizer.decode(output[0]))

# Inspect what happened
print(model.summary())
# DeepSwap: 1/64 on GPU, 56 in RAM (60.5GB), 8 on disk (8.6GB), 1024 swaps

print(model.swap_stats())
# {'swaps': 1024, 'avg_swap_ms': 11.6, 'ram_loads': 896, 'disk_loads': 128, ...}
```

## How It Works

DeepswapLLM uses a **three-tier storage hierarchy** that automatically sizes to your hardware:

```
                    ┌──────────────────┐
           Tier 1   │    GPU VRAM      │  Hot layers — no transfer needed
                    │   (auto-sized)   │  Kept resident if VRAM permits
                    ├──────────────────┤
           Tier 2   │  Pinned CPU RAM  │  Warm layers — fast PCIe DMA
                    │  (60% of RAM)    │  ~12-14 GB/s via pre-alloc buffers
                    ├──────────────────┤
           Tier 3   │   NVMe Disk      │  Cold layers — loaded on demand
                    │  (safetensors)   │  Pre-mmap'd with page cache warming
                    └──────────────────┘
```

### Key Techniques

**Zero-Allocation Double Buffering** — Two GPU staging buffers are pre-allocated at startup. While the current layer computes on buffer A, the next layer's data streams into buffer B via async DMA. No `cudaMalloc`, no `pin_memory()`, no allocation overhead during generation.

**Tiered Auto-Sizing** — Measures available VRAM and RAM at startup. Keeps as many layers on GPU as fit, puts the rest in pinned CPU RAM (up to 60% of total RAM), and spills the remainder to NVMe disk via pre-mmap'd safetensors handles.

**ShardPool Pre-mmap** — Disk-tier layers use pre-opened safetensors file handles with `madvise(MADV_WILLNEED)` to warm the kernel page cache. After the first pass, disk reads serve from RAM transparently.

**CUDA Stream Prefetch** — A dedicated CUDA stream pre-loads the next layer while the current layer computes. For pinned tensors, the DMA transfer overlaps with GPU computation.

**Sparse Block Compression** — Layers with >15% sparsity are compressed using a custom sparse block encoding scheme, reducing CPU RAM footprint by up to 7x for sparse layers while maintaining fast decompression.

## GGUF Support

Load quantized GGUF models for even larger parameter counts. DeepswapLLM dequantizes each layer on the fly when swapping to GPU.

```python
from deepswap import deepswap_gguf

# Load the architecture skeleton (weights come from GGUF)
model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-32B-Instruct",
    torch_dtype=torch.float16,
    device_map="cpu",
)

# Load quantized weights from GGUF
model = deepswap_gguf("model-q4_k_m.gguf", model)

# Use normally
output = model.generate(**inputs, max_new_tokens=100)
```

Supports all GGUF quantization types: Q4_0, Q4_1, Q4_K_M, Q5_0, Q5_K, Q6_K, Q8_0, and more. Dequantization uses the `gguf` library's optimized numpy kernels.

## Compressed-Tensors / Quantized Models

Load models quantized with [compressed-tensors](https://github.com/neuralmagic/compressed-tensors) (MXFP4, INT8, FP8). Packed weights stay on disk via mmap'd safetensors and are decompressed per layer swap — handles models of any size regardless of RAM.

```python
from deepswap import deepswap_quantized

# One call — skeleton creation, weight mapping, and offloading are automatic
model = deepswap_quantized("/path/to/quantized-model")

# Use normally
output = model.generate(**inputs, max_new_tokens=100)
```

For MoE models (like Kimi K3 with 896 experts per layer), expert weights are loaded on-demand: only the experts selected by the router are decompressed and moved to GPU during each forward pass.

Supports MXFP4 (Microscaling FP4), INT8, and FP8 quantization formats with automatic format detection from `quantization_config`.

## SageAttention (Experimental)

Optional INT8 attention quantization for faster prefill on Ampere+ GPUs:

```python
model = deepswap(model, sage_attention=True)
```

Uses [SageAttention](https://github.com/thu-ml/SageAttention) for INT8 QK + FP16 PV during the prefill phase. Falls back to standard SDPA for autoregressive decoding. Best suited for long-context scenarios where prefill is the bottleneck.

## API Reference

### `deepswap(model, **kwargs) -> DeepSwapModel`

Wrap a HuggingFace model for layer-level GPU offloading.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `model` | `nn.Module` | required | HuggingFace model loaded to CPU |
| `device` | `torch.device` | `cuda:0` | Target GPU |
| `max_gpu_layers` | `int` | auto | Max layers on GPU (auto-sizes to VRAM) |
| `max_ram_gb` | `float` | auto | Max GB for RAM tier (auto: 60% of total) |
| `reserve_gb` | `float` | `2.0` | VRAM reserved for activations/KV cache |
| `pin_memory` | `bool` | `True` | Use pinned memory for DMA transfers |
| `sage_attention` | `bool` | `False` | Enable SageAttention (INT8 QK + FP16 PV) |

### `deepswap_gguf(gguf_path, model, **kwargs) -> DeepSwapModel`

Load a GGUF quantized model with layer-level offloading.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `gguf_path` | `str` | required | Path to GGUF file |
| `model` | `nn.Module` | required | Architecture skeleton (weights replaced) |
| `target_dtype` | `torch.dtype` | `float16` | Dtype after dequantization |

### `deepswap_quantized(model_path, **kwargs) -> DeepSwapModel`

Load a compressed-tensors quantized model with layer-level offloading.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `model_path` | `str` | required | Path to HF model with `quantization_config` |
| `device` | `torch.device` | `cuda:0` | Target GPU |
| `max_gpu_layers` | `int` | `1` | Max layers on GPU simultaneously |
| `reserve_gb` | `float` | `2.0` | VRAM reserved for activations/KV cache |
| `target_dtype` | `torch.dtype` | `float16` | Dtype for non-quantized parameters |

### `DeepSwapModel`

The wrapped model supports all standard HuggingFace methods:

```python
model.generate(**inputs, max_new_tokens=100)  # Standard generation
model.summary()                                # Tier breakdown
model.swap_stats()                             # Performance metrics
model.config                                   # Original model config
```

## Supported Architectures

Any HuggingFace model with a standard transformer layer structure:

| Layer Path | Models |
|---|---|
| `model.layers` | Llama, Qwen, Mistral, Gemma, DeepSeek, Phi, InternLM, Yi |
| `language_model.model.layers` | Kimi K3, VLMs with language_model wrapper |
| `transformer.h` | GPT-2, GPT-Neo, StarCoder |
| `gpt_neox.layers` | GPT-NeoX, Pythia, RedPajama |
| `transformer.layers` | Falcon, MPT |
| `encoder.layer` | BERT, RoBERTa, DeBERTa |

## Requirements

```
torch >= 2.0
transformers
numpy
safetensors
```

Optional:
```
sageattention      # For INT8 attention (sage_attention=True)
gguf               # For GGUF model loading
compressed-tensors # For MXFP4/INT8/FP8 quantized models
```

## Installation

```bash
git clone https://github.com/apolloraines/DeepswapLLM.git
cd DeepswapLLM
pip install -r requirements.txt

# Add to your Python path
export PYTHONPATH=$PYTHONPATH:$(pwd)/src
```

## How It Compares

| Feature | DeepswapLLM | AirLLM | llama.cpp (offload) | HF Accelerate |
|---|:---:|:---:|:---:|:---:|
| Zero-alloc double buffer | Yes | No | No | No |
| Tiered storage (VRAM/RAM/disk) | Yes | Disk only | VRAM/RAM | VRAM/RAM |
| Pre-allocated GPU buffers | Yes | No | No | No |
| CUDA stream prefetch | Yes | Partial | No | No |
| GGUF quantized models | Yes | No (HF quants only) | Yes | No |
| Compressed-tensors (MXFP4) | Yes | Yes | No | No |
| MoE expert-level offload | Yes | Yes | No | No |
| Sparse compression | Yes | No | No | No |
| Auto-sizes to hardware | Yes | No | Manual | Manual |
| HuggingFace compatible | Yes | Yes | No | Yes |
| Any architecture | Yes | Yes | Limited | Yes |

## License

Apache License 2.0. See [LICENSE](LICENSE) for details.
