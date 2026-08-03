# DeepSwapLLM

Run oversized LLMs on undersized GPUs. Actually fast.

DeepSwapLLM keeps transformer layers compressed in CPU RAM and swaps them onto GPU on demand. Unlike disk-based offloading solutions, DeepSwapLLM operates entirely from memory with intelligent GPU residency management — keeping as many layers on GPU as physically fit and only swapping the rest.

## Quick Start

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from deepswap import deepswap

model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-32B",
    torch_dtype=torch.float16,
    device_map="cpu",
)
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-32B")

model = deepswap(model)  # that's it

inputs = tokenizer("The future of AI is", return_tensors="pt").to("cuda")
output = model.generate(**inputs, max_new_tokens=100)
print(tokenizer.decode(output[0]))

print(model.summary())
print(model.swap_stats())
```

## How It Works

1. **Compress** — Each transformer layer is sparse-block compressed and stored in pinned CPU RAM
2. **Auto-size** — Measures free VRAM, keeps as many layers on GPU as fit (minus a reserve for KV cache and activations)
3. **Swap** — When a layer not on GPU is needed, the oldest resident layer is evicted and the new one is decompressed + uploaded via PCIe DMA
4. **Prefetch** — While the current layer computes, the next layer is pre-decompressed in a background thread

## Why Not AirLLM?

| | AirLLM | DeepSwapLLM |
|---|---|---|
| **Offload to** | Disk (NVMe) | CPU RAM (pinned) |
| **Transfer speed** | ~5 GB/s | ~25 GB/s (PCIe 4.0) |
| **GPU residency** | 1 layer, always | Auto-sized to fill VRAM |
| **Prefetch** | Llama2 only | All architectures |
| **Compression** | None | Sparse block encoding |
| **Published benchmarks** | None | See below |
| **Pinned memory DMA** | No | Yes |

AirLLM streams layers from disk. DeepSwapLLM streams from RAM. RAM is 5x faster than NVMe. That's the whole story.

## Estimated Performance (RTX 3090, 24GB)

| Model | Params | Precision | Layers on GPU | Layers Swapped | Est. tok/s |
|---|---|---|---|---|---|
| Qwen2.5-7B | 7B | fp16 | 28/28 | 0 | Full speed (fits entirely) |
| Qwen2.5-32B | 32B | Q8 | 44/64 | 20 | ~2.5 |
| Llama-3-70B | 70B | Q4 | 50/80 | 30 | ~1.5 |
| Llama-3-70B | 70B | fp16 | 10/80 | 70 | ~0.4 |

## API

### `deepswap(model, device=None, max_gpu_layers=None, reserve_gb=2.0, pin_memory=True)`

Wraps a HuggingFace model for layer-level GPU offloading.

- **model** — Any `AutoModelForCausalLM` loaded to CPU
- **device** — Target GPU (default: `cuda:0`)
- **max_gpu_layers** — Override auto-sizing. `None` = fill available VRAM
- **reserve_gb** — VRAM to keep free for activations/KV cache (default: 2GB)
- **pin_memory** — Use pinned CPU memory for DMA transfers (default: True)

Returns a `DeepSwapModel` with `.generate()`, `.summary()`, and `.swap_stats()`.

## Supported Architectures

Any HuggingFace model with a standard layer structure:
- `model.layers` — Llama, Qwen, Mistral, Gemma, DeepSeek, Phi
- `transformer.h` — GPT-2, GPT-Neo
- `gpt_neox.layers` — GPT-NeoX, Pythia
- `transformer.layers` — Falcon, MPT
- `encoder.layer` — BERT, RoBERTa

## Requirements

```
torch >= 2.0
transformers
numpy
```

## License

Copyright (c) 2025 Apollo Raines / Robert Rice. All Rights Reserved.
Proprietary and Confidential.
