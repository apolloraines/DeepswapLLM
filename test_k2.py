#!/usr/bin/env python3
"""Go/no-go feasibility test: DeepswapLLM loading + generating on Kimi-K2-Thinking.

K2 is DeepseekV3ForCausalLM, INT4 pack-quantized (compressed-tensors), 61 layers,
384 experts/layer routing 8 per token. Unlike K3 (MXFP4), K2's routed experts need
the new INT4 decode path in deepswap.py (_decompress_int4_packed_tensor). This runs
on the HDD copy first; if it produces a token, the 554GB NVMe copy is worth doing.
"""
import sys
import os
import time
import traceback

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
os.environ["TRANSFORMERS_VERBOSITY"] = "warning"

import torch
import logging

logging.basicConfig(level=logging.WARNING, stream=sys.stderr)
log = logging.getLogger("k2_test")
log.setLevel(logging.INFO)


def main():
    t_start = time.time()
    # Default HDD copy; override with argv[1] or K2_MODEL_PATH for the NVMe run.
    model_path = (
        sys.argv[1] if len(sys.argv) > 1
        else os.environ.get("K2_MODEL_PATH",
                            "/mnt/model_storage/stored_models/Kimi-K2-Thinking")
    )
    log.info("Model path: %s", model_path)
    storage = "nvme" if "_nvme_" in model_path or "/nvme" in model_path else "hdd"

    log.info("Loading tokenizer...")
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    log.info("Tokenizer loaded (%.1fs)", time.time() - t_start)

    log.info("Setting up DeepswapLLM + K2 (INT4)...")
    from deepswap import deepswap_quantized
    t_setup = time.time()
    model = deepswap_quantized(
        model_path,
        device=torch.device("cuda:0"),
        max_gpu_layers=1,
        reserve_gb=4.0,
        target_dtype=torch.float16,
    )
    setup_time = time.time() - t_setup
    log.info("Setup: %.1fs", setup_time)

    prompt = "Hello"
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"].to("cuda:0")
    log.info("Generating 1 token from: '%s' (shape %s)", prompt, input_ids.shape)

    swap_count = [0]
    original_restore = model._manager.restore_to_gpu.__func__

    def traced_restore(mgr, layer_name, module):
        swap_count[0] += 1
        vram_mb = torch.cuda.memory_allocated() // (1024 * 1024)
        if swap_count[0] <= 5 or swap_count[0] % 10 == 0:
            log.info("  Layer swap #%d: %s (VRAM: %dMB)", swap_count[0], layer_name, vram_mb)
        original_restore(mgr, layer_name, module)

    import types
    model._manager.restore_to_gpu = types.MethodType(traced_restore, model._manager)

    # One forward pass = one token's worth of per-layer disk streaming, which is the
    # metric. We bypass generate() because K2's vendored modeling_deepseek.py calls the
    # removed DynamicCache.seen_tokens under transformers 4.56; a direct forward with
    # use_cache=False avoids that cache path while doing the identical disk work.
    seq_len = input_ids.shape[1]
    attn_mask = torch.ones((1, seq_len), dtype=torch.long, device="cuda:0")
    position_ids = torch.arange(seq_len, device="cuda:0").unsqueeze(0)
    t_gen = time.time()
    with torch.no_grad():
        out = model(
            input_ids,
            attention_mask=attn_mask,
            position_ids=position_ids,
            use_cache=False,
        )
    logits = out.logits if hasattr(out, "logits") else out[0]
    next_id = int(logits[0, -1].argmax())
    gen_time = time.time() - t_gen
    gen_text = tokenizer.decode([next_id])

    stats = model.swap_stats()
    log.info("=" * 60)
    log.info("  RESULTS (DeepswapLLM / Kimi-K2, %s)", storage.upper())
    log.info("=" * 60)
    log.info("  Setup: %.1fs", setup_time)
    log.info("  Generation: %.1fs for 1 token", gen_time)
    log.info("  Swaps: %d", stats["swaps"])
    log.info("  Avg swap: %.1fms", stats.get("avg_swap_ms", 0))
    log.info("  RAM loads: %d, Disk loads: %d", stats["ram_loads"], stats["disk_loads"])
    log.info("  Output: '%s'", gen_text)
    log.info("  VRAM: %dMB", torch.cuda.memory_allocated() // (1024 * 1024))
    log.info("=" * 60)
    log.info("  Total time: %.1fs", time.time() - t_start)
    log.info("=" * 60)
    print(f"RESULT deepswap_k2_{storage} gen={gen_time:.1f} setup={setup_time:.1f} "
          f"vram={torch.cuda.memory_allocated() // (1024 * 1024)} out={gen_text!r}")
    print(f"SUCCESS: {gen_text}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        traceback.print_exc()
        sys.exit(1)
