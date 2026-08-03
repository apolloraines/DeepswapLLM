#!/usr/bin/env python3
"""
DeepswapLLM GPU integration test.

Loads a small model normally, runs reference inference, then loads with
deepswap() and compares outputs. Requires GPU.

Usage:
    python tests/test_deepswap.py
"""

import gc
import logging
import sys
import time

sys.path.insert(0, "/home/nova/DEVELOPMENTS/DeepswapLLM/src")

import torch

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)


def get_vram_mb():
    return torch.cuda.memory_allocated() / 1e6


def main():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from deepswap import deepswap

    model_name = "Qwen/Qwen2.5-0.5B"
    prompt = "The capital of France is"

    print("=" * 60)
    print("  DeepswapLLM Integration Test")
    print("=" * 60)

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Reference
    print(f"\n[1/4] Reference inference ({model_name} on GPU)...")
    ref_model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float16,
        device_map="cuda:0", trust_remote_code=True,
    )
    ref_model.eval()
    ref_vram = get_vram_mb()

    inputs = tokenizer(prompt, return_tensors="pt").to("cuda:0")
    with torch.no_grad():
        ref_out = ref_model.generate(**inputs, max_new_tokens=32, do_sample=False)
        ref_logits = ref_model(**inputs).logits.cpu()
    ref_text = tokenizer.decode(ref_out[0], skip_special_tokens=True)
    print(f"  VRAM: {ref_vram:.0f}MB | Output: {ref_text!r}")

    del ref_model
    torch.cuda.empty_cache()
    gc.collect()

    # DeepSwap
    print(f"\n[2/4] DeepSwap inference...")
    cpu_model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float16,
        device_map="cpu", trust_remote_code=True,
    )
    cpu_model.eval()

    t0 = time.perf_counter()
    swapped = deepswap(cpu_model, max_gpu_layers=2)
    setup_s = time.perf_counter() - t0
    swap_vram = get_vram_mb()

    inputs = tokenizer(prompt, return_tensors="pt").to("cuda:0")
    t0 = time.perf_counter()
    with torch.no_grad():
        swap_out = swapped.generate(**inputs, max_new_tokens=32, do_sample=False)
        swap_logits = swapped._model(**inputs).logits.cpu()
    gen_s = time.perf_counter() - t0
    swap_text = tokenizer.decode(swap_out[0], skip_special_tokens=True)
    print(f"  VRAM: {swap_vram:.0f}MB | Setup: {setup_s:.1f}s | Gen: {gen_s:.2f}s")
    print(f"  Output: {swap_text!r}")
    print(f"  {swapped.summary()}")
    print(f"  Stats: {swapped.swap_stats()}")

    # Compare
    print(f"\n[3/4] Comparing...")
    cos = torch.nn.functional.cosine_similarity(
        ref_logits.flatten().unsqueeze(0).float(),
        swap_logits.flatten().unsqueeze(0).float(),
    ).item()
    text_match = ref_text == swap_text
    print(f"  Cosine similarity: {cos:.6f}")
    print(f"  Text match: {text_match}")
    print(f"  VRAM: {ref_vram:.0f}MB (normal) -> {swap_vram:.0f}MB (swapped)")

    # Verdict
    print(f"\n[4/4] Result")
    passed = cos > 0.99
    if passed:
        print(f"  PASS (cosine={cos:.6f})")
    else:
        print(f"  FAIL (cosine={cos:.6f} < 0.99)")
    print("=" * 60)
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
