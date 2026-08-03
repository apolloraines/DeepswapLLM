#!/usr/bin/env python3
"""
DeepswapLLM GGUF integration test.

Loads a quantized GGUF model and verifies it produces coherent output.
Requires GPU and a Qwen2.5-0.5B-Instruct Q4_K_M GGUF file.

Usage:
    python tests/test_gguf.py
"""

import sys
import logging

sys.path.insert(0, "/home/nova/DEVELOPMENTS/DeepSwapLLM/src")

import torch

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)


def main():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from deepswap import deepswap_gguf

    gguf_path = "/mnt/model_storage/.hf_cache/hub/models--Qwen--Qwen2.5-0.5B-Instruct-GGUF/snapshots/9217f5db79a29953eb74d5343926648285ec7e67/qwen2.5-0.5b-instruct-q4_k_m.gguf"
    model_name = "Qwen/Qwen2.5-0.5B-Instruct"
    prompt = "The capital of France is"

    print("=" * 60)
    print("  DeepswapLLM GGUF Integration Test")
    print("=" * 60)

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    cpu_model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float16,
        device_map="cpu", trust_remote_code=True,
    )
    cpu_model.eval()

    print(f"\n[1/3] Loading GGUF: {gguf_path.split('/')[-1]}")
    swapped = deepswap_gguf(gguf_path, cpu_model, max_gpu_layers=2, target_dtype=torch.float16)
    print(f"  {swapped.summary()}")

    print(f"\n[2/3] Generating...")
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda:0")
    with torch.no_grad():
        out = swapped.generate(**inputs, max_new_tokens=32, do_sample=False)
    text = tokenizer.decode(out[0], skip_special_tokens=True)
    print(f"  Output: {text!r}")
    print(f"  Stats: {swapped.swap_stats()}")

    print(f"\n[3/3] Validation")
    has_paris = "paris" in text.lower()
    has_tokens = len(text.split()) > 5
    passed = has_paris and has_tokens

    if passed:
        print(f"  PASS (coherent output with 'Paris')")
    else:
        print(f"  FAIL (output: {text!r})")
    print("=" * 60)
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
