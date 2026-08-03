#!/usr/bin/env python3
"""DeepSwapLLM compression format tests. No GPU required."""

import sys
sys.path.insert(0, "/home/nova/DEVELOPMENTS/DeepSwapLLM/src")

import numpy as np
from deepswap_compress import compress_chunked, decompress_chunked


def test_roundtrip(name, data, min_ratio=None):
    compressed = compress_chunked(data)
    recovered = decompress_chunked(compressed)
    np_dtype = {2: np.float16, 4: np.float32}[data.dtype.itemsize]
    arr = np.frombuffer(recovered, dtype=np_dtype)
    ratio = data.nbytes / len(compressed)
    assert np.array_equal(data.ravel(), arr), f"{name}: data mismatch"
    if min_ratio:
        assert ratio >= min_ratio, f"{name}: ratio {ratio:.2f}x < {min_ratio}x"
    print(f"  PASS {name}: {data.nbytes:,} -> {len(compressed):,} ({ratio:.2f}x)")


def main():
    print("=" * 60)
    print("  DeepSwapLLM Compression Tests")
    print("=" * 60)

    rng = np.random.default_rng(42)

    # Sparse data
    d = rng.standard_normal(100_000).astype(np.float32)
    d[rng.random(100_000) < 0.9] = 0.0
    test_roundtrip("sparse fp32 90%", d, min_ratio=3.0)

    d = rng.standard_normal(50_000).astype(np.float16)
    d[rng.random(50_000) < 0.8] = np.float16(0)
    test_roundtrip("sparse fp16 80%", d, min_ratio=2.0)

    # Dense
    test_roundtrip("dense fp32", rng.standard_normal(10_000).astype(np.float32))

    # All zeros
    test_roundtrip("all zeros", np.zeros(100_000, dtype=np.float32), min_ratio=50.0)

    # Simulated weight matrix
    w = rng.standard_normal((1536, 1536)).astype(np.float16)
    w[np.abs(w) < 0.5] = np.float16(0)
    test_roundtrip("weight matrix", w.ravel())

    print("\n  All tests passed.")
    print("=" * 60)


if __name__ == "__main__":
    main()
