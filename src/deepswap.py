"""
DeepswapLLM — Run oversized LLMs on undersized GPUs.
Copyright 2025 Apollo Raines
Licensed under the Apache License, Version 2.0

Tiered layer storage with intelligent GPU swapping.
Layers live across three tiers:
  1. GPU VRAM  — hot layers, no transfer needed
  2. Pinned RAM — warm layers, fast PCIe DMA (~12 GB/s)
  3. NVMe disk — cold layers, loaded on demand (~3.5 GB/s)

Auto-sizes each tier based on available VRAM and RAM.
Only the active transformer layer occupies GPU memory.

Usage:
    from deepswap import deepswap

    model = AutoModelForCausalLM.from_pretrained(name, torch_dtype=torch.float16)
    model = deepswap(model)
    output = model.generate(input_ids, max_new_tokens=64)
"""

from __future__ import annotations

import ctypes
import gc
import json
import logging
import mmap as mmap_mod
import os
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

import torch
import torch.nn as nn

from deepswap_compress import compress_chunked, decompress_chunked, _probe_sparsity

logger = logging.getLogger("deepswap")

SPARSITY_THRESHOLD = 0.15
PIN_RESERVE_GB = 12.0
RAM_RESERVE_GB = 20.0

# ---------------------------------------------------------------------------
#  Huge pages (2MB) for pinned CUDA memory — +26% DMA bandwidth
# ---------------------------------------------------------------------------

_libc = ctypes.CDLL("libc.so.6", use_errno=True)
_libc.mmap.restype = ctypes.c_void_p
_libc.mmap.argtypes = [
    ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int,
    ctypes.c_int, ctypes.c_int, ctypes.c_long,
]
_libc.munmap.argtypes = [ctypes.c_void_p, ctypes.c_size_t]

_MAP_PRIVATE = 0x02
_MAP_ANONYMOUS = 0x20
_MAP_HUGETLB = 0x40000
_MAP_HUGE_2MB = 21 << 26
_PROT_RW = 0x03
_PAGE_2MB = 2 * 1024 * 1024
_CUDA_HOST_REGISTER_PORTABLE = 1


def _hugepages_available() -> int:
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("HugePages_Free"):
                    return int(line.split()[1])
    except (OSError, ValueError):
        pass
    return 0


class HugePinnedBuffer:
    """2MB-huge-page-backed CUDA-pinned buffer for faster DMA."""

    def __init__(self, nbytes: int, dtype: torch.dtype = torch.bfloat16) -> None:
        alloc_size = (nbytes + _PAGE_2MB - 1) & ~(_PAGE_2MB - 1)
        flags = _MAP_PRIVATE | _MAP_ANONYMOUS | _MAP_HUGETLB | _MAP_HUGE_2MB
        ptr = _libc.mmap(None, alloc_size, _PROT_RW, flags, -1, 0)
        if ptr == ctypes.c_void_p(-1).value:
            raise MemoryError("MAP_HUGETLB failed")
        ret = int(torch.cuda.cudart().cudaHostRegister(
            ptr, alloc_size, _CUDA_HOST_REGISTER_PORTABLE,
        ))
        if ret != 0:
            _libc.munmap(ptr, alloc_size)
            raise RuntimeError(f"cudaHostRegister failed: CUDA error {ret}")
        self._ptr = ptr
        self._alloc_size = alloc_size
        elem_size = torch.tensor([], dtype=dtype).element_size()
        numel = nbytes // elem_size
        raw_dtype = {2: ctypes.c_int16, 4: ctypes.c_int32, 1: ctypes.c_int8}[elem_size]
        arr = (raw_dtype * numel).from_address(ptr)
        view_dtype = {2: torch.int16, 4: torch.int32, 1: torch.int8}[elem_size]
        self.tensor = torch.frombuffer(arr, dtype=view_dtype).view(dtype)

    def __del__(self) -> None:
        if hasattr(self, "_ptr") and self._ptr is not None:
            try:
                torch.cuda.cudart().cudaHostUnregister(self._ptr)
            except Exception:
                pass
            _libc.munmap(self._ptr, self._alloc_size)
            self._ptr = None


def _try_huge_pin(cpu_data: torch.Tensor) -> Tuple[torch.Tensor, Optional[HugePinnedBuffer]]:
    """Try to allocate a huge-page-pinned copy. Falls back to regular pin_memory."""
    pages_needed = (cpu_data.nbytes + _PAGE_2MB - 1) // _PAGE_2MB
    free_pages = _hugepages_available()
    if free_pages < pages_needed + 10:
        return cpu_data.pin_memory(), None
    # Also check that enough regular RAM remains (huge pages don't use regular RAM,
    # but we need headroom for the OS and other allocations)
    if _mem_available_bytes() < int(PIN_RESERVE_GB * 1024**3):
        return cpu_data.pin_memory(), None
    try:
        buf = HugePinnedBuffer(cpu_data.nbytes, cpu_data.dtype)
        buf.tensor.copy_(cpu_data)
        return buf.tensor.view(cpu_data.shape), buf
    except (MemoryError, RuntimeError):
        return cpu_data.pin_memory(), None


# ---------------------------------------------------------------------------
#  SageAttention monkey-patch
# ---------------------------------------------------------------------------

_sage_patched = False
_original_sdpa = None


def enable_sage_attention() -> bool:
    """Replace F.scaled_dot_product_attention with SageAttention (INT8 QK + FP16 PV).

    Returns True if patching succeeded, False if SageAttention is not installed.
    """
    global _sage_patched, _original_sdpa
    if _sage_patched:
        return True
    try:
        from sageattention import sageattn
    except ImportError:
        logger.warning("sageattention not installed — pip install sageattention")
        return False

    import torch.nn.functional as F
    _original_sdpa = F.scaled_dot_product_attention

    def _sage_compat(q, k, v, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None, **kw):
        # SageAttention (INT8 QK + FP16 PV) for prefill; SDPA for decode.
        # Only use sageattn when q_len == k_len (prefill) and q_len > 1.
        # Decode steps (q_len=1) are memory-bound, not compute-bound,
        # so sage's speedup doesn't help there and the SDPA kernel is fine.
        use_sage = (
            attn_mask is None
            and q.shape[-2] == k.shape[-2]
            and q.shape[-2] > 1
        )
        if use_sage:
            return sageattn(q, k, v, tensor_layout="HND", is_causal=is_causal, sm_scale=scale)
        return _original_sdpa(q, k, v, attn_mask=attn_mask, dropout_p=dropout_p,
                              is_causal=is_causal, scale=scale, **kw)

    F.scaled_dot_product_attention = _sage_compat
    _sage_patched = True
    logger.info("SageAttention enabled (INT8 QK + FP16 PV)")
    return True


def disable_sage_attention() -> None:
    """Restore original F.scaled_dot_product_attention."""
    global _sage_patched, _original_sdpa
    if _sage_patched and _original_sdpa is not None:
        import torch.nn.functional as F
        F.scaled_dot_product_attention = _original_sdpa
        _sage_patched = False
        logger.info("SageAttention disabled, original SDPA restored")


def _mem_available_bytes() -> int:
    """Read MemAvailable from /proc/meminfo (Linux only)."""
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemAvailable"):
                    return int(line.split()[1]) * 1024
    except (OSError, ValueError):
        pass
    return 0


def _mem_total_bytes() -> int:
    """Read MemTotal from /proc/meminfo (Linux only)."""
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemTotal"):
                    return int(line.split()[1]) * 1024
    except (OSError, ValueError):
        pass
    return 0


# ---------------------------------------------------------------------------
#  Storage types
# ---------------------------------------------------------------------------

@dataclass
class StoredTensor:
    """A single tensor stored in CPU RAM (compressed or raw)."""
    cpu_tensor: Optional[torch.Tensor]
    compressed_bytes: Optional[bytes]
    shape: torch.Size
    dtype: torch.dtype
    numel: int
    original_bytes: int
    stored_bytes: int
    is_sparse: bool

    @property
    def ratio(self) -> float:
        if self.stored_bytes == 0:
            return 0.0
        return self.original_bytes / self.stored_bytes


@dataclass
class StoredLayer:
    """All parameters of one transformer layer, stored in CPU RAM."""
    tensors: Dict[str, StoredTensor] = field(default_factory=dict)
    total_original: int = 0
    total_stored: int = 0
    _huge_bufs: List[Any] = field(default_factory=list)

    @property
    def ratio(self) -> float:
        if self.total_stored == 0:
            return 0.0
        return self.total_original / self.total_stored


@dataclass
class DiskLayer:
    """Layer stored on NVMe — tensors loaded on demand from safetensors."""
    tensor_map: Dict[str, Tuple[str, str]]  # param_name -> (shard_path, full_key)
    total_bytes: int = 0


@dataclass
class GGUFLayer:
    """Layer stored in a GGUF file — tensors dequantized on demand."""
    gguf_path: str
    layer_idx: int
    tensor_map: Dict[str, str]  # param_name -> gguf_tensor_name
    total_bytes: int = 0


@dataclass
class QuantizedLayer:
    """Layer with compressed-tensors packed weights — decompressed on GPU per-swap."""
    packed_groups: Dict[str, Dict[str, torch.Tensor]]
    plain_tensors: Dict[str, torch.Tensor]
    compressor: Any
    scheme: Any
    total_packed_bytes: int = 0


@dataclass
class DiskQuantizedLayer:
    """Layer with MXFP4/compressed-tensors packed weights on disk via mmap'd safetensors.

    For large MoE models (e.g., K3 with 896 experts/layer) where packed data
    per layer exceeds RAM. Tensors are loaded and decompressed per swap using
    ShardPool mmap handles.
    """
    packed_map: Dict[str, Dict[str, Tuple[str, str]]]
    plain_map: Dict[str, Tuple[str, str]]
    quant_format: str
    group_size: int = 32
    target_dtype: torch.dtype = torch.float16
    total_packed_bytes: int = 0


class ShardPool:
    """Keeps safetensors shard files pre-mmap'd with warm page cache.

    Eliminates per-load open/close/mmap overhead by holding persistent
    handles to shard files. Uses madvise(MADV_WILLNEED) to pre-populate
    the kernel page cache so disk-tier layers serve from RAM after the
    first pass.
    """

    def __init__(self) -> None:
        self._handles: Dict[str, Any] = {}
        self._mmaps: List[mmap_mod.mmap] = []

    def open_shard(self, shard_path: str) -> None:
        if shard_path in self._handles:
            return
        from safetensors import safe_open
        self._handles[shard_path] = safe_open(
            shard_path, framework="pt", device="cpu",
        )

    def get_tensor(self, shard_path: str, key: str) -> torch.Tensor:
        return self._handles[shard_path].get_tensor(key)

    def warm(self, shard_paths: Set[str]) -> None:
        """Trigger read-ahead into page cache via madvise(MADV_WILLNEED)."""
        for path in shard_paths:
            try:
                fd = os.open(path, os.O_RDONLY)
                mm = mmap_mod.mmap(fd, 0, access=mmap_mod.ACCESS_READ)
                mm.madvise(mmap_mod.MADV_WILLNEED)
                self._mmaps.append(mm)
                os.close(fd)
                logger.debug("Warming page cache: %s", os.path.basename(path))
            except (OSError, ValueError) as exc:
                logger.debug("madvise skipped for %s: %s", path, exc)

    def close(self) -> None:
        for mm in self._mmaps:
            try:
                mm.close()
            except Exception:
                pass
        self._handles.clear()
        self._mmaps.clear()


# ---------------------------------------------------------------------------
#  Helpers
# ---------------------------------------------------------------------------

def _to_numpy_flat(tensor: torch.Tensor):
    """Convert tensor to flat numpy array, handling bfloat16."""
    import numpy as np
    if tensor.dtype == torch.bfloat16:
        return tensor.view(torch.int16).view(-1).numpy().view(np.uint16)
    return tensor.view(-1).numpy()


def _store_state_dict(
    state_dict: Dict[str, torch.Tensor],
    pin_memory: bool = False,
) -> StoredLayer:
    """Store all tensors from a state dict in CPU RAM."""
    layer = StoredLayer()
    for name, param in state_dict.items():
        cpu_data = param.detach().cpu().contiguous()
        flat = _to_numpy_flat(cpu_data)
        sparsity = _probe_sparsity(flat)

        if sparsity >= SPARSITY_THRESHOLD:
            compressed = compress_chunked(flat)
            st = StoredTensor(
                cpu_tensor=None,
                compressed_bytes=compressed,
                shape=cpu_data.shape,
                dtype=cpu_data.dtype,
                numel=cpu_data.numel(),
                original_bytes=cpu_data.nbytes,
                stored_bytes=len(compressed),
                is_sparse=True,
            )
        else:
            if pin_memory and torch.cuda.is_available():
                avail = _mem_available_bytes()
                need = cpu_data.nbytes + int(PIN_RESERVE_GB * 1024**3)
                if avail > need:
                    try:
                        pinned, huge_buf = _try_huge_pin(cpu_data)
                        cpu_data = pinned
                        if huge_buf is not None:
                            layer._huge_bufs.append(huge_buf)
                    except (RuntimeError, OSError):
                        pass
            st = StoredTensor(
                cpu_tensor=cpu_data,
                compressed_bytes=None,
                shape=cpu_data.shape,
                dtype=cpu_data.dtype,
                numel=cpu_data.numel(),
                original_bytes=cpu_data.nbytes,
                stored_bytes=cpu_data.nbytes,
                is_sparse=False,
            )

        layer.tensors[name] = st
        layer.total_original += st.original_bytes
        layer.total_stored += st.stored_bytes

    return layer


def _restore_to_device(
    slayer: StoredLayer,
    device: torch.device,
    pin_memory: bool = True,
    non_blocking: bool = False,
) -> Dict[str, torch.Tensor]:
    """Restore a StoredLayer's tensors to the target device."""
    import numpy as np

    result = {}
    for name, st in slayer.tensors.items():
        if st.cpu_tensor is not None:
            tensor = st.cpu_tensor
            if tensor.is_pinned():
                result[name] = tensor.to(device=device, non_blocking=non_blocking)
            elif device.type == "cuda" and pin_memory:
                tensor = tensor.pin_memory()
                result[name] = tensor.to(device=device)
            else:
                result[name] = tensor.to(device=device)
        else:
            raw = decompress_chunked(st.compressed_bytes)
            elem_size = st.original_bytes // st.numel

            if st.dtype == torch.bfloat16:
                arr = np.frombuffer(raw, dtype=np.uint16).copy().reshape(st.shape)
                tensor = torch.from_numpy(arr.view(np.int16)).view(torch.bfloat16)
            else:
                np_dtype = {2: np.float16, 4: np.float32}.get(elem_size, np.float16)
                arr = np.frombuffer(raw, dtype=np_dtype).copy().reshape(st.shape)
                tensor = torch.from_numpy(arr)

            if device.type == "cuda" and pin_memory:
                tensor = tensor.pin_memory()

            result[name] = tensor.to(dtype=st.dtype, device=device)

    return result


def _load_disk_layer(
    dlayer: DiskLayer,
    device: torch.device,
    pool: Optional[ShardPool] = None,
) -> Dict[str, torch.Tensor]:
    """Load a DiskLayer's tensors from safetensors files to device."""
    by_shard: Dict[str, List[Tuple[str, str]]] = {}
    for param_name, (shard_path, full_key) in dlayer.tensor_map.items():
        by_shard.setdefault(shard_path, []).append((param_name, full_key))

    result = {}
    if pool is not None:
        for shard_path, items in by_shard.items():
            for param_name, full_key in items:
                tensor = pool.get_tensor(shard_path, full_key)
                if device.type == "cuda":
                    tensor = tensor.pin_memory()
                result[param_name] = tensor.to(device=device)
    else:
        from safetensors import safe_open
        for shard_path, items in by_shard.items():
            with safe_open(shard_path, framework="pt", device="cpu") as f:
                for param_name, full_key in items:
                    tensor = f.get_tensor(full_key)
                    if device.type == "cuda":
                        tensor = tensor.pin_memory()
                    result[param_name] = tensor.to(device=device)
    return result


def _load_gguf_layer(
    glayer: GGUFLayer,
    tensor_index: Dict[str, Any],
    device: torch.device,
    target_dtype: torch.dtype = torch.float16,
) -> Dict[str, torch.Tensor]:
    """Load a GGUFLayer's tensors, dequantizing on the fly."""
    from gguf.quants import dequantize
    import numpy as np

    result = {}
    for param_name, gguf_name in glayer.tensor_map.items():
        t = tensor_index[gguf_name]
        np_fp32 = dequantize(t.data, t.tensor_type)
        logical_shape = tuple(reversed(t.shape.tolist()))
        tensor = torch.from_numpy(np_fp32.reshape(logical_shape).copy())
        if device.type == "cuda":
            tensor = tensor.pin_memory()
        result[param_name] = tensor.to(dtype=target_dtype, device=device)
    return result


# GGUF tensor name patterns (HuggingFace param suffix -> GGUF suffix candidates)
# Some architectures use different GGUF names; list alternates in order of preference.
_GGUF_PARAM_MAP: Dict[str, List[str]] = {
    "input_layernorm.weight": ["attn_norm.weight"],
    "self_attn.q_proj.weight": ["attn_q.weight"],
    "self_attn.k_proj.weight": ["attn_k.weight"],
    "self_attn.v_proj.weight": ["attn_v.weight"],
    "self_attn.o_proj.weight": ["attn_output.weight"],
    "post_attention_layernorm.weight": ["ffn_norm.weight", "post_attention_norm.weight"],
    "mlp.gate_proj.weight": ["ffn_gate.weight"],
    "mlp.up_proj.weight": ["ffn_up.weight"],
    "mlp.down_proj.weight": ["ffn_down.weight"],
    "self_attn.q_proj.bias": ["attn_q.bias"],
    "self_attn.k_proj.bias": ["attn_k.bias"],
    "self_attn.v_proj.bias": ["attn_v.bias"],
    "self_attn.o_proj.bias": ["attn_output.bias"],
}


def _build_gguf_layer(
    layer_idx: int,
    gguf_path: str,
    tensor_names: Set[str],
    param_names: List[str],
) -> GGUFLayer:
    """Map HF param names to GGUF tensor names for a single layer."""
    tensor_map = {}
    total_bytes = 0

    for param_name in param_names:
        candidates = _GGUF_PARAM_MAP.get(param_name)
        matched = False
        if candidates:
            for suffix in candidates:
                gguf_name = f"blk.{layer_idx}.{suffix}"
                if gguf_name in tensor_names:
                    tensor_map[param_name] = gguf_name
                    matched = True
                    break
        if not matched:
            gguf_name = f"blk.{layer_idx}.{param_name}"
            if gguf_name in tensor_names:
                tensor_map[param_name] = gguf_name

    return GGUFLayer(
        gguf_path=gguf_path,
        layer_idx=layer_idx,
        tensor_map=tensor_map,
        total_bytes=total_bytes,
    )


def _decompress_quantized_layer(
    qlayer: QuantizedLayer,
    device: torch.device,
) -> Dict[str, torch.Tensor]:
    """Decompress a QuantizedLayer's packed tensors on GPU."""
    restored = {}

    for prefix, packed in qlayer.packed_groups.items():
        gpu_packed = {k: v.to(device) for k, v in packed.items()}
        decompressed = qlayer.compressor.decompress(gpu_packed, qlayer.scheme)
        if "weight" in decompressed:
            restored[f"{prefix}.weight"] = decompressed["weight"]
        else:
            for k, v in decompressed.items():
                restored[f"{prefix}.{k}"] = v
        del gpu_packed

    for name, tensor in qlayer.plain_tensors.items():
        restored[name] = tensor.to(device)

    return restored


def _decompress_mxfp4_tensor(
    packed: torch.Tensor,
    scale: torch.Tensor,
    group_size: int,
    target_dtype: torch.dtype,
) -> torch.Tensor:
    """Decompress a single MXFP4 packed tensor to target dtype on CPU."""
    from compressed_tensors import unpack_fp4_from_uint8
    from compressed_tensors.compressors.mx_utils import decompress_mx_scale

    m, n_packed = packed.shape
    n = n_packed * 2
    unpacked = unpack_fp4_from_uint8(packed, m, n, dtype=torch.bfloat16)
    scale_fp = decompress_mx_scale(scale)
    scale_expanded = scale_fp.repeat_interleave(group_size, dim=1)[:, :n]
    return (unpacked * scale_expanded).to(target_dtype)


def _decompress_int4_packed_tensor(
    packed: torch.Tensor,
    scale: torch.Tensor,
    weight_shape: Optional[torch.Tensor],
    group_size: int,
    target_dtype: torch.dtype,
) -> torch.Tensor:
    """Decompress a compressed-tensors INT4 pack-quantized weight (e.g. Kimi-K2).

    Each int32 packs eight symmetric 4-bit ints; weight_scale holds one bf16 scale
    per group_size columns. Reuses compressed-tensors' own unpacker so the nibble
    order and signed offset match the packer exactly.
    """
    from compressed_tensors.compressors.pack_quantized.helpers import unpack_from_int32

    pack_factor = 32 // 4
    if weight_shape is not None:
        out_features = int(weight_shape[0].item())
        in_features = int(weight_shape[1].item())
    else:
        out_features = packed.shape[0]
        in_features = packed.shape[1] * pack_factor
    unpacked = unpack_from_int32(
        packed, 4, torch.Size([out_features, in_features]), packed_dim=1,
    )
    scale_expanded = scale.repeat_interleave(group_size, dim=1)[:, :in_features]
    return unpacked.to(target_dtype) * scale_expanded.to(target_dtype)


def _load_dq_tensors(
    tensor_refs: Dict[str, Tuple[str, str]],
    pool: Optional[ShardPool],
    group_size: int,
    target_dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    """Load and decompress a single packed weight group from safetensors."""
    packed_sp, packed_key = tensor_refs["weight_packed"]
    scale_sp, scale_key = tensor_refs["weight_scale"]

    def _read(sp, key):
        if pool is not None:
            return pool.get_tensor(sp, key)
        from safetensors import safe_open
        with safe_open(sp, framework="pt") as f:
            return f.get_tensor(key)

    packed = _read(packed_sp, packed_key)
    scale = _read(scale_sp, scale_key)

    # int32-packed => compressed-tensors INT4 pack-quantized (Kimi-K2);
    # uint8-packed => MXFP4/FP4 (Kimi-K3).
    if packed.dtype == torch.int32:
        # Decompress on the GPU, not the CPU. unpack_from_int32 is pure-Python per-nibble
        # shifting; on the CPU it dominates the token (~56% in profiling). Moving the compact
        # packed int32 + scale to the card first runs the unpack/scale-expand as fast GPU ops
        # and transfers ~4x fewer bytes than shipping the expanded fp16 weight across PCIe.
        # weight_shape stays on the CPU: it only feeds .item() for the output dimensions.
        packed = packed.to(device, non_blocking=True)
        scale = scale.to(device, non_blocking=True)
        weight_shape = None
        if "weight_shape" in tensor_refs:
            ws_sp, ws_key = tensor_refs["weight_shape"]
            weight_shape = _read(ws_sp, ws_key)
        weight = _decompress_int4_packed_tensor(
            packed, scale, weight_shape, group_size, target_dtype,
        )
    else:
        weight = _decompress_mxfp4_tensor(packed, scale, group_size, target_dtype)
    result = weight.to(device)
    del packed, scale, weight
    return result


def _load_disk_quantized_layer(
    dqlayer: DiskQuantizedLayer,
    device: torch.device,
    pool: Optional[ShardPool] = None,
) -> Dict[str, torch.Tensor]:
    """Load and decompress a DiskQuantizedLayer's tensors from mmap'd safetensors."""
    restored = {}

    for prefix, tensor_refs in dqlayer.packed_map.items():
        restored[f"{prefix}.weight"] = _load_dq_tensors(
            tensor_refs, pool, dqlayer.group_size, dqlayer.target_dtype, device,
        )

    for name, (shard_path, full_key) in dqlayer.plain_map.items():
        if pool is not None:
            tensor = pool.get_tensor(shard_path, full_key)
        else:
            from safetensors import safe_open
            with safe_open(shard_path, framework="pt") as f:
                tensor = f.get_tensor(full_key)
        restored[name] = tensor.to(dtype=dqlayer.target_dtype, device=device)

    return restored


def _assign_tensor_by_path(root: nn.Module, dotted_name: str, tensor: torch.Tensor) -> bool:
    """Assign ``tensor`` to ``root.<dotted_name>``.

    Returns False (skipping the assignment) if an intermediate submodule is absent.
    That happens when a checkpoint stores a buffer the installed model recomputes
    instead of holding as a submodule — e.g. Kimi-K2 ships
    ``self_attn.rotary_emb.inv_freq`` per attention, but native DeepseekV3 builds RoPE
    at the model level, so ``self_attn`` has no ``rotary_emb`` child. inv_freq is
    derived from config, so dropping the stored copy is lossless.
    """
    parts = dotted_name.split(".")
    target = root
    for part in parts[:-1]:
        if not hasattr(target, part):
            return False
        target = getattr(target, part)
    attr = getattr(target, parts[-1], None)
    if isinstance(attr, nn.Parameter):
        attr.data = tensor
    else:
        setattr(target, parts[-1], tensor)
    return True


def _install_expert_offload_hooks(
    module: nn.Module,
    dqlayer: DiskQuantizedLayer,
    device: torch.device,
    pool: Optional[ShardPool] = None,
) -> None:
    """Install per-expert hooks for on-demand GPU loading of MoE expert weights.

    For each expert submodule (e.g., experts.0, experts.1, ...):
    - pre_forward: decompress packed weights from disk and move to GPU
    - post_forward: move weights back to CPU and free GPU memory

    Non-expert weights (attention, norms) are loaded to GPU once.
    """
    expert_prefix_map: Dict[str, Dict[str, Dict[str, Tuple[str, str]]]] = {}
    for prefix, tensor_refs in dqlayer.packed_map.items():
        parts = prefix.split(".")
        expert_key = None
        for j, part in enumerate(parts):
            if part == "experts" and j + 1 < len(parts):
                expert_key = ".".join(parts[:j+2])
                break
        if expert_key:
            expert_prefix_map.setdefault(expert_key, {})[prefix] = tensor_refs

    non_expert_packed = {
        p: r for p, r in dqlayer.packed_map.items()
        if not any(p.startswith(ek) for ek in expert_prefix_map)
    }
    for prefix, tensor_refs in non_expert_packed.items():
        weight = _load_dq_tensors(
            tensor_refs, pool, dqlayer.group_size, dqlayer.target_dtype, device,
        )
        _assign_tensor_by_path(module, prefix + ".weight", weight)

    for name, (shard_path, full_key) in dqlayer.plain_map.items():
        if pool is not None:
            tensor = pool.get_tensor(shard_path, full_key)
        else:
            from safetensors import safe_open
            with safe_open(shard_path, framework="pt") as f:
                tensor = f.get_tensor(full_key)
        tensor = tensor.to(dtype=dqlayer.target_dtype, device=device)
        _assign_tensor_by_path(module, name, tensor)

    for expert_key, packed_refs in expert_prefix_map.items():
        parts = expert_key.split(".")
        expert_mod = module
        for p in parts:
            expert_mod = getattr(expert_mod, p)

        expert_mod._dq_packed_refs = packed_refs
        expert_mod._dq_pool = pool
        expert_mod._dq_group_size = dqlayer.group_size
        expert_mod._dq_target_dtype = dqlayer.target_dtype
        expert_mod._dq_device = device

        def make_pre_hook(em, ek):
            def hook(m, args):
                for prefix, tensor_refs in em._dq_packed_refs.items():
                    rel = prefix[len(ek) + 1:] if prefix.startswith(ek + ".") else prefix
                    weight = _load_dq_tensors(
                        tensor_refs, em._dq_pool,
                        em._dq_group_size, em._dq_target_dtype, em._dq_device,
                    )
                    param_name = rel + ".weight"
                    p = param_name.split(".")
                    target = m
                    for pp in p[:-1]:
                        target = getattr(target, pp)
                    attr = getattr(target, p[-1])
                    if isinstance(attr, nn.Parameter):
                        attr.data = weight
                    else:
                        setattr(target, p[-1], weight)
            return hook

        def make_post_hook(em, ek):
            def hook(m, args, output):
                for prefix in em._dq_packed_refs:
                    rel = prefix[len(ek) + 1:] if prefix.startswith(ek + ".") else prefix
                    param_name = rel + ".weight"
                    p = param_name.split(".")
                    target = m
                    for pp in p[:-1]:
                        target = getattr(target, pp)
                    attr = getattr(target, p[-1])
                    if isinstance(attr, nn.Parameter):
                        attr.data = torch.empty(0, dtype=em._dq_target_dtype)
                    else:
                        setattr(target, p[-1], torch.empty(0, dtype=em._dq_target_dtype))
                return output
            return hook

        expert_mod.register_forward_pre_hook(make_pre_hook(expert_mod, expert_key))
        expert_mod.register_forward_hook(make_post_hook(expert_mod, expert_key))


def _estimate_layer_bytes(module: nn.Module) -> int:
    """Estimate GPU memory needed for a single layer."""
    total = 0
    for param in module.parameters():
        total += param.numel() * param.element_size()
    for buf in module.buffers():
        total += buf.numel() * buf.element_size()
    return total


def _auto_max_layers(
    layers: List[Tuple[str, nn.Module]],
    device: torch.device,
    reserve_gb: float = 2.0,
    non_layer_bytes: int = 0,
) -> int:
    """Compute how many layers can fit on GPU simultaneously."""
    if not layers or device.type != "cuda":
        return len(layers)

    torch.cuda.empty_cache()
    free_mem, total_mem = torch.cuda.mem_get_info(device)
    available = free_mem - int(reserve_gb * 1024**3) - non_layer_bytes

    if available <= 0:
        return 1

    per_layer = _estimate_layer_bytes(layers[0][1])
    if per_layer == 0:
        return len(layers)

    available -= per_layer

    if available <= 0:
        return 1

    max_fit = max(1, available // per_layer)
    return min(max_fit, len(layers))


def _build_safetensors_map(model_path: str) -> Optional[Dict[str, str]]:
    """Build tensor_name -> shard_file_path mapping from safetensors index."""
    index_path = os.path.join(model_path, "model.safetensors.index.json")
    if not os.path.exists(index_path):
        single = os.path.join(model_path, "model.safetensors")
        if os.path.exists(single):
            return None
        return None

    with open(index_path) as f:
        index = json.load(f)

    weight_map = index.get("weight_map", {})
    return {
        tensor_name: os.path.join(model_path, shard_file)
        for tensor_name, shard_file in weight_map.items()
    }


def _find_layer_prefix(
    module: nn.Module,
    safetensors_map: Dict[str, str],
) -> Optional[str]:
    """Find the safetensors key prefix for a layer module."""
    first_param = next(iter(module.state_dict()), None)
    if first_param is None:
        return None
    suffix = f".{first_param}"
    for full_key in safetensors_map:
        if full_key.endswith(suffix):
            return full_key[: -len(suffix)]
    return None


def _build_disk_layer(
    module: nn.Module,
    prefix: str,
    safetensors_map: Dict[str, str],
) -> DiskLayer:
    """Create a DiskLayer referencing safetensors files for all params."""
    tensor_map = {}
    total_bytes = 0
    for param_name, param in module.state_dict().items():
        full_key = f"{prefix}.{param_name}"
        if full_key in safetensors_map:
            tensor_map[param_name] = (safetensors_map[full_key], full_key)
            total_bytes += param.numel() * param.element_size()
    return DiskLayer(tensor_map=tensor_map, total_bytes=total_bytes)


# ---------------------------------------------------------------------------
#  GPU Buffer Pool — eliminates per-swap allocation and pin_memory overhead
# ---------------------------------------------------------------------------

class GPUBufferPool:
    """Pre-allocated GPU buffers for double-buffered layer swapping.

    Two flat GPU buffers alternate roles: one holds the active layer's params
    during compute, the other receives the next layer's data via DMA.
    Eliminates per-swap cudaHostRegister (pin_memory) and cudaMalloc overhead.
    """

    def __init__(self, layer_bytes: int, dtype: torch.dtype, device: torch.device) -> None:
        numel = layer_bytes // torch.tensor([], dtype=dtype).element_size()
        self._bufs = [
            torch.empty(numel, dtype=dtype, device=device),
            torch.empty(numel, dtype=dtype, device=device),
        ]
        self._active = 0

    def __del__(self) -> None:
        self._bufs.clear()

    def set_layout(self, param_shapes: Dict[str, Tuple[int, torch.Size]]) -> None:
        """Set the parameter layout: name -> (offset_in_elements, shape)."""
        self._layout = param_shapes

    @property
    def loading_buf(self) -> torch.Tensor:
        return self._bufs[1 - self._active]

    @property
    def active_buf(self) -> torch.Tensor:
        return self._bufs[self._active]

    def swap(self) -> None:
        self._active = 1 - self._active

    def load_from_stored(
        self,
        slayer: "StoredLayer",
        stream: Optional[torch.cuda.Stream] = None,
    ) -> Dict[str, torch.Tensor]:
        """Copy stored layer into the loading buffer, return param views."""
        buf = self.loading_buf
        result = {}
        offset = 0
        ctx = torch.cuda.stream(stream) if stream else _nullcontext()
        with ctx:
            for name, st in slayer.tensors.items():
                cpu_t = st.cpu_tensor
                if cpu_t is not None:
                    numel = cpu_t.numel()
                    dst = buf[offset:offset + numel]
                    if cpu_t.is_pinned():
                        dst.copy_(cpu_t.view(-1), non_blocking=True)
                    else:
                        dst.copy_(cpu_t.view(-1))
                    result[name] = dst.view(cpu_t.shape)
                    offset += numel
                else:
                    import numpy as np
                    raw = decompress_chunked(st.compressed_bytes)
                    elem_size = st.original_bytes // st.numel
                    if st.dtype == torch.bfloat16:
                        arr = np.frombuffer(raw, dtype=np.uint16).copy().reshape(st.shape)
                        cpu_t = torch.from_numpy(arr.view(np.int16)).view(torch.bfloat16)
                    else:
                        np_dtype = {2: np.float16, 4: np.float32}.get(elem_size, np.float16)
                        arr = np.frombuffer(raw, dtype=np_dtype).copy().reshape(st.shape)
                        cpu_t = torch.from_numpy(arr).to(dtype=st.dtype)
                    numel = cpu_t.numel()
                    dst = buf[offset:offset + numel]
                    dst.copy_(cpu_t.view(-1))
                    result[name] = dst.view(cpu_t.shape)
                    offset += numel
        return result

    def load_from_dict(
        self,
        tensors: Dict[str, torch.Tensor],
        stream: Optional[torch.cuda.Stream] = None,
    ) -> Dict[str, torch.Tensor]:
        """Copy a dict of CPU tensors into the loading buffer, return views."""
        buf = self.loading_buf
        result = {}
        offset = 0
        ctx = torch.cuda.stream(stream) if stream else _nullcontext()
        with ctx:
            for name, cpu_t in tensors.items():
                numel = cpu_t.numel()
                dst = buf[offset:offset + numel]
                if cpu_t.is_pinned():
                    dst.copy_(cpu_t.view(-1), non_blocking=True)
                else:
                    dst.copy_(cpu_t.view(-1))
                result[name] = dst.view(cpu_t.shape)
                offset += numel
        return result


class _nullcontext:
    def __enter__(self):
        return self
    def __exit__(self, *args):
        pass


# ---------------------------------------------------------------------------
#  Manager
# ---------------------------------------------------------------------------

class LayerSwapManager:
    """Manages tiered layer storage (RAM + disk) and GPU residency."""

    def __init__(
        self,
        device: torch.device,
        max_gpu_layers: int = 1,
        pin_memory: bool = True,
        pin_storage: bool = False,
        shard_pool: Optional[ShardPool] = None,
        gpu_buffer_pool: Optional[GPUBufferPool] = None,
    ):
        self.device = device
        self.max_gpu_layers = max_gpu_layers
        self.pin_memory = pin_memory
        self.pin_storage = pin_storage
        self.shard_pool = shard_pool
        self.gpu_buffer_pool = gpu_buffer_pool

        self.layer_store: Dict[str, StoredLayer] = {}
        self.disk_store: Dict[str, DiskLayer] = {}
        self.gguf_store: Dict[str, GGUFLayer] = {}
        self.quantized_store: Dict[str, QuantizedLayer] = {}
        self.disk_quantized_store: Dict[str, DiskQuantizedLayer] = {}
        self._gguf_index: Optional[Dict[str, Any]] = None
        self._gguf_dtype: torch.dtype = torch.float16
        self.gpu_resident: OrderedDict[str, bool] = OrderedDict()
        self._modules: Dict[str, nn.Module] = {}

        self._prefetch_cache: Dict[str, Dict[str, torch.Tensor]] = {}
        self._prefetch_lock = threading.Lock()
        self._prefetch_stream: Optional[torch.cuda.Stream] = None
        if device.type == "cuda":
            self._prefetch_stream = torch.cuda.Stream(device=device)

        self._stats = {
            "swaps": 0,
            "evictions": 0,
            "cache_hits": 0,
            "ram_loads": 0,
            "disk_loads": 0,
            "total_swap_ms": 0.0,
        }

    def store_layer(self, layer_name: str, module: nn.Module) -> StoredLayer:
        """Store a layer's parameters in CPU RAM."""
        state = module.state_dict()
        slayer = _store_state_dict(state, pin_memory=self.pin_storage)
        self.layer_store[layer_name] = slayer
        self._modules[layer_name] = module
        return slayer

    def store_disk_layer(
        self, layer_name: str, module: nn.Module, dlayer: DiskLayer,
    ) -> None:
        """Register a layer for on-demand disk loading."""
        self.disk_store[layer_name] = dlayer
        self._modules[layer_name] = module

    def store_gguf_layer(
        self, layer_name: str, module: nn.Module, glayer: GGUFLayer,
    ) -> None:
        """Register a layer for on-demand GGUF loading + dequantization."""
        self.gguf_store[layer_name] = glayer
        self._modules[layer_name] = module

    def store_quantized_layer(
        self, layer_name: str, module: nn.Module, qlayer: QuantizedLayer,
    ) -> None:
        """Register a layer with compressed-tensors packed weights."""
        self.quantized_store[layer_name] = qlayer
        self._modules[layer_name] = module

    def store_disk_quantized_layer(
        self, layer_name: str, module: nn.Module, dqlayer: DiskQuantizedLayer,
    ) -> None:
        """Register a layer with disk-backed MXFP4/compressed-tensors weights."""
        self.disk_quantized_store[layer_name] = dqlayer
        self._modules[layer_name] = module

    def _apply_params(self, module: nn.Module, restored: Dict[str, torch.Tensor]) -> None:
        """Apply restored parameters to a module."""
        for name, param_data in restored.items():
            _assign_tensor_by_path(module, name, param_data)

    def restore_to_gpu(self, layer_name: str, module: nn.Module) -> None:
        """Restore a layer's parameters onto GPU from RAM, disk, or GGUF."""
        if layer_name in self.gpu_resident:
            self._stats["cache_hits"] += 1
            return

        if (layer_name not in self.layer_store
                and layer_name not in self.disk_store
                and layer_name not in self.gguf_store
                and layer_name not in self.quantized_store
                and layer_name not in self.disk_quantized_store):
            raise KeyError(f"Layer {layer_name} not in store")

        while len(self.gpu_resident) >= self.max_gpu_layers:
            oldest = next(iter(self.gpu_resident))
            self._evict(oldest)

        t0 = time.perf_counter()

        with self._prefetch_lock:
            prefetched = self._prefetch_cache.pop(layer_name, None)

        if prefetched is not None:
            if self._prefetch_stream is not None:
                torch.cuda.current_stream().wait_stream(self._prefetch_stream)
            self._apply_params(module, prefetched)
            if self.gpu_buffer_pool is not None:
                self.gpu_buffer_pool.swap()
        elif layer_name in self.layer_store:
            slayer = self.layer_store[layer_name]
            if self.gpu_buffer_pool is not None:
                restored = self.gpu_buffer_pool.load_from_stored(slayer)
                self.gpu_buffer_pool.swap()
            else:
                restored = _restore_to_device(slayer, self.device, self.pin_memory)
            self._apply_params(module, restored)
            self._stats["ram_loads"] += 1
        elif layer_name in self.disk_store:
            dlayer = self.disk_store[layer_name]
            if self.gpu_buffer_pool is not None:
                cpu_tensors = _load_disk_layer(dlayer, torch.device("cpu"), pool=self.shard_pool)
                restored = self.gpu_buffer_pool.load_from_dict(cpu_tensors)
                self.gpu_buffer_pool.swap()
                del cpu_tensors
            else:
                restored = _load_disk_layer(dlayer, self.device, pool=self.shard_pool)
            self._apply_params(module, restored)
            self._stats["disk_loads"] += 1
        elif layer_name in self.gguf_store:
            glayer = self.gguf_store[layer_name]
            if self.gpu_buffer_pool is not None:
                cpu_tensors = _load_gguf_layer(
                    glayer, self._gguf_index, torch.device("cpu"), self._gguf_dtype,
                )
                restored = self.gpu_buffer_pool.load_from_dict(cpu_tensors)
                self.gpu_buffer_pool.swap()
                del cpu_tensors
            else:
                restored = _load_gguf_layer(
                    glayer, self._gguf_index, self.device, self._gguf_dtype,
                )
            self._apply_params(module, restored)
            self._stats["disk_loads"] += 1
        elif layer_name in self.quantized_store:
            qlayer = self.quantized_store[layer_name]
            restored = _decompress_quantized_layer(qlayer, self.device)
            self._apply_params(module, restored)
            self._stats["ram_loads"] += 1
        else:
            dqlayer = self.disk_quantized_store[layer_name]
            has_experts = any("experts" in p for p in dqlayer.packed_map)
            if has_experts:
                _install_expert_offload_hooks(
                    module, dqlayer, self.device, pool=self.shard_pool,
                )
            else:
                restored = _load_disk_quantized_layer(
                    dqlayer, self.device, pool=self.shard_pool,
                )
                self._apply_params(module, restored)
            self._stats["disk_loads"] += 1

        swap_ms = (time.perf_counter() - t0) * 1000
        self.gpu_resident[layer_name] = True
        self._stats["swaps"] += 1
        self._stats["total_swap_ms"] += swap_ms

        with self._prefetch_lock:
            late = self._prefetch_cache.pop(layer_name, None)
        if late is not None:
            del late
            torch.cuda.empty_cache()

        logger.debug("Swapped in %s (%.1fms)", layer_name, swap_ms)

    def prefetch(self, layer_name: str) -> None:
        """Pre-restore a layer using buffer pool, CUDA streams, or threads."""
        if layer_name in self.gpu_resident:
            return
        if (layer_name not in self.layer_store
                and layer_name not in self.disk_store
                and layer_name not in self.gguf_store
                and layer_name not in self.quantized_store
                and layer_name not in self.disk_quantized_store):
            return

        # Buffer pool + CUDA stream path: async copy into pre-allocated buffer
        if (self.gpu_buffer_pool is not None
                and self._prefetch_stream is not None
                and layer_name in self.layer_store):
            slayer = self.layer_store[layer_name]
            try:
                restored = self.gpu_buffer_pool.load_from_stored(
                    slayer, stream=self._prefetch_stream,
                )
                with self._prefetch_lock:
                    if layer_name in self.gpu_resident:
                        del restored
                    else:
                        self._prefetch_cache[layer_name] = restored
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
            return

        # Pinned CUDA stream path (no buffer pool)
        if layer_name in self.layer_store:
            slayer = self.layer_store[layer_name]
            all_pinned = all(
                st.cpu_tensor is not None and st.cpu_tensor.is_pinned()
                for st in slayer.tensors.values()
            )

            if all_pinned and self._prefetch_stream is not None:
                try:
                    with torch.cuda.stream(self._prefetch_stream):
                        restored = _restore_to_device(
                            slayer, self.device, pin_memory=False, non_blocking=True,
                        )
                        with self._prefetch_lock:
                            if layer_name in self.gpu_resident:
                                del restored
                            else:
                                self._prefetch_cache[layer_name] = restored
                except torch.cuda.OutOfMemoryError:
                    torch.cuda.empty_cache()
                return

        # Thread-based fallback for disk/GGUF layers
        def _do_prefetch():
            try:
                if layer_name in self.layer_store:
                    restored = _restore_to_device(
                        self.layer_store[layer_name], self.device, self.pin_memory,
                    )
                elif layer_name in self.disk_store:
                    restored = _load_disk_layer(
                        self.disk_store[layer_name], self.device,
                        pool=self.shard_pool,
                    )
                elif layer_name in self.gguf_store:
                    restored = _load_gguf_layer(
                        self.gguf_store[layer_name], self._gguf_index,
                        self.device, self._gguf_dtype,
                    )
                else:
                    return
                with self._prefetch_lock:
                    if layer_name in self.gpu_resident:
                        del restored
                        torch.cuda.empty_cache()
                    else:
                        self._prefetch_cache[layer_name] = restored
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()

        thread = threading.Thread(target=_do_prefetch, daemon=True)
        thread.start()

    def _evict(self, layer_name: str) -> None:
        """Remove a layer from GPU and free its memory."""
        if layer_name in self.gpu_resident:
            del self.gpu_resident[layer_name]
            if self.gpu_buffer_pool is None:
                module = self._modules.get(layer_name)
                if module is not None:
                    for param in module.parameters():
                        param.data = torch.empty(0, dtype=param.dtype, device=self.device)
                torch.cuda.empty_cache()
            self._stats["evictions"] += 1
            logger.debug("Evicted %s from GPU", layer_name)

    @property
    def stats(self) -> dict:
        s = dict(self._stats)
        if s["swaps"] > 0:
            s["avg_swap_ms"] = s["total_swap_ms"] / s["swaps"]
        return s

    def summary(self) -> str:
        n_ram = len(self.layer_store)
        n_disk = len(self.disk_store)
        n_gguf = len(self.gguf_store)
        n_quant = len(self.quantized_store)
        n_dquant = len(self.disk_quantized_store)
        n_total = n_ram + n_disk + n_gguf + n_quant + n_dquant
        n_resident = len(self.gpu_resident)
        total_ram_bytes = sum(c.total_original for c in self.layer_store.values())
        total_disk_bytes = sum(d.total_bytes for d in self.disk_store.values())
        total_quant_bytes = sum(q.total_packed_bytes for q in self.quantized_store.values())
        total_dquant_bytes = sum(q.total_packed_bytes for q in self.disk_quantized_store.values())
        parts = [
            f"DeepSwap: {n_resident}/{n_total} on GPU",
            f"{n_ram} in RAM ({total_ram_bytes / 1e9:.1f}GB)",
        ]
        if n_disk > 0:
            parts.append(f"{n_disk} on disk ({total_disk_bytes / 1e9:.1f}GB)")
        if n_gguf > 0:
            parts.append(f"{n_gguf} GGUF layers")
        if n_quant > 0:
            parts.append(f"{n_quant} quantized ({total_quant_bytes / 1e9:.1f}GB packed)")
        if n_dquant > 0:
            parts.append(f"{n_dquant} disk-quantized ({total_dquant_bytes / 1e9:.1f}GB packed)")
        parts.append(f"{self._stats['swaps']} swaps")
        return ", ".join(parts)


# ---------------------------------------------------------------------------
#  Layer discovery
# ---------------------------------------------------------------------------

_LAYER_ATTRS = (
    "model.layers", "language_model.model.layers",
    "transformer.h", "gpt_neox.layers",
    "transformer.layers", "encoder.layer",
)


def _find_transformer_layers(
    model: nn.Module,
) -> Tuple[List[Tuple[str, nn.Module]], str]:
    """Find transformer layer modules and the attribute path used."""
    for attr in _LAYER_ATTRS:
        parts = attr.split(".")
        obj = model
        try:
            for p in parts:
                obj = getattr(obj, p)
        except AttributeError:
            continue

        result = []
        for i, layer in enumerate(obj):
            result.append((f"layer_{i}", layer))
        return result, attr

    raise ValueError(
        "Could not find transformer layers. "
        "Supported: " + ", ".join(_LAYER_ATTRS)
    )


# ---------------------------------------------------------------------------
#  Wrapper model
# ---------------------------------------------------------------------------

class DeepSwapModel(nn.Module):
    """Wrapper that intercepts forward passes to swap layers on/off GPU."""

    def __init__(
        self,
        model: nn.Module,
        manager: LayerSwapManager,
        layer_modules: List[Tuple[str, nn.Module]],
    ):
        super().__init__()
        self._model = model
        self._manager = manager
        self._layer_modules = layer_modules
        self._layer_index = {name: i for i, (name, _) in enumerate(layer_modules)}
        self._hooks = []
        self._install_hooks()

    def _install_hooks(self) -> None:
        for layer_name, module in self._layer_modules:
            def make_pre_hook(name, mod):
                def hook(m, args):
                    self._manager.restore_to_gpu(name, mod)
                    idx = self._layer_index[name]
                    if idx + 1 < len(self._layer_modules):
                        next_name = self._layer_modules[idx + 1][0]
                        self._manager.prefetch(next_name)
                return hook
            handle = module.register_forward_pre_hook(make_pre_hook(layer_name, module))
            self._hooks.append(handle)

    def forward(self, *args, **kwargs):
        return self._model(*args, **kwargs)

    def generate(self, *args, **kwargs):
        return self._model.generate(*args, **kwargs)

    @property
    def config(self):
        return self._model.config

    @property
    def device(self):
        return self._manager.device

    def summary(self) -> str:
        return self._manager.summary()

    def swap_stats(self) -> dict:
        return self._manager.stats


# ---------------------------------------------------------------------------
#  Public API
# ---------------------------------------------------------------------------

def deepswap(
    model: nn.Module,
    device: Optional[torch.device] = None,
    max_gpu_layers: Optional[int] = None,
    max_ram_gb: Optional[float] = None,
    reserve_gb: float = 2.0,
    pin_memory: bool = True,
    sage_attention: bool = False,
) -> DeepSwapModel:
    """Wrap a HuggingFace model for layer-level GPU offloading.

    Uses tiered storage (GPU VRAM / pinned RAM / NVMe disk) sized
    automatically to the available hardware. Pass pin_memory=True
    (default) for fastest transfers on systems with sufficient RAM.

    Parameters
    ----------
    model : nn.Module
        A HuggingFace transformers model (loaded to CPU).
    device : torch.device, optional
        Target GPU device. Defaults to cuda:0.
    max_gpu_layers : int, optional
        Max layers to keep on GPU. If None, auto-sizes to fill
        available VRAM minus reserve_gb.
    max_ram_gb : float, optional
        Max GB of layer data to keep in RAM. Remaining layers are
        loaded on-demand from disk. If None, auto-sizes to 60% of
        total RAM.
    reserve_gb : float
        VRAM to reserve for activations/KV cache (default 2GB).
    pin_memory : bool
        Use pinned CPU memory for faster DMA transfers (default True).
    sage_attention : bool
        Replace F.scaled_dot_product_attention with SageAttention
        (INT8 QK + FP16 PV). 2-5x attention speedup on Ampere+.

    Returns
    -------
    DeepSwapModel
        Wrapped model that transparently swaps layers.
    """
    if sage_attention:
        enable_sage_attention()
    if device is None:
        device = torch.device("cuda:0")

    layers, layer_attr = _find_transformer_layers(model)
    n_layers = len(layers)
    logger.info("Found %d transformer layers", n_layers)

    if max_gpu_layers is None:
        layer_ids = {id(m) for _, m in layers}
        non_layer_bytes = sum(
            p.numel() * p.element_size()
            for m in model.modules() if id(m) not in layer_ids
            for p in m.parameters(recurse=False)
        )
        max_gpu_layers = _auto_max_layers(layers, device, reserve_gb, non_layer_bytes)
    max_gpu_layers = min(max_gpu_layers, n_layers)

    n_offloaded = n_layers - max_gpu_layers
    logger.info(
        "GPU residency: %d/%d layers on GPU, %d offloaded",
        max_gpu_layers, n_layers, n_offloaded,
    )

    # --- Tiered storage budget ---
    total_model_bytes = sum(
        p.numel() * p.element_size() for p in model.parameters()
    )
    total_ram = _mem_total_bytes()
    per_layer_bytes = _estimate_layer_bytes(layers[0][1]) if layers else 0

    # Try to find safetensors on disk for the cold tier
    model_path = getattr(getattr(model, "config", None), "_name_or_path", None)
    safetensors_map = None
    if model_path and os.path.isdir(str(model_path)):
        safetensors_map = _build_safetensors_map(str(model_path))

    # Decide how many layers fit in RAM (warm tier)
    # Use 60% of total RAM for layer storage — leaves 40% for OS, page cache,
    # GPU transfers, and overhead. Prevents swap thrashing on tight systems.
    if max_ram_gb is not None and per_layer_bytes > 0:
        ram_budget = int(max_ram_gb * 1024**3)
        max_ram_layers = max(1, min(n_layers, ram_budget // per_layer_bytes))
        use_tiered = max_ram_layers < n_layers and safetensors_map is not None
    elif safetensors_map and total_ram > 0 and per_layer_bytes > 0:
        ram_budget = int(total_ram * 0.60)
        max_ram_layers = max(1, min(n_layers, ram_budget // per_layer_bytes))
        use_tiered = max_ram_layers < n_layers
    else:
        max_ram_layers = n_layers
        use_tiered = False

    if use_tiered and safetensors_map is None:
        logger.warning("Tiered storage requested but no safetensors found, using all-RAM")
        max_ram_layers = n_layers
        use_tiered = False

    # Pin storage only when model fits comfortably in RAM
    pin_at_store = pin_memory and (max_ram_layers == n_layers) and (
        total_ram == 0 or total_model_bytes < total_ram * 0.55
    )

    if use_tiered:
        logger.info(
            "Tiered storage: %d layers in RAM (%.1fGB), %d layers on disk (%.1fGB)",
            max_ram_layers,
            max_ram_layers * per_layer_bytes / 1e9,
            n_layers - max_ram_layers,
            (n_layers - max_ram_layers) * per_layer_bytes / 1e9,
        )

    shard_pool = None
    if use_tiered and safetensors_map is not None:
        shard_pool = ShardPool()
        disk_shards: Set[str] = set()
        for i in range(max_ram_layers, n_layers):
            prefix = f"{layer_attr}.{i}"
            for param_name in layers[i][1].state_dict():
                full_key = f"{prefix}.{param_name}"
                if full_key in safetensors_map:
                    disk_shards.add(safetensors_map[full_key])
        for shard_path in sorted(disk_shards):
            shard_pool.open_shard(shard_path)
        shard_pool.warm(disk_shards)
        logger.info("ShardPool: %d shards pre-opened, page cache warming started", len(disk_shards))

    # Pre-allocate GPU buffer pool for zero-alloc layer swapping
    gpu_pool = None
    if device.type == "cuda" and per_layer_bytes > 0:
        layer_dtype = next(layers[0][1].parameters()).dtype
        try:
            gpu_pool = GPUBufferPool(per_layer_bytes, layer_dtype, device)
            logger.info(
                "GPU buffer pool: 2x%.0fMB pre-allocated (%s)",
                per_layer_bytes / 1e6, layer_dtype,
            )
        except torch.cuda.OutOfMemoryError:
            logger.warning("Not enough VRAM for buffer pool, using per-swap allocation")
            gpu_pool = None

    manager = LayerSwapManager(
        device=device,
        max_gpu_layers=max_gpu_layers,
        pin_memory=pin_memory,
        pin_storage=pin_at_store,
        shard_pool=shard_pool,
        gpu_buffer_pool=gpu_pool,
    )

    # --- Store layers across tiers ---
    total_ram_stored = 0
    total_disk_stored = 0

    for i, (layer_name, module) in enumerate(layers):
        if i < max_ram_layers:
            # Warm tier: store in CPU RAM (optionally pinned)
            slayer = manager.store_layer(layer_name, module)
            total_ram_stored += slayer.total_original

            logger.debug(
                "RAM  %s: %.1fMB",
                layer_name, slayer.total_original / 1e6,
            )
        else:
            # Cold tier: reference safetensors files on disk
            prefix = f"{layer_attr}.{i}"
            dlayer = _build_disk_layer(module, prefix, safetensors_map)
            manager.store_disk_layer(layer_name, module, dlayer)
            total_disk_stored += dlayer.total_bytes

            logger.debug(
                "Disk %s: %.1fMB (%d tensors)",
                layer_name, dlayer.total_bytes / 1e6, len(dlayer.tensor_map),
            )

        # Free original parameters from CPU
        for param in module.parameters():
            param.data = torch.empty(0, dtype=param.dtype, device=device)

        # Periodic GC to release freed memory during setup
        if (i + 1) % 16 == 0:
            gc.collect()

    # Pinning stats
    pinned_bytes = sum(
        st.stored_bytes
        for slayer in manager.layer_store.values()
        for st in slayer.tensors.values()
        if st.cpu_tensor is not None and st.cpu_tensor.is_pinned()
    )

    logger.info(
        "Offload complete: %.1fGB in RAM + %.1fGB on disk",
        total_ram_stored / 1e9,
        total_disk_stored / 1e9,
    )
    if pinned_bytes > 0:
        logger.info(
            "Page-locked: %.1fGB / %.1fGB pinned (%.0f%%)",
            pinned_bytes / 1e9, total_ram_stored / 1e9,
            100 * pinned_bytes / max(total_ram_stored, 1),
        )

    # Move non-layer params (embeddings, lm_head, norms) to GPU
    non_layer_ids = {id(m) for _, m in layers}
    for name, module in model.named_modules():
        if id(module) not in non_layer_ids and module is not model:
            for param in module.parameters(recurse=False):
                if param.device.type == "cpu":
                    param.data = param.data.to(device)

    gc.collect()
    torch.cuda.empty_cache()

    return DeepSwapModel(model, manager, layers)


def _native_config_from_remote(config) -> Optional[Any]:
    """Rebuild a native transformers config from a remote-code config.

    Some checkpoints vendor a modeling_*.py written for an older transformers and
    crash on import (e.g. Kimi-K2's modeling_deepseek.py imports the removed
    is_torch_fx_available). When the checkpoint's architecture is natively supported
    by the installed transformers, reconstruct a native config from the same fields
    so the maintained implementation is used instead of the stale vendored one.
    Returns None if no native equivalent exists.
    """
    import transformers

    fields = config.to_dict()
    fields.pop("auto_map", None)
    fields.pop("model_type", None)
    for arch in fields.get("architectures") or []:
        model_cls = getattr(transformers, arch, None)
        cfg_cls = getattr(model_cls, "config_class", None) if model_cls else None
        if cfg_cls is None:
            continue
        try:
            return cfg_cls(**fields)
        except Exception:
            continue
    return None


def _make_empty_model(model_name_or_path: str, target_dtype: torch.dtype) -> nn.Module:
    """Create an empty HF model skeleton without loading weights."""
    from transformers import AutoConfig, AutoModelForCausalLM

    config = AutoConfig.from_pretrained(model_name_or_path, trust_remote_code=True)

    def _from_config(cfg, trust_remote_code):
        with torch.device("meta"):
            model = AutoModelForCausalLM.from_config(
                cfg, torch_dtype=target_dtype, trust_remote_code=trust_remote_code,
            )
        return model.to_empty(device="cpu")

    try:
        model = _from_config(config, True)
    except Exception:
        native = _native_config_from_remote(config)
        if native is not None:
            model = _from_config(native, False)
        else:
            model = AutoModelForCausalLM.from_pretrained(
                model_name_or_path, torch_dtype=target_dtype,
                device_map="cpu", low_cpu_mem_usage=True,
                trust_remote_code=True,
            )

    model.eval()
    return model


def deepswap_gguf(
    gguf_path: str,
    model: Any = None,
    device: Optional[torch.device] = None,
    max_gpu_layers: Optional[int] = None,
    reserve_gb: float = 2.0,
    target_dtype: torch.dtype = torch.float16,
    sage_attention: bool = False,
    model_name: Optional[str] = None,
) -> DeepSwapModel:
    """Wrap a HuggingFace model with GGUF quantized weights.

    Loads quantized weights from a GGUF file, dequantizing each layer
    on the fly when swapping to GPU. Enables running 4-bit/5-bit/8-bit
    quantized models with the same offloading engine.

    Parameters
    ----------
    gguf_path : str
        Path to the GGUF model file.
    model : nn.Module or str, optional
        A HuggingFace transformers model (loaded to CPU) or a model
        name/path string. If a string is provided (or model_name is
        set), an empty skeleton is created automatically, avoiding
        the RAM cost of loading full pretrained weights.
    device : torch.device, optional
        Target GPU device. Defaults to cuda:0.
    max_gpu_layers : int, optional
        Max layers to keep on GPU simultaneously.
    reserve_gb : float
        VRAM to reserve for activations/KV cache (default 2GB).
    target_dtype : torch.dtype
        Dtype to dequantize into (default float16).
    sage_attention : bool
        Enable SageAttention (INT8 QK + FP16 PV).
    model_name : str, optional
        HF model name/path. Creates an empty skeleton. Alternative
        to passing a pre-loaded model.

    Returns
    -------
    DeepSwapModel
        Wrapped model that loads GGUF weights on demand.
    """
    from gguf.gguf_reader import GGUFReader

    if sage_attention:
        enable_sage_attention()

    if device is None:
        device = torch.device("cuda:0")

    if isinstance(model, str):
        model = _make_empty_model(model, target_dtype)
    elif model is None and model_name is not None:
        model = _make_empty_model(model_name, target_dtype)
    elif model is None:
        raise ValueError("Provide a model (nn.Module or str) or model_name")

    reader = GGUFReader(gguf_path)
    tensor_index = {t.name: t for t in reader.tensors}
    tensor_names = set(tensor_index.keys())

    layers, layer_attr = _find_transformer_layers(model)
    n_layers = len(layers)
    logger.info("GGUF: %s — %d tensors, %d transformer layers",
                os.path.basename(gguf_path), len(tensor_index), n_layers)

    if max_gpu_layers is None:
        max_gpu_layers = _auto_max_layers(layers, device, reserve_gb, 0)
    max_gpu_layers = min(max_gpu_layers, n_layers)

    layer_bytes = _estimate_layer_bytes(layers[0][1]) if layers else 0
    elem_size = torch.finfo(target_dtype).bits // 8
    gpu_buffer_pool = None
    if layer_bytes > 0 and device.type == "cuda":
        try:
            gpu_buffer_pool = GPUBufferPool(layer_bytes, target_dtype, device)
            logger.info("GGUF: GPUBufferPool allocated (%.1f MB x2)",
                        layer_bytes / 1e6)
        except Exception as e:
            logger.warning("GGUF: GPUBufferPool failed: %s", e)

    manager = LayerSwapManager(
        device=device,
        max_gpu_layers=max_gpu_layers,
        pin_memory=True,
    )
    manager._gguf_index = tensor_index
    manager._gguf_dtype = target_dtype
    if gpu_buffer_pool is not None:
        manager.gpu_buffer_pool = gpu_buffer_pool

    from gguf.quants import dequantize as gguf_dequantize
    import numpy as np

    mapped_count = 0
    for i, (layer_name, module) in enumerate(layers):
        param_names = list(module.state_dict().keys())
        glayer = _build_gguf_layer(i, gguf_path, tensor_names, param_names)

        if not glayer.tensor_map:
            logger.warning("GGUF: no tensors mapped for layer %d", i)
            for param in module.parameters():
                param.data = torch.empty(0, dtype=param.dtype, device=device)
            continue

        mapped_count += len(glayer.tensor_map)

        cpu_tensors = {}
        for param_name, gguf_name in glayer.tensor_map.items():
            t = tensor_index[gguf_name]
            np_fp32 = gguf_dequantize(t.data, t.tensor_type)
            logical_shape = tuple(reversed(t.shape.tolist()))
            tensor = torch.from_numpy(
                np_fp32.reshape(logical_shape).astype(
                    np.float16 if target_dtype == torch.float16 else np.float32,
                    copy=False,
                )
            ).contiguous()
            cpu_tensors[param_name] = tensor

        slayer = _store_state_dict(cpu_tensors, pin_memory=False)
        manager.layer_store[layer_name] = slayer
        manager._modules[layer_name] = module

        for param in module.parameters():
            param.data = torch.empty(0, dtype=param.dtype, device=device)

        del cpu_tensors
        if (i + 1) % 8 == 0:
            logger.info("GGUF: pre-dequantized %d/%d layers (%.1f GB stored)",
                        i + 1, n_layers, sum(
                            s.total_stored for s in manager.layer_store.values()
                        ) / 1e9)

    logger.info("GGUF: mapped %d tensors across %d layers", mapped_count, n_layers)

    _GGUF_GLOBAL_MAP = {
        "model.embed_tokens.weight": "token_embd.weight",
        "model.norm.weight": "output_norm.weight",
        "lm_head.weight": "output.weight",
    }
    for hf_name, gguf_name in _GGUF_GLOBAL_MAP.items():
        if gguf_name in tensor_index:
            t = tensor_index[gguf_name]
            np_fp32 = gguf_dequantize(t.data, t.tensor_type)
            logical_shape = tuple(reversed(t.shape.tolist()))
            tensor = torch.from_numpy(np_fp32.reshape(logical_shape).copy())
            tensor = tensor.to(dtype=target_dtype, device=device)

            parts = hf_name.split(".")
            target_mod = model
            try:
                for p in parts[:-1]:
                    target_mod = getattr(target_mod, p)
                attr = getattr(target_mod, parts[-1])
                if isinstance(attr, nn.Parameter):
                    attr.data = tensor
                else:
                    setattr(target_mod, parts[-1], tensor)
                logger.debug("GGUF global: %s -> %s", gguf_name, hf_name)
            except AttributeError:
                logger.debug("GGUF global: skipped %s (not found in model)", hf_name)

    del reader, tensor_index
    gc.collect()
    torch.cuda.empty_cache()

    return DeepSwapModel(model, manager, layers)


def deepswap_quantized(
    model_path: str,
    device: Optional[torch.device] = None,
    max_gpu_layers: Optional[int] = None,
    reserve_gb: float = 2.0,
    target_dtype: torch.dtype = torch.float16,
    sage_attention: bool = False,
) -> DeepSwapModel:
    """Wrap a HuggingFace pre-quantized model for layer-level offloading.

    Supports models quantized with compressed-tensors (MXFP4, INT8, FP8).
    Packed weights stay on disk via mmap'd safetensors and are decompressed
    per layer swap — handles models of any size regardless of RAM.

    Parameters
    ----------
    model_path : str
        Path to a HuggingFace model with quantization_config in its config.
    device : torch.device, optional
        Target GPU device. Defaults to cuda:0.
    max_gpu_layers : int, optional
        Max layers to keep on GPU simultaneously.
    reserve_gb : float
        VRAM to reserve for activations/KV cache (default 2GB).
    target_dtype : torch.dtype
        Dtype for non-quantized parameters (default float16).
    sage_attention : bool
        Enable SageAttention (INT8 QK + FP16 PV).

    Returns
    -------
    DeepSwapModel
        Wrapped model with quantized layer swapping.
    """
    from transformers import AutoConfig
    from safetensors import safe_open

    if sage_attention:
        enable_sage_attention()

    if device is None:
        device = torch.device("cuda:0")

    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    quant_config = getattr(config, "quantization_config", None)
    if quant_config is None:
        quant_config = getattr(
            getattr(config, "text_config", None), "quantization_config", None,
        )
    if quant_config is None:
        raise ValueError(f"No quantization_config found in {model_path}")

    quant_format = quant_config.get("format", quant_config.get("quant_method", ""))
    fmt_str = getattr(quant_format, "value", str(quant_format))
    group_size = 32
    cg = quant_config.get("config_groups", {})
    for g in cg.values():
        w = g.get("weights", {})
        if "group_size" in w:
            group_size = w["group_size"]
            break

    logger.info("Quantized model: %s, format: %s, group_size: %d",
                model_path, fmt_str, group_size)

    model = _make_empty_model(model_path, target_dtype)

    layers, layer_attr = _find_transformer_layers(model)
    n_layers = len(layers)

    max_gpu_layers = min(max_gpu_layers or 1, n_layers)

    safetensors_map = _build_safetensors_map(model_path)

    shard_pool = ShardPool()
    all_shards = set(safetensors_map.values())
    for sp in all_shards:
        shard_pool.open_shard(sp)
    shard_pool.warm(all_shards)

    manager = LayerSwapManager(
        device=device,
        max_gpu_layers=max_gpu_layers,
        pin_memory=False,
        shard_pool=shard_pool,
    )

    layer_prefixes = []
    import re as _re
    layer_idx_pattern = _re.compile(r'\.layers\.(\d+)\.')
    idx_to_prefix: Dict[int, str] = {}
    idx_to_keys: Dict[int, List[str]] = {}
    for fk in safetensors_map:
        m = layer_idx_pattern.search(fk)
        if m:
            idx = int(m.group(1))
            if idx not in idx_to_prefix:
                idx_to_prefix[idx] = fk[:m.end() - 1]
            idx_to_keys.setdefault(idx, []).append(fk)
    for i in range(n_layers):
        layer_prefixes.append(idx_to_prefix.get(i))

    mapped_count = 0
    for i, (layer_name, module) in enumerate(layers):
        prefix = layer_prefixes[i]
        if not prefix:
            logger.warning("Quantized: no safetensors prefix for layer %d", i)
            continue

        all_full_keys = idx_to_keys.get(i, [])

        packed_prefixes = set()
        for fk in all_full_keys:
            if fk.endswith(".weight_packed"):
                rel = fk[len(prefix) + 1:]
                packed_prefixes.add(rel[:-len(".weight_packed")])

        packed_map: Dict[str, Dict[str, Tuple[str, str]]] = {}
        plain_map: Dict[str, Tuple[str, str]] = {}
        n_packed_tensors = 0

        for pp in packed_prefixes:
            refs: Dict[str, Tuple[str, str]] = {}
            for fk in all_full_keys:
                rel = fk[len(prefix) + 1:]
                if rel.startswith(pp + ".weight"):
                    shard_path = safetensors_map[fk]
                    suffix = rel[len(pp) + 1:]
                    refs[suffix] = (shard_path, fk)
                    n_packed_tensors += 1
            if refs:
                packed_map[pp] = refs

        for fk in all_full_keys:
            rel = fk[len(prefix) + 1:]
            is_packed = any(rel.startswith(pp + ".weight") for pp in packed_prefixes)
            if not is_packed:
                shard_path = safetensors_map[fk]
                plain_map[rel] = (shard_path, fk)

        if not packed_map and not plain_map:
            logger.warning("Quantized: no tensors for layer %d", i)
            continue

        dqlayer = DiskQuantizedLayer(
            packed_map=packed_map,
            plain_map=plain_map,
            quant_format=fmt_str,
            group_size=group_size,
            target_dtype=target_dtype,
            total_packed_bytes=n_packed_tensors,
        )
        manager.store_disk_quantized_layer(layer_name, module, dqlayer)
        mapped_count += len(packed_map) + len(plain_map)

        for param in module.parameters():
            param.data = torch.empty(0, dtype=param.dtype)

        if (i + 1) % 10 == 0 or i == n_layers - 1:
            logger.info("Quantized: mapped %d/%d layers (%d tensor groups)",
                        i + 1, n_layers, mapped_count)

    logger.info("Quantized: %d tensor groups across %d layers", mapped_count, n_layers)

    layer_prefix_set = {p for p in layer_prefixes if p}
    for fk in safetensors_map:
        if any(fk.startswith(p + ".") for p in layer_prefix_set):
            continue
        if fk.endswith("_packed") or fk.endswith("_scale"):
            continue
        try:
            shard_path = safetensors_map[fk]
            tensor = shard_pool.get_tensor(shard_path, fk)
            if tensor.is_floating_point() and tensor.element_size() > 1:
                tensor = tensor.to(dtype=target_dtype)
            tensor = tensor.to(device=device)

            parts = fk.split(".")
            target_mod = model
            for p in parts[:-1]:
                target_mod = getattr(target_mod, p)
            attr = getattr(target_mod, parts[-1])
            if isinstance(attr, nn.Parameter):
                attr.data = tensor
            else:
                setattr(target_mod, parts[-1], tensor)
        except (AttributeError, KeyError):
            pass

    gc.collect()
    torch.cuda.empty_cache()

    return DeepSwapModel(model, manager, layers)
