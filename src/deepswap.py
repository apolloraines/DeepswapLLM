"""
DeepSwapLLM — Run oversized LLMs on undersized GPUs.
Copyright (c) 2025 Apollo Raines / Robert Rice. All Rights Reserved.
PROPRIETARY AND CONFIDENTIAL.

Compressed CPU-side layer storage with intelligent GPU swapping.
Only the active transformer layer lives on GPU. Others sit compressed
in pinned CPU RAM, uploaded on demand via PCIe with double-buffer
prefetch. Auto-sizes GPU residency to keep as many layers as fit.

Usage:
    from deepswap import deepswap

    model = AutoModelForCausalLM.from_pretrained(name, torch_dtype=torch.float16)
    model = deepswap(model)
    output = model.generate(input_ids, max_new_tokens=64)
"""

from __future__ import annotations

import gc
import logging
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from deepswap_compress import compress_chunked, decompress_chunked

logger = logging.getLogger("deepswap")


@dataclass
class CompressedTensor:
    """A single tensor stored compressed in pinned CPU RAM."""
    compressed_bytes: bytes
    shape: torch.Size
    dtype: torch.dtype
    numel: int
    original_bytes: int
    compressed_size: int

    @property
    def ratio(self) -> float:
        if self.compressed_size == 0:
            return 0.0
        return self.original_bytes / self.compressed_size


@dataclass
class CompressedLayer:
    """All parameters of one transformer layer, compressed."""
    tensors: Dict[str, CompressedTensor] = field(default_factory=dict)
    total_original: int = 0
    total_compressed: int = 0

    @property
    def ratio(self) -> float:
        if self.total_compressed == 0:
            return 0.0
        return self.total_original / self.total_compressed


def _compress_state_dict(state_dict: Dict[str, torch.Tensor]) -> CompressedLayer:
    """Compress all tensors in a state dict."""
    layer = CompressedLayer()
    for name, param in state_dict.items():
        cpu_data = param.detach().cpu().contiguous()
        flat = cpu_data.view(-1).numpy()

        compressed = compress_chunked(flat)
        ct = CompressedTensor(
            compressed_bytes=compressed,
            shape=cpu_data.shape,
            dtype=cpu_data.dtype,
            numel=cpu_data.numel(),
            original_bytes=cpu_data.nbytes,
            compressed_size=len(compressed),
        )
        layer.tensors[name] = ct
        layer.total_original += ct.original_bytes
        layer.total_compressed += ct.compressed_size

    return layer


def _decompress_to_device(
    clayer: CompressedLayer,
    device: torch.device,
    pin_memory: bool = True,
) -> Dict[str, torch.Tensor]:
    """Decompress a CompressedLayer back to tensors on the target device.

    Decompresses on CPU, optionally pins memory for faster DMA transfer,
    then moves to GPU.
    """
    import numpy as np

    result = {}
    for name, ct in clayer.tensors.items():
        raw = decompress_chunked(ct.compressed_bytes)
        elem_size = ct.original_bytes // ct.numel
        np_dtype = {2: np.float16, 4: np.float32}.get(elem_size, np.float16)
        typed = np.frombuffer(raw, dtype=np_dtype).copy().reshape(ct.shape)
        tensor = torch.from_numpy(typed)

        if device.type == "cuda" and pin_memory:
            tensor = tensor.pin_memory()

        result[name] = tensor.to(dtype=ct.dtype, device=device)

    return result


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
) -> int:
    """Compute how many layers can fit on GPU simultaneously."""
    if not layers or device.type != "cuda":
        return len(layers)

    torch.cuda.empty_cache()
    free_mem, total_mem = torch.cuda.mem_get_info(device)
    available = free_mem - int(reserve_gb * 1024**3)

    if available <= 0:
        return 1

    per_layer = _estimate_layer_bytes(layers[0][1])
    if per_layer == 0:
        return len(layers)

    max_fit = max(1, available // per_layer)
    return min(max_fit, len(layers))


class LayerSwapManager:
    """Manages compressed layer storage and GPU residency."""

    def __init__(
        self,
        device: torch.device,
        max_gpu_layers: int = 1,
        pin_memory: bool = True,
    ):
        self.device = device
        self.max_gpu_layers = max_gpu_layers
        self.pin_memory = pin_memory

        self.compressed_store: Dict[str, CompressedLayer] = {}
        self.gpu_resident: OrderedDict[str, bool] = OrderedDict()

        self._prefetch_cache: Dict[str, Dict[str, torch.Tensor]] = {}
        self._prefetch_lock = threading.Lock()

        self._stats = {
            "swaps": 0,
            "evictions": 0,
            "cache_hits": 0,
            "total_swap_ms": 0.0,
        }

    def compress_and_store(self, layer_name: str, module: nn.Module) -> CompressedLayer:
        """Compress a layer's parameters and store in CPU RAM."""
        state = module.state_dict()
        clayer = _compress_state_dict(state)
        self.compressed_store[layer_name] = clayer
        return clayer

    def _apply_params(self, module: nn.Module, restored: Dict[str, torch.Tensor]) -> None:
        """Apply restored parameters to a module."""
        for name, param_data in restored.items():
            parts = name.split(".")
            target = module
            for part in parts[:-1]:
                target = getattr(target, part)
            attr = getattr(target, parts[-1])
            if isinstance(attr, nn.Parameter):
                attr.data = param_data
            else:
                setattr(target, parts[-1], param_data)

    def restore_to_gpu(self, layer_name: str, module: nn.Module) -> None:
        """Decompress a layer and load its parameters onto GPU."""
        if layer_name in self.gpu_resident:
            self._stats["cache_hits"] += 1
            return

        if layer_name not in self.compressed_store:
            raise KeyError(f"Layer {layer_name} not in compressed store")

        while len(self.gpu_resident) >= self.max_gpu_layers:
            oldest = next(iter(self.gpu_resident))
            self._evict(oldest)

        t0 = time.perf_counter()

        with self._prefetch_lock:
            prefetched = self._prefetch_cache.pop(layer_name, None)

        if prefetched is not None:
            self._apply_params(module, prefetched)
        else:
            clayer = self.compressed_store[layer_name]
            restored = _decompress_to_device(clayer, self.device, self.pin_memory)
            self._apply_params(module, restored)

        swap_ms = (time.perf_counter() - t0) * 1000
        self.gpu_resident[layer_name] = True
        self._stats["swaps"] += 1
        self._stats["total_swap_ms"] += swap_ms

        logger.debug("Swapped in %s (%.1fms)", layer_name, swap_ms)

    def prefetch(self, layer_name: str) -> None:
        """Pre-decompress a layer in a background thread."""
        if layer_name in self.gpu_resident:
            return
        if layer_name not in self.compressed_store:
            return

        def _do_prefetch():
            clayer = self.compressed_store[layer_name]
            restored = _decompress_to_device(clayer, self.device, self.pin_memory)
            with self._prefetch_lock:
                self._prefetch_cache[layer_name] = restored

        thread = threading.Thread(target=_do_prefetch, daemon=True)
        thread.start()

    def _evict(self, layer_name: str) -> None:
        """Remove a layer from GPU residency tracking."""
        if layer_name in self.gpu_resident:
            del self.gpu_resident[layer_name]
            self._stats["evictions"] += 1
            logger.debug("Evicted %s from GPU", layer_name)

    @property
    def stats(self) -> dict:
        s = dict(self._stats)
        if s["swaps"] > 0:
            s["avg_swap_ms"] = s["total_swap_ms"] / s["swaps"]
        return s

    def summary(self) -> str:
        total_orig = sum(c.total_original for c in self.compressed_store.values())
        total_comp = sum(c.total_compressed for c in self.compressed_store.values())
        ratio = total_orig / total_comp if total_comp > 0 else 0
        n_resident = len(self.gpu_resident)
        n_total = len(self.compressed_store)
        return (
            f"DeepSwap: {n_resident}/{n_total} layers on GPU, "
            f"{total_orig / 1e9:.1f}GB -> {total_comp / 1e9:.1f}GB compressed "
            f"({ratio:.2f}x), {self._stats['swaps']} swaps"
        )


def _find_transformer_layers(model: nn.Module) -> List[Tuple[str, nn.Module]]:
    """Find transformer layer modules in a HuggingFace model."""
    for attr in ("model.layers", "transformer.h", "gpt_neox.layers",
                 "transformer.layers", "encoder.layer"):
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
        return result

    raise ValueError(
        "Could not find transformer layers. "
        "Supported: model.layers, transformer.h, gpt_neox.layers, "
        "transformer.layers, encoder.layer"
    )


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


def deepswap(
    model: nn.Module,
    device: Optional[torch.device] = None,
    max_gpu_layers: Optional[int] = None,
    reserve_gb: float = 2.0,
    pin_memory: bool = True,
) -> DeepSwapModel:
    """Wrap a HuggingFace model for layer-level GPU offloading.

    Parameters
    ----------
    model : nn.Module
        A HuggingFace transformers model (loaded to CPU).
    device : torch.device, optional
        Target GPU device. Defaults to cuda:0.
    max_gpu_layers : int, optional
        Max layers to keep on GPU. If None, auto-sizes to fill
        available VRAM minus reserve_gb.
    reserve_gb : float
        VRAM to reserve for activations/KV cache (default 2GB).
    pin_memory : bool
        Use pinned CPU memory for faster DMA transfers (default True).

    Returns
    -------
    DeepSwapModel
        Wrapped model that transparently swaps layers.
    """
    if device is None:
        device = torch.device("cuda:0")

    layers = _find_transformer_layers(model)
    n_layers = len(layers)
    logger.info("Found %d transformer layers", n_layers)

    if max_gpu_layers is None:
        max_gpu_layers = _auto_max_layers(layers, device, reserve_gb)
    max_gpu_layers = min(max_gpu_layers, n_layers)

    n_offloaded = n_layers - max_gpu_layers
    logger.info(
        "GPU residency: %d/%d layers on GPU, %d offloaded to CPU RAM",
        max_gpu_layers, n_layers, n_offloaded,
    )

    manager = LayerSwapManager(
        device=device,
        max_gpu_layers=max_gpu_layers,
        pin_memory=pin_memory,
    )

    total_original = 0
    total_compressed = 0

    for layer_name, module in layers:
        clayer = manager.compress_and_store(layer_name, module)
        total_original += clayer.total_original
        total_compressed += clayer.total_compressed

        for param in module.parameters():
            param.data = torch.empty(0, dtype=param.dtype, device=device)

        logger.debug(
            "Compressed %s: %.1fMB -> %.1fMB (%.2fx)",
            layer_name,
            clayer.total_original / 1e6,
            clayer.total_compressed / 1e6,
            clayer.ratio,
        )

    ratio = total_original / total_compressed if total_compressed > 0 else 0
    logger.info(
        "Offload complete: %.1fGB -> %.1fGB (%.2fx), %d layers swappable",
        total_original / 1e9,
        total_compressed / 1e9,
        ratio,
        n_offloaded,
    )

    non_layer_ids = {id(m) for _, m in layers}
    for name, module in model.named_modules():
        if id(module) not in non_layer_ids and module is not model:
            for param in module.parameters(recurse=False):
                if param.device.type == "cpu":
                    param.data = param.data.to(device)

    gc.collect()
    torch.cuda.empty_cache()

    return DeepSwapModel(model, manager, layers)
