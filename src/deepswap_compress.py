"""
DeepSwapLLM Compression Engine
Copyright (c) 2025 Apollo Raines / Robert Rice. All Rights Reserved.
PROPRIETARY AND CONFIDENTIAL.

CPU-side sparse block compression and decompression.
Format is self-describing and portable.
"""

from __future__ import annotations

import struct

import numpy as np

CHUNK_MAGIC = 0x4C544348
CHUNK_SIZE_ELEMENTS = 16384
BLOCK_SIZE_ELEMENTS = 64

BLOCK_ZERO = 0
BLOCK_SPARSE = 1
BLOCK_RAW = 2


def compress_chunked(data: np.ndarray) -> bytes:
    """Compress a numpy array using sparse block encoding.

    Format: ChunkHeader + per-chunk offsets + block data.
    Each 64-element block is classified as ZERO, SPARSE (bitmap + values),
    or RAW (uncompressed).
    """
    if data.ndim != 1:
        data = data.ravel()

    elem_size = data.dtype.itemsize
    raw_bytes = data.tobytes()
    num_elements = len(data)
    num_chunks = (num_elements + CHUNK_SIZE_ELEMENTS - 1) // CHUNK_SIZE_ELEMENTS

    data_blob = bytearray()
    chunk_offsets = []

    for chunk_i in range(num_chunks):
        chunk_offsets.append(len(data_blob))
        start = chunk_i * CHUNK_SIZE_ELEMENTS
        end = min(start + CHUNK_SIZE_ELEMENTS, num_elements)
        chunk_len = end - start

        elements_done = 0
        while elements_done < chunk_len:
            remaining = chunk_len - elements_done
            block_len = min(remaining, BLOCK_SIZE_ELEMENTS)
            block_start = (start + elements_done) * elem_size
            block_bytes_raw = raw_bytes[block_start:block_start + block_len * elem_size]

            if all(b == 0 for b in block_bytes_raw):
                data_blob.append(BLOCK_ZERO)
                elements_done += block_len
                continue

            bitmap = 0
            nnz = 0
            nonzero_vals = bytearray()
            for k in range(block_len):
                elem = block_bytes_raw[k * elem_size:(k + 1) * elem_size]
                if any(b != 0 for b in elem):
                    bitmap |= (1 << k)
                    nnz += 1
                    nonzero_vals.extend(elem)

            sparse_size = 1 + 8 + nnz * elem_size
            raw_size = 1 + block_len * elem_size

            if sparse_size < raw_size:
                data_blob.append(BLOCK_SPARSE)
                data_blob.extend(struct.pack("<Q", bitmap))
                data_blob.extend(nonzero_vals)
            else:
                data_blob.append(BLOCK_RAW)
                data_blob.extend(block_bytes_raw)

            elements_done += block_len

    header = struct.pack(
        "<IIQI I",
        CHUNK_MAGIC,
        num_chunks,
        num_elements,
        elem_size,
        0,
    )
    offsets_bytes = struct.pack(f"<{num_chunks}I", *chunk_offsets)

    return bytes(header) + offsets_bytes + bytes(data_blob)


def decompress_chunked(blob: bytes) -> bytearray:
    """Decompress a sparse block encoded blob back to raw bytes."""
    header_size = 24
    magic, num_chunks, total_elements, elem_size, _ = struct.unpack_from(
        "<IIQI I", blob, 0
    )
    assert magic == CHUNK_MAGIC, f"Bad magic: {hex(magic)}"

    offsets_start = header_size
    offsets = struct.unpack_from(f"<{num_chunks}I", blob, offsets_start)
    data_start = offsets_start + num_chunks * 4

    output = bytearray(total_elements * elem_size)

    for chunk_i in range(num_chunks):
        pos = data_start + offsets[chunk_i]
        chunk_start = chunk_i * CHUNK_SIZE_ELEMENTS
        chunk_end = min(chunk_start + CHUNK_SIZE_ELEMENTS, total_elements)
        chunk_len = chunk_end - chunk_start

        elements_done = 0
        while elements_done < chunk_len:
            remaining = chunk_len - elements_done
            block_len = min(remaining, BLOCK_SIZE_ELEMENTS)
            block_type = blob[pos]
            pos += 1

            out_off = (chunk_start + elements_done) * elem_size

            if block_type == BLOCK_ZERO:
                pass
            elif block_type == BLOCK_SPARSE:
                bitmap = struct.unpack_from("<Q", blob, pos)[0]
                pos += 8
                for k in range(block_len):
                    if (bitmap >> k) & 1:
                        dst = out_off + k * elem_size
                        output[dst:dst + elem_size] = blob[pos:pos + elem_size]
                        pos += elem_size
            elif block_type == BLOCK_RAW:
                n = block_len * elem_size
                output[out_off:out_off + n] = blob[pos:pos + n]
                pos += n

            elements_done += block_len

    return output
