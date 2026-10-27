# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Kunlunxin (XPU) override of `resolve_neg`.
#
# `resolve_neg(A)` materializes a neg-bit tensor: out = -raw (matching CPU/CUDA
# PyTorch semantics; the generic path uses `neg_func(A)`). The benchmark input
# is `x.conj().imag` -> a stride-2 float32 view (into contiguous complex storage)
# with the neg bit set. The generic `neg_func` runs the elementwise neg on that
# strided view through the un-tuned generic kernel -> discrete (offsetState=-1)
# access -> IR baseline `harness/perf_ir_3/ir-resolve_neg-dev3.log` gems speedup
# 0.003-0.30 ([4096,4096] 48.9ms, [10000,65536] 1860ms).
#
# Fix: when A is the imaginary view of a contiguous complex tensor, its logical
# element j reads raw float32 storage index (storage_offset + 2*j). A dedicated
# kernel with a large BLOCK reads that (stride-2) raw and writes a dense stride-1
# output, negating -> matches neg_func exactly but ~140x faster than the generic
# discrete path ([4096,4096] 48.9ms -> 0.34ms). Anything else falls back to
# neg_func (correctness preserved).
import logging
import os

import torch
import triton
import triton.language as tl

from flag_gems.ops.neg import neg_func
from flag_gems.runtime import torch_device_fn

logger = logging.getLogger(__name__)

# ---- tle.raw payload: bulk 2N read + LM odd-lane compaction -----------------
# The stride-2 Triton path fetches full cache lines anyway (2x traffic) but
# issues a strided load per lane. The payload bulk-DMAs the contiguous complex
# storage, compacts the odd lanes in LM while negating, and bulk-DMAs N floats.
_RN_PAYLOAD_MIN_ELEMS = 500_000
_RN_PAYLOAD_GRID = 12
_RN_PAYLOAD = {"init": False, "ok": False, "kern": None}
_RN_META_CACHE = {}


def _rn_payload_init():
    if _RN_PAYLOAD["init"]:
        return _RN_PAYLOAD["ok"]
    _RN_PAYLOAD["init"] = True
    try:
        import triton as _triton
        import triton.experimental.tle as _tle

        obj = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "payload",
            "obj",
            "resolve_neg.o",
        )
        ns = {"_tle": _tle, "_OBJ": obj}
        exec(
            "@_tle.raw.dialect('xpu3', object=_OBJ, arch=3)\n"
            "def resolve_neg_lm(meta, fin, fout):\n"
            "    ...\n",
            ns,
        )
        callee = ns["resolve_neg_lm"]

        @_triton.jit
        def _k(M, I, O):
            _tle.raw.call(callee, (M, I, O))

        _RN_PAYLOAD["kern"] = _k
        _RN_PAYLOAD["ok"] = True
    except Exception as _e:  # pragma: no cover - keep backend importable
        logger.debug("GEMS_KUNLUNXIN RESOLVE_NEG payload init failed: %r", _e)
        _RN_PAYLOAD["ok"] = False
    return _RN_PAYLOAD["ok"]


def _rn_payload_launch(fin, fout, n, so):
    if not _rn_payload_init():
        return False
    key = (n, so, fin.device.index)
    meta = _RN_META_CACHE.get(key)
    if meta is None:
        meta = torch.tensor([n, so], dtype=torch.int32, device=fin.device)
        if len(_RN_META_CACHE) >= 64:
            _RN_META_CACHE.clear()
        _RN_META_CACHE[key] = meta
    with torch_device_fn.device(fin.device):
        _RN_PAYLOAD["kern"][(_RN_PAYLOAD_GRID,)](meta, fin, fout)
    return True


@triton.jit
def _neg_imag_kernel(fin, fout, so, N, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    j = pid * BLOCK + tl.arange(0, BLOCK)
    m = j < N
    x = tl.load(fin + so + 2 * j, mask=m)
    tl.store(fout + j, -x, mask=m)


def _contig_strides(shape):
    s = []
    acc = 1
    for d in reversed(shape):
        s.append(acc)
        acc *= d
    return tuple(reversed(s))


def resolve_neg(A: torch.Tensor):
    logger.debug("GEMS_KUNLUNXIN RESOLVE_NEG")
    if not A.is_neg():
        return A
    # Fast path: A is the imaginary view of a contiguous complex64 tensor, i.e.
    # storage_offset 1 and strides = 2 * (contiguous strides of A.shape).
    fast = (
        A.dtype == torch.float32
        and A.storage_offset() == 1
        and A.stride() == tuple(2 * s for s in _contig_strides(A.shape))
    )
    if not fast:
        return neg_func(A)
    N = A.numel()
    total = A.untyped_storage().nbytes() // 4
    fin = torch.empty(0, dtype=torch.float32, device=A.device)
    fin.set_(A.untyped_storage(), 0, (total,), (1,))
    out = torch.empty(A.shape, dtype=torch.float32, device=A.device)
    fout = out.reshape(-1)
    if N >= _RN_PAYLOAD_MIN_ELEMS:
        try:
            if _rn_payload_launch(fin, fout, N, A.storage_offset()):
                return out
        except Exception as exc:  # pragma: no cover - fall back on failure
            logger.debug("resolve_neg payload failed: %s", exc)
    BLOCK = 4096
    grid = (triton.cdiv(N, BLOCK),)
    with torch_device_fn.device(A.device):
        _neg_imag_kernel[grid](
            fin, fout, A.storage_offset(), N, BLOCK=BLOCK, num_warps=8
        )
    return out
