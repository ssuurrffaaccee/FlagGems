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

import logging
import os

import torch
import triton
import triton.language as tl
import triton.experimental.tle as tle

from flag_gems.runtime import torch_device_fn

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# tle.raw payload for the sparse backward fast case (mode=0 / sum, no
# per-sample weights, padding_idx < 0).  Sparse backward does not deduplicate:
#   values[s] = grad[offset2bag[s]],  out_indices = indices
# so the whole op is one row copy per sample.  The 16-bit entry is a raw copy,
# so fp16 and bf16 share it; fp32 uses the 32-bit entry.
# ---------------------------------------------------------------------------
_EB_OBJ = os.path.join(
    os.path.dirname(__file__), "..", "payload", "obj", "embedding_bag.o"
)
_EB_GRID = 12
_EB_DMAX = 256
_EB_META = {}
_HAS_EB_RAW = False
try:

    @tle.raw.dialect("xpu3", object=_EB_OBJ, arch=3)
    def ebs_copy16(meta, grad, o2b, out): ...

    @tle.raw.dialect("xpu3", object=_EB_OBJ, arch=3)
    def ebs_copy32(meta, grad, o2b, out): ...

    @tle.raw.dialect("xpu3", object=_EB_OBJ, arch=3)
    def ebs_copy16l(meta, grad, o2b, out): ...

    @tle.raw.dialect("xpu3", object=_EB_OBJ, arch=3)
    def ebs_copy32l(meta, grad, o2b, out): ...

    @triton.jit
    def _ebs16_kernel(M, G, O2, O):
        tle.raw.call(ebs_copy16, (M, G, O2, O))

    @triton.jit
    def _ebs32_kernel(M, G, O2, O):
        tle.raw.call(ebs_copy32, (M, G, O2, O))

    @triton.jit
    def _ebs16l_kernel(M, G, O2, O):
        tle.raw.call(ebs_copy16l, (M, G, O2, O))

    @triton.jit
    def _ebs32l_kernel(M, G, O2, O):
        tle.raw.call(ebs_copy32l, (M, G, O2, O))

    _HAS_EB_RAW = True
except Exception:  # pragma: no cover - tle.raw object= unavailable
    _HAS_EB_RAW = False


def _eb_meta(n, d, device):
    key = (int(n), int(d), str(device))
    meta = _EB_META.get(key)
    if meta is None:
        if len(_EB_META) >= 64:
            _EB_META.clear()
        meta = torch.tensor([n, d], dtype=torch.int32, device=device)
        _EB_META[key] = meta
    return meta


def _generic(*args):
    from flag_gems.ops._embedding_bag_sparse_backward import (
        _embedding_bag_sparse_backward as _g,
    )

    return _g(*args)


def _embedding_bag_sparse_backward(
    grad: torch.Tensor,
    indices: torch.Tensor,
    offsets: torch.Tensor,
    offset2bag: torch.Tensor,
    bag_size: torch.Tensor,
    num_weights: int,
    scale_grad_by_freq: bool,
    mode: int,
    per_sample_weights: torch.Tensor = None,
    padding_idx: int = -1,
) -> torch.Tensor:
    logger.debug("GEMS_KUNLUNXIN _EMBEDDING_BAG_SPARSE_BACKWARD")

    if (
        _HAS_EB_RAW
        and int(mode) == 0
        and per_sample_weights is None
        and (padding_idx is None or padding_idx < 0)
        and grad.dim() == 2
        and grad.dtype in (torch.float16, torch.bfloat16, torch.float32)
        and indices.dtype in (torch.int32, torch.int64)
        and int(num_weights) >= 1
        and indices.device == grad.device
        and offset2bag.device == grad.device
    ):
        d = grad.shape[1]
        n = indices.numel()
        if 0 < d <= _EB_DMAX and n > 0 and offset2bag.numel() == n:
            try:
                o2b = offset2bag
                if o2b.dtype not in (torch.int32, torch.int64):
                    raise TypeError
                if not o2b.is_contiguous():
                    o2b = o2b.contiguous()
                i64 = o2b.dtype == torch.int64
                meta = _eb_meta(n, d, grad.device)
                values = torch.empty(
                    (n, d), dtype=grad.dtype, device=grad.device
                )
                with torch_device_fn.device(grad.device):
                    if grad.dtype == torch.float32:
                        k = _ebs32l_kernel if i64 else _ebs32_kernel
                        k[(_EB_GRID,)](meta, grad, o2b, values)
                    else:
                        k = _ebs16l_kernel if i64 else _ebs16_kernel
                        k[(_EB_GRID,)](
                            meta,
                            grad.view(torch.int16),
                            o2b,
                            values.view(torch.int16),
                        )
                vi = indices if indices.dtype == torch.int64 else indices.to(torch.int64)
                return torch.sparse_coo_tensor(
                    vi.unsqueeze(0),
                    values,
                    (int(num_weights), d),
                    dtype=grad.dtype,
                    device=grad.device,
                    is_coalesced=False,
                    check_invariants=False,
                )
            except Exception:
                pass

    return _generic(
        grad,
        indices,
        offsets,
        offset2bag,
        bag_size,
        num_weights,
        scale_grad_by_freq,
        mode,
        per_sample_weights,
        padding_idx,
    )
