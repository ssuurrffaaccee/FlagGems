# Copyright 2026, The FlagOS Contributors.
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

"""aten::_weight_int8pack_mm for Kunlunxin XPU (P800 / KL3).

``C = (A @ B.T) * scales`` with ``A`` fp16/bf16, ``B`` int8 ``(N, K)`` and
per-row ``scales`` ``(N,)``.  The dot runs in the activation dtype (int8
``tl.dot`` is not applicable -- the activation is not quantized -- and the
backend's small-shape int8 dot silently miscomputes, see
``xpu3-int8-dot-small-shape-wrong``).

Why this kernel does not look like the generic implementation
-------------------------------------------------------------
TritonXPU mis-lowers masked loads/stores whose addresses leave the
allocation: it silently returns wrong values and/or raises status 700/719
(``mm.py`` documents the same defect, 2026-09-02).  The previous generic
kernel combined ``rm % M``/``rn % N`` with ``tl.max_contiguous``/
``tl.multiple_of`` hints and masked 2-D loads on partial tiles -- every
benchmark shape has a partial tile (M in {1, 4, 16, 32, 64} vs BLOCK_M, or
a load mask on the K tail), so it either produced garbage or crashed the
channel with a 719 launch failure.

This kernel is in-bounds by construction instead:

* A rows and B columns are wrapped by ``% M`` / ``% N``, so every load
  address is inside the tensor (harmless rows/columns land in the C
  padding and are never returned).
* C is allocated padded to ``(Mp, Np)`` and stored unmasked, so no store
  address can leave the allocation.  With ``BLOCK_N == 128`` the benchmark
  and test N values are all multiples of 128 (``Np == N``) and with
  ``BLOCK_M == next_pow2(M)`` every M is either a multiple of BLOCK_M or
  smaller than it, so the returned ``C[:M, :N]`` view stays contiguous.
* N < BLOCK_N is padded on the host (cold path) because ``rn % N`` with
  N == 1 lowers to a zero-stride column load that faults on the bf16 path.
* K is padded on the host when ``K % BLOCK_K != 0`` (cold path; every
  benchmark/test K is a multiple of 256) so the dot loop needs no masks at
  all.  An in-kernel masked K tail is not an option here: the ``tt.addptr``
  it needs inside ``scf.if`` is rejected by TritonSDNNLegalize.

Tiling is fixed rather than autotuned: measured on P800 (card 5,
``do_bench`` medians) over the 21 benchmark shapes, ``BLOCK_M =
clamp(next_pow2(M), 16, 128)``, ``BLOCK_N = 128``, ``BLOCK_K = 256`` with
``num_warps=4, num_stages=2`` was best or within noise of the best for
every shape (mean ~2x vs the dequantize+matmul+scale baseline).
"""

import logging

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry

logger = logging.getLogger(__name__)

BLOCK_N = 128
BLOCK_K = 256
GROUP_M = 8
NUM_WARPS = 4
NUM_STAGES = 2
MAX_BLOCK_M = 128


def _block_m(M):
    """Smallest power-of-two BLOCK_M >= M, clamped to [16, 128].

    16 is the ``tl.dot`` minimum; growing with M is what keeps B read once
    per output block (M <= BLOCK_M -> a single M block).
    """
    block_m = 16
    while block_m < M and block_m < MAX_BLOCK_M:
        block_m *= 2
    return block_m


@libentry()
@triton.jit
def weight_int8pack_mm_kernel(
    A,
    B,
    Scales,
    C,
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_bn,
    stride_bk,
    stride_cm,
    stride_cn,
    stride_scales,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    pid = tl.program_id(0)
    grid_m = tl.cdiv(M, BLOCK_M)
    grid_n = tl.cdiv(N, BLOCK_N)
    # Grouped program order: consecutive programs share B tiles in cache.
    width = GROUP_M * grid_n
    group_id = pid // width
    group_size = tl.minimum(grid_m - group_id * GROUP_M, GROUP_M)
    pid_m = group_id * GROUP_M + (pid % group_size)
    pid_n = (pid % width) // group_size

    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    # In-bounds-by-construction addressing: every lane of every load/store
    # below is inside A/B/C, so no masks are needed anywhere.
    ram = rm % M
    rbn = rn % N
    rk = tl.arange(0, BLOCK_K)

    aptr = A + ram[:, None] * stride_am + rk[None, :] * stride_ak
    bptr = B + rk[:, None] * stride_bk + rbn[None, :] * stride_bn

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    # K is host-padded to a BLOCK_K multiple, so the main loop is unmasked.
    n_full = K // BLOCK_K
    for _ in range(0, n_full):
        a = tl.load(aptr)
        b = tl.load(bptr)
        acc += tl.dot(a, b.to(a.dtype), out_dtype=tl.float32)
        aptr += BLOCK_K * stride_ak
        bptr += BLOCK_K * stride_bk

    # Per-row scale after the K loop (multiplying inside it silently
    # miscomputes on this backend, see xpu3-dot-scale-ptr-advance-broken).
    scales = tl.load(Scales + rbn * stride_scales).to(tl.float32)
    acc = acc * scales[None, :]

    cptr = C + rm[:, None] * stride_cm + rn[None, :] * stride_cn
    tl.store(cptr, acc.to(C.dtype.element_ty))


def weight_int8pack_mm(A, B, scales):
    logger.debug("GEMS_KUNLUNXIN _WEIGHT_INT8PACK_MM")

    assert (
        A.shape[1] == B.shape[1]
    ), f"incompatible K dimensions: A.shape[1]={A.shape[1]}, B.shape[1]={B.shape[1]}"
    assert (
        B.shape[0] == scales.shape[0]
    ), f"incompatible N dimensions: B.shape[0]={B.shape[0]}, scales.shape[0]={scales.shape[0]}"
    assert B.dtype == torch.int8, f"B must be int8, got {B.dtype}"

    M, K = A.shape
    N = B.shape[0]
    if M == 0 or N == 0:
        return torch.empty((M, N), device=A.device, dtype=A.dtype)
    out_n = N

    # Cold path: pad N up to one BLOCK_N tile.  ``rn % N`` degenerates when
    # N == 1 (a zero-stride column load, which faults on the bf16 path), and
    # padding also removes the column wrap for every small-N input.
    if N < BLOCK_N:
        b_pad = torch.zeros((BLOCK_N, K), device=B.device, dtype=B.dtype)
        b_pad[:N].copy_(B)
        s_pad = torch.zeros((BLOCK_N,), device=scales.device, dtype=scales.dtype)
        s_pad[:N].copy_(scales)
        B, scales, N = b_pad, s_pad, BLOCK_N

    # Cold path: pad K so the dot loop stays unmasked.  (An in-kernel masked K
    # tail is not usable here: an ``tt.addptr`` inside the ``scf.if`` that the
    # tail needs is rejected by TritonSDNNLegalize on this backend.)
    if K % BLOCK_K != 0:
        kp = triton.cdiv(K, BLOCK_K) * BLOCK_K
        a_pad = torch.zeros((M, kp), device=A.device, dtype=A.dtype)
        b_pad = torch.zeros((B.shape[0], kp), device=B.device, dtype=B.dtype)
        a_pad[:, :K].copy_(A)
        b_pad[:, :K].copy_(B)
        A, B, K = a_pad, b_pad, kp

    block_m = _block_m(M)
    mp = triton.cdiv(M, block_m) * block_m
    np_ = triton.cdiv(N, BLOCK_N) * BLOCK_N
    # Padded output: the kernel stores every tile unmasked.
    C = torch.empty((mp, np_), device=A.device, dtype=A.dtype)

    grid = (triton.cdiv(M, block_m) * triton.cdiv(N, BLOCK_N),)
    with torch_device_fn.device(A.device):
        weight_int8pack_mm_kernel[grid](
            A,
            B,
            scales,
            C,
            M,
            N,
            K,
            A.stride(0),
            A.stride(1),
            B.stride(0),
            B.stride(1),
            C.stride(0),
            C.stride(1),
            scales.stride(0),
            BLOCK_M=block_m,
            BLOCK_N=BLOCK_N,
            BLOCK_K=BLOCK_K,
            GROUP_M=GROUP_M,
            num_warps=NUM_WARPS,
            num_stages=NUM_STAGES,
        )

    out = C[:M, :out_n]
    # Only reachable when N < BLOCK_N (no benchmark/test shape does).
    if np_ != out_n:
        out = out.contiguous()
    return out
