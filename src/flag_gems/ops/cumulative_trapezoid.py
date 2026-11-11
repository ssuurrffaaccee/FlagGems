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

import logging
import math

import torch
import triton
import triton.language as tl

from flag_gems import runtime
from flag_gems.runtime import torch_device_fn

logger = logging.getLogger(__name__)

# The vectorized `tl.cumsum` scan is the default implementation and stays in
# charge on every backend that can compile it. Kunlunxin (TritonXPU) cannot:
# its scan input is an elementwise combination of two vectorized loads, which
# lowers to a `triton_xpu.unpack` that fails to legalize ("reached lowering
# without a bufPtr"). Only on that backend is the scan replaced by a
# sequential fp32 accumulator.
_SERIAL_SCAN = runtime.device.vendor_name == "kunlunxin"


_MAX_RANK = 8
_BLOCK_CAP = 4096  # widest block along the integration axis (8192 spills)
_BLOCK_CAP_FP32 = 1024  # fp32 keeps the shorter block for accuracy
_BLOCK_MIN = 16
_P_MIN = 8
_P_MAX = 1024  # cap on the persistent grid size


@triton.jit
def _row_offsets(
    r,
    S0,
    SI0,
    SO0,
    S1,
    SI1,
    SO1,
    S2,
    SI2,
    SO2,
    S3,
    SI3,
    SO3,
    S4,
    SI4,
    SO4,
    S5,
    SI5,
    SO5,
    S6,
    SI6,
    SO6,
    S7,
    SI7,
    SO7,
):
    """Flat row id -> (input base, output base) element offsets."""
    rem = r
    j = rem % S0
    rem = rem // S0
    b_in = j * SI0
    b_out = j * SO0
    j = rem % S1
    rem = rem // S1
    b_in += j * SI1
    b_out += j * SO1
    j = rem % S2
    rem = rem // S2
    b_in += j * SI2
    b_out += j * SO2
    j = rem % S3
    rem = rem // S3
    b_in += j * SI3
    b_out += j * SO3
    j = rem % S4
    rem = rem // S4
    b_in += j * SI4
    b_out += j * SO4
    j = rem % S5
    rem = rem // S5
    b_in += j * SI5
    b_out += j * SO5
    j = rem % S6
    rem = rem // S6
    b_in += j * SI6
    b_out += j * SO6
    j = rem % S7
    b_in += j * SI7
    b_out += j * SO7
    return b_in, b_out


@triton.jit
def _row_scan(
    y_ptr,
    out_ptr,
    NR,
    K,
    dx,
    P,
    S0,
    SI0,
    SO0,
    S1,
    SI1,
    SO1,
    S2,
    SI2,
    SO2,
    S3,
    SI3,
    SO3,
    S4,
    SI4,
    SO4,
    S5,
    SI5,
    SO5,
    S6,
    SI6,
    SO6,
    S7,
    SI7,
    SO7,
    STEP_IN: tl.constexpr,
    STEP_OUT: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pid = tl.program_id(0)
    cols = tl.arange(0, BLOCK_D)
    n_full = K // BLOCK_D
    # Persistent row loop: one program owns rows pid, pid+P, pid+2P, ...
    for r in range(pid, NR, P):
        b_in, b_out = _row_offsets(
            r,
            S0,
            SI0,
            SO0,
            S1,
            SI1,
            SO1,
            S2,
            SI2,
            SO2,
            S3,
            SI3,
            SO3,
            S4,
            SI4,
            SO4,
            S5,
            SI5,
            SO5,
            S6,
            SI6,
            SO6,
            S7,
            SI7,
            SO7,
        )
        y_base = y_ptr + b_in
        o_base = out_ptr + b_out
        carry = 0.0
        for i in range(0, n_full):
            c = i * BLOCK_D + cols
            a = tl.load(y_base + c * STEP_IN).to(tl.float32)
            b = tl.load(y_base + (c + 1) * STEP_IN).to(tl.float32)
            t = a + b
            v = tl.cumsum(t, axis=0) + carry
            tl.store(o_base + c * STEP_OUT, v * dx)
            carry = carry + tl.sum(t, axis=0)
        # Tail block; fully masked when K is a multiple of BLOCK_D.
        c = n_full * BLOCK_D + cols
        m = c < K
        a = tl.load(y_base + c * STEP_IN, mask=m, other=0.0).to(tl.float32)
        b = tl.load(y_base + (c + 1) * STEP_IN, mask=m, other=0.0).to(tl.float32)
        t = a + b
        v = tl.cumsum(t, axis=0) + carry
        tl.store(o_base + c * STEP_OUT, v * dx, mask=m)


_CT_PAYLOAD_GRID = 12  # physical clusters
_CT_PAYLOAD = {"init": False, "ok": False, "kern": {}}
_CT_META_CACHE = {}


def _ct_payload_init():
    if _CT_PAYLOAD["init"]:
        return _CT_PAYLOAD["ok"]
    _CT_PAYLOAD["init"] = True
    try:
        import os
        import triton as _triton
        import triton.experimental.tle as _tle

        obj = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "runtime",
            "backend",
            "_kunlunxin",
            "payload",
            "obj",
            "cumtrapz.o",
        )
        for sym in ("cumtrapz_16", "cumtrapz_bf16", "cumtrapz_32"):
            ns = {"_tle": _tle, "_OBJ": obj}
            exec(
                "@_tle.raw.dialect('xpu3', object=_OBJ, arch=3)\n"
                f"def {sym}(meta, y, out):\n"
                "    ...\n",
                ns,
            )
            callee = ns[sym]

            def _mk(callee=callee):
                @_triton.jit
                def _k(M_, Y, O):
                    _tle.raw.call(callee, (M_, Y, O))

                return _k

            _CT_PAYLOAD["kern"][sym] = _mk()
        _CT_PAYLOAD["ok"] = True
    except Exception as _e:  # pragma: no cover - keep backend importable
        logger.debug("GEMS CUMPTRAPZ payload init failed: %r", _e)
        _CT_PAYLOAD["ok"] = False
    return _CT_PAYLOAD["ok"]


def _ct_payload_launch(y, out, rows, N, dx):
    """True when the payload handled the op."""
    if N < 2 or rows <= 0:
        return False
    if not _ct_payload_init():
        return False
    import struct

    key = (rows, N, float(dx), y.device.index)
    meta = _CT_META_CACHE.get(key)
    if meta is None:
        meta = torch.tensor(
            [rows, N, struct.unpack("<i", struct.pack("<f", float(dx)))[0]],
            dtype=torch.int32,
            device=y.device,
        )
        if len(_CT_META_CACHE) >= 64:
            _CT_META_CACHE.clear()
        _CT_META_CACHE[key] = meta
    if y.dtype == torch.float32:
        sym, yi, oi = "cumtrapz_32", y, out
    elif y.dtype == torch.float16:
        sym, yi, oi = "cumtrapz_16", y.view(torch.int16), out.view(torch.int16)
    elif y.dtype == torch.bfloat16:
        sym, yi, oi = "cumtrapz_bf16", y.view(torch.int16), out.view(torch.int16)
    else:
        return False
    kern = _CT_PAYLOAD["kern"][sym]
    with torch_device_fn.device(y.device):
        kern[(_CT_PAYLOAD_GRID,)](meta, yi.view(-1), oi.view(-1))
    return True


@triton.jit
def _cumtrapz_serial_kernel(
    y_ptr,
    out_ptr,
    N,
    inner,
    stride_row_in,
    stride_row_out,
    dx,
):
    """One program per (outer, inner) pair, walking the row one element at a time.

    Used on XPU only. The reference there accumulates the scan serially in fp32,
    so this reproduces it without a `tl.cumsum` scan.
    """
    pid = tl.program_id(0)
    o = pid // inner
    i = pid % inner
    in_base = o * stride_row_in + i
    out_base = o * stride_row_out + i

    acc = tl.zeros((), dtype=tl.float32)
    prev = tl.load(y_ptr + in_base).to(tl.float32)
    for j in range(0, N - 1):
        nxt = tl.load(y_ptr + in_base + (j + 1) * inner).to(tl.float32)
        acc = acc + (prev + nxt) * dx
        prev = nxt
        tl.store(out_ptr + out_base + j * inner, acc)


def cumulative_trapezoid(y, *, dx=1, dim=-1):
    logger.debug("GEMS CUMULATIVE_TRAPEZOID")

    ndim = y.dim()
    if ndim == 0:
        raise IndexError("cumulative_trapezoid: dim out of range for a 0-dim tensor")
    d = dim + ndim if dim < 0 else dim
    if d < 0 or d >= ndim:
        raise IndexError(
            "Dimension out of range (expected to be in range of "
            "[{}, {}], but got {})".format(-ndim, ndim - 1, dim)
        )

    shape = list(y.shape)
    D = shape[d]
    K = D - 1 if D > 1 else 0
    nrows = 1
    for k, s in enumerate(shape):
        if k != d:
            nrows *= s

    out_shape = list(shape)
    out_shape[d] = K
    out = torch.empty(out_shape, dtype=y.dtype, device=y.device)
    if K == 0 or nrows == 0:
        return out

    if _SERIAL_SCAN:
        # XPU path: one program per (outer, inner) pair over a contiguous copy.
        N = D
        outer = 1
        for s in shape[:d]:
            outer *= s
        inner = 1
        for s in shape[d + 1 :]:
            inner *= s
        # tle.raw payload: one core per row, LM-resident sequential fp32 scan.
        # It reproduces the reference's summation order, which the Triton tree
        # scan cannot (2.6e-6 vs the 1.3e-6 tolerance on the largest shape).
        if inner == 1 and y.is_contiguous() and D > 1:
            if _ct_payload_launch(y, out, outer, D, float(dx)):
                return out

        _cumtrapz_serial_kernel[(outer * inner,)](
            y.contiguous(),
            out,
            N,
            inner,
            N * inner,
            (N - 1) * inner,
            0.5 * float(dx),
        )
        return out

    # Non-integration axes: (size, input stride, output stride).
    meta = []
    for k in range(ndim):
        if k != d:
            meta.append((shape[k], y.stride(k), out.stride(k)))
    while len(meta) < _MAX_RANK:
        meta.append((1, 0, 0))
    flat = []
    for s, si, so in meta:
        flat.extend((s, si, so))

    # Persistent grid: one program per ~4*sqrt(elements) elements.  Wide grids
    # only pay off once the scan work dominates the 0.46us-per-program dispatch.
    p = int(0.25 * math.sqrt(nrows * K)) + 1
    p = max(_P_MIN, min(p, min(nrows, _P_MAX)))

    # The 0.5 of the trapezoid rule is folded into the spacing constant so the
    # inner loop does one multiply per element instead of two.
    # BLOCK_D is capped per dtype: 4096 is 6-8% faster than 1024 on the largest
    # cases for the 2-byte dtypes, but a 4096-long fp32 block scan drifts past
    # the fp32 tolerance (max err 6.4e-4 vs 2.1e-4 for 1024 against float64),
    # so fp32 keeps 1024.
    block_cap = _BLOCK_CAP if y.element_size() <= 2 else _BLOCK_CAP_FP32
    _row_scan[(p,)](
        y,
        out,
        nrows,
        K,
        0.5 * float(dx),
        p,
        *flat,
        STEP_IN=y.stride(d),
        STEP_OUT=out.stride(d),
        BLOCK_D=max(_BLOCK_MIN, min(block_cap, triton.next_power_of_2(K))),
        num_warps=1,
    )
    return out
