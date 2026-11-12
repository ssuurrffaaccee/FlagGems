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

"""Kunlunxin tle.raw fast path for fake_quantize_per_tensor_affine_cachemask.

The payload (payload/obj/fake_quantize.o, entries fqv3_{16,32}) quantizes in
the integer domain:
    r  = rne(x * inv)                       # vfloat2fix.rn
    rc = clamp(r, qmin-zp, qmax-zp)
    mask = (r == rc) && finite(x*inv)
    out  = rc * scale                       # (0-zp)*scale for NaN/inf
with a 2-deep ping-pong (load next / store previous overlapped with compute).
"""

import logging
import os
import struct

import torch
import triton
import triton.language as tl
import triton.experimental.tle as tle

from flag_gems.runtime import torch_device_fn

logger = logging.getLogger(__name__)

_OBJ_DIR = os.path.join(os.path.dirname(__file__), "..", "payload", "obj")
_OBJ16 = os.path.join(_OBJ_DIR, "fake_quantize_16.o")
_OBJBF16 = os.path.join(_OBJ_DIR, "fake_quantize_bf16.o")
_OBJ32 = os.path.join(_OBJ_DIR, "fake_quantize_32.o")
_CHUNK = 512
_META = {}
from flag_gems.ops.fake_quantize_per_tensor_affine_cachemask import (  # noqa: E402
    _split_scale,
)

_HAS_RAW = False
try:

    @tle.raw.dialect("xpu3", object=_OBJ16, arch=3)
    def fqv3_16(x, out, msk, n, scale, inv, zp, qmin, qmax): ...

    @tle.raw.dialect("xpu3", object=_OBJBF16, arch=3)
    def fqv3_bf16(x, out, msk, n, scale, inv, zp, qmin, qmax): ...

    @tle.raw.dialect("xpu3", object=_OBJ32, arch=3)
    def fqv3_32(x, out, msk, n, scale, inv, zp, qmin, qmax): ...

    @triton.jit(do_not_specialize=["n", "qmin", "qmax"])
    def _fq16_kernel(X, O, MS, n, scale, inv, zp, qmin, qmax):
        tle.raw.call(fqv3_16, (
            X, O, MS, tl.cast(n, tl.int32),
            tl.full((), scale, tl.float32),
            tl.full((), inv, tl.float32),
            tl.full((), zp, tl.float32),
            tl.full((), qmin, tl.int32),
            tl.full((), qmax, tl.int32),
        ))

    @triton.jit(do_not_specialize=["n", "qmin", "qmax"])
    def _fqbf16_kernel(X, O, MS, n, scale, inv, zp, qmin, qmax):
        tle.raw.call(fqv3_bf16, (
            X, O, MS, tl.cast(n, tl.int32),
            tl.full((), scale, tl.float32),
            tl.full((), inv, tl.float32),
            tl.full((), zp, tl.float32),
            tl.full((), qmin, tl.int32),
            tl.full((), qmax, tl.int32),
        ))

    @triton.jit(do_not_specialize=["n", "qmin", "qmax"])
    def _fq32_kernel(X, O, MS, n, scale, inv, zp, qmin, qmax):
        tle.raw.call(fqv3_32, (
            X, O, MS, tl.cast(n, tl.int32),
            tl.full((), scale, tl.float32),
            tl.full((), inv, tl.float32),
            tl.full((), zp, tl.float32),
            tl.full((), qmin, tl.int32),
            tl.full((), qmax, tl.int32),
        ))

    _HAS_RAW = True
except Exception:  # pragma: no cover - tle.raw object= unavailable
    _HAS_RAW = False


def _generic(name):
    import importlib

    g = importlib.import_module(
        "flag_gems.ops.fake_quantize_per_tensor_affine_cachemask"
    )
    return getattr(g, name)


def _fast(x, scale, zero_point, quant_min, quant_max, out0, out1):
    if not _HAS_RAW:
        return None
    if x.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        return None
    if not x.is_contiguous():
        return None
    qmin, qmax, zp = int(quant_min), int(quant_max), int(zero_point)
    if qmin > qmax or zp < qmin or zp > qmax:
        return None
    n = x.numel()
    if n <= 0:
        return None
    if out0 is None:
        out0 = torch.empty_like(x)
    if out1 is None:
        out1 = torch.empty_like(x, dtype=torch.bool)
    if (
        out0.dtype != x.dtype
        or out1.dtype != torch.bool
        or out0.shape != x.shape
        or out1.shape != x.shape
        or not out0.is_contiguous()
        or not out1.is_contiguous()
        or out0.device != x.device
        or out1.device != x.device
    ):
        return None

    scale_f, inv = _split_scale(scale)
    chunk = 256 if x.dtype == torch.float32 else 512
    nch = (n + chunk - 1) // chunk
    grid = max(1, min(12, (nch + 63) // 64))

    try:
        with torch_device_fn.device(x.device):
            if x.dtype == torch.float32:
                _fq32_kernel[(grid,)](x, out0, out1, n, scale_f, inv, float(zp),
                                      qmin, qmax)
            elif x.dtype == torch.float16:
                _fq16_kernel[(grid,)](x, out0, out1, n, scale_f, inv, float(zp),
                                      qmin, qmax)
            else:
                _fqbf16_kernel[(grid,)](x, out0, out1, n, scale_f, inv, float(zp),
                                        qmin, qmax)
    except Exception:
        return None
    return out0, out1


def fake_quantize_per_tensor_affine_cachemask(
    input, scale, zero_point, quant_min, quant_max
):
    logger.debug("GEMS_KUNLUNXIN FAKE_QUANTIZE_PER_TENSOR_AFFINE_CACHEMASK")
    r = _fast(input, scale, zero_point, quant_min, quant_max, None, None)
    if r is not None:
        return r
    return _generic("fake_quantize_per_tensor_affine_cachemask")(
        input, scale, zero_point, quant_min, quant_max
    )


def fake_quantize_per_tensor_affine_cachemask_out(
    input, scale, zero_point, quant_min, quant_max, *, out0, out1
):
    logger.debug("GEMS_KUNLUNXIN FAKE_QUANTIZE_PER_TENSOR_AFFINE_CACHEMASK_OUT")
    r = _fast(input, scale, zero_point, quant_min, quant_max, out0, out1)
    if r is not None:
        return r
    return _generic("fake_quantize_per_tensor_affine_cachemask_out")(
        input, scale, zero_point, quant_min, quant_max, out0=out0, out1=out1
    )
