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
from _kunlunxin.utils.codegen_config_utils import CodeGenConfig

from flag_gems.runtime import torch_device_fn

from ..utils.pointwise_dynamic import pointwise_dynamic

logger = logging.getLogger(__name__)

# Two codegen variants for the two latency regimes on XPU:
# - small tensors (numel <= 4096): launch-bound -> 1-CTA (kunlunAutoGrid) path
#   keeps the whole tensor in one CTA.
# - medium/large tensors: 12-CTA grid-stride path with explicit unroll 16;
#   unroll 16 is the sweet spot measured for this op family on XPU
#   (sibling bitwise_left_shift_ closure, harness/solution/performance/).
config_small = CodeGenConfig(
    512,
    (65536, 65536, 65536),
    32,
    True,
    prefer_1d_tile=True,
    kunlunAutoGrid=True,
    unroll_num=16,
)
config_large = CodeGenConfig(
    512,
    (65536, 65536, 65536),
    32,
    True,
    prefer_1d_tile=True,
    unroll_num=16,
)


@pointwise_dynamic(promotion_methods=[(0, 1, "DEFAULT")], config=config_small)
@triton.jit
def bitwise_right_shift_small(a, b):
    return a >> b


@pointwise_dynamic(promotion_methods=[(0, 1, "DEFAULT")], config=config_large)
@triton.jit
def bitwise_right_shift_large(a, b):
    return a >> b


# ---- tle.raw payload: LM-staged vector arithmetic right shift --------------
# The pointwise codegen plateaus at 130.7us (int16) / 147.2us (int32) for 16.8M
# elements. The payload stages 512 int32 / 1024 int16 of a and b (4KB), shifts
# with vvsra_int32x16 (int16 = two halves packed per int32 lane), and bulk-writes
# the result. Pure Triton has no per-lane variable-shift SIMD path.
_RS_PAYLOAD_MIN = 500_000
_RS_PAYLOAD_GRID = 12
_RS_PAYLOAD = {"init": False, "ok": False, "kern": {}}
_RS_META_CACHE = {}


def _rs_payload_init():
    if _RS_PAYLOAD["init"]:
        return _RS_PAYLOAD["ok"]
    _RS_PAYLOAD["init"] = True
    try:
        import triton as _triton
        import triton.experimental.tle as _tle

        obj = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "payload",
            "obj",
            "rshift.o",
        )
        for sym in ("rshift_16", "rshift_32"):
            ns = {"_tle": _tle, "_OBJ": obj}
            exec(
                "@_tle.raw.dialect('xpu3', object=_OBJ, arch=3)\n"
                f"def {sym}(meta, a, b, out):\n"
                "    ...\n",
                ns,
            )
            callee = ns[sym]

            def _mk(callee=callee):
                @_triton.jit
                def _k(M, A, B, O):
                    _tle.raw.call(callee, (M, A, B, O))

                return _k

            _RS_PAYLOAD["kern"][sym] = _mk()
        _RS_PAYLOAD["ok"] = True
    except Exception as _e:  # pragma: no cover - keep backend importable
        logger.debug("GEMS_KUNLUNXIN RS payload init failed: %r", _e)
        _RS_PAYLOAD["ok"] = False
    return _RS_PAYLOAD["ok"]


def _rs_payload_launch(a, b, out):
    if a.dtype not in (torch.int16, torch.int32) or b.dtype != a.dtype:
        return False
    if a.shape != b.shape:
        return False
    if not (a.is_contiguous() and b.is_contiguous() and out.is_contiguous()):
        return False
    n = a.numel()
    if n < _RS_PAYLOAD_MIN or n >= 2**31:
        return False
    if not _rs_payload_init():
        return False
    key = (n, a.device.index)
    meta = _RS_META_CACHE.get(key)
    if meta is None:
        meta = torch.tensor([n], dtype=torch.int32, device=a.device)
        if len(_RS_META_CACHE) >= 64:
            _RS_META_CACHE.clear()
        _RS_META_CACHE[key] = meta
    kern = _RS_PAYLOAD["kern"][
        "rshift_16" if a.dtype == torch.int16 else "rshift_32"
    ]
    with torch_device_fn.device(a.device):
        kern[(_RS_PAYLOAD_GRID,)](meta, a.view(-1), b.view(-1), out.view(-1))
    return True


def _right_shift_kernel(a, b, *, out=None, out0=None):
    if out is None and out0 is None:
        if a.numel() > 4096:
            target = torch.empty_like(a)
            try:
                if _rs_payload_launch(a, b, target):
                    return target
            except Exception as exc:  # pragma: no cover - fall back on failure
                logger.debug("bitwise_right_shift payload failed: %s", exc)
            return bitwise_right_shift_large(a, b)
        # The 1-CTA small wrapper has no `out` argument.
        return bitwise_right_shift_small(a, b)
    target = out if out is not None else out0
    try:
        if _rs_payload_launch(a, b, target):
            return target
    except Exception as exc:  # pragma: no cover - fall back on failure
        logger.debug("bitwise_right_shift payload failed: %s", exc)
    large = a.numel() > 4096
    if out is not None:
        if large:
            return bitwise_right_shift_large(a, b, out=out)
        return bitwise_right_shift_small(a, b, out=out)
    if out0 is not None:
        if large:
            return bitwise_right_shift_large(a, b, out0=out0)
        return bitwise_right_shift_small(a, b, out0=out0)
    if large:
        return bitwise_right_shift_large(a, b)
    return bitwise_right_shift_small(a, b)


def bitwise_right_shift(self, other, *, out=None):
    logger.debug("GEMS_KUNLUNXIN BITWISE_RIGHT_SHIFT")
    return _right_shift_kernel(self, other, out=out)


def bitwise_right_shift_(self, other, *, out=None):
    return _right_shift_kernel(self, other, out0=self)
