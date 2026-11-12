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
#
# kunlunxin (XPU) convolution_backward / convolution_backward.out.
#
# The generic triton implementation is both wrong and orders of magnitude
# slower on this backend: the full accuracy suite reports 660 failed / 81
# passed (plus a status=700 illegal memory access at teardown) and the
# grad_input/grad_weight-only benchmark cases run at 0.000-0.006x of the
# vendor kernel (2.8-110 ms vs 6-50 us).  The generic kernels go through the
# SDNN tl.dot path with tiny tiles, which the P800 memory model cannot
# sustain (see conv_transpose2d for the same finding).
#
# Delegate to the native aten kernels instead -- captured before FlagGems
# registers its own overrides, the same pattern as
# flag_gems/runtime/backend/_kunlunxin/ops/conv_transpose{1,2,3}d.py.
#
# Measured quirks of the vendor kernel (all verified against a CPU fp64
# reference, see new_batch_1112/ops/convolution_backward.md):
#
#   * non-transposed backward is correct for bf16/fp32 and for fp16 except two
#     cases where a single fp16 rounding of the partial sums drifts outside the
#     test tolerance: grad_input of a 2D/3D convolution, and grad_weight of a
#     grouped convolution.  Those two run the same native kernels in fp32.
#   * transposed backward (transposed=True) is only correct for
#     stride == 1 and dilation == 1 and output_padding == 0; with any other
#     combination the vendor kernel returns garbage (max error ~1e5) and can
#     abort with CUDNN_STATUS_EXECUTION_FAILED.
#   * empty batch / empty channel makes the vendor kernel raise std::bad_alloc,
#     so zero-filled outputs are produced here.
#
# For transposed backward outside that verified-safe corner the gradient is
# expressed through non-transposed primitives (which are correct):
#
#     dx = conv2d(grad_output, weight, stride, padding, dilation, groups)
#     dw = weight-gradient of conv2d(grad_output, weight) w.r.t. weight,
#          i.e. convolution_backward(grad_output=grad_input_padded,
#                                    input=grad_output, weight=weight,
#                                    transposed=False, mask=(False, True, False))
#     db = sum(grad_output, dim=(N, *spatial))
#
# The swap pads the "grad_output" argument with zeros whose spatial size the
# vendor kernel expects; the extra positions contribute nothing.
import logging

import torch

from flag_gems import runtime
from flag_gems.ops.convolution_backward import (
    _empty_gradient,
    _normalize_parameter,
)

logger = logging.getLogger(__name__)

_CAN_CAPTURE_NATIVE_KERNEL = hasattr(torch.library, "get_kernel")

if _CAN_CAPTURE_NATIVE_KERNEL:
    # Capture before FlagGems installs its override for these operator names.
    _NATIVE_CONV_BWD = torch.library.get_kernel(
        torch.ops.aten.convolution_backward.default, runtime.device.dispatch_key
    )
    _NATIVE_CONV_BWD_OUT = torch.library.get_kernel(
        torch.ops.aten.convolution_backward.out, runtime.device.dispatch_key
    )
    _NATIVE_CONV = torch.library.get_kernel(
        torch.ops.aten.convolution.default, runtime.device.dispatch_key
    )

_NATIVE_KEY_SET = torch._C.DispatchKeySet(
    getattr(torch._C.DispatchKey, runtime.device.dispatch_key)
)

_DTYPES = (torch.float16, torch.bfloat16, torch.float32, torch.float64)

# Validation only depends on the argument descriptors below.  Benchmark and
# training loops replay the same descriptors thousands of times, and the ~30 us
# of Python checks would otherwise dominate the vendor kernel launch (~40 us on
# this stack).  Same idea as the generic implementation's _LAUNCH_PLANS cache.
_VALIDATION_CACHE = {}


def _param_key(value):
    if isinstance(value, (list, tuple)):
        return tuple(value)
    return value


def _validation_key(
    grad_output,
    input,
    weight,
    bias_sizes,
    stride,
    padding,
    dilation,
    transposed,
    output_padding,
    groups,
    output_mask,
):
    return (
        grad_output.shape,
        input.shape,
        weight.shape,
        grad_output.dtype,
        input.dtype,
        weight.dtype,
        grad_output.device,
        input.device,
        weight.device,
        grad_output.layout,
        input.layout,
        weight.layout,
        None if bias_sizes is None else _param_key(bias_sizes),
        _param_key(stride),
        _param_key(padding),
        _param_key(dilation),
        transposed,
        _param_key(output_padding),
        groups,
        _param_key(output_mask),
    )


def _check_common(
    grad_output,
    input,
    weight,
    bias_sizes,
    stride,
    padding,
    dilation,
    transposed,
    output_padding,
    groups,
    output_mask,
):
    """Validate the aten contract and return normalized convolution metadata.

    The generic implementation raises RuntimeError for malformed arguments and
    the test suite asserts that; the vendor kernel either accepts some of them
    silently (dilation=0) or aborts the device (groups=0), so validate first.
    """
    try:
        key = _validation_key(
            grad_output,
            input,
            weight,
            bias_sizes,
            stride,
            padding,
            dilation,
            transposed,
            output_padding,
            groups,
            output_mask,
        )
        cached = _VALIDATION_CACHE.get(key)
    except TypeError:
        key = None
        cached = None
    if cached is not None:
        return cached
    dimensions = input.ndim - 2
    if (
        dimensions not in (1, 2, 3)
        or weight.ndim != input.ndim
        or grad_output.ndim != input.ndim
    ):
        raise RuntimeError(
            "convolution_backward: expected matching batched 1D/2D/3D tensors"
        )
    if (
        input.layout != torch.strided
        or weight.layout != torch.strided
        or grad_output.layout != torch.strided
    ):
        raise RuntimeError("convolution_backward: expected strided tensors")
    if input.dtype not in _DTYPES:
        raise RuntimeError("convolution_backward: expected real floating-point tensors")
    if input.dtype == torch.float64 and not runtime.device.support_fp64:
        raise RuntimeError(
            "convolution_backward: this backend does not support FP64 arithmetic"
        )
    if input.dtype != weight.dtype or input.dtype != grad_output.dtype:
        raise RuntimeError("convolution_backward: dtype mismatch")
    if input.device != weight.device or input.device != grad_output.device:
        raise RuntimeError("convolution_backward: device mismatch")
    if not isinstance(groups, int) or isinstance(groups, bool) or groups <= 0:
        raise RuntimeError("convolution_backward: groups must be positive")
    if len(output_mask) != 3:
        raise RuntimeError(
            "convolution_backward: output_mask must contain three booleans"
        )
    for m in output_mask:
        if type(m) is not bool:
            raise RuntimeError(
                "convolution_backward: output_mask must contain three booleans"
            )
    stride = _normalize_parameter(stride, dimensions, "stride", 1)
    padding = _normalize_parameter(padding, dimensions, "padding", 0)
    dilation = _normalize_parameter(dilation, dimensions, "dilation", 1)
    output_padding = _normalize_parameter(
        output_padding, dimensions, "output_padding", 0
    )
    for shape in (input.shape, weight.shape, grad_output.shape):
        for s in shape[2:]:
            if s <= 0:
                raise RuntimeError(
                    "convolution_backward: spatial dimensions must be positive"
                )
    if weight.shape[0] < groups or weight.shape[0] % groups:
        raise RuntimeError("convolution_backward: invalid weight channels for groups")
    if transposed:
        channels = weight.shape[1] * groups
        if input.shape[1] != weight.shape[0]:
            raise RuntimeError("convolution_backward: input/weight channel mismatch")
        for op, s, d in zip(output_padding, stride, dilation):
            if op >= s and op >= d:
                raise RuntimeError(
                    "convolution_backward: output_padding must be smaller than stride or dilation"
                )
    else:
        channels = weight.shape[0]
        if input.shape[1] != weight.shape[1] * groups:
            raise RuntimeError("convolution_backward: input/weight channel mismatch")
        for op in output_padding:
            if op:
                raise RuntimeError(
                    "convolution_backward: ordinary convolution requires zero output_padding"
                )
    spatial = []
    for i, k, s, p, d, op in zip(
        input.shape[2:], weight.shape[2:], stride, padding, dilation, output_padding
    ):
        spatial.append(
            (i - 1) * s - 2 * p + d * (k - 1) + op + 1
            if transposed
            else (i + 2 * p - d * (k - 1) - 1) // s + 1
        )
    if input.shape[1] == 0:
        channels = 0
    for s in spatial:
        if s <= 0:
            raise RuntimeError(
                "convolution_backward: grad_output shape does not match convolution"
            )
    if tuple(grad_output.shape) != (input.shape[0], channels, *spatial):
        raise RuntimeError(
            "convolution_backward: grad_output shape does not match convolution"
        )
    bias_channels = weight.shape[1] * groups if transposed else weight.shape[0]
    if bias_sizes is not None and tuple(bias_sizes) != (bias_channels,):
        raise RuntimeError("convolution_backward: invalid bias_sizes")
    result = (
        dimensions,
        tuple(spatial),
        bias_channels,
        stride,
        padding,
        dilation,
        output_padding,
    )
    if key is not None:
        if len(_VALIDATION_CACHE) >= 256:
            _VALIDATION_CACHE.pop(next(iter(_VALIDATION_CACHE)))
        _VALIDATION_CACHE[key] = result
    return result


def _channels_last_format(dimensions):
    if dimensions == 2:
        return torch.channels_last
    if dimensions == 3:
        return torch.channels_last_3d
    return None


def _empty_inputs(input, weight, bias_channels, output_mask, dimensions):
    outputs = [
        _empty_gradient(input) if output_mask[0] else None,
        _empty_gradient(weight) if output_mask[1] else None,
        (
            torch.empty(
                (bias_channels,), dtype=input.dtype, device=input.device
            )
            if output_mask[2]
            else None
        ),
    ]
    return outputs


def _zero_gradients(outputs):
    for output in outputs:
        if output is not None and output.numel():
            output.zero_()


def _native_transposed_safe(stride, dilation, output_padding):
    # The vendor transposed backward was only verified correct (and free of
    # CUDNN_STATUS_EXECUTION_FAILED) for this corner of the parameter space.
    return (
        all(step == 1 for step in stride)
        and all(step == 1 for step in dilation)
        and all(pad == 0 for pad in output_padding)
    )


def _needs_fp32(dtype, mask, transposed, safe_transposed, groups):
    # The vendor fp16 kernel is not accurate enough on every gradient:
    #   * non-transposed grad_input drifts outside the 1e-3 rtol for 2D/3D,
    #   * grouped grad_weight drifts by up to 3 orders of magnitude,
    #   * the decomposed transposed path builds conv2d / weight-gradient /
    #     reduction primitives whose bf16 accumulation is likewise too coarse.
    # Where a precise gradient is requested, run the native kernels in fp32 and
    # round once at the end.  Masks that avoid those gradients (e.g. the
    # bias-only case) stay in the native dtype and skip all casts.
    if dtype == torch.float16:
        return (
            not safe_transposed
            or (mask[0] and not transposed)
            or (mask[1] and groups > 1)
        )
    if dtype == torch.bfloat16:
        return not safe_transposed
    return False


def _conv_backward_raw(
    grad_output,
    input,
    weight,
    bias_sizes,
    stride,
    padding,
    dilation,
    transposed,
    output_padding,
    groups,
    output_mask,
    cast_outputs=True,
):
    dtype = grad_output.dtype
    dimensions = input.ndim - 2
    stride = list(stride)
    padding = list(padding)
    dilation = list(dilation)
    output_padding = list(output_padding)
    mask = tuple(bool(m) for m in output_mask)
    zeros = (0,) * dimensions
    safe_transposed = not transposed or _native_transposed_safe(
        stride, dilation, output_padding
    )
    need_fp32 = _needs_fp32(dtype, mask, transposed, safe_transposed, groups)
    if need_fp32:
        compute_go = grad_output.float()
        compute_x = input.float()
        compute_w = weight.float()
    else:
        compute_go, compute_x, compute_w = grad_output, input, weight
    wants_db = mask[2]
    # aten's bias gradient is sum(grad_output) over batch and spatial dims even
    # when bias_sizes is None; the vendor kernel instead rejects None with
    # std::bad_alloc, so reduce it here in that case.  Otherwise the vendor
    # reduction is both accurate (it accumulates in fp32 even for fp16/bf16
    # inputs) and faster than a separate sum.
    compute_db_here = bias_sizes is None
    native_mask = (mask[0], mask[1], mask[2] and not compute_db_here)
    if bias_sizes is None:
        native_bias_sizes = [
            weight.shape[1] * groups if transposed else weight.shape[0]
        ]
    else:
        native_bias_sizes = list(bias_sizes)

    if not (mask[0] or mask[1] or (wants_db and not compute_db_here)):
        # Bias-only with bias_sizes given: the native reduction below is used
        # instead, so no separate launch is needed here.
        outputs = [None, None, None]
    elif safe_transposed:
        result = _NATIVE_CONV_BWD.call_boxed(
            _NATIVE_KEY_SET,
            compute_go,
            compute_x,
            compute_w,
            native_bias_sizes,
            stride,
            padding,
            dilation,
            transposed,
            output_padding,
            groups,
            native_mask,
        )
        outputs = [result[i] if native_mask[i] else None for i in range(3)]
    else:
        outputs = [None, None, None]
        if mask[0]:
            # Adjoint of the forward conv: conv2d(grad_output, weight).  The
            # extra output_padding positions have no counterpart in the input;
            # crop them away.
            dx = _NATIVE_CONV.call_boxed(
                _NATIVE_KEY_SET,
                compute_go,
                compute_w,
                None,
                stride,
                padding,
                dilation,
                False,
                zeros,
                groups,
            )
            wanted_spatial = tuple(compute_x.shape[2:])
            if tuple(dx.shape[2:]) != wanted_spatial:
                dx = dx[
                    (slice(None), slice(None))
                    + tuple(slice(0, size) for size in wanted_spatial)
                ].contiguous()
            outputs[0] = dx
        if mask[1]:
            # weight-gradient with the x/grad_output roles exchanged.
            pad = []
            needs_pad = False
            for i in reversed(range(dimensions)):
                kernel = compute_w.shape[2 + i]
                expected = (
                    compute_go.shape[2 + i]
                    + 2 * padding[i]
                    - dilation[i] * (kernel - 1)
                    - 1
                ) // stride[i] + 1
                extra = expected - compute_x.shape[2 + i]
                pad.extend((0, max(0, extra)))
                needs_pad = needs_pad or extra > 0
            source = compute_x
            if needs_pad:
                source = torch.nn.functional.pad(compute_x, tuple(pad))
            outputs[1] = _NATIVE_CONV_BWD.call_boxed(
                _NATIVE_KEY_SET,
                source,
                compute_go,
                compute_w,
                [compute_w.shape[0]],
                stride,
                padding,
                dilation,
                False,
                zeros,
                groups,
                (False, True, False),
            )[1]
    if wants_db and outputs[2] is None:
        outputs[2] = grad_output.sum(
            dim=(0, *range(2, grad_output.ndim)), dtype=torch.float32
        ).to(dtype)
    if need_fp32 and cast_outputs:
        outputs = [
            None if output is None else output.to(dtype) for output in outputs
        ]
    return outputs


def convolution_backward(
    grad_output,
    input,
    weight,
    bias_sizes,
    stride,
    padding,
    dilation,
    transposed,
    output_padding,
    groups,
    output_mask,
):
    logger.debug("GEMS_KUNLUNXIN CONVOLUTION_BACKWARD")

    if not _CAN_CAPTURE_NATIVE_KERNEL:
        from flag_gems.ops.convolution_backward import (
            convolution_backward as _generic,
        )

        return _generic(
            grad_output,
            input,
            weight,
            bias_sizes,
            stride,
            padding,
            dilation,
            transposed,
            output_padding,
            groups,
            output_mask,
        )

    (
        dimensions,
        _,
        bias_channels,
        stride_n,
        padding_n,
        dilation_n,
        output_padding_n,
    ) = _check_common(
        grad_output,
        input,
        weight,
        bias_sizes,
        stride,
        padding,
        dilation,
        transposed,
        output_padding,
        groups,
        output_mask,
    )
    if input.numel() == 0 or grad_output.numel() == 0 or weight.numel() == 0:
        outputs = _empty_inputs(input, weight, bias_channels, output_mask, dimensions)
        _zero_gradients(outputs)
        return tuple(outputs)
    results = _conv_backward_raw(
        grad_output,
        input,
        weight,
        bias_sizes,
        stride_n,
        padding_n,
        dilation_n,
        transposed,
        output_padding_n,
        groups,
        output_mask,
    )
    fmt = _channels_last_format(dimensions)
    if fmt is not None and input.is_contiguous(memory_format=fmt):
        # The test contract keeps channels-last gradients in that layout.
        results[0] = (
            None if results[0] is None else results[0].contiguous(memory_format=fmt)
        )
        results[1] = (
            None if results[1] is None else results[1].contiguous(memory_format=fmt)
        )
    return tuple(results)


def convolution_backward_out(
    grad_output,
    input,
    weight,
    bias_sizes,
    stride,
    padding,
    dilation,
    transposed,
    output_padding,
    groups,
    output_mask,
    *,
    out0,
    out1,
    out2,
):
    logger.debug("GEMS_KUNLUNXIN CONVOLUTION_BACKWARD_OUT")

    if not _CAN_CAPTURE_NATIVE_KERNEL:
        from flag_gems.ops.convolution_backward import (
            convolution_backward_out as _generic,
        )

        return _generic(
            grad_output,
            input,
            weight,
            bias_sizes,
            stride,
            padding,
            dilation,
            transposed,
            output_padding,
            groups,
            output_mask,
            out0=out0,
            out1=out1,
            out2=out2,
        )

    if not all(output_mask):
        raise RuntimeError(
            "convolution_backward.out: a defined out tensor cannot represent an undefined masked gradient"
        )
    (
        dimensions,
        _,
        bias_channels,
        stride_l,
        padding_l,
        dilation_l,
        output_padding_l,
    ) = _check_common(
        grad_output,
        input,
        weight,
        bias_sizes,
        stride,
        padding,
        dilation,
        transposed,
        output_padding,
        groups,
        output_mask,
    )
    stride_l = list(stride_l)
    padding_l = list(padding_l)
    dilation_l = list(dilation_l)
    output_padding_l = list(output_padding_l)
    outputs = (out0, out1, out2)
    for out, shape in zip(
        outputs, (input.shape, weight.shape, (bias_channels,))
    ):
        if out.dtype != input.dtype or out.device != input.device:
            raise RuntimeError("convolution_backward.out: output dtype/device mismatch")
        if out.shape != shape:
            out.resize_(shape)
        for size, step in zip(out.shape, out.stride()):
            if size > 1 and step == 0:
                raise RuntimeError("convolution_backward.out: overlapping output")

    if input.numel() == 0 or grad_output.numel() == 0 or weight.numel() == 0:
        _zero_gradients(outputs)
        return outputs

    safe_transposed = not transposed or _native_transposed_safe(
        stride_l, dilation_l, output_padding_l
    )
    aliases = False
    for out in outputs:
        for tensor in (grad_output, input, weight):
            if torch._C._is_alias_of(out, tensor):
                aliases = True
                break
        if aliases:
            break
    if not aliases:
        for i in range(3):
            for j in range(i):
                if torch._C._is_alias_of(outputs[i], outputs[j]):
                    aliases = True
                    break
            if aliases:
                break
    # Write straight into the caller's buffers when the native kernel is
    # accurate for this configuration (it also honours arbitrary out strides);
    # otherwise compute into fresh tensors and let copy_ convert the dtype.
    if (
        bias_sizes is not None
        and safe_transposed
        and not aliases
        and not _needs_fp32(
            input.dtype, (True, True, True), transposed, safe_transposed, groups
        )
    ):
        _NATIVE_CONV_BWD_OUT.call_boxed(
            _NATIVE_KEY_SET,
            grad_output,
            input,
            weight,
            list(bias_sizes),
            stride_l,
            padding_l,
            dilation_l,
            transposed,
            output_padding_l,
            groups,
            [True, True, True],
            out0=out0,
            out1=out1,
            out2=out2,
        )
        return outputs

    results = _conv_backward_raw(
        grad_output,
        input,
        weight,
        bias_sizes,
        stride_l,
        padding_l,
        dilation_l,
        transposed,
        output_padding_l,
        groups,
        output_mask,
        cast_outputs=False,
    )
    for out, result in zip(outputs, results):
        if result is not None:
            out.copy_(result)
    return outputs
