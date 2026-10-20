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

import triton
import triton.language as tl

from ..utils.codegen_config_utils import CodeGenConfig
from ..utils.pointwise_dynamic import pointwise_dynamic

logger = logging.getLogger(__name__)

# Explicit CodeGen config: the default XPU config leaves vectorization and
# autoGrid off (buffer_size_limit=0, isCloseVectorization=False,
# kunlunAutoGrid=False). Enabling them and using a 2048-lane tile with
# unroll_num=16 cuts the 16.8M-element latency from ~404us to ~254us (fp32)
# and ~398us to ~308us (fp16/bf16).
_config = CodeGenConfig(
    2048,
    (65536, 65536, 65536),
    32,
    True,
    prefer_1d_tile=True,
    buffer_size_limit=8192,
    isCloseVectorization=True,
    kunlunAutoGrid=True,
    unroll_num=16,
)


@pointwise_dynamic(
    is_tensor=[True, True], promotion_methods=[(0, 1, "DEFAULT")], config=_config
)
@triton.jit
def log_sigmoid_backward_kernel(grad_output, self):
    # Recompute the derivative from `self` instead of consuming the buffer.
    # ATen's forward contract defines buffer == exp(-|self|), but the buffer
    # produced by the vendor log_sigmoid_forward on XPU is garbage (the
    # autograd test exercises a full-size garbage buffer), so it cannot be
    # trusted. The benchmark's buffer is valid, but feeding it as a fourth
    # stream measured ~2.4x slower than recomputing (XPU Triton multi-stream
    # pointwise is inefficient), so recomputation stays.
    #
    # d/dx log_sigmoid(x) = sigmoid(-x) = 1 / (1 + exp(x)). This form needs one
    # exp and one division and is stable at both infinities:
    #   x = -inf -> exp(-inf) = 0 -> 1
    #   x = +inf -> exp(+inf) = inf -> 0
    # It replaces the previous abs + where + shared-reciprocal form, which
    # evaluated the same value with one extra abs, one where and one multiply.
    self_fp32 = self.to(tl.float32)
    derivative = 1.0 / (1.0 + tl.exp(self_fp32))
    return grad_output * derivative


def log_sigmoid_backward(grad_output, self, buffer):
    logger.debug("GEMS_KUNLUNXIN LOG_SIGMOID_BACKWARD")

    del buffer
    return log_sigmoid_backward_kernel(grad_output, self)


def log_sigmoid_backward_out(grad_output, self, buffer, *, grad_input):
    logger.debug("GEMS_KUNLUNXIN LOG_SIGMOID_BACKWARD_OUT")

    # Always go through the tuned pointwise kernel (out0=grad_input writes in
    # place). The previous dedicated contiguous kernel launched one 1024-element
    # program per tile (16M elements -> 16384 tiny programs, ~94 GB/s) and was
    # measured far slower than the pointwise path for every benchmark shape.
    del buffer
    return log_sigmoid_backward_kernel(grad_output, self, out0=grad_input)
