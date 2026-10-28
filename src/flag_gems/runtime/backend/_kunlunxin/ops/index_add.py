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
import struct

import torch
import triton
import triton.language as tl

from flag_gems import runtime
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry

logger = logging.getLogger(__name__)


def _contig_fast(t):
    """Contiguous (copy-if-needed) via ``aten::_copy_from``.

    ``t.clone()`` / ``t.contiguous()`` (on a non-contiguous tensor) funnel into
    ``aten::copy_``, which gems overrides with a Triton pointwise kernel
    (measured ~3.4 ms for 1M elements, i.e. the dominant fixed cost of an
    index_add call inside the gems dispatch). ``aten::_copy_from`` is never
    overridden and dispatches to the vendor strided-copy engine (same pattern
    as the slice_backward / resize / constant_pad_nd fixes).
    """
    if t.is_contiguous():
        return t
    dst = torch.empty_like(t, memory_format=torch.contiguous_format)
    torch.ops.aten._copy_from(t, dst, False)
    return dst


def _clone_fast(t):
    """Always-copy variant (``t.clone()`` semantics), ``_copy_from`` based."""
    dst = torch.empty_like(t, memory_format=torch.contiguous_format)
    torch.ops.aten._copy_from(t, dst, False)
    return dst


# def cfggen():
#     block_m = [1, 2, 4]
#     block_n = [128, 1024, 2048, 4096]
#     configs = [
#         triton.Config({"BLOCK_M": m, "BLOCK_N": n}, num_warps=4)
#         for m in block_m
#         for n in block_n
#     ]
#     return configs


@libentry()
# @triton.autotune(configs=cfggen(), key=["M", "N"])
@triton.heuristics(runtime.get_heuristic_config("index_add"))
# @triton.autotune(
#     configs=[], generate_configs="index_add", op_affiliation="cluster", row_sign="M", col_sign="N",
#     key=["M", "N"],
# )
@triton.jit
def index_add_kernel(
    index,
    src,
    inp_cont,
    M: tl.constexpr,
    N: tl.constexpr,
    alpha,
    inp_len,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    # Fast path: index is unique. Each destination cell is written by exactly
    # one program, so plain (non-atomic) load-add-store is safe and fast.
    pid_x = tl.program_id(axis=0)
    pid_y = tl.program_id(axis=1)
    rows_offsets = pid_x * BLOCK_M + tl.arange(0, BLOCK_M)[:, None]
    cols_offsets = pid_y * BLOCK_N + tl.arange(0, BLOCK_N)

    rows_mask = rows_offsets < M
    index_mask = cols_offsets < N
    block_mask = rows_mask and index_mask

    cur_indices = tl.load(index + cols_offsets, mask=index_mask, other=0)
    inp_off = rows_offsets * inp_len + cur_indices[None, :]
    cur_inp = tl.load(inp_cont + inp_off, mask=block_mask, other=0.0)
    src_off = rows_offsets * N + cols_offsets[None, :]
    cur_src = tl.load(src + src_off, mask=block_mask, other=0.0)
    cur_inp += alpha * cur_src

    tl.store(inp_cont + inp_off, cur_inp, mask=block_mask)


@libentry()
@triton.jit
def index_add_seg_kernel(
    src_ptr,
    perm_ptr,
    v_ptr,
    head_ptr,
    len_ptr,
    out_ptr,
    M,
    N,
    out_len,
    U,
    alpha,
    BLOCK_S: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    # Deterministic duplicate-safe path. The index is sorted once (host side);
    # equal values form adjacent runs ("segments"). One program handles one row
    # and BLOCK_S segments; each segment's source columns are summed with a
    # bounded 1D column-tile reduction and accumulated into the unique
    # destination column of that segment. There are no atomics and no
    # cross-program writes to the same cell. (1D-only structure: the 2D
    # gather+reduce variant fails the XPU uni_sram lowering pass.)
    pid_m = tl.program_id(axis=0)
    pid_s = tl.program_id(axis=1)
    r = pid_m
    cc = tl.arange(0, BLOCK_C)
    for s in tl.static_range(BLOCK_S):
        gs = pid_s * BLOCK_S + s
        ok_s = gs < U
        c0_t = tl.load(head_ptr + gs)
        l_s = tl.load(len_ptr + gs)
        j_s = tl.load(v_ptr + c0_t)
        m_s = (cc < l_s) & ok_s
        # XPU masked loads may ignore `other=0` for runtime masks; read within
        # the padded buffers, then zero the out-of-segment lanes explicitly.
        cols = tl.load(perm_ptr + c0_t + cc)  # padded, always in-bounds
        col_clamped = tl.minimum(tl.where(m_s, cols, 0), N - 1)
        vals = tl.load(src_ptr + r.to(tl.int64) * N + col_clamped)
        vals = tl.where(m_s, vals, 0.0)
        seg_sum = tl.sum(vals, axis=0)
        dst = r.to(tl.int64) * out_len + j_s
        cur = tl.load(out_ptr + dst, mask=ok_s, other=0.0)
        tl.store(out_ptr + dst, cur + alpha * seg_sum, mask=ok_s)


@libentry()
@triton.jit
def index_add_seg_loop_kernel(
    src_ptr,
    perm_ptr,
    v_ptr,
    head_ptr,
    len_ptr,
    out_ptr,
    M,
    N,
    out_len,
    U,
    alpha,
    BLOCK_S: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    # Same deterministic segment reduction for segments longer than BLOCK_C:
    # each segment's source columns are accumulated in column tiles.
    pid_m = tl.program_id(axis=0)
    pid_s = tl.program_id(axis=1)
    r = pid_m
    cc = tl.arange(0, BLOCK_C)
    for s in tl.static_range(BLOCK_S):
        gs = pid_s * BLOCK_S + s
        ok_s = gs < U
        c0_t = tl.load(head_ptr + gs)
        l_s = tl.load(len_ptr + gs)
        j_s = tl.load(v_ptr + c0_t)
        acc = tl.zeros((BLOCK_C,), dtype=tl.float32)
        for k in range(0, l_s, BLOCK_C):
            m_s = ((k + cc) < l_s) & ok_s
            cols = tl.load(perm_ptr + c0_t + k + cc)  # padded, in-bounds
            col_clamped = tl.minimum(tl.where(m_s, cols, 0), N - 1)
            vals = tl.load(src_ptr + r.to(tl.int64) * N + col_clamped)
            vals = tl.where(m_s, vals, 0.0)
            acc += vals
        seg_sum = tl.sum(acc, axis=0)
        dst = r.to(tl.int64) * out_len + j_s
        cur = tl.load(out_ptr + dst, mask=ok_s, other=0.0)
        tl.store(out_ptr + dst, cur + alpha * seg_sum, mask=ok_s)


# Duplicate-safe segment-reduction tile sizes (bounded: BLOCK_S * BLOCK_C
# stays well under the XPU tl.sum safety ceiling of 8192 elements).
index_add_seg_block_s = 64
index_add_seg_block_c = 64

# Fixed tiles reused by the non-atomic fast kernel when invoked without
# heuristics (bounded, mirrors the index_add heuristics caps).
index_add_fast_block_m = 8
index_add_fast_block_n = 512


def _index_add_sort(index):
    """Deterministic sort of an index returning (sorted_values, perm).

    Reuses the backend-local radix sort (deterministic, works inside the gems
    dispatch context without the tl.argsort lowering that fails on XPU).
    """
    N = index.numel()
    if N <= 1:
        return index, torch.arange(N, dtype=torch.int64, device=index.device)
    from .sort import radix_sort_low_mem  # backend-local reuse, no dispatch

    flat = index.reshape(1, -1)
    v, perm = radix_sort_low_mem(flat, k_bits=4)
    return v.reshape(-1), perm.reshape(-1)


@libentry()
@triton.jit
def index_mark_unique_kernel(index_ptr, mark_ptr, N, BLOCK: tl.constexpr):
    # Writes 1 at mark[index[i]] for every index element. Concurrent writes of
    # the same value 1 to one slot are benign (no atomicity requirement), so
    # after this kernel sum(mark) == number of distinct index values.
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    v = tl.load(index_ptr + offs, mask=mask, other=0).to(tl.int64)
    tl.store(mark_ptr + v, tl.full((BLOCK,), 1, dtype=tl.int32), mask=mask)


def _index_unique_gate(index, inp_len):
    """Sound duplicate check in O(N): distinct_count == N  <=>  no duplicates.

    A per-value marker buffer is set to 1 (same-value concurrent writes are
    deterministic); sum(markers) equals the number of distinct index values,
    so sum == N iff the index contains no duplicate. No sorting required and
    the path stays entirely on XPU.
    """
    N = index.numel()
    if N <= 1:
        return True
    block = 256
    grid = (triton.cdiv(N, block),)
    mark = torch.zeros((inp_len,), dtype=torch.int32, device=index.device)
    index_mark_unique_kernel[grid](index, mark, N, BLOCK=block)
    return bool(mark.sum().item() == N)


@libentry()
@triton.jit
def index_seg_flags_kernel(v_ptr, flags_ptr, N, BLOCK: tl.constexpr):
    # flags[i] = 1 when sorted position i starts a new segment (v[i] != v[i-1]).
    # Written as a device kernel so the comparison never goes through the
    # pointwise codegen path (which fails to lower large N tiles on XPU).
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    cur = tl.load(v_ptr + offs, mask=mask, other=-1)
    prev = tl.load(v_ptr + offs - 1, mask=mask & (offs > 0))
    is_head = tl.where(offs == 0, True, cur != prev)
    tl.store(flags_ptr + offs, is_head.to(tl.int64), mask=mask)


def _index_add_segments(index, block_s, block_c):
    """Deterministic duplicate classification of an index tensor.

    Returns sorted values ``v``, the sort permutation ``perm`` and padded
    per-segment tables (head column offset / length) used by the duplicate-safe
    kernel. See ``index_add_seg_kernel``.
    """
    N = index.numel()
    v, perm = _index_add_sort(index)
    # int64 marker channel (device kernel, avoids the XPU pointwise ne crash).
    flag_blk = 256
    flags = torch.zeros((N,), dtype=torch.int64, device=index.device)
    index_seg_flags_kernel[(triton.cdiv(N, flag_blk),)](v, flags, N, BLOCK=flag_blk)
    heads = torch.nonzero(flags).reshape(-1)  # (U,) start column of each segment
    U = heads.numel()
    lens = torch.cat((heads[1:] - heads[:-1], (N - heads[-1]).reshape(1))).to(
        torch.int32
    )
    # Padded copies so masked lanes never read past the physical buffers.
    n_pad = ((N + block_c - 1) // block_c) * block_c
    u_pad = ((U + block_s - 1) // block_s) * block_s
    perm_pad = torch.zeros((n_pad,), dtype=torch.int64, device=index.device)
    perm_pad[:N] = perm
    v_pad = torch.zeros((n_pad,), dtype=torch.int64, device=index.device)
    v_pad[:N] = v
    head_pad = torch.zeros((u_pad,), dtype=torch.int64, device=index.device)
    head_pad[:U] = heads
    len_pad = torch.zeros((u_pad,), dtype=torch.int32, device=index.device)
    len_pad[:U] = lens
    return v_pad, perm_pad, head_pad, len_pad, U, int(lens.max().item())


# ---- tle.raw payload: one core per row, LM-resident scatter-add -------------
# Small/medium 2-D dim=1 rows fit one row in the per-core LM (16-bit <= 1280
# elems, 32-bit <= 1024). One core owns a whole row, so duplicate indices
# accumulate exactly without atomics or the uniqueness gate, and the host
# bound-check/gate syncs are skipped. Larger rows keep the existing kernels.
_IA_PAYLOAD_GRID = 12  # physical clusters
_IA_PAYLOAD = {"init": False, "ok": False, "kern": {}}
_IA_META_CACHE = {}


def _ia_payload_init():
    if _IA_PAYLOAD["init"]:
        return _IA_PAYLOAD["ok"]
    _IA_PAYLOAD["init"] = True
    try:
        import triton as _triton
        import triton.experimental.tle as _tle

        obj = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "payload",
            "obj",
            "index_add_lm.o",
        )
        # C signature: (int* meta, long long* idx, T* src, T* out);
        # meta = [rows, w_out, w_idx, alpha_bits, is_bf16].
        for sym in (
            "index_add_lm_16",
            "index_add_lm_32",
            "index_add_row_16",
            "index_add_row_32",
            "index_add_rowb_16",
            "index_add_rowb_32",
        ):
            ns = {"_tle": _tle, "_OBJ": obj}
            exec(
                "@_tle.raw.dialect('xpu3', object=_OBJ, arch=3)\n"
                f"def {sym}(meta, idx, src, base, out):\n"
                "    ...\n",
                ns,
            )
            callee = ns[sym]

            def _mk(callee=callee):
                @_triton.jit
                def _k(M, I, S, B, O):
                    _tle.raw.call(callee, (M, I, S, B, O))

                return _k

            _IA_PAYLOAD["kern"][sym] = _mk()
        _IA_PAYLOAD["ok"] = True
    except Exception as _e:  # pragma: no cover - keep backend importable
        logger.debug("GEMS_KUNLUNXIN INDEX_ADD payload init failed: %r", _e)
        _IA_PAYLOAD["ok"] = False
    return _IA_PAYLOAD["ok"]


def _ia_payload_launch(inp, dim, index, src, alpha):
    """Return the result tensor, or None to use the existing path."""
    if not _IA_PAYLOAD["init"] and not _ia_payload_init():
        return None
    if inp.dim() != 2 or (int(dim) % 2) != 1:
        return None
    if inp.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        return None
    if index.dtype != torch.int64:
        return None
    if not (inp.is_contiguous() and index.is_contiguous() and src.is_contiguous()):
        return None
    rows, w_out = int(inp.shape[0]), int(inp.shape[1])
    w_idx = int(index.numel())
    if index.dim() != 1 or tuple(src.shape) != (rows, w_idx):
        return None
    if rows <= 0 or w_idx <= 0 or w_out <= 0:
        return None
    is32 = inp.dtype == torch.float32
    lm_limit = 1024 if is32 else 1280
    sm_limit = 65536 if is32 else 131072
    batch = 1
    if w_out <= lm_limit:
        mode = "lm"
    elif w_out <= sm_limit:
        # The 64 cores of a cluster split the row's j range; duplicate indices
        # would race, so the cluster path is gated on uniqueness (the gate is
        # negligible at these row widths). Duplicates fall back to the segment
        # kernel, which is deterministic.
        if not _index_unique_gate(index, w_out):
            return None
        mode = "rowb"
        # B rows per sync epoch: the rows must fit the cluster SM together.
        batch = max(1, min(16, sm_limit // w_out))
    else:
        return None
    if rows * w_out >= 2**31 or rows * w_idx >= 2**31:
        return None
    is_bf16 = 1 if inp.dtype == torch.bfloat16 else 0
    alpha_bits = struct.unpack("<i", struct.pack("<f", float(alpha)))[0]
    key = (rows, w_out, w_idx, alpha_bits, is_bf16, batch, inp.device.index)
    meta = _IA_META_CACHE.get(key)
    if meta is None:
        meta = torch.tensor(
            [rows, w_out, w_idx, alpha_bits, is_bf16, batch],
            dtype=torch.int32,
            device=inp.device,
        )
        if len(_IA_META_CACHE) >= 64:
            _IA_META_CACHE.clear()
        _IA_META_CACHE[key] = meta
    out = torch.empty_like(inp)
    sym = f"index_add_{mode}_{'32' if is32 else '16'}"
    kern = _IA_PAYLOAD["kern"][sym]
    if is32:
        sv, bv, ov = src.view(-1), inp.view(-1), out.view(-1)
    else:
        sv = src.view(torch.int16).view(-1)
        bv = inp.view(torch.int16).view(-1)
        ov = out.view(torch.int16).view(-1)
    with torch_device_fn.device(inp.device):
        kern[(_IA_PAYLOAD_GRID,)](meta, index.view(-1), sv, bv, ov)
    return out


def index_add(inp, dim, index, src, alpha=1):
    logger.debug("GEMS_KUNLUNXIN INDEX_ADD")
    _payload_out = _ia_payload_launch(inp, dim, index, src, alpha)
    if _payload_out is not None:
        return _payload_out
    # Cheap host-side bound check. A full tensor compare against a freshly
    # allocated ones(...) tensor (the previous form) was measured at ~5ms per
    # call on XPU (cross-device compare + allocation + sync), i.e. the single
    # dominant fixed cost for small shapes. .all() is one device reduction +
    # one sync and preserves the assert semantics (0 <= index < size(dim)).
    assert bool(
        ((index >= 0) & (index < inp.size(dim))).all()
    ), "0 <= index < self.size(dim)"
    assert dim >= -inp.ndim and dim < inp.ndim, "Invalid dim"
    assert index.numel() == src.size(
        dim
    ), "The dimth dimension of source must have the same size as the length of index"
    assert (
        inp.ndim == src.ndim
    ), "Self and source should have the same number of dimensions"
    assert (
        ((inp.size(i) == src.size(i)) or i == dim) for i in range(0, inp.ndim)
    ), "src.size(d) == self.size(d) for all dimensions d != dim"

    inp = _contig_fast(inp)
    index = _contig_fast(index)
    src = _contig_fast(src)

    dim = dim % inp.ndim
    inp_len = inp.size(dim)
    N = index.numel()
    M = src.numel() // N if N > 0 else 0
    fine_dim = inp.ndim - 1
    if dim != fine_dim:
        # Same shape transform as dim_compress (move `dim` last + contiguous),
        # but with the vendor strided-copy engine instead of the gems copy_
        # override (perf: contiguous() on the permuted view is O(numel) Triton).
        order = [i for i in range(inp.ndim) if i != dim] + [dim]
        inp = _contig_fast(inp.permute(order))
        src = _contig_fast(src.permute(order))
    inp_cont = _clone_fast(inp)

    if N > 0 and M > 0:
        if _index_unique_gate(index, inp_len):
            # Index has no duplicates: exactly one writer per destination
            # cell, so the plain non-atomic load-add-store kernel is safe
            # (heuristics supply the bounded BLOCK_M/BLOCK_N at launch).
            grid = lambda meta: (
                triton.cdiv(M, min(index_add_fast_block_m, meta["BLOCK_M"])),
                triton.cdiv(N, min(index_add_fast_block_n, meta["BLOCK_N"])),
            )
            index_add_kernel[grid](index, src, inp_cont, M, N, alpha, inp_len)
        else:
            # Duplicate index values: deterministic segment reduction, no
            # atomics, no cross-program writes to the same cell.
            v_pad, perm_pad, head_pad, len_pad, U, len_max = _index_add_segments(
                index, index_add_seg_block_s, index_add_seg_block_c
            )
            grid = (M, triton.cdiv(U, index_add_seg_block_s))
            seg_args = (
                src,
                perm_pad,
                v_pad,
                head_pad,
                len_pad,
                inp_cont,
                M,
                N,
                inp_len,
                U,
                alpha,
            )
            if len_max <= index_add_seg_block_c:
                index_add_seg_kernel[grid](
                    *seg_args,
                    BLOCK_S=index_add_seg_block_s,
                    BLOCK_C=index_add_seg_block_c,
                )
            else:
                index_add_seg_loop_kernel[grid](
                    *seg_args,
                    BLOCK_S=index_add_seg_block_s,
                    BLOCK_C=index_add_seg_block_c,
                )
    if dim != fine_dim:
        order = [i for i in range(inp_cont.ndim - 1)]
        order.insert(dim, fine_dim)
        return _contig_fast(inp_cont.permute(order))
    else:
        return inp_cont


def index_add_(inp, dim, index, src, alpha=1):
    logger.debug("GEMS_KUNLUNXIN INDEX_ADD_")
    # Cheap host-side bound check (identical semantics to the previous
    # .equal(torch.ones(..., device="cuda")) form; the old form measured
    # ~5ms per call on XPU because it allocates a cross-device ones-tensor
    # and does a full elementwise compare, i.e. the dominant fixed cost for
    # small shapes). .all() is one device reduction + one sync. Guarded for
    # empty index: t.all() on a 0-element tensor goes through the vendor
    # `all` override which divides by zero.
    if index.numel() > 0:
        assert bool(
            ((index >= 0) & (index < inp.size(dim))).all()
        ), "0 <= index < self.size(dim)"
    assert dim >= -inp.ndim and dim < inp.ndim, "Invalid dim"
    assert index.numel() == src.size(
        dim
    ), "The dimth dimension of source must have the same size as the length of index"
    assert (
        inp.ndim == src.ndim
    ), "Self and source should have the same number of dimensions"
    assert (
        ((inp.size(i) == src.size(i)) or i == dim) for i in range(0, inp.ndim)
    ), "src.size(d) == self.size(d) for all dimensions d != dim"

    # Empty index is a no-op in torch semantics. Must come before the bound
    # check: t.all() on a 0-element tensor goes through the vendor `all`
    # override which divides by zero (and src.numel() // 0 would raise too).
    if index.numel() == 0:
        return inp

    # clone/contig via the vendor strided-copy engine (aten::_copy_from is
    # never overridden): torch.clone()/contiguous() funnel into the gems
    # copy_ override (Triton pointwise, ~3.4ms/1M elements on XPU).
    inp_cont = _clone_fast(inp)
    index = _contig_fast(index)
    src = _contig_fast(src)

    dim = dim % inp_cont.ndim
    inp_len = inp_cont.size(dim)
    N = index.numel()
    M = src.numel() // N if N > 0 else 0
    fine_dim = inp_cont.ndim - 1
    if dim != fine_dim:
        # Same shape transform as dim_compress (move `dim` last + contiguous),
        # but with the vendor strided-copy engine instead of the gems copy_
        # override (perf: contiguous() on the permuted view is O(numel) Triton).
        order = [i for i in range(inp_cont.ndim) if i != dim] + [dim]
        inp_cont = _contig_fast(inp_cont.permute(order))
        src = _contig_fast(src.permute(order))

    if N > 0 and M > 0:
        if _index_unique_gate(index, inp_len):
            # Index has no duplicates: exactly one writer per destination
            # cell, so the plain non-atomic load-add-store kernel is safe
            # (heuristics supply the bounded BLOCK_M/BLOCK_N at launch).
            grid = lambda meta: (
                triton.cdiv(M, min(index_add_fast_block_m, meta["BLOCK_M"])),
                triton.cdiv(N, min(index_add_fast_block_n, meta["BLOCK_N"])),
            )
            index_add_kernel[grid](index, src, inp_cont, M, N, alpha, inp_len)
        else:
            # Duplicate index values: deterministic segment reduction.
            v_pad, perm_pad, head_pad, len_pad, U, len_max = _index_add_segments(
                index, index_add_seg_block_s, index_add_seg_block_c
            )
            grid = (M, triton.cdiv(U, index_add_seg_block_s))
            seg_args = (
                src,
                perm_pad,
                v_pad,
                head_pad,
                len_pad,
                inp_cont,
                M,
                N,
                inp_len,
                U,
                alpha,
            )
            if len_max <= index_add_seg_block_c:
                index_add_seg_kernel[grid](
                    *seg_args,
                    BLOCK_S=index_add_seg_block_s,
                    BLOCK_C=index_add_seg_block_c,
                )
            else:
                index_add_seg_loop_kernel[grid](
                    *seg_args,
                    BLOCK_S=index_add_seg_block_s,
                    BLOCK_C=index_add_seg_block_c,
                )
    if dim != fine_dim:
        order = [i for i in range(inp_cont.ndim - 1)]
        order.insert(dim, fine_dim)
        inp_cont = _contig_fast(inp_cont.permute(order))
        torch.ops.aten._copy_from(inp_cont, inp, False)
        return inp
    else:
        torch.ops.aten._copy_from(inp_cont, inp, False)
        return inp
