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
"""Staging compatibility layer: `torch.compile(backend="flagos")` with FlagGems enabled.

Status: **staging** — this is a collection of diagnostics gathered on MetaX C550 and
Hygon DCU while trying to compile a real model (Jev / OCR-RM-2B, Qwen3.5-VL + seq-cls)
with FlagGems left fully enabled. It is *not* a proposed fix: every patch below works
around a place where FlagGems' Python implementation and TorchDynamo disagree, and the
real fixes belong in FlagGems (opaque custom ops with fake impls) and in torch_fl
(`torch_device_fn` stubs, FakeTensor support). See `tools/day0/README.md` for the
measurements and the failures that remain.

Nothing here changes operator routing: no op is pinned back to `cuda`, and no flag is
set to disable FlagGems.

The patches, in the order they are needed:

1. ``ext`` alias — inductor captures a FlagGems kernel body as a custom Triton kernel but
   drops the module-level ``from flag_gems.utils import triton_lang_extension as ext``, so
   the generated source raises ``NameError('ext is not defined')``.
2. ``torch.cuda._get_device_index`` — FlagGems guards work with ``torch_device_fn.device(...)``,
   which routes into ``torch.cuda``; the CPU-only wheel rejects ``flagos:N``.
3. ``torch_device_fn`` proxy — on a CPU-only torch the ``is_current_stream_capturing`` and
   ``_DeviceGuard`` attributes are dummy classes that raise on instantiation.
4. Dynamo boundary — FlagGems entry points are marked ``torch._dynamo.disable`` so dynamo
   breaks the graph there instead of tracing into ``.item()`` / ``data_ptr()`` /
   ``get_device_properties()``, none of which can run on fake tensors.
5. Optional: ``get_rope_index`` compat (``len() of a 0-d tensor``) — same defect the Hygon
   bring-up hit in eager mode; needed before the compiled path can even reach the graph.

Usage::

    import torch_fl  # first on MetaX
    from torch_fl.compile.flaggems_dynamo_compat import install
    install()                    # idempotent; returns a dict of what was applied

    model = torch.compile(model, backend="flagos")

Caveats measured on 2026-10-26 (see tools/day0/README.md):

* Hygon DCU: compiles and is numerically correct, but **slower** than eager
  (440 ms vs 340 ms for Jev s02) — the graph is cut at every FlagGems call.
* MetaX C550: ``sdpa`` still fails inside fake tensor propagation; ``eager`` attention
  compiles and is ~4x faster but produced a **wrong** result (0.0614 vs 0.9954).
  Do not use the compiled path on MetaX until that is root-caused.
"""

from __future__ import annotations

import sys
from typing import Any

import torch

__all__ = ["install", "install_ext_alias_patch", "install_device_index_patch",
           "install_device_fn_proxy", "install_dynamo_boundary", "install_rope_index_compat"]

_APPLIED: dict[str, Any] = {}


# --------------------------------------------------------------------------------------
# 1. `ext` alias
# --------------------------------------------------------------------------------------
def install_ext_alias_patch() -> bool:
    """Re-add the ``ext`` import to captured FlagGems kernel sources.

    Inductor's custom-kernel capture copies the ``@triton.jit`` function body but not the
    module-level imports, so any FlagGems kernel that spells ``ext.program_id(0)`` fails to
    compile. Prepending the same import restores the exact semantics the kernel was written
    with; nothing else about the source is touched.
    """
    if _APPLIED.get("ext_alias"):
        return True
    try:
        import torch._inductor.async_compile as ac
    except Exception:  # noqa: BLE001
        return False

    original = ac.AsyncCompile.triton
    if getattr(original, "_flagos_ext_alias", False):
        _APPLIED["ext_alias"] = True
        return True

    def triton(self, kernel_name, source_code, device_str="cuda"):
        if (isinstance(source_code, str) and "ext." in source_code
                and "as ext" not in source_code):
            source_code = (
                "from flag_gems.utils import triton_lang_extension as ext\n" + source_code
            )
        return original(self, kernel_name, source_code, device_str)

    triton._flagos_ext_alias = True
    ac.AsyncCompile.triton = triton
    _APPLIED["ext_alias"] = True
    return True


# --------------------------------------------------------------------------------------
# 2. `_get_device_index` accepts `flagos:N`
# --------------------------------------------------------------------------------------
def install_device_index_patch() -> bool:
    """Teach ``torch.cuda._get_device_index`` about ``flagos`` device strings/objects."""
    if _APPLIED.get("device_index"):
        return True
    try:
        import torch.cuda._utils as cu
    except Exception:  # noqa: BLE001
        return False

    original = cu._get_device_index
    if getattr(original, "_flagos_aware", False):
        _APPLIED["device_index"] = True
        return True

    def _get_device_index(device, optional=True, allow_cpu=False):
        if isinstance(device, torch.device) and device.type == "flagos":
            idx = device.index
            return torch.flagos.current_device() if idx is None else idx
        if isinstance(device, str) and device.startswith("flagos"):
            return int(device.split(":")[1]) if ":" in device else torch.flagos.current_device()
        return original(device, optional, allow_cpu)

    _get_device_index._flagos_aware = True
    cu._get_device_index = _get_device_index
    torch.cuda._get_device_index = _get_device_index
    _APPLIED["device_index"] = True
    return True


# --------------------------------------------------------------------------------------
# 3. `torch_device_fn` proxy
# --------------------------------------------------------------------------------------
class _NoOpDeviceGuard:
    def __init__(self, *args, **kwargs):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


class _NoOpDevice:
    def __init__(self, device=None):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


class _DeviceFnProxy:
    """Wrap the vendor ``torch.cuda`` handle, overriding what a CPU-only wheel stubs out.

    ``flag_gems.runtime.torch_device_fn`` is a ``torch.cuda``-like object assembled at import
    time. Under the CPU-only torch that Torch-FL pairs with, ``is_current_stream_capturing``
    and ``_DeviceGuard`` are ``_utils.err_fn`` stubs (instantiation raises), and ``device()``
    only accepts CUDA device specs. Everything else is forwarded unchanged.
    """

    def __init__(self, inner):
        object.__setattr__(self, "_inner", inner)

    def device(self, device=None):
        return _NoOpDevice(device)

    def is_current_stream_capturing(self):
        return False

    @property
    def _DeviceGuard(self):
        return _NoOpDeviceGuard

    def __getattr__(self, name):
        return getattr(object.__getattribute__(self, "_inner"), name)


def install_device_fn_proxy() -> int:
    """Swap ``torch_device_fn`` in ``flag_gems.runtime`` and every already-imported module.

    Module-level ``from flag_gems.runtime import torch_device_fn`` binds the object, so the
    replacement has to be written into every module that already imported it. Returns the
    number of modules rewritten.
    """
    if _APPLIED.get("device_fn") is not None:
        return _APPLIED["device_fn"]
    try:
        import flag_gems.runtime as fgr
    except Exception:  # noqa: BLE001
        return 0

    inner = getattr(fgr, "torch_device_fn", None)
    if inner is None:
        return 0
    if isinstance(inner, _DeviceFnProxy):
        _APPLIED["device_fn"] = -1
        return -1

    proxy = _DeviceFnProxy(inner)
    fgr.torch_device_fn = proxy
    count = 0
    for name, module in list(sys.modules.items()):
        if not name.startswith("flag_gems") or module is None:
            continue
        if getattr(module, "torch_device_fn", None) is inner:
            setattr(module, "torch_device_fn", proxy)
            count += 1
    _APPLIED["device_fn"] = count
    return count


# --------------------------------------------------------------------------------------
# 4. Dynamo boundary
# --------------------------------------------------------------------------------------
def install_dynamo_boundary() -> int:
    """Mark imported FlagGems / Triton entry points as untraceable.

    Dynamo inlines ordinary Python callables, so it walks straight into FlagGems kernels and
    trips on ``Tensor.item()``, ``data_ptr()`` and ``torch.cuda.get_device_properties()``.
    ``torch._dynamo.disable`` turns each of them into a graph break: they run eagerly on real
    tensors, and inductor still compiles the pieces around them.

    The scan deliberately uses ``sys.modules`` rather than ``pkgutil``: importing every
    ``flag_gems.runtime.backend._*`` package pulls in other vendors' kernels and fails on
    the current stack (e.g. Kunlunxin's ``triton.autotune(generate_configs=...)``).
    """
    if _APPLIED.get("dynamo_boundary") is not None:
        return _APPLIED["dynamo_boundary"]

    count = 0
    seen: set[int] = set()
    for name, module in list(sys.modules.items()):
        if not name.startswith("flag_gems.") or module is None:
            continue
        for obj in list(vars(module).values()):
            if id(obj) in seen or isinstance(obj, type):
                continue
            if callable(obj):
                try:
                    torch._dynamo.disable(obj, recursive=True)
                    seen.add(id(obj))
                    count += 1
                except Exception:  # noqa: BLE001
                    pass
    try:
        import triton.runtime.autotuner as autotuner
        import triton.runtime.jit as triton_jit

        for raw in (autotuner.Autotuner.run, triton_jit.JITFunction.run):
            torch._dynamo.disable(raw)
            count += 1
    except Exception:  # noqa: BLE001
        pass
    _APPLIED["dynamo_boundary"] = count
    return count


# --------------------------------------------------------------------------------------
# 5. get_rope_index compat
# --------------------------------------------------------------------------------------
def install_rope_index_compat() -> list[str]:
    """``torch.tensor([<flagos 0-d tensor>])`` raises ``len() of a 0-d tensor``.

    Scoped to ``get_rope_index`` only: replacing ``torch.tensor`` process-wide corrupts
    unrelated ops (see the Hygon bring-up notes).
    """
    if _APPLIED.get("rope_index") is not None:
        return _APPLIED["rope_index"]
    try:
        import transformers.models.qwen3_5.modeling_qwen3_5 as q35
    except Exception:  # noqa: BLE001
        _APPLIED["rope_index"] = []
        return []

    original_tensor = torch.tensor

    def _tensor_compat(data=None, *args, **kwargs):
        if isinstance(data, (list, tuple)) and any(torch.is_tensor(x) for x in data):
            try:
                return original_tensor(data, *args, **kwargs)
            except TypeError as exc:
                if "0-d tensor" not in str(exc):
                    raise
                data = [x.item() if torch.is_tensor(x) else x for x in data]
        return original_tensor(data, *args, **kwargs)

    patched = []
    for cls in list(vars(q35).values()):
        if not isinstance(cls, type) or "get_rope_index" not in cls.__dict__:
            continue
        original = cls.__dict__["get_rope_index"]

        def wrapper(self, *args, _orig=original, **kwargs):
            prev = torch.tensor
            torch.tensor = _tensor_compat
            try:
                return _orig(self, *args, **kwargs)
            finally:
                torch.tensor = prev

        setattr(cls, "get_rope_index", wrapper)
        patched.append(cls.__name__)
    _APPLIED["rope_index"] = patched
    return patched


# --------------------------------------------------------------------------------------
def install(*, rope_compat: bool = True) -> dict[str, Any]:
    """Apply every patch and report what happened. Idempotent."""
    install_ext_alias_patch()
    install_device_index_patch()
    install_device_fn_proxy()
    if rope_compat:
        install_rope_index_compat()
    install_dynamo_boundary()
    return dict(_APPLIED)
