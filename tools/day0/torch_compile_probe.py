#!/usr/bin/env python3
"""Minimal probes for `torch.compile(backend="flagos")` on a FlagGems-served model.

One case per defect; each prints REPRODUCED / NOT REPRODUCED plus the tail of the error so
the output can be pasted into an upstream issue.

    python tools/day0/torch_compile_probe.py            # all cases
    python tools/day0/torch_compile_probe.py -k ext-alias
"""
from __future__ import annotations

import argparse
import traceback

import torch_fl  # noqa: F401  — must precede torch on MetaX (cu-bridge ABI)
import torch

DEV = "flagos:0"


def _report(name: str, fn) -> None:
    print(f"\n=== {name} ===", flush=True)
    try:
        fn()
    except Exception:  # noqa: BLE001
        tb = traceback.format_exc()
        print("REPRODUCED")
        print("\n".join(tb.strip().splitlines()[-14:]))
        return
    print("NOT REPRODUCED")


def case_ext_alias() -> None:
    """A FlagGems kernel body that spells `ext.program_id(0)` under inductor capture."""

    def f(x):
        return torch.repeat_interleave(x, 2)

    x = torch.arange(1024, device=DEV, dtype=torch.int64)
    eager = f(x).sum().item()
    out = torch.compile(f, backend="flagos")(x)
    torch.flagos.synchronize()
    compiled = out.sum().item()
    print(f"eager={eager} compiled={compiled}")
    assert eager == compiled


def case_device_fn() -> None:
    """`torch_device_fn` is a torch.cuda stand-in; two attributes are dummy stubs."""
    from flag_gems.runtime import torch_device_fn

    print("is_current_stream_capturing:", torch_device_fn.is_current_stream_capturing())
    with torch_device_fn.device(DEV):
        print("device(flagos:0) entered")


def case_fake_tensor_sdpa() -> None:
    """SDPA on flagos bf16 + a following view: fake propagation is missing."""
    import torch.nn.functional as F

    def f(q, k, v):
        out = F.scaled_dot_product_attention(q, k, v, is_causal=False)
        return out.transpose(1, 2).contiguous()

    q = torch.randn(1, 16, 280, 64, device=DEV, dtype=torch.bfloat16)
    print("eager:", tuple(f(q, q, q).shape))
    out = torch.compile(f, backend="flagos")(q, q, q)
    torch.flagos.synchronize()
    print("compiled:", tuple(out.shape))


def case_device_index() -> None:
    """`torch.cuda._get_device_index` rejects a flagos device."""
    print(torch.cuda._get_device_index("flagos:0"))


CASES = {
    "ext-alias": case_ext_alias,
    "device-fn": case_device_fn,
    "fake-tensor-sdpa": case_fake_tensor_sdpa,
    "device-index": case_device_index,
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("-k", "--case", choices=sorted(CASES), action="append")
    ap.add_argument("--with-compat", action="store_true",
                    help="install torch_fl.compile.flaggems_dynamo_compat first")
    args = ap.parse_args()

    print(f"torch {torch.__version__}  flagos devs {torch.flagos.device_count()}")

    if args.with_compat:
        from torch_fl.compile.flaggems_dynamo_compat import install

        print("compat:", install())

    for name in args.case or list(CASES):
        _report(name, CASES[name])


if __name__ == "__main__":
    main()
