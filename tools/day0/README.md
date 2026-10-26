# Day-0 staging: `torch.compile(backend="flagos")` with FlagGems enabled

Scratch branch for the investigations behind
[flagos-ai/day0](https://github.com/flagos-ai) model bring-ups on MetaX C550 and Hygon DCU.
**Nothing here is proposed as a fix yet** — it is a staging area so the probes, the exact
failures, and the reproduction steps live next to the code they exercise.

Model under test: **Jev / OCR-RM-2B** (Qwen3.5-VL 2B + sequence-classification value head),
12 delivery samples, batch=1, bf16, `attn_implementation="sdpa"`.

## What is here

| Path | What it does |
|---|---|
| `torch_fl/compile/flaggems_dynamo_compat.py` | Five scoped patches that let dynamo trace a model whose ops are still served by FlagGems (no op is pinned back to `cuda`) |
| `tools/day0/torch_compile_probe.py` | Minimal probes, one per defect — safe to run on any vendor |
| `tools/day0/torch_compile_jev.py` | End-to-end probe on the Jev model (eager vs compiled, numerics + latency) |

## Results (2026-10-26)

| Platform | attention | compiles? | latency (s02) | numerics |
|---|---|---|---|---|
| Hygon DCU (`torch 2.10.0+cpu`, `torch_fl 0.1.0+dtk`) | sdpa | yes, 18.2 s | **440 ms** (eager 340 ms) | correct, Δ=+7.4e-05 |
| MetaX C550 (`torch 2.10.0+cpu`, `torch_fl 2.10.0+metax3.8.0`) | eager | yes, 176 s | **86–97 ms** (eager 347 ms) | **wrong** — `prob_1` 0.0614 vs 0.9954 |
| MetaX C550 | sdpa | no | — | fails in fake-tensor propagation |

So "FlagGems and `torch.compile` are mutually exclusive" is not the case — Hygon compiles
and is numerically right. The open problems are the ones below.

## Defects, each with a minimal trigger

Run `python tools/day0/torch_compile_probe.py --case <name>`.

### `ext-alias` (all vendors)

Inductor captures a FlagGems kernel body as a custom Triton kernel but does not carry the
module-level import, so the generated source fails:

```
NameError: ext is not defined
  pid = ext.program_id(0)
```

Seen inside the Jev trace (`flag_gems/ops/repeat_interleave.py:80`). Standalone
`torch.repeat_interleave` under `torch.compile` does **not** reproduce it, so the trigger is
the custom-kernel capture path rather than the op itself — the probe therefore reports
"not reproduced" and the evidence is the generated module
(`/tmp/torchinductor_*/**/*.py`, header `# Original path: .../flag_gems/ops/repeat_interleave.py:79`).

### `device-fn` (Hygon)

`flag_gems.utils.pointwise_dynamic` and friends use `torch_device_fn` as a `torch.cuda`-like
handle. On the CPU-only wheel that Torch-FL pairs with, two of its attributes are `err_fn`
stubs and `device()` rejects `flagos`:

```
RuntimeError: Tried to instantiate dummy base class _cuda_isCurrentStreamCapturing
ValueError: Expected a cuda device, but got: flagos:0
```

This reproduces in plain eager code; `torch.compile` only makes it fatal because dynamo
traces into it.

### `fake-tensor-sdpa` (MetaX, and Hygon pre-boundary)

```
torch._dynamo.exc.TorchRuntimeError: Dynamo failed to run FX node with fake tensors:
  call_function <function sum_dim_comm ...> ...: got RuntimeError(
  "Cannot access data pointer of Tensor (e.g. FakeTensor, FunctionalTensor)")
```

Trigger: `F.scaled_dot_product_attention` on `flagos:0` bf16 under `torch.compile`, followed
by `transpose(1,2).contiguous()`. The transpose itself traces fine; the failure is that the
SDPA result's fake propagation is missing, so the following view/materialisation walks into
`data_ptr()` (Hygon) or `from_meta_and_device` (MetaX, `tensor's device must be "meta", got
flagos instead`).

### `wrong-numerics` (MetaX, unresolved)

With `attn_implementation="eager"` the compiled model runs 4x faster and returns
`prob_1 = 0.0614` where eager returns `0.9954`. Not minimised. Until it is, the MetaX
compiled path is unusable and its latency number must not be quoted.

## Reproducing

```bash
# Hygon DCU (container from the Jev bring-up, corex bind-mounts required)
python tools/day0/torch_compile_jev.py --attn sdpa \
  --base /public-flash/models/Jev-OCR-RM-2B/rm_hf_infer_demo_v1_20261015 \
  --ckpt /public-flash/models/Jev-OCR-RM-2B/qwen35_2b_l2_triple_value_checkpoint-3000

# MetaX C550 (activate the Torch-FL venv first)
python tools/day0/torch_compile_jev.py --attn eager \
  --base /models/Jev-OCR-RM-2B/rm_hf_infer_demo_v1_20261015 \
  --ckpt /models/Jev-OCR-RM-2B/qwen35_2b_l2_triple_value_checkpoint-3000
```
