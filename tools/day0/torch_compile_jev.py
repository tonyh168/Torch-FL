#!/usr/bin/env python3
"""Jev / OCR-RM-2B: eager vs `torch.compile(backend="flagos")`, one sample.

Prints the score from each path plus the compiled latency, which is what the day-0 notes
compare. `--sample` picks a manifest entry (default s02).

    python tools/day0/torch_compile_jev.py --attn sdpa --base ... --ckpt ...

`--no-compat` skips `torch_fl.compile.flaggems_dynamo_compat.install()` to show the first
failure an unpatched stack hits.
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import time

import torch_fl  # noqa: F401  — must precede torch on MetaX
import torch
from PIL import Image
from transformers import AutoProcessor, Qwen3_5ForSequenceClassification

PROMPT = (
    "<|im_start|>user\n"
    "<|vision_start|><|image_pad|><|vision_end|>OCR文本：\n{ocr}\n"
    "<|vision_start|><|image_pad|><|vision_end|>"
    "第一张图是原图，第二张图是上述 OCR 文本渲染得到的图片。\n"
    "请判断该 OCR 文本及其对应渲染图是否与原图内容一致。\n"
    "这是一个二分类任务：一致标为 1，不一致标为 0。<|im_end|>\n"
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True, help="delivery bundle (has samples/manifest.json)")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--sample", default="s02")
    ap.add_argument("--attn", default="sdpa", choices=["sdpa", "eager", "flash_attention_2"])
    ap.add_argument("--device", default="flagos:0")
    ap.add_argument("--no-compat", action="store_true")
    ap.add_argument("--mode", default=None, help="torch.compile mode")
    ap.add_argument("--reps", type=int, default=5)
    args = ap.parse_args()

    if not args.no_compat:
        from torch_fl.compile.flaggems_dynamo_compat import install

        print("compat:", install(), flush=True)

    samples = json.load(open(f"{args.base}/samples/manifest.json"))["samples"]
    sample = next(s for s in samples if s["name"] == args.sample)

    processor = AutoProcessor.from_pretrained(args.ckpt)
    model = Qwen3_5ForSequenceClassification.from_pretrained(
        args.ckpt, dtype=torch.bfloat16, attn_implementation=args.attn, device_map=None
    ).to(args.device).eval()

    ocr = open(f"{args.base}/{sample['ocr_text_file']}", encoding="utf-8").read().rstrip("\r\n")
    images = [Image.open(f"{args.base}/{sample['orig_image']}").convert("RGB"),
              Image.open(f"{args.base}/{sample['render_image']}").convert("RGB")]
    inputs = processor(text=[PROMPT.format(ocr=ocr)], images=images, return_tensors="pt")
    inputs = {k: (v.to(args.device) if hasattr(v, "to") else v) for k, v in inputs.items()}
    print(f"sample={args.sample} attn={args.attn} n_tokens={inputs['input_ids'].shape[1]}", flush=True)

    def run(mod):
        with torch.inference_mode():
            logits = mod(**inputs).logits.float()
        torch.flagos.synchronize()
        return float(torch.softmax(logits, dim=-1)[0, 1])

    def timed(mod) -> list[float]:
        out = []
        for _ in range(args.reps):
            t = time.perf_counter()
            run(mod)
            out.append((time.perf_counter() - t) * 1e3)
        return out

    eager_prob = run(model)
    eager_ms = statistics.median(timed(model))
    print(f"eager    prob_1={eager_prob:.8f}  median={eager_ms:.1f} ms", flush=True)

    opts = {"mode": args.mode} if args.mode else {}
    t = time.perf_counter()
    compiled = torch.compile(model, backend="flagos", **opts)
    try:
        run(compiled)
    except Exception:  # noqa: BLE001
        import traceback

        traceback.print_exc()
        print("COMPILE FAILED", flush=True)
        raise SystemExit(1)
    compile_s = time.perf_counter() - t

    compiled_prob = run(compiled)
    compiled_ms = statistics.median(timed(compiled))
    ref = sample["ref_prob_1"]
    print(f"compiled prob_1={compiled_prob:.8f}  median={compiled_ms:.1f} ms  "
          f"(compile {compile_s:.1f} s)", flush=True)
    print(f"delta vs eager = {compiled_prob - eager_prob:+.2e}   "
          f"delta vs ref = {compiled_prob - ref:+.2e}   "
          f"speedup = {eager_ms / compiled_ms:.2f}x", flush=True)


if __name__ == "__main__":
    main()
