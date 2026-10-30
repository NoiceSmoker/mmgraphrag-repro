"""Batched Qwen2.5-VL passes for LILaC graph construction (vLLM).

The official pipeline runs Qwen2.5-VL-7B through Hugging Face ``generate``
one image at a time for three passes:

  step 0   page-level summary of every image          -> image_summaries/dev/<stem>.txt
  step 5a  open-vocabulary detection + crops           -> image_components_sub/dev/<stem>/{detections.json, i_cls.png}
  step 5b  caption of every crop                       -> image_components_sub/dev/<stem>/i_cls.txt

This module produces the same files with the same prompts, the same model
and greedy decoding, but batches images through vLLM. Afterwards the
official ``step5_add_subimages.py --mode add_to_parsed_documents`` registers
the crops in the parsed documents, unchanged.

One deliberate difference (configurable): images are resized to at most
``--max-pixels`` (default 1280*28*28 ~ 1.0 MP) before entering the vision
encoder. Qwen2.5-VL returns boxes in the resized frame, so boxes are mapped
back to the original resolution before cropping (the official code crops
with the raw model coordinates). Pass ``--max-pixels 12845056`` to use the
processor default of the official run.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from mmrag.common.gpu import vllm_dtype
from mmrag.lilac import official

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
# Prompts of the pinned official code (step0 summary / step5 detection / step5 crop caption).
SUMMARY_PROMPT = "Generate a summary of the given image. Include all the texts within the image."
DETECT_PROMPT = ("Detect all objects in the image and return ONLY a JSON list "
                 "of {class, bbox_2d:[x1,y1,x2,y2]}. Do NOT include markdown or extra text.")
CROP_PROMPT = "Generate a summary of the given image region. Include all the texts within it."

QWEN_VL_TEMPLATE = (
    "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
    "<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>{prompt}<|im_end|>\n"
    "<|im_start|>assistant\n"
)


def smart_resize(height: int, width: int, factor: int = 28, min_pixels: int = 56 * 56,
                 max_pixels: int = 12845056) -> tuple[int, int]:
    """Qwen2.5-VL resizing rule (qwen_vl_utils.smart_resize)."""
    h_bar = max(factor, round(height / factor) * factor)
    w_bar = max(factor, round(width / factor) * factor)
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = max(factor, math.floor(height / beta / factor) * factor)
        w_bar = max(factor, math.floor(width / beta / factor) * factor)
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = math.ceil(height * beta / factor) * factor
        w_bar = math.ceil(width * beta / factor) * factor
    return h_bar, w_bar


def extract_json_objects(s: str) -> list[dict]:
    """Same balanced-brace scanner as the official ``_extract_json_objects``."""
    objs, depth, start = [], 0, None
    for i, ch in enumerate(s):
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}" and depth:
            depth -= 1
            if depth == 0 and start is not None:
                try:
                    objs.append(json.loads(s[start : i + 1]))
                except json.JSONDecodeError:
                    pass
                start = None
    return objs


def rescale_box(box, orig_wh: tuple[int, int], resized_hw: tuple[int, int]) -> tuple[int, int, int, int]:
    w, h = orig_wh
    rh, rw = resized_hw
    sx, sy = w / rw, h / rh
    x1, y1, x2, y2 = (float(v) for v in box)
    x1, x2 = sorted((x1 * sx, x2 * sx))
    y1, y2 = sorted((y1 * sy, y2 * sy))
    return (max(0, int(x1)), max(0, int(y1)), min(w, int(round(x2))), min(h, int(round(y2))))


class QwenVL:
    def __init__(self, model: str, max_pixels: int, max_model_len: int = 8192,
                 gpu_memory_utilization: float = 0.9, max_num_seqs: int = 16):
        from vllm import LLM

        self.max_pixels = max_pixels
        self.llm = LLM(model=model, max_model_len=max_model_len, max_num_seqs=max_num_seqs,
                       gpu_memory_utilization=gpu_memory_utilization, dtype=vllm_dtype(),
                       limit_mm_per_prompt={"image": 1},
                       mm_processor_kwargs={"min_pixels": 56 * 56, "max_pixels": max_pixels})

    def run(self, images, prompt: str, max_tokens: int) -> list[str]:
        from vllm import SamplingParams

        sp = SamplingParams(temperature=0.0, repetition_penalty=1.05, max_tokens=max_tokens)
        reqs = [{"prompt": QWEN_VL_TEMPLATE.format(prompt=prompt), "multi_modal_data": {"image": im}}
                for im in images]
        return [o.outputs[0].text for o in self.llm.generate(reqs, sampling_params=sp, use_tqdm=True)]


def _load_rgb(path: Path):
    from PIL import Image

    with Image.open(path) as im:
        return im.convert("RGB")


def list_images(dataset: str) -> list[Path]:
    img_dir = official.dataset_dir(dataset) / "image_components" / "dev"
    return sorted(p for p in img_dir.iterdir() if p.is_file() and p.suffix.lower() in IMG_EXTS)


def summarize(vl: QwenVL, dataset: str, chunk: int, max_tokens: int) -> None:
    out_dir = official.artifact_dir(dataset) / "image_summaries" / "dev"
    out_dir.mkdir(parents=True, exist_ok=True)
    todo = [p for p in list_images(dataset) if not (out_dir / f"{p.stem}.txt").exists()]
    print(f"[summaries] {len(todo)} images to caption")
    for i in range(0, len(todo), chunk):
        batch, imgs = [], []
        for p in todo[i : i + chunk]:
            try:
                imgs.append(_load_rgb(p))
                batch.append(p)
            except Exception as exc:
                print(f"[summaries] skip unreadable {p.name}: {exc}")
        for p, txt in zip(batch, vl.run(imgs, SUMMARY_PROMPT, max_tokens)):
            (out_dir / f"{p.stem}.txt").write_text(txt.strip(), encoding="utf-8")


def detect(vl: QwenVL, dataset: str, chunk: int, max_tokens: int) -> None:
    sub_root = official.artifact_dir(dataset) / "image_components_sub" / "dev"
    sub_root.mkdir(parents=True, exist_ok=True)
    todo = [p for p in list_images(dataset) if not (sub_root / p.stem / "detections.json").exists()]
    print(f"[detect] {len(todo)} images")
    for i in range(0, len(todo), chunk):
        batch, imgs = [], []
        for p in todo[i : i + chunk]:
            try:
                imgs.append(_load_rgb(p))
                batch.append(p)
            except Exception as exc:
                print(f"[detect] skip unreadable {p.name}: {exc}")
        outs = vl.run(imgs, DETECT_PROMPT, max_tokens)
        for p, im, txt in zip(batch, imgs, outs):
            dets = extract_json_objects(txt)
            out_dir = sub_root / p.stem
            out_dir.mkdir(parents=True, exist_ok=True)
            resized = smart_resize(im.height, im.width, max_pixels=vl.max_pixels)
            for k, det in enumerate(dets, start=1):
                box = det.get("bbox_2d") or det.get("bbox")
                cls = det.get("class", "object")
                if not box or len(box) != 4:
                    continue
                try:
                    x1, y1, x2, y2 = rescale_box(box, (im.width, im.height), resized)
                    if x2 <= x1 or y2 <= y1:
                        continue
                    im.crop((x1, y1, x2, y2)).save(out_dir / f"{k}_{cls}.png")
                except Exception as exc:  # e.g. class names containing '/'
                    print(f"[detect] crop error {p.name} #{k}: {exc}")
            tmp = out_dir / "detections.json.tmp"
            tmp.write_text(json.dumps(dets, indent=2), encoding="utf-8")
            tmp.replace(out_dir / "detections.json")


def caption_crops(vl: QwenVL, dataset: str, chunk: int, max_tokens: int) -> None:
    sub_root = official.artifact_dir(dataset) / "image_components_sub" / "dev"
    crops = [c for d in sorted(sub_root.iterdir()) if d.is_dir()
             for c in sorted(d.glob("*.png")) if c.name.lower() != "boxed.png"
             and not c.with_suffix(".txt").exists()]
    print(f"[crop-captions] {len(crops)} crops")
    for i in range(0, len(crops), chunk):
        batch, imgs = [], []
        for c in crops[i : i + chunk]:
            try:
                imgs.append(_load_rgb(c))
                batch.append(c)
            except Exception as exc:
                print(f"[crop-captions] skip {c}: {exc}")
        for c, txt in zip(batch, vl.run(imgs, CROP_PROMPT, max_tokens)):
            c.with_suffix(".txt").write_text(txt.strip(), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("passes", nargs="+", choices=["summarize", "detect", "caption-crops"])
    ap.add_argument("--dataset", default="MMCoQA")
    ap.add_argument("--model", default="Qwen/Qwen2.5-VL-7B-Instruct")
    ap.add_argument("--max-pixels", type=int, default=1280 * 28 * 28)
    ap.add_argument("--chunk", type=int, default=64, help="images per vLLM call (results are flushed per chunk)")
    ap.add_argument("--gpu-mem", type=float, default=0.9)
    ap.add_argument("--allow-model-change", action="store_true",
                    help="resume although earlier outputs came from another VLM checkpoint")
    args = ap.parse_args()
    # Outputs of different VLM weights must not be mixed (resumed runs skip finished images).
    out = official.artifact_dir(args.dataset)
    out.mkdir(parents=True, exist_ok=True)
    meta_path = out / "vlm_meta.json"
    if meta_path.exists() and not args.allow_model_change:
        before = json.loads(meta_path.read_text()).get("model")
        if before and before != args.model:
            raise SystemExit(f"VLM outputs in {out} were produced by {before}, but this run uses {args.model}. "
                             "Delete the artefacts (or start a fresh Drive store) or pass --allow-model-change.")
    vl = QwenVL(args.model, args.max_pixels, gpu_memory_utilization=args.gpu_mem)
    # record which VLM produced the texts (the experiment runs log it to W&B)
    (out / "vlm_meta.json").write_text(json.dumps({"model": args.model, "dtype": vllm_dtype(),
                                                   "max_pixels": args.max_pixels, "passes": args.passes}, indent=1))
    for p in args.passes:
        if p == "summarize":
            summarize(vl, args.dataset, args.chunk, max_tokens=2048)
        elif p == "detect":
            detect(vl, args.dataset, args.chunk, max_tokens=2048)
        else:
            caption_crops(vl, args.dataset, args.chunk, max_tokens=512)


if __name__ == "__main__":
    main()
