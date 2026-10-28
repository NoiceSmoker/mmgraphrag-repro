"""Serialise image nodes *without* VLM-generated text (extra ablation).

In the official pipeline an image node is embedded as
    pixels + "title [SEP] section [SEP] caption [SEP] <Qwen2.5-VL summary>"
and every sub-image (object crop) replaces the human caption by a
Qwen2.5-VL caption of the crop and re-uses the page summary. The paper
describes image nodes as embedded "as images", so this ablation measures how
much of LILaC's image retrieval relies on the generated text:

    image     -> pixels + "title [SEP] section [SEP] human caption"
    subimage  -> crop pixels + "title [SEP] section [SEP] human caption of the parent image"

Output: ``embeddings/serializations/{image,subimage}+image.json``; embed them
with the official runner into ``image_novlm.pt`` / ``subimage_novlm.pt`` (see
scripts/lilac/50_novlm_ablation.sh).
"""
from __future__ import annotations

import argparse
import copy
import os

from mmrag.common.io import read_json, write_json
from mmrag.lilac import official


def run(dataset: str) -> None:
    official.ensure_importable()
    from src.lilac.basic_class.image import Image  # official serializer
    from src.utils.utils import read_yaml, REPO_ROOT, parsed_documents_path

    cfg = read_yaml(f"{REPO_ROOT}/config/lcg_constructor/a.yaml")
    pd_dir = parsed_documents_path(cfg, dataset)
    images_dir = official.dataset_dir(dataset) / "image_components" / "dev"
    sub_dir = official.artifact_dir(dataset) / "image_components_sub" / "dev"
    summaries = official.artifact_dir(dataset) / "image_summaries" / "dev"
    out_dir = official.artifact_dir(dataset) / "embeddings" / "serializations"

    img_rows, sub_rows = [], []
    for fn in sorted(os.listdir(pd_dir)):
        doc = read_json(os.path.join(pd_dir, fn))
        title, hier = doc["title"], doc["hierarchy"]
        for cid, comp in doc.get("image", {}).items():
            img = Image(fn, title, hier, cid, comp, images_dir=str(images_dir),
                        image_summaries_dir=str(summaries))
            img_rows.append(img.serialize(["image"]))
        for sid, comp in doc.get("subimage", {}).items():
            parent_id = sid.rsplit("_", 1)[0]
            parent = doc.get("image", {}).get(parent_id, {})
            comp = copy.deepcopy(comp)
            comp["caption"] = copy.deepcopy(parent.get("caption", {"text": " "}))
            img = Image(fn, title, hier, sid, comp, images_dir=str(sub_dir),
                        image_summaries_dir=str(summaries))
            sub_rows.append(img.serialize(["image"]))
    write_json(img_rows, out_dir / "image+image.json", indent=1)
    write_json(sub_rows, out_dir / "subimage+image.json", indent=1)
    print(f"[novlm] {len(img_rows)} images, {len(sub_rows)} sub-images -> {out_dir}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default="MMCoQA")
    run(ap.parse_args().dataset)


if __name__ == "__main__":
    main()
