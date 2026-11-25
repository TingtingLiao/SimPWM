"""--link style LTX dataset (ltx-trainer convention) from GR1 proxy items (gr1/run.py output, one directory per item).

  OUT/videos/<idx>.mp4  -> item video.mp4 (real clip, 1280x768, 10 fps)
  OUT/refs/<idx>.mp4    -> item ref.mp4 (RGB proxy, 800x480)
  OUT/first/<idx>.png   -> item first.png (1280x768)
  OUT/dataset.jsonl     {caption, video, reference_video, idx} rows
Modes (ITEMS: the gr1/select.py item list of the mode):
  train  idx = episode; items with arm IoU > --min-iou, in ITEMS order (round-robin over sessions), the first
         --max-items
  val    as train, skipping the episodes of the --exclude dataset (the training set): held-out validation samples
  eval   idx = test window k (w<k> item directories), every item

usage: python -m simpwm.video.dataset_gr1 {train,val,eval} PX_DIR OUT --items ITEMS_JSON [--max-items N] [--min-iou F]
           [--exclude DS]
"""

import json
import os
from pathlib import Path

from simpwm.video.common import parser, read_jsonl, write_jsonl

MAX_ITEMS = {"train": 3000, "val": 4}
MIN_IOU = 0.6
LINKS = (("videos", "video.mp4", "mp4"), ("refs", "ref.mp4", "mp4"), ("first", "first.png", "png"))


def build(mode, px, out, items, max_items=None, min_iou=MIN_IOU, exclude=()):
    px, out = Path(px).resolve(), Path(out)
    for d in ("videos", "refs", "first"):
        (out / d).mkdir(parents=True, exist_ok=True)
    rows, ious = [], []
    for it in items:
        d = px / (f"w{it['k']:03d}" if mode == "eval" else str(it["ep"]))
        idx = it["k"] if mode == "eval" else it["ep"]
        if idx in exclude or not (d / "info.json").exists():
            continue
        info = json.loads((d / "info.json").read_text())
        if info.get("skipped") or not (d / "ref.mp4").exists():
            continue
        if mode != "eval" and not (info["arm_iou"] is not None and info["arm_iou"] > min_iou):
            continue
        for sub, src, ext in LINKS:
            dst = out / sub / f"{idx}.{ext}"
            if not dst.exists():
                os.symlink(d / src, dst)
        rows.append(dict(caption=info["caption"], video=f"videos/{idx}.mp4", reference_video=f"refs/{idx}.mp4",
                         idx=idx))
        ious.append(info["arm_iou"])
        if max_items and len(rows) >= max_items:
            break
    write_jsonl(out / "dataset.jsonl", rows)
    print(f"{len(rows)} rows -> {out / 'dataset.jsonl'}; arm IoU mean "
          f"{sum(i or 0 for i in ious) / max(len(ious), 1):.3f}")


def main():
    p = parser(__doc__)
    p.add_argument("mode", choices=["train", "val", "eval"])
    p.add_argument("px", help="proxy item directory (gr1/run.py output)")
    p.add_argument("out", help="dataset directory to write")
    p.add_argument("--items", required=True, type=Path, help="item list of the mode (gr1/select.py output)")
    p.add_argument("--max-items", type=int, help="rows to keep (default: train 3000, val 4, eval all)")
    p.add_argument("--min-iou", type=float, default=MIN_IOU, help="arm IoU filter of train / val (default %(default)s)")
    p.add_argument("--exclude", type=Path, help="val: dataset whose episodes are skipped (the training set)")
    a = p.parse_args()
    exclude = {r["idx"] for r in read_jsonl(a.exclude / "dataset.jsonl")} if a.exclude else set()
    build(a.mode, a.px, a.out, json.loads(a.items.read_text()), a.max_items or MAX_ITEMS.get(a.mode), a.min_iou,
          exclude)


if __name__ == "__main__":
    main()
