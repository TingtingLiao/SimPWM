"""VLM pass over the first frame of DROID episodes: what the gripper closes on at each closing point.

Inputs are test-time inputs only: the first ext1 frame, the task text and the closing points of the action (the lowest
TCP pose of each closing window projected with the episode's setup camera, as for the grasp proposals). The VLM sees
the frame with numbered markers at those points and returns
  objects  short noun phrases of the movable things on the work surface
  grasps   per marker, the thing the gripper will close on (or null)
written to OUT/<idx>.json (+ the marked frame OUT/<idx>_marked.jpg). droid/run.py (--vlm-dir) reads the grasp
phrases: a closing point that names a deformable thing (sim/pipeline.is_soft) makes the object there soft (draped by
the rule engine).

usage: python -m simpwm.perception.vlm_objects IDX [IDX ...] [--out runs/vlm_objects] [--setups SETUPS_JSON] [--model M]
"""

from __future__ import annotations

import argparse
import json
import re
import traceback
from pathlib import Path

import cv2
import torch
from PIL import Image

from simpwm import paths
from simpwm.core.embodiment import Embodiment
from simpwm.droid import data as droid_data
from simpwm.droid import setup_calib
from simpwm.sim import pipeline

MODEL = "Qwen/Qwen3-VL-32B-Instruct"  # bf16, ~66 GB of GPU memory
UPSCALE = 3  # DROID frames are 320x180: shown to the VLM at 3x

PROMPT = """This is the first frame of a robot manipulation video. Task given to the robot: "{task}".
Red numbered circles mark the image points where the robot's gripper will close later in the video.

1. List every movable object on the tables / counters / work surfaces, including cloths or towels lying flat,
   transparent objects, lids and small items. Use short noun phrases (1-3 words, colour + name, e.g. "orange cup",
   "white towel", "glass pitcher"). At most 15. Do not list the robot, furniture, walls or the table itself.
2. For each numbered marker, name the object the gripper will most likely grasp at that point, as a short noun
   phrase, or null if the marker is not on or next to a graspable object.

Answer with JSON only: {{"objects": ["..."], "grasps": [{{"marker": 1, "phrase": "..."}}]}}"""


def load_vlm(model_id):
    from transformers import AutoModelForImageTextToText, AutoProcessor

    proc = AutoProcessor.from_pretrained(model_id)
    model = AutoModelForImageTextToText.from_pretrained(model_id, dtype=torch.bfloat16, device_map="cuda").eval()
    return proc, model


def marked_frame(frame, pts):
    """The frame upscaled by UPSCALE with a numbered red circle at each closing point (pixels of `frame`)."""
    im = cv2.resize(frame, None, fx=UPSCALE, fy=UPSCALE, interpolation=cv2.INTER_CUBIC)
    for k, (u, v) in enumerate(pts, 1):
        c = (int(u * UPSCALE), int(v * UPSCALE))
        cv2.circle(im, c, 14, (255, 0, 0), 3)
        cv2.putText(im, str(k), (c[0] + 16, c[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 0), 3)
    return im


@torch.no_grad()
def ask(proc, model, im, task):
    msgs = [{"role": "user", "content": [{"type": "image", "image": Image.fromarray(im)},
                                         {"type": "text", "text": PROMPT.format(task=task or "unknown")}]}]
    inp = proc.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True, return_dict=True,
                                   return_tensors="pt").to(model.device)
    out = model.generate(**inp, max_new_tokens=400, do_sample=False)
    return proc.batch_decode(out[:, inp["input_ids"].shape[1]:], skip_special_tokens=True)[0]


def parse(text, n_markers):
    """VLM answer -> (object phrases, one grasp phrase or None per marker)."""
    m = re.search(r"\{.*\}", text, re.S)
    d = json.loads(m.group(0)) if m else {}
    objs = [str(o).strip() for o in d.get("objects", []) if str(o).strip()][:15]
    by_marker = {int(g.get("marker", 0)): g.get("phrase") for g in d.get("grasps", []) if isinstance(g, dict)}
    return objs, [(by_marker.get(k + 1) or None) for k in range(n_markers)]


def run(out, idxs, setups=None, model_id=MODEL):
    """Query the VLM for episodes `idxs` (skipping those already written). setups: {idx: [other episodes of its
    setup]} for the setup camera, as for droid/run.py (None: from the setup index)."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    ok = [r for r in droid_data.manifest() if r["status"] == "ok"]
    emb = Embodiment("franka_robotiq")
    proc, model = load_vlm(model_id)
    for idx in idxs:
        f = out / f"{idx}.json"
        if f.exists():
            continue
        try:
            ep = droid_data.load(ok[idx])
            others = setup_calib.same_setup(idx) if setups is None else setups[str(idx)]
            cam, _ = setup_calib.episode_camera(idx, others, ok, emb, ep["cam"])
            pts = pipeline.closing_points(ep["arm_q"], ep["grip"], cam, emb)
            im = marked_frame(ep["frames"][0], pts)
            raw = ask(proc, model, im, ep["text"])
            objs, gph = parse(raw, len(pts))
            f.write_text(json.dumps(dict(idx=idx, task=ep["text"], model=model_id, objects=objs,
                                         grasps=[dict(marker=k + 1, px=[int(u), int(v)], phrase=p)
                                                 for k, ((u, v), p) in enumerate(zip(pts, gph))],
                                         raw=raw), indent=1))
            cv2.imwrite(str(out / f"{idx}_marked.jpg"), im[..., ::-1])
            print("vlm", idx, objs, gph, flush=True)
        except Exception:  # one failing episode must not end the list
            print(f"FAILED vlm {idx}\n{traceback.format_exc()}", flush=True)


def main():
    ap = argparse.ArgumentParser(description="Name what the gripper closes on at each closing point of DROID episodes "
                                             "(first frame + task text, Qwen3-VL); droid/run.py reads the result.")
    ap.add_argument("idxs", type=int, nargs="+", metavar="IDX", help="episode indices (droid/data.py manifest)")
    ap.add_argument("--out", type=Path, default=paths.RUNS / "vlm_objects",
                    help="output directory, <idx>.json per episode (droid/run.py --vlm-dir; default: %(default)s)")
    ap.add_argument("--setups", type=Path,
                    help="JSON {idx: [other episodes of its setup]} for setup calibration (default: the setup index)")
    ap.add_argument("--model", default=MODEL, help="Hugging Face id or local path of the VLM (default: %(default)s)")
    a = ap.parse_args()
    run(a.out, a.idxs, json.loads(a.setups.read_text()) if a.setups else None, a.model)


if __name__ == "__main__":
    main()
