"""VLM grasp judge ("does the robot pick up and lift an object, and which?") on DROID test videos: Table 3 (the
sim-to-real ablation of the video model, and the Table 1 baselines) on the generated videos, Table 2 on the RGB proxy
videos themselves. GT = the answers on the real videos (a video set of its own, by default ROOT/real).

Judge: vlm.py (Qwen3-VL-8B-Instruct, greedy). Input per video: 8 frames at recorded steps
np.linspace(0, 96, 8).round() = 0, 14, 27, 41, 55, 69, 82, 96. Frame k of a video is recorded step k * stride; a step
that is not a multiple of the stride (Ctrl-World, stride 3) is replaced by the nearest available one (ties to the
earlier); the steps used are stored. Each frame is resized to 1280 wide keeping the video's aspect (bicubic when
upsampling, area otherwise) and the 8 go in as one multi-image input followed by the question Q_GRASP.
Answers are parsed with PARSE_RULE (parse()); a generation agrees with the GT under SAME_OBJECT_RULE (match stage).
Metrics per video set, over the episodes with both answers: grasp acc. = agree fraction; halluc. = among GT "none",
the fraction naming an object; miss = among GT objects, the fraction saying none; bootstrap 95% intervals (2000
resamples of episodes, seed 0).

Stages (per video set NAME: ROOT/NAME/<idx>.json):
  answer   raw + parsed answer per video of VIDEOS: a directory of <idx>.mp4 or a path pattern with {idx} / {i5} (the
           5-digit index), e.g. RUN/droid_{i5}/conditions/ref_rgb_1280x704.mp4 for the proxies of a run
  match    same-object decisions against the GT answers -> fields added to the records
  summary  -> ROOT/summary.json
  review   blind 8-frame grids for a human check (anonymous file names, no answer drawn) -> ROOT/human_review/
  parse    re-parse the stored raw answers after a rule change (drops the match fields)
usage: python -m simpwm.eval.grasp_vlm answer ROOT NAME VIDEOS --ids IDS_TXT [--stride S] [--shard T N] [--vlm MODEL]
       python -m simpwm.eval.grasp_vlm match ROOT NAME [NAME ...] --test-set DS [--gt GT_DIR] [--vlm MODEL]
       python -m simpwm.eval.grasp_vlm summary ROOT NAME [NAME ...] [--gt GT_DIR]
       python -m simpwm.eval.grasp_vlm review ROOT NAME [NAME ...] [--total 60 | --per-name K] [--prefix r]
       python -m simpwm.eval.grasp_vlm parse ROOT
  DS: the test video dataset (its dataset.jsonl captions give the task instruction of Q_SAME)
"""

import csv
import json
import re
from pathlib import Path

import cv2
import numpy as np

from simpwm.eval import vlm
from simpwm.eval.common import ci, each, grid, load_dir, parser, read_ids, read_indices, task_texts, write_jpg

STEPS = [int(s) for s in np.linspace(0, 96, 8).round()]
WIDTH = 1280
ANS_TOKENS, SAME_TOKENS = 32, 4
Q_GRASP = ("Does the robot pick up and lift an object off its support in these frames? "
           "If yes, name the object; if no, answer none.")
# a stricter wording ("Do A and B refer to the same physical object?" without the rewording hint) rejected rewordings
# of one object ("green towel" / "green cloth")
Q_SAME = ("Task: {task}. Two short descriptions, possibly worded differently or with different detail, each of an "
          "object a robot lifts in this scene. Object A: {a}. Object B: {b}. "
          "Do A and B refer to the same physical object in this scene? Answer yes or no.")
GENERIC = {"object", "an object", "objects", "item", "an item", "thing", "something"}
NONE_WORDS = {"none", "nothing", "no object", "no", "n/a", "na", "no objects", "none.", "nothing is lifted",
              "no object is lifted", "no object is picked up"}
FILLER = [r"the robot (?:arm )?(?:is )?(?:picks?|picking|lifts?|lifting|grasps?|grasping|picked|lifted)"
          r"(?: up)?(?: and (?:lifts?|lifting|lifted|picks?|picked)(?: up)?)?(?: up)?",
          r"it (?:picks?|lifts?|grasps?|picked|lifted)(?: up)?(?: and (?:lifts?|lifted)(?: up)?)?",
          r"(?:picks?|lifts?|picked|lifted)(?: up)?(?: and (?:lifts?|lifted)(?: up)?)?",
          r"the (?:object|item) (?:is|being lifted is|picked up is|lifted is)", r"object:", r"the object name is"]
PARSE_RULE = """deterministic, on the raw answer:
  1. a = first non-empty line, lower case, markdown '*' / quotes removed, surrounding whitespace and . , ; : ! stripped.
  2. a starts with the word "no" or "none", or a is one of NONE_WORDS ("nothing", "no object", "n/a", ...) -> "none".
  3. a starts with the word "yes": drop it and following punctuation / whitespace; then drop leading filler phrases
     (FILLER: "the robot picks up and lifts", "it picks up", "the object is", ...) and a leading article; the rest up
     to the first '.' ';' '(' newline, ' from ' / ' off ' / ' out of ' (the support), ' - ', or an optionally
     comma-led ' which / that / likely / possibly / located / resembling / probably' clause is the object, with its
     remaining commas (adjective lists: "a small, pink, bowl-shaped object") replaced by spaces; empty -> "object".
  4. otherwise the whole of a (filler / article / support cut as in 3) is the object name; if it is in NONE_WORDS ->
     "none".
  5. if the object from 3/4 is generic (GENERIC: "object", "item", ...), the first "the object is / appears to be X"
     anywhere in the full raw answer gives X (cut as in 3) instead; failing that it stays "object"."""
SAME_OBJECT_RULE = """GT parsed g, generation parsed p:
  both "none"                       -> agree
  exactly one "none"                -> disagree (halluc. if g is none, miss if p is none)
  both objects, identical strings   -> agree (no VLM call)
  both objects, different strings   -> text-only VLM with Q_SAME (the episode's task instruction), greedy; agree iff
                                       the answer, lower-cased and stripped, starts with "yes"."""
RECORD = ("idx", "method", "video", "stride", "steps", "native_size", "input_size", "raw")


# ----------------------------------------------------------------------------------------------- parsing
def _clean(a):
    a = next((line for line in a.splitlines() if line.strip()), "").lower()
    a = a.replace("*", "").replace('"', "").replace("`", "")
    return a.strip().strip(".,;:! ").strip()


def _object(a):
    for f in FILLER:
        a = re.sub(r"^" + f + r"\b[\s,:]*", "", a).strip()
    a = re.sub(r"^(?:the|a|an|some)\s+", "", a)
    a = re.split(r"[.;(\n]| from | off | out of | - |,? (?:which|that|likely|possibly|located|resembling|probably)\b",
                 a)[0]
    return re.sub(r"\s*,\s*", " ", a).strip(" ,")


def parse(raw):
    """'none' or the object name (PARSE_RULE)."""
    a = _clean(raw)
    if a in NONE_WORDS or re.match(r"^(?:no|none)\b", a):
        return "none"
    if re.match(r"^yes\b", a):
        o = _object(re.sub(r"^yes\b[\s,.:;!-]*", "", a))
        o = o if o and o not in NONE_WORDS else "object"
    else:
        o = _object(a)
        if not o or o in NONE_WORDS:
            return "none"
    if o in GENERIC:  # "yes, the robot lifts an object off its support. The object is a white box, ..."
        m = re.search(r"the (?:object|item) (?:is|appears to be|being lifted is)\s+(.+)", raw.lower().replace("*", ""))
        if m:
            o2 = _object(m.group(1).strip())
            o = o2 if o2 and o2 not in NONE_WORDS else o
    return o


def same_object(task, a, b):
    """(same, raw VLM answer or "identical") for two object names a, b in an episode with this task instruction."""
    if a == b:
        return True, "identical"
    raw = vlm.ask([{"type": "text", "text": Q_SAME.format(task=task, a=a, b=b)}], SAME_TOKENS)
    return raw.strip().lower().startswith("yes"), raw


# ----------------------------------------------------------------------------------------------- frames
def frame_index(step, stride):
    """Nearest frame (ties to the earlier) to a recorded step."""
    return int(np.floor(step / stride + 0.5 - 1e-9))


def video_path(videos, idx):
    """VIDEOS/<idx>.mp4, or the pattern VIDEOS with {idx} / {i5} filled in."""
    return str(videos).format(idx=idx, i5=f"{idx:05d}") if "{" in str(videos) else Path(videos) / f"{idx}.mp4"


def frames(videos, stride, idx):
    """The 8 RGB frames the VLM sees (1280 wide, own aspect), the recorded steps they show, the native size."""
    want = [frame_index(s, stride) for s in STEPS]
    got = read_indices(video_path(videos, idx), want)
    missing = [k for k in want if k not in got]
    if missing:
        raise RuntimeError(f"{videos} {idx}: frames {missing} missing")
    h, w = got[want[0]].shape[:2]
    H = int(round(WIDTH * h / w))
    interp = cv2.INTER_CUBIC if w < WIDTH else cv2.INTER_AREA
    imgs = [got[k] if (w, h) == (WIDTH, H) else cv2.resize(got[k], (WIDTH, H), interpolation=interp) for k in want]
    return imgs, [k * stride for k in want], (w, h)


# ----------------------------------------------------------------------------------------------- stages
def answer_stage(root, name, videos, ids, stride=1):
    from PIL import Image

    od = Path(root) / name
    od.mkdir(parents=True, exist_ok=True)

    def run(idx):
        imgs, steps, size = frames(videos, stride, idx)
        raw = vlm.ask([*[{"type": "image", "image": Image.fromarray(i)} for i in imgs],
                       {"type": "text", "text": Q_GRASP}], ANS_TOKENS)
        (od / f"{idx}.json").write_text(json.dumps(dict(
            idx=idx, method=name, video=str(videos), stride=stride, steps=steps, native_size=size,
            input_size=[imgs[0].shape[1], imgs[0].shape[0]], raw=raw, parsed=parse(raw))))
        print("done", name, idx, repr(raw), "->", parse(raw), flush=True)

    each(ids, run, done=lambda idx: (od / f"{idx}.json").exists())


def match_stage(root, names, gt_dir, tasks):
    gt = load_dir(gt_dir)
    for name in names:
        for p in sorted((Path(root) / name).glob("*.json")):
            r = json.loads(p.read_text())
            idx = r["idx"]
            if idx not in gt or "agree" in r:
                continue
            g, q = parse(gt[idx]["raw"]), parse(r["raw"])
            r["parsed"] = q
            r.update(gt_raw=gt[idx]["raw"], gt_parsed=g)
            if g == "none" or q == "none":
                r.update(agree=(g == q), same_raw=None)
            else:
                s, raw = same_object(tasks.get(int(idx), ""), g, q)
                r.update(agree=bool(s), same_raw=raw)
            r.update(gt_none=(g == "none"), halluc=(g == "none" and q != "none"), miss=(g != "none" and q == "none"))
            p.write_text(json.dumps(r))
        print("matched", name, flush=True)


def matched(d):
    """The matched records of a video set directory."""
    return [r for r in load_dir(d).values() if "agree" in r]


def metrics(rs):
    """Grasp metrics of matched records: grasp acc., binary acc. (object identity ignored), halluc., miss."""
    return dict(grasp_acc=ci([r["agree"] for r in rs]),
                grasp_binary_acc=ci([(r["gt_parsed"] == "none") == (r["parsed"] == "none") for r in rs]),
                halluc=ci([r["halluc"] for r in rs if r["gt_none"]]),
                miss=ci([r["miss"] for r in rs if not r["gt_none"]]))


def summary_stage(root, names, gt_dir):
    gt = load_dir(gt_dir)
    res = dict(
        judge=dict(model=vlm.name(), decoding=f"greedy (do_sample=False), max_new_tokens {ANS_TOKENS} / {SAME_TOKENS}",
                   frames=f"recorded steps {STEPS} (nearest available frame for strided videos), resized to {WIDTH} "
                          "wide keeping the video's aspect (bicubic upsampling for low-res), multi-image input"),
        prompt_grasp=Q_GRASP, prompt_same_object=Q_SAME, parse_rule=PARSE_RULE, same_object_rule=SAME_OBJECT_RULE,
        gt=dict(n=len(gt), none=sum(r["parsed"] == "none" for r in gt.values()),
                object=sum(r["parsed"] != "none" for r in gt.values())),
        methods={})
    for name in names:
        rs = matched(Path(root) / name)
        res["methods"][name] = dict(**metrics(rs), says_object=float(np.mean([r["parsed"] != "none" for r in rs]))
                                    if rs else None,
                                    same_object_vlm_calls=sum(r["same_raw"] not in (None, "identical") for r in rs))
    (Path(root) / "summary.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


def review_stage(root, names, total=60, per_name=None, seed=0, prefix="r"):
    """Blind grids: per_name episodes of each set, or `total` balanced over the sets (numpy default_rng(seed))."""
    root = Path(root)
    rd = root / "human_review"
    rd.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    if per_name:
        counts = [per_name] * len(names)
    else:
        counts = [total // len(names)] * len(names)
        for j in rng.permutation(len(names))[: total - sum(counts)]:
            counts[j] += 1
    items = []
    for m, c in zip(names, counts):
        have = sorted(int(p.stem) for p in (root / m).glob("*.json"))
        items += [(m, int(idx)) for idx in rng.choice(have, c, replace=False)]
    items = [items[i] for i in rng.permutation(len(items))]
    rows = []
    for k, (m, idx) in enumerate(items):
        r = json.loads((root / m / f"{idx}.json").read_text())
        imgs, _, _ = frames(r["video"], r["stride"], idx)
        fn = f"{prefix}{k:03d}.jpg"
        write_jpg(rd / fn, grid(imgs))
        rows.append(dict(file=fn, method=m, idx=idx, vlm_answer=r["raw"], parsed=parse(r["raw"]), human=""))
    with open(rd / "index.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["file", "method", "idx", "vlm_answer", "parsed", "human"])
        w.writeheader()
        w.writerows(rows)
    (rd / "README.txt").write_text(
        "Each image: the 8 frames the VLM saw (recorded steps " + ", ".join(map(str, STEPS)) +
        ", left-to-right, top-to-bottom; 640 wide each; magenta in proxy videos = unknown). Question: " + Q_GRASP +
        "\nFill index.csv 'human' with none or the object name.\n")
    print("wrote", len(rows), "to", rd)


def parse_stage(root):
    for p in Path(root).glob("*/*.json"):
        r = json.loads(p.read_text())
        r = {k: v for k, v in r.items() if k in RECORD}
        r["parsed"] = parse(r["raw"])
        p.write_text(json.dumps(r))


def main():
    p = parser(__doc__)
    sub = p.add_subparsers(dest="stage", required=True)
    s = sub.add_parser("answer", help="VLM answers of one video set")
    s.add_argument("root", type=Path)
    s.add_argument("name")
    s.add_argument("videos", help="directory of <idx>.mp4, or a pattern with {idx} / {i5}")
    s.add_argument("--ids", required=True, help="episode id list")
    s.add_argument("--stride", type=int, default=1, help="recorded steps per video frame (default 1)")
    s.add_argument("--shard", nargs=2, type=int, default=(0, 1), metavar=("T", "N"), help="T-th of N shards")
    s.add_argument("--vlm", help=f"VLM judge model (default {vlm.MODEL})")
    s = sub.add_parser("match", help="same-object decisions against the GT")
    s.add_argument("root", type=Path)
    s.add_argument("names", nargs="+", metavar="NAME")
    s.add_argument("--test-set", type=Path, required=True, help="test video dataset (task instructions)")
    s.add_argument("--gt", type=Path, help="GT answers (default ROOT/real)")
    s.add_argument("--vlm", help=f"VLM judge model (default {vlm.MODEL})")
    s = sub.add_parser("summary", help="metrics per video set -> ROOT/summary.json")
    s.add_argument("root", type=Path)
    s.add_argument("names", nargs="+", metavar="NAME")
    s.add_argument("--gt", type=Path, help="GT answers (default ROOT/real)")
    s.add_argument("--vlm", help="judge model recorded in the summary (default: the default judge)")
    s = sub.add_parser("review", help="blind grids for a human check")
    s.add_argument("root", type=Path)
    s.add_argument("names", nargs="+", metavar="NAME")
    s.add_argument("--total", type=int, default=60, help="grids balanced over the sets (default %(default)s)")
    s.add_argument("--per-name", type=int, help="grids per set instead of --total")
    s.add_argument("--prefix", default="r", help="grid file name prefix (default %(default)s)")
    s = sub.add_parser("parse", help="re-parse the stored answers")
    s.add_argument("root", type=Path)
    a = p.parse_args()
    vlm.use(getattr(a, "vlm", None))
    gt = (getattr(a, "gt", None) or a.root / "real")
    if a.stage == "answer":
        answer_stage(a.root, a.name, a.videos, read_ids(a.ids, *a.shard), a.stride)
    elif a.stage == "match":
        match_stage(a.root, a.names, gt, task_texts(a.test_set / "dataset.jsonl"))
    elif a.stage == "summary":
        summary_stage(a.root, a.names, gt)
    elif a.stage == "review":
        review_stage(a.root, a.names, a.total, a.per_name, prefix=a.prefix)
    else:
        parse_stage(a.root)


if __name__ == "__main__":
    main()
