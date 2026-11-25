"""No-task-text analysis: does the video model take what to grasp from the caption's task sentence or from the proxy?

Generations: the Table 3 models rendered again (same training config, so the same validation settings, first frames
and reference videos) on a copy of the test set whose captions drop the task sentence ("The robot arm performs the
task: <instruction>." - the only place the task enters the generation), judged with grasp_vlm.py unchanged (GT = the
real-video answers). Compared with the with-text answers of the same models on the same episodes: the grasp metrics
with and without text, paired per episode (2x2 counts of "agrees with GT", the difference with a paired bootstrap 95%
interval, 2000 resamples, seed 0), the rate the two answers name the same object, and, for --follow-model, the follow
analysis of grasp_vlm_follow.py (proxy = --proxy answers) with and without text.

usage: python -m simpwm.eval.grasp_vlm_notext OUT_JSON --ids IDS_TXT --with-text ROOT --no-text ROOT --proxy DIR
           --test-set DS [--gt GT_DIR] [--models robot rgb noproxy] [--follow-model rgb] [--cache F] [--vlm MODEL]
  ROOT: grasp_vlm.py roots holding one answer set per model
"""

import json
from pathlib import Path

import numpy as np

from simpwm.eval import grasp_vlm as gv
from simpwm.eval import vlm
from simpwm.eval.common import ci, load_dir, paired_bootstrap, parser, read_ids, task_texts
from simpwm.eval.grasp_vlm_follow import agree

MODELS = ["robot", "rgb", "noproxy"]


def paired(a, b, seed=0):
    """Paired comparison of per-episode 0/1 outcomes with text (a) and without (b)."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    d = paired_bootstrap(a, b, np.random.default_rng(seed))
    return dict(n=len(a), with_text=float(a.mean()), no_text=float(b.mean()), diff_no_minus_with=float((b - a).mean()),
                ci95=[float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))],
                both_ok=int(((a == 1) & (b == 1)).sum()), only_with=int(((a == 1) & (b == 0)).sum()),
                only_no=int(((a == 0) & (b == 1)).sum()), both_wrong=int(((a == 0) & (b == 0)).sum()))


def follow(gen, real, prx, cache, tasks):
    """grasp_vlm_follow's mismatch analysis on matched records ({idx: follows} of the mismatch episodes too)."""
    mm, ctrl = [], []
    for idx in sorted(set(gen) & set(real) & set(prx)):
        R, X, G = gv.parse(real[idx]["raw"]), gv.parse(prx[idx]["raw"]), gv.parse(gen[idx]["raw"])
        xr, gr, gx = prx[idx]["agree"], gen[idx]["agree"], agree(idx, X, G, cache, tasks)
        if xr:
            ctrl.append(gr)
        else:
            mm.append(dict(idx=idx, type="proxy_only" if R == "none" else "real_only" if X == "none" else "diff_object",
                           follows="proxy" if gx and not gr else "real" if gr and not gx else "neither" if not gx
                           else "both"))
    out = dict(n_mismatch=len(mm), control_gen_agrees=ci(ctrl))
    for k in ("all", "diff_object", "proxy_only", "real_only"):
        es = [e for e in mm if k == "all" or e["type"] == k]
        out[k] = dict(n=len(es),
                      counts={c: sum(e["follows"] == c for e in es) for c in ("proxy", "real", "neither", "both")},
                      **{f"follows_{c}": ci([e["follows"] == c for e in es]) for c in ("proxy", "real", "neither")})
    return out, {e["idx"]: e["follows"] for e in mm}


def main():
    p = parser(__doc__)
    p.add_argument("out", type=Path, help="output JSON")
    p.add_argument("--ids", required=True, help="episodes of the no-text generations")
    p.add_argument("--with-text", type=Path, required=True, help="grasp_vlm.py root of the with-text answers")
    p.add_argument("--no-text", type=Path, required=True, help="grasp_vlm.py root of the no-text answers")
    p.add_argument("--proxy", type=Path, required=True, help="answer set of the RGB proxy (follow analysis)")
    p.add_argument("--test-set", type=Path, required=True, help="test video dataset (task instructions)")
    p.add_argument("--gt", type=Path, help="GT answers (default WITH_TEXT/real)")
    p.add_argument("--models", nargs="+", default=MODELS, help="answer sets to compare (default %(default)s)")
    p.add_argument("--follow-model", default="rgb", help="model of the follow analysis (default %(default)s)")
    p.add_argument("--cache", type=Path, help="same-object decision cache (default OUT's directory)")
    p.add_argument("--vlm", help=f"VLM judge model (default {vlm.MODEL})")
    a = p.parse_args()
    vlm.use(a.vlm)
    ids = read_ids(a.ids)
    real = load_dir(a.gt or a.with_text / "real")
    tasks = task_texts(a.test_set / "dataset.jsonl")
    cf = a.cache or a.out.parent / "follow_same_cache.json"
    cache = json.loads(cf.read_text()) if cf.exists() else {}
    res = dict(doc=__doc__, ids=str(a.ids), prompts=dict(grasp=gv.Q_GRASP, same_object=gv.Q_SAME), models={})
    gt = [real[i] for i in ids]
    res["gt"] = dict(n=len(gt), none=sum(gv.parse(r["raw"]) == "none" for r in gt))
    res["gt"]["object"] = res["gt"]["n"] - res["gt"]["none"]
    for m in a.models:
        wt, nt = load_dir(a.with_text / m), load_dir(a.no_text / m)
        common = [i for i in ids if i in wt and i in nt and "agree" in nt[i] and "agree" in wt[i]]
        x = dict(n_common=len(common), with_text=gv.metrics([wt[i] for i in common]),
                 no_text=gv.metrics([nt[i] for i in common]),
                 paired_grasp_acc=paired([wt[i]["agree"] for i in common], [nt[i]["agree"] for i in common]),
                 paired_says_object=paired([wt[i]["parsed"] != "none" for i in common],
                                           [nt[i]["parsed"] != "none" for i in common]),
                 same_answer_with_vs_no_text=ci([agree(i, gv.parse(wt[i]["raw"]), gv.parse(nt[i]["raw"]), cache, tasks)
                                                 for i in common]))
        if m == a.follow_model:
            prx = load_dir(a.proxy)
            fw, ew = follow({i: wt[i] for i in common}, real, prx, cache, tasks)
            fn, en = follow({i: nt[i] for i in common}, real, prx, cache, tasks)
            both = sorted(set(ew) & set(en))
            x["follow"] = dict(proxy_source=str(a.proxy), with_text=fw, no_text=fn,
                               **{f"paired_follows_{c}": paired([ew[i] == c for i in both], [en[i] == c for i in both])
                                  for c in ("proxy", "real")})
        res["models"][m] = x
        cf.write_text(json.dumps(cache))
    a.out.write_text(json.dumps(res, indent=1))
    for m, x in res["models"].items():
        f = lambda d: f"{d['mean']:.3f}" if d["mean"] is not None else "-"  # noqa: E731
        print(m, x["n_common"], "acc with/no", f(x["with_text"]["grasp_acc"]), f(x["no_text"]["grasp_acc"]),
              "paired", x["paired_grasp_acc"])


if __name__ == "__main__":
    main()
