"""Which does the generated video follow when the proxy's grasp and the real grasp disagree? (Table 3 models whose
proxy carries object motion; evaluation only, from grasp_vlm.py answers.)

Answers (grasp_vlm.py, same judge, question, frames and parse rule):
  real       the GT answers on the real videos
  generated  GEN_ROOT/<variant>: the answers on a model's generations
  proxy      --proxy VARIANT=DIR: the answers on the RGB proxy video the model was conditioned on (Table 2's answer
             sets; models conditioned on the same simulator run share one, e.g. rgb, depth and rgb_depth all use the
             full system's RGB proxy answer: the VLM is not run on depth videos)
Agreement of two answers: grasp_vlm's same-object rule (both none; or both objects and identical strings, or the
text-only VLM answers yes to Q_SAME with A = the first, B = the second answer and the task instruction). Proxy vs real
and generation vs real reuse the stored match decisions; generation vs proxy is decided here (A = proxy, B =
generation; decisions cached in --cache).
Mismatch episodes (proxy disagrees with real): the generation follows the proxy (agrees with it), the real video or
neither, by type: diff_object (both name different objects), proxy_only (proxy grasps, real none), real_only (proxy
none, real grasps). Control: the episodes where the proxy agrees with real, the rate the generation agrees with real.
Bootstrap 95% intervals (2000 resamples, seed 0).

usage: python -m simpwm.eval.grasp_vlm_follow OUT_JSON GEN_ROOT --proxy VARIANT=DIR [...] --test-set DS [--gt GT_DIR]
           [--cache F] [--vlm MODEL]
"""

import json
from pathlib import Path

from simpwm.eval import grasp_vlm as gv
from simpwm.eval import vlm
from simpwm.eval.common import ci, load_dir, parser, task_texts


def agree(idx, a, b, cache, tasks):
    """grasp_vlm's same-object rule for answers a, b of episode idx; VLM decisions are cached under "idx|a|b"."""
    if a == "none" or b == "none":
        return a == b
    k = f"{idx}|{a}|{b}"
    if k not in cache:
        s, raw = gv.same_object(tasks.get(int(idx), ""), a, b)
        cache[k] = [bool(s), raw]
    return cache[k][0]


def follow(gen, real, prx, cache, tasks):
    """Per-episode records of the episodes all three answer sets have, the mismatch summary and the control."""
    per, mm, ctrl = [], {"all": [], "diff_object": [], "proxy_only": [], "real_only": []}, []
    for idx in sorted(set(real) & set(gen) & set(prx)):
        R, X, G = gv.parse(real[idx]["raw"]), gv.parse(prx[idx]["raw"]), gv.parse(gen[idx]["raw"])
        xr = prx[idx]["agree"] if "agree" in prx[idx] else agree(idx, R, X, cache, tasks)
        gr = gen[idx]["agree"] if "agree" in gen[idx] else agree(idx, R, G, cache, tasks)
        gx = agree(idx, X, G, cache, tasks)
        e = dict(idx=idx, real=R, proxy=X, gen=G, proxy_eq_real=bool(xr), gen_eq_real=bool(gr), gen_eq_proxy=bool(gx))
        if xr:
            ctrl.append(e)
        else:
            e["type"] = "proxy_only" if R == "none" else "real_only" if X == "none" else "diff_object"
            # gx and gr cannot both hold unless the same-object calls are inconsistent
            e["follows"] = "both (inconsistent same-object calls)" if gx and gr else \
                "proxy" if gx else "real" if gr else "neither"
            mm["all"].append(e)
            mm[e["type"]].append(e)
        per.append(e)
    res = {}
    for k, es in mm.items():
        res[k] = dict(n=len(es), counts={c: sum(e["follows"] == c for e in es) for c in
                                         ("proxy", "real", "neither", "both (inconsistent same-object calls)")},
                      follows_proxy=ci([e["follows"] == "proxy" for e in es]),
                      follows_real=ci([e["follows"] == "real" for e in es]),
                      neither=ci([e["follows"] == "neither" for e in es]))
    return dict(n=len(per), n_mismatch=len(mm["all"]), mismatch=res,
                control_proxy_eq_real=dict(gen_agrees=ci([e["gen_eq_real"] for e in ctrl]), n=len(ctrl)),
                episodes=per)


def main():
    p = parser(__doc__)
    p.add_argument("out", type=Path, help="output JSON")
    p.add_argument("gen_root", type=Path, help="grasp_vlm.py root of the generations' answer sets")
    p.add_argument("--proxy", action="append", required=True, metavar="VARIANT=DIR",
                   help="a model (answer set GEN_ROOT/VARIANT) and the answer set of its proxy")
    p.add_argument("--test-set", type=Path, required=True, help="test video dataset (task instructions)")
    p.add_argument("--gt", type=Path, help="GT answers (default GEN_ROOT/real)")
    p.add_argument("--cache", type=Path, help="same-object decision cache (default OUT's directory)")
    p.add_argument("--vlm", help=f"VLM judge model (default {vlm.MODEL})")
    a = p.parse_args()
    vlm.use(a.vlm)
    proxy = dict(x.split("=", 1) for x in a.proxy)
    cf = a.cache or a.out.parent / "follow_same_cache.json"
    cache = json.loads(cf.read_text()) if cf.exists() else {}
    tasks = task_texts(a.test_set / "dataset.jsonl")
    real = load_dir(a.gt or a.gen_root / "real")
    proxies = {d: load_dir(d) for d in set(proxy.values())}
    out = dict(doc=__doc__, proxy_source=proxy, variants={})
    for v, pd in proxy.items():
        out["variants"][v] = follow(load_dir(a.gen_root / v), real, proxies[pd], cache, tasks)
        cf.write_text(json.dumps(cache))
    a.out.write_text(json.dumps(out, indent=1))
    for v, x in out["variants"].items():
        print(v, "n", x["n"], "mismatch", x["n_mismatch"], x["mismatch"]["all"]["counts"], "ctrl",
              x["control_proxy_eq_real"])


if __name__ == "__main__":
    main()
