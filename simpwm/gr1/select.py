"""GR1 item selection; no DreamDojo test episode is used for camera calibration or training.

Writes into OUT:
  sessions.json      {session: [up to 3 episodes]} for the per-session camera refit (gr1/session_calib.py), every
                     session of the dataset (the test sessions are also recorded in training episodes)
  candidates.json    [{ep, start, session}] training candidates: non-test episodes with >= 195 frames (a 97-frame
                     stride-2 clip), round-robin over sessions (random order within a session), clip start uniform in
                     [0, length - 195] (DreamDojo's test windows also start anywhere in an episode)
  test_windows.json  [{k, ep, base, start}] DreamDojo's 100 test windows; frames raw base + 2 + 2j, j = 0..48
sessions.json and test_windows.json are shipped in resources/gr1 (identical: the selection is seeded).

usage: python -m simpwm.gr1.select [--out runs/gr1] [--num-candidates 4800]
"""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

from simpwm.gr1 import data as gr1_data
from simpwm.paths import RUNS

CLIP_SPAN = 195  # raw frames covered by a 97-frame stride-2 clip (with margin)


def select_items(out, num_candidates):
    rng = random.Random(0)
    E, test = gr1_data.episodes(), gr1_data.test_episodes()
    by_s = defaultdict(list)
    for i in E:
        if i not in test:
            by_s[gr1_data.session(i)].append(i)
    sessions = {}
    for s, eps in sorted(by_s.items()):
        long = [i for i in eps if E[i]["length"] >= CLIP_SPAN] or [i for i in eps if E[i]["length"] >= 60]
        if long:
            sessions[s] = sorted(rng.sample(long, min(3, len(long))))
    queues = {}
    for s, eps in sorted(by_s.items()):
        long = [i for i in eps if E[i]["length"] >= CLIP_SPAN]
        queues[s] = rng.sample(long, len(long))
    order = list(queues)
    rng.shuffle(order)
    cand = []
    while len(cand) < num_candidates and any(queues.values()):
        for s in order:
            if queues[s] and len(cand) < num_candidates:
                i = queues[s].pop()
                cand.append(dict(ep=i, start=rng.randint(0, E[i]["length"] - CLIP_SPAN), session=s))
    split = json.loads(gr1_data.TEST_SPLIT.read_text())
    windows = [dict(k=k, ep=e, base=b, start=b + 2) for k, (e, b) in enumerate(split["samples"])]
    out.mkdir(parents=True, exist_ok=True)
    (out / "sessions.json").write_text(json.dumps(sessions, indent=0))
    (out / "candidates.json").write_text(json.dumps(cand))
    (out / "test_windows.json").write_text(json.dumps(windows, indent=0))
    test_sessions = {gr1_data.session(i) for i in test}
    print(f"sessions {len(sessions)} (test sessions covered: {len(test_sessions & set(sessions))}/"
          f"{len(test_sessions)}), candidates {len(cand)} from {len({c['session'] for c in cand})} sessions, "
          f"test windows {len(windows)}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", type=Path, default=RUNS / "gr1", help="output dir")
    ap.add_argument("--num-candidates", type=int, default=4800, help="number of training candidates")
    a = ap.parse_args()
    select_items(a.out, a.num_candidates)


if __name__ == "__main__":
    main()
