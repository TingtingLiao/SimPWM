"""Episode selection for the LTX IC-LoRA training, validation and test sets (raw DROID source, droid/data.py indices).
The paper's sets are shipped in resources/droid/splits/{train,val,test}.txt; this module records how they were chosen.

Camera setups (droid/setup_calib.py keys, which depend only on the episode metadata) tie the raw episodes to the
setup-level sets of resources/droid/*.json (RLDS indices). The DROID-500 dev/test setups (droid500.json "setups") are
never used for training or validation.
  candidates  long-clip candidates (>= CLIP_FRAMES recorded steps) from every other setup: val = N_VAL calibrated setups
              whose calibration reached a robot IoU >= VAL_MIN_IOU, one episode each; train = up to PER_SETUP episodes
              per remaining setup, round-robin over setups (any prefix spreads over setups), calibrated setups first
              -> OUT/sel.json {train, val, keys, n}, OUT/setups.json {idx: [other episodes of its setup]},
                 OUT/ids_train.txt, OUT/ids_val.txt
  pick        after the per-episode camera fits (droid/episode_calib.py): the candidates with arm IoU >= MIN_IOU and
              first-frame stereo depth, in candidate order, up to MAX_TRAIN -> OUT/pick.json, OUT/ids_px.txt
  val         a larger held-out set: fitted candidates outside the pick whose gripper starts closing between step 10 and
              MAX_GRIP -> OUT/val.json, OUT/ids_val_px.txt
  test        test candidates from setups without a training episode -> OUT/test_cand.json, OUT/ids_test_cand.txt
  pick-test   after their camera fits: the well-aligned ones with stereo depth -> OUT/test.json, OUT/ids_test.txt
  short       97-frame clips from the calibrated setups of video_train2k.json (train) and other calibrated setups (val)
              -> OUT/sel.json, OUT/setups.json

usage: python -m simpwm.droid.select {candidates,pick,val,test,pick-test,short} OUT [options]
"""

import argparse
import json
import random
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from simpwm import paths
from simpwm.droid import data, setup_calib

SETS = paths.RESOURCES / "droid"
VAL_MIN_IOU = 0.5  # a badly calibrated validation setup measures the camera, not the LoRA
MIN_T_SHORT = 100  # `short`: recorded steps needed for a 97-frame clip


# ----------------------------------------------------------------------------------------------- helpers
def calibrated_setups():
    return {f.stem for f in setup_calib.CACHE.glob("*.json")}


def setup_calib_iou(key):
    """Robot IoU the setup calibration reached."""
    return json.loads((setup_calib.CACHE / f"{key}.json").read_text())["report"].get("iou_after", 0)


def droid500_setups():
    """Setup keys of the DROID-500 dev / test episodes."""
    d500 = json.loads((SETS / "droid500.json").read_text())
    return set(d500.get("setups", {}).get("dev", [])) | set(d500.get("setups", {}).get("test", []))


def interleave(lists, n=None):
    """Round-robin over lists: the r-th element of every list before any (r + 1)-th, for r < n (default: all)."""
    n = max(map(len, lists), default=0) if n is None else n
    return [q[r] for r in range(n) for q in lists if r < len(q)]


def round_robin(ids, key, rng):
    """ids interleaved over their groups `key(i)`, groups and members in random order."""
    by = {}
    for i in ids:
        by.setdefault(key(i), []).append(i)
    keys = sorted(by, key=lambda k: rng.random())
    for v in by.values():
        rng.shuffle(v)
    return interleave([by[k] for k in keys])


def write_ids(f, ids):
    f.write_text(" ".join(map(str, ids)) + "\n")


def write_selection(out, train, val, key_of, members, **extra):
    sel = train + val
    keys = {str(i): key_of[i] for i in sel}
    (out / "sel.json").write_text(json.dumps(dict(train=train, val=val, keys=keys, **extra)))
    (out / "setups.json").write_text(json.dumps({str(i): [j for j in members[key_of[i]] if j != i] for i in sel}))


def lengths(out):
    """Recorded length of every episode (metadata.json trajectory_length), cached in OUT/lengths.json."""
    f = out / "lengths.json"
    if not f.exists():
        ok = setup_calib.ok_entries()
        meta = lambda e: json.loads((data.DROID_RAW / e["raw_dir"] / "metadata.json").read_text())["trajectory_length"]  # noqa: E731
        with ThreadPoolExecutor(64) as ex:  # tens of thousands of small files
            f.write_text(json.dumps(list(ex.map(meta, ok))))
    return json.loads(f.read_text())


def first_grip(i, ok, thr=0.2, n=241):
    """First step at which the gripper starts closing (recorded gripper_position > thr), -1 if not within n steps."""
    import h5py
    import numpy as np

    with h5py.File(data.DROID_RAW / ok[i]["raw_dir"] / "trajectory.h5", "r") as h:
        g = h["observation/robot_state/gripper_position"][:n]
    c = np.nonzero(g > thr)[0]
    return int(c[0]) if len(c) and g[0] <= thr else -1


def first_grips(ids):
    ok = setup_calib.ok_entries()
    with ThreadPoolExecutor(64) as ex:
        return dict(zip(ids, ex.map(lambda i: first_grip(i, ok), ids)))


def camera_ious(camera_dir):
    """{idx: arm IoU} of the per-episode camera fits (droid/episode_calib.py)."""
    return {int(f.stem): json.loads(f.read_text())["iou_after"] for f in Path(camera_dir).glob("*.json")}


def has_stereo(stereo_dir, i):
    return (Path(stereo_dir) / f"{i}_ext1.npy").exists()


# ----------------------------------------------------------------------------------------------- long clips
def candidates(out, clip_frames=241, per_setup=8, n_val=8, seed=0):
    rng = random.Random(seed)
    L = lengths(out)
    _, members = setup_calib.load_index()
    calibrated = calibrated_setups()
    held = droid500_setups()
    long = {k: [i for i in m if L[i] - 1 >= clip_frames] for k, m in members.items() if k not in held}
    cal_keys = sorted(k for k in long if long[k] and k in calibrated)
    raw_keys = sorted(k for k in long if long[k] and k not in calibrated)
    rng.shuffle(cal_keys)
    rng.shuffle(raw_keys)
    val_keys = [k for k in cal_keys if setup_calib_iou(k) >= VAL_MIN_IOU][:n_val]
    val = [rng.choice(long[k]) for k in val_keys]
    train = []
    for ks in ([k for k in cal_keys if k not in val_keys], raw_keys):
        train += interleave([rng.sample(long[k], min(per_setup, len(long[k]))) for k in ks], per_setup)
    keys = cal_keys + raw_keys
    key_of = {i: k for k in keys for i in members[k]}
    write_selection(out, train, val, key_of, members, n=clip_frames)
    write_ids(out / "ids_train.txt", train)
    write_ids(out / "ids_val.txt", val)
    n_cal = sum(min(per_setup, len(long[k])) for k in cal_keys if k not in val_keys)
    print(f"clip {clip_frames} frames: setups with long episodes {len(keys)} ({len(cal_keys)} calibrated; held-out "
          f"dev/test {len(held)}); long episodes in them {sum(len(long[k]) for k in keys)}; train {len(train)} "
          f"({n_cal} from calibrated setups), val {len(val)}")


def pick(out, camera_dir, stereo_dir, max_train=3000, min_iou=0.6):
    sel = json.loads((out / "sel.json").read_text())
    iou = camera_ious(camera_dir)
    ok = lambda i: iou.get(i, 0) >= min_iou and has_stereo(stereo_dir, i)  # noqa: E731
    train = [i for i in sel["train"] if ok(i)][:max_train]
    val = [i for i in sel["val"] if ok(i)]
    write_ids(out / "ids_px.txt", val + train)
    (out / "pick.json").write_text(json.dumps(dict(train=train, val=val, min_iou=min_iou,
                                                   iou={i: iou[i] for i in val + train})))
    n = len(sel["train"])
    print(f"refined {sum(i in iou for i in sel['train'])}/{n} train candidates, arm IoU >= {min_iou}: "
          f"{sum(iou.get(i, 0) >= min_iou for i in sel['train'])}, with stereo depth: {sum(map(ok, sel['train']))}; "
          f"picked train {len(train)}, val {len(val)}/{len(sel['val'])}")


def val(out, camera_dir, stereo_dir, n_val=200, max_grip=72, min_iou=0.6, seed=0):
    """Fitted candidates not in the training pick (arm IoU >= min_iou, stereo depth) whose gripper starts closing
    between step 10 and max_grip, so the grasp and what follows are inside a 97-frame clip. Episodes from setups absent
    from training come first, then round-robin over the other setups, one episode per setup per round."""
    rng = random.Random(seed)
    sel, p = json.loads((out / "sel.json").read_text()), json.loads((out / "pick.json").read_text())
    train, keys = set(p["train"]), sel["keys"]
    iou = camera_ious(camera_dir)
    cand = sorted(i for i, v in iou.items() if v >= min_iou and i not in train and has_stereo(stereo_dir, i))
    grip = first_grips(cand)
    cand = [i for i in cand if 10 <= grip[i] <= max_grip]
    seen = {keys[str(i)] for i in train}
    by = {}
    for i in cand:
        by.setdefault(keys[str(i)], []).append(i)
    for v in by.values():
        rng.shuffle(v)
    unseen = sorted((k for k in by if k not in seen), key=lambda k: rng.random())
    rest = sorted((k for k in by if k in seen), key=lambda k: rng.random())
    order = [by[k][0] for k in unseen] + interleave([by[k] for k in rest])
    order += [i for k in unseen for i in by[k][1:]]
    picked = order[:n_val]
    write_ids(out / "ids_val_px.txt", picked)
    (out / "val.json").write_text(json.dumps(dict(val=picked,
                                                  unseen_setup=[i for i in picked if keys[str(i)] not in seen],
                                                  grip={i: grip[i] for i in picked}, iou={i: iou[i] for i in picked})))
    print(f"held-out candidates {len(grip)}, grasp at steps 10-{max_grip}: {len(cand)} in {len(by)} setups "
          f"({len(unseen)} not in training); picked {len(picked)} from {len({keys[str(i)] for i in picked})} setups, "
          f"{sum(keys[str(i)] not in seen for i in picked)} from unseen setups")


def test(out, clip_frames=241, n_cand=1000, max_grip=72, seed=0):
    """Test candidates from setups with no training episode (the DROID-500 dev/test setups and every setup the training
    pick did not reach): >= clip_frames recorded steps and the gripper starting to close between step 10 and max_grip,
    round-robin over setups. Their cameras are then fitted (droid/episode_calib.py) and `pick-test` keeps the
    well-aligned ones."""
    rng = random.Random(seed)
    L = lengths(out)
    _, members = setup_calib.load_index()
    key_of = {i: k for k, m in members.items() for i in m}
    seen = {key_of[i] for i in json.loads((out / "pick.json").read_text())["train"]}
    cand = sorted(i for k, m in members.items() if k not in seen for i in m if L[i] - 1 >= clip_frames)
    grip = first_grips(cand)
    good = [i for i in cand if 10 <= grip[i] <= max_grip]
    picked = round_robin(good, key_of.get, rng)[:n_cand]
    (out / "test_cand.json").write_text(json.dumps(dict(test=picked, keys={str(i): key_of[i] for i in picked},
                                                        grip={str(i): grip[i] for i in picked})))
    write_ids(out / "ids_test_cand.txt", picked)
    print(f"unseen setups {len({key_of[i] for i in cand})}: {len(cand)} episodes >= {clip_frames} steps, grasp at "
          f"10-{max_grip}: {len(good)} in {len({key_of[i] for i in good})} setups; candidates {len(picked)} from "
          f"{len({key_of[i] for i in picked})} setups")


def pick_test(out, camera_dir, stereo_dir, n_test=500, min_iou=0.6, seed=0):
    c = json.loads((out / "test_cand.json").read_text())
    iou = camera_ious(camera_dir)
    ok = [i for i in c["test"] if iou.get(i, 0) >= min_iou and has_stereo(stereo_dir, i)]
    picked = round_robin(ok, lambda i: c["keys"][str(i)], random.Random(seed))[:n_test]
    (out / "test.json").write_text(json.dumps(dict(test=picked, keys={str(i): c["keys"][str(i)] for i in picked},
                                                   iou={str(i): iou[i] for i in picked},
                                                   grip={str(i): c["grip"][str(i)] for i in picked})))
    write_ids(out / "ids_test.txt", picked)
    print(f"candidates {len(c['test'])}: refined {sum(i in iou for i in c['test'])}, arm IoU >= {min_iou} "
          f"{sum(iou.get(i, 0) >= min_iou for i in c['test'])}, + stereo {len(ok)}; test {len(picked)} from "
          f"{len({c['keys'][str(i)] for i in picked})} setups")


# ----------------------------------------------------------------------------------------------- 97-frame clips
def short(out, per_setup=8, n_val=8, seed=0):
    """train: up to per_setup episodes with >= MIN_T_SHORT steps from each calibrated setup of video_train2k.json,
    round-robin over setups; val: n_val other calibrated setups (not DROID-500 dev/test) whose calibration reached a
    robot IoU >= VAL_MIN_IOU, one episode each."""
    import h5py

    ok = setup_calib.ok_entries()

    def long_enough(i):
        with h5py.File(data.RAW / ok[i]["raw_dir"] / "trajectory.h5", "r") as h:
            return len(h["observation/robot_state/joint_positions"]) >= MIN_T_SHORT

    def take(members, k):  # up to k episodes of a setup, random order
        found = []
        for i in rng.sample(members, len(members)):
            if long_enough(i):
                found.append(i)
                if len(found) == k:
                    break
        return found

    rng = random.Random(seed)
    _, members = setup_calib.load_index()
    usable = sorted(k for k in calibrated_setups() if k in members)
    train_set_keys = set(json.loads((SETS / "video_train2k.json").read_text())["keys"].values())
    train_keys = [k for k in usable if k in train_set_keys]
    held = droid500_setups()
    rest = [k for k in usable if k not in train_set_keys and k not in held]
    rng.shuffle(train_keys)
    rng.shuffle(rest)
    train = interleave([take(members[k], per_setup) for k in train_keys], per_setup)
    # filtered after the shuffles, so the train list does not depend on it
    rest = [k for k in rest if setup_calib_iou(k) >= VAL_MIN_IOU]
    val = []
    for k in rest:
        val += take(members[k], 1)
        if len(val) == n_val:
            break
    key_of = {i: k for k in train_keys + rest for i in members[k]}
    write_selection(out, train, val, key_of, members)
    print(f"calibrated setups with raw episodes {len(usable)} (video_train2k setups {len(train_keys)}, held-out "
          f"dev/test setups {len(held)}); train {len(train)} val {len(val)}")


def main():
    ap = argparse.ArgumentParser(prog="python -m simpwm.droid.select", description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)

    def cmd(name, help, fits=False, seed=True):
        p = sub.add_parser(name, help=help, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        p.add_argument("out", type=Path, help="selection directory")
        if fits:
            p.add_argument("--camera-dir", type=Path, required=True,
                           help="per-episode camera fits (droid/episode_calib.py)")
            p.add_argument("--stereo-dir", type=Path, required=True, help="first-frame stereo depth <idx>_ext1.npy")
            p.add_argument("--min-iou", type=float, default=0.6, help="minimum arm IoU of the fitted camera")
        if seed:
            p.add_argument("--seed", type=int, default=0, help="random seed")
        return p

    p = cmd("candidates", "long-clip training / validation candidates")
    p.add_argument("--clip-frames", type=int, default=241, help="recorded steps an episode must cover")
    p.add_argument("--per-setup", type=int, default=8, help="training candidates per setup")
    p.add_argument("--n-val", type=int, default=8, help="validation episodes (one per setup)")
    p = cmd("pick", "training set from the fitted candidates", fits=True, seed=False)
    p.add_argument("--max-train", type=int, default=3000, help="training episodes")
    p = cmd("val", "larger held-out set from the fitted candidates", fits=True)
    p.add_argument("--n-val", type=int, default=200, help="held-out episodes")
    p.add_argument("--max-grip", type=int, default=72, help="latest step at which the gripper starts closing")
    p = cmd("test", "test candidates from setups without training episodes")
    p.add_argument("--clip-frames", type=int, default=241, help="recorded steps an episode must cover")
    p.add_argument("--n-cand", type=int, default=1000, help="test candidates")
    p.add_argument("--max-grip", type=int, default=72, help="latest step at which the gripper starts closing")
    p = cmd("pick-test", "test set from the fitted test candidates", fits=True)
    p.add_argument("--n-test", type=int, default=500, help="test episodes")
    p = cmd("short", "97-frame clips from the video_train2k.json setups")
    p.add_argument("--per-setup", type=int, default=8, help="training episodes per setup")
    p.add_argument("--n-val", type=int, default=8, help="validation episodes (one per setup)")
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    if a.cmd == "candidates":
        candidates(a.out, a.clip_frames, a.per_setup, a.n_val, a.seed)
    elif a.cmd == "pick":
        pick(a.out, a.camera_dir, a.stereo_dir, a.max_train, a.min_iou)
    elif a.cmd == "val":
        val(a.out, a.camera_dir, a.stereo_dir, a.n_val, a.max_grip, a.min_iou, a.seed)
    elif a.cmd == "test":
        test(a.out, a.clip_frames, a.n_cand, a.max_grip, a.seed)
    elif a.cmd == "pick-test":
        pick_test(a.out, a.camera_dir, a.stereo_dir, a.n_test, a.min_iou, a.seed)
    else:
        short(a.out, a.per_setup, a.n_val, a.seed)


if __name__ == "__main__":
    main()
