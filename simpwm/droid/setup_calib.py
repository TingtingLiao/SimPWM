"""Per-setup camera calibration: the test-time camera protocol.

A camera setup = same lab, ext1 camera serial and rounded logged ext1 extrinsic (`key_from_meta`; it depends only on the
episode metadata, so it is the same for the RLDS and the raw DROID source). The camera is calibrated ONCE per setup from
robot silhouettes in up to N_OTHER OTHER episodes of the setup (FRAMES_PER_EP frames each; the evaluated episode's
frames are never used), then reused by every episode of the setup, like calibrating once after mounting. Calibrations
are cached in resources/droid/setup_cameras/<key>.json (shipped for the paper's setups); a missing one is computed on
first use, with a lock file so that parallel workers calibrate each setup once.

  episode_camera(idx, others, ...)  pre-calibrated camera of an episode (calibrates its setup on first use)
  load_index()                      setup key of every usable episode and the members of every setup (built on first
                                    use into `paths.DATA`/droid_setup_index.json)
"""

from __future__ import annotations

import hashlib
import json
import os
import threading
import time

import numpy as np

from simpwm import paths

CACHE = paths.RESOURCES / "droid" / "setup_cameras"
INDEX = paths.DATA / "droid_setup_index.json"
N_OTHER, FRAMES_PER_EP = 6, 5
LOCK_STALE_S = 600  # the calibrating worker touches its lock every LOCK_BEAT_S; an older lock belongs to a dead worker
LOCK_BEAT_S = 60


def key_from_meta(entry, m):
    ext = m["ext1_cam_extrinsics"]
    ext = json.loads(ext) if isinstance(ext, str) else ext
    ext = np.round(np.array(ext), 4).tolist()
    return f"{entry['lab']}_{m['ext1_cam_serial']}_{hashlib.md5(json.dumps(ext).encode()).hexdigest()[:10]}"


def setup_key(entry):
    from simpwm.droid import data

    return key_from_meta(entry, json.loads(next((data.RAW / entry["raw_dir"]).glob("metadata*.json")).read_text()))


def ok_entries():
    """Manifest rows of the usable episodes; the position in this list is the episode index."""
    from simpwm.droid import data

    return [r for r in data.manifest() if r["status"] == "ok"]


def build_index():
    from simpwm.droid import data

    key = {}
    for i, e in enumerate(ok_entries()):
        try:
            m = json.loads((data.RAW / e["raw_dir"] / e["metadata_files"][0]).read_text())
            key[i] = key_from_meta(e, m)
        except Exception:
            key[i] = None
    members = {}
    for i, k in key.items():
        if k:
            members.setdefault(k, []).append(i)
    INDEX.parent.mkdir(parents=True, exist_ok=True)
    INDEX.write_text(json.dumps(dict(key={str(i): k for i, k in key.items()}, members=members)))
    return key, members


def load_index():
    """({idx: setup key or None}, {setup key: [idx]})."""
    if not INDEX.exists():
        return build_index()
    d = json.loads(INDEX.read_text())
    return {int(i): k for i, k in d["key"].items()}, d["members"]


def same_setup(idx):
    """The other episodes of idx's setup (the input of its setup calibration)."""
    key, members = load_index()
    return [j for j in members[key[idx]] if j != idx]


def calibrate_setup(key, members, ok, emb, cam0):
    """(camera, report) of setup `key`: the cached calibration, else one fitted on `members` and cached."""
    from simpwm.core.camera import Camera

    CACHE.mkdir(parents=True, exist_ok=True)
    f = CACHE / f"{key}.json"
    if f.exists():
        c = json.loads(f.read_text())
        emb.set_gripper_yaw(c["gripper_yaw"])
        cam = Camera(c["K"], c["T_wc"], cam0.H, cam0.W, c.get("dist", (0.0, 0.0)))
        return cam, dict(c["report"], cached=True, setup=key)
    lock = CACHE / f"{key}.lock"
    try:  # one calibration per setup across workers
        fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        os.close(fd)
    except FileExistsError:
        def age():  # None once the other worker finished (or gave up) and removed its lock
            try:
                return time.time() - lock.stat().st_mtime
            except FileNotFoundError:
                return None

        a = age()
        if a is None:
            return calibrate_setup(key, members, ok, emb, cam0)
        if a < LOCK_STALE_S:  # someone is calibrating this setup right now: wait for it
            while True:
                if f.exists():
                    return calibrate_setup(key, members, ok, emb, cam0)
                a = age()
                if a is None:
                    return calibrate_setup(key, members, ok, emb, cam0)
                if a >= LOCK_STALE_S:
                    break
                time.sleep(15)
        lock.touch()  # stale (dead worker): take over
    stop = threading.Event()

    def beat():  # heartbeat: keeps the lock fresh while this worker is alive
        while not stop.wait(LOCK_BEAT_S):
            try:
                lock.touch()
            except OSError:
                pass

    threading.Thread(target=beat, daemon=True).start()
    try:
        return _calibrate(key, members, ok, emb, cam0, f)
    finally:
        stop.set()
        lock.unlink(missing_ok=True)


def _calibrate(key, members, ok, emb, cam0, f):
    from simpwm.droid import data
    from simpwm.sim import camera_calib

    pick = [members[int(i)] for i in np.linspace(0, len(members) - 1, min(N_OTHER, len(members)))]
    frames, arm_q, grip = [], [], []
    for j in pick:
        e = data.load(ok[j])
        t = np.linspace(0, len(e["frames"]) - 1, FRAMES_PER_EP).astype(int)
        frames.append(e["frames"][t]), arm_q.append(e["arm_q"][t]), grip.append(e["grip"][t])
    frames, arm_q, grip = np.concatenate(frames), np.concatenate(arm_q), np.concatenate(grip)
    cam, rep = camera_calib.calibrate_camera(frames, arm_q, grip, cam0, emb, n=len(frames))
    rep = {k: v for k, v in rep.items() if k not in ("x", "coarse_stage")}
    rep.update(setup=key, calib_episodes=[int(j) for j in pick], n_frames=int(len(frames)))
    f.write_text(json.dumps(dict(K=cam.K.tolist(), T_wc=cam.T_wc.tolist(), dist=list(cam.dist),
                                 gripper_yaw=rep.get("gripper_yaw", 90), report=rep), indent=1, default=float))
    return cam, dict(rep, cached=False)


def episode_camera(idx, others, ok, emb, cam0):
    """Pre-calibrated camera for episode idx: its setup's calibration from `others` (episodes of the same setup)."""
    return calibrate_setup(setup_key(ok[idx]), others, ok, emb, cam0)
