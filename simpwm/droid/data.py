"""DROID episode loader. Two sources, chosen by what exists on disk:
  rlds  RLDS 1.0.1 shards under `paths.DROID_ROOT`/1.0.1 (images, joints, gripper, language) with the raw metadata
        (camera extrinsics) and its manifest under `paths.DROID_ROOT`/raw_lowdim_1.0.1
  raw   per-episode raw directories under `paths.DROID_RAW` (ext1.mp4, ext2.mp4 at 1280x720, trajectory.h5,
        metadata.json); the MP4s are resized to RLDS' 320x180 (640x360 with `config.cfg.hires`, sim/scale.py).
        Episode index i is line i of the shipped list resources/droid/episodes.txt (the 42,927 usable episodes of the
        release), so indices do not depend on which episodes were downloaded.
        Instruction: language_instruction1 of the DROID language annotations (`paths.DATA`/
        droid_language_annotations.json, from the HF dataset KarlP/droid), else the metadata's current_task.
The rlds source is used when its manifest exists. Episode indices differ between the two sources.

Episode dict (the generic format every dataset loader returns):
  frames   (T, H, W, 3) uint8      the chosen camera's RGB stream
  arm_q    (T, n_arm) float        joint positions
  grip     (T,) float in [0, 1]    observed gripper closure (1 = closed)
  cam      Camera                  logged intrinsics + camera-to-world (world = robot base frame)
  fps      float
  text     str                     task instruction
  meta     dict
"""

from __future__ import annotations

import glob
import json

import cv2
import numpy as np

from simpwm import paths
from simpwm.core.camera import Camera
from simpwm.sim import scale

RLDS = paths.DROID_ROOT / "1.0.1"
RAW_LOWDIM = paths.DROID_ROOT / "raw_lowdim_1.0.1"
DROID_RAW = paths.DROID_RAW
EPISODES = paths.RESOURCES / "droid/episodes.txt"  # raw-release directory names; line i = episode index i
LANGUAGE = paths.DATA / "droid_language_annotations.json"
SOURCE = "rlds" if (RAW_LOWDIM / "manifest.jsonl").exists() else "raw"
RAW = RAW_LOWDIM if SOURCE == "rlds" else DROID_RAW  # per-episode dirs (metadata*.json + trajectory.h5)
# ZED 2 rectified HD720 focal length is ~530 px (refined per setup / episode by robot-silhouette alignment).
ZED_F720 = 530.0
CAM_KEYS = {"ext1": "exterior_image_1_left", "ext2": "exterior_image_2_left", "wrist": "wrist_image_left"}


def manifest():
    """Episode rows {raw_dir, lab, status, metadata_files, ...}; the index into the "ok" rows is the episode index."""
    if SOURCE == "rlds":
        return [json.loads(line) for line in open(RAW_LOWDIM / "manifest.jsonl")]
    return [dict(raw_dir=d, lab=d.split("+")[0], status="ok", metadata_files=["metadata.json"])
            for d in EPISODES.read_text().split()]


_LANG = None


def _instruction(meta):
    global _LANG
    if _LANG is None:
        _LANG = json.loads(LANGUAGE.read_text()) if LANGUAGE.exists() else {}
    return (_LANG.get(meta["uuid"]) or {}).get("language_instruction1") or meta["current_task"]


def _read_record(shard, idx):
    from tfrecord import example_pb2
    from tfrecord.reader import tfrecord_iterator

    path = glob.glob(str(RLDS / f"droid_101-train.tfrecord-{shard.split('-', 1)[1]}"))[0]
    for i, rec in enumerate(tfrecord_iterator(path)):
        if i == idx:
            ex = example_pb2.Example()
            ex.ParseFromString(rec)
            return ex.features.feature
    raise IndexError(f"{shard} has no record {idx}")


def _vals(f, k):
    v = f[k]
    return getattr(v, v.WhichOneof("kind")).value


def logged_camera(meta, camera, H, W):
    """The camera as DROID logged it: the metadata's extrinsic and the nominal ZED focal length, at H x W."""
    ext = meta[f"{camera}_cam_extrinsics"]
    ext = np.array(json.loads(ext) if isinstance(ext, str) else ext, float)
    s = W / 1280.0
    K = np.array([[ZED_F720 * s, 0, W / 2], [0, ZED_F720 * s, H / 2], [0, 0, 1]])
    return Camera.from_xyz_euler(ext, K, H, W)


def _raw_video(d, camera, n=None):
    cap, frames = cv2.VideoCapture(str(d / f"{camera}.mp4")), []
    wh = scale.sim_hw()[::-1]  # the simulation frame: RLDS' 320x180, or 640x360 with config.cfg.hires
    while n is None or len(frames) < n:
        ok, im = cap.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(cv2.resize(im, wh, interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB))
    cap.release()
    return frames


def first_frames(entry, cams=("ext1", "ext2", "wrist")):
    """t = 0 image of each camera (left views), the episode length and the camera serials."""
    if SOURCE == "raw":  # the raw copy has no wrist recording
        import h5py

        d = DROID_RAW / entry["raw_dir"]
        meta = json.loads((d / "metadata.json").read_text())
        cams = [c for c in cams if (d / f"{c}.mp4").exists()]
        with h5py.File(d / "trajectory.h5", "r") as h:
            T = len(h["observation/robot_state/joint_positions"])
        cap = cv2.VideoCapture(str(d / "ext1.mp4"))
        T = min(T, int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))  # as load(): the recordings have one frame fewer
        cap.release()
        return dict(frames={c: _raw_video(d, c, 1)[0] for c in cams}, T=T,
                    serials={c: meta[f"{c}_cam_serial"] for c in cams})
    f = _read_record(entry["shard"], entry["i"])
    meta = json.loads(next((RAW / entry["raw_dir"]).glob("metadata_*.json")).read_text())
    frames = {c: cv2.cvtColor(cv2.imdecode(np.frombuffer(_vals(f, f"steps/observation/{CAM_KEYS[c]}")[0], np.uint8),
                                           cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB) for c in cams}
    return dict(frames=frames, T=len(_vals(f, "steps/is_first")), serials={c: meta[f"{c}_cam_serial"] for c in cams})


def _episode(frames, arm_q, grip, cart, text, meta, camera):
    H, W = frames.shape[1:3]
    return dict(frames=frames, arm_q=arm_q, grip=grip, cam=logged_camera(meta, camera, H, W), fps=15.0, text=text,
                meta=dict(dataset="droid", uuid=meta["uuid"], success=meta["success"], lab=meta["lab"],
                          task=meta["current_task"], camera=camera, serial=meta[f"{camera}_cam_serial"], cart=cart))


def load(entry, camera="ext1"):
    """Episode dict (module docstring) of a manifest row."""
    return (_load_rlds if SOURCE == "rlds" else _load_raw)(entry, camera)


def _load_rlds(entry, camera):
    f = _read_record(entry["shard"], entry["i"])
    T = len(_vals(f, "steps/is_first"))
    frames = np.stack([cv2.cvtColor(cv2.imdecode(np.frombuffer(b, np.uint8), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
                       for b in _vals(f, f"steps/observation/{CAM_KEYS[camera]}")])
    arm_q = np.array(_vals(f, "steps/observation/joint_position")).reshape(T, 7)
    # the observed closure, not the command: it stops short on a held object, which the gripper-signal rules rely on
    grip = np.array(_vals(f, "steps/observation/gripper_position")).reshape(T)
    text = next((t.decode() for t in _vals(f, "steps/language_instruction") if t), "")
    meta = json.loads(next((RAW / entry["raw_dir"]).glob("metadata_*.json")).read_text())
    cart = np.array(_vals(f, "steps/observation/cartesian_position")).reshape(T, 6)
    return _episode(frames, arm_q, grip, cart, text, meta, camera)


def _load_raw(entry, camera):
    import h5py

    d = DROID_RAW / entry["raw_dir"]
    meta = json.loads((d / "metadata.json").read_text())
    frames = _raw_video(d, camera)
    with h5py.File(d / "trajectory.h5", "r") as h:
        rs = h["observation/robot_state"]
        arm_q, cart, grip = rs["joint_positions"][:], rs["cartesian_position"][:], rs["gripper_position"][:]
    # the mp4 has one frame fewer than the trajectory: frame i <-> step i
    T = min(len(frames), len(arm_q))
    return _episode(np.stack(frames[:T]), arm_q[:T], grip[:T], cart[:T], _instruction(meta), meta, camera)
