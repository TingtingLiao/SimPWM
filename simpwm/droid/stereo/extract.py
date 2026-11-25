"""First-frame stereo pairs + FoundationStereo depth of the downloaded DROID recordings (droid/stereo/download.py).

Stages, each run in its own environment (pyzed needs numpy 1 and the ZED SDK, FoundationStereo its own venv; see
docs/dataset.md), each over shard TASK of N_TASKS of the downloaded episodes:
  rlds    main env, CPU: first frames of the chosen cameras as the loader sees them -> OUT/rlds/<idx>.npz (+ length T)
  svo     ZED venv, GPU: per episode and camera, .svo frame 0 (= recorded step 0 when the lengths agree; otherwise the
          frame whose left view best matches the loader's first frame among the first SEARCH frames; the match errors
          are recorded either way) as its full-resolution rectified left/right pair -> OUT/pairs/<idx>_<cam>_{left,
          right}.png + <idx>_<cam>.json (rectified fx/fy/cx/cy, baseline, matched frame, match errors, lengths,
          `flipped` for upside-down mounts whose recorded image is the rotated .svo right view)
  svoseq  ZED venv, GPU: later ext1 frames of recordings already paired by `svo` (evaluation depth), see `svoseq`
  fs      FoundationStereo venv, GPU: metric depth of every pair in OUT/pairs -> OUT/depth/<key>.npy (float16 metres,
          NaN = no stereo support); OUT/depth/<idx>_ext1.npy is the input of `droid/run.py --stereo-dir`

usage: python -m simpwm.droid.stereo.extract {rlds,svo,svoseq,fs} [--out DIR] [--task T --n-tasks N] [options]
"""

import argparse
import glob
import json
import traceback
from pathlib import Path

import numpy as np

from simpwm import paths
from simpwm.droid.stereo.download import DOWNLOAD_DIR

SEARCH = 40
SMALL = (320, 180)


def _episodes(download_dir):
    st = [json.loads(line) for line in open(download_dir / "status.jsonl")]
    return sorted({r["idx"] for r in st if r["ok"]})


def _metadata(download_dir, index, idx):
    return json.loads(Path(glob.glob(str(download_dir / index[str(idx)] / "metadata_*.json"))[0]).read_text())


def _svo_path(download_dir, index, idx, meta, cam):
    return download_dir / index[str(idx)] / "recordings/SVO" / Path(meta[f"{cam}_svo_path"]).name


def _open_svo(sl, svo_path, settings):
    init = sl.InitParameters()
    init.optional_settings_path = str(settings) + "/"
    init.set_from_svo_file(str(svo_path))
    init.svo_real_time_mode = False
    init.depth_mode = sl.DEPTH_MODE.NONE
    init.coordinate_units = sl.UNIT.METER  # the default is millimetres (baseline_m and the depth)
    zed = sl.Camera()
    err = zed.open(init)
    if err != sl.ERROR_CODE.SUCCESS:
        raise RuntimeError(f"open {svo_path}: {err}")
    return zed


def _save_pair(sl, zed, prefix):
    """Write the current frame's rectified views as <prefix>_left.png / <prefix>_right.png."""
    import cv2

    for name, view in (("left", sl.VIEW.LEFT), ("right", sl.VIEW.RIGHT)):
        m = sl.Mat()
        zed.retrieve_image(m, view)
        cv2.imwrite(f"{prefix}_{name}.png", m.get_data()[:, :, :3])


def rlds(out, cams, download_dir, task=0, n_tasks=1):
    from simpwm.droid import data, setup_calib

    ok = setup_calib.ok_entries()
    d = out / "rlds"
    d.mkdir(parents=True, exist_ok=True)
    for idx in _episodes(download_dir)[task::n_tasks]:
        f = d / f"{idx}.npz"
        if f.exists():
            continue
        try:
            ep = data.first_frames(ok[idx], cams)
            np.savez_compressed(f, T=ep["T"], **ep["frames"], serials=json.dumps(ep["serials"]))
            print("rlds", idx, flush=True)
        except Exception:
            print(f"FAILED rlds {idx}\n{traceback.format_exc()}", flush=True)


def _match(zed, sl, ref):
    """index of the .svo frame (among the first SEARCH) whose left view best matches the loader's frame (RGB, small)."""
    import cv2

    m, errs = sl.Mat(), []
    zed.set_svo_position(0)
    for _ in range(SEARCH):
        if zed.grab(sl.RuntimeParameters()) != sl.ERROR_CODE.SUCCESS:
            break
        zed.retrieve_image(m, sl.VIEW.LEFT)
        im = cv2.resize(cv2.cvtColor(m.get_data()[:, :, :3], cv2.COLOR_BGR2RGB), SMALL, interpolation=cv2.INTER_AREA)
        errs.append(float(np.abs(im.astype(np.float32) - cv2.resize(ref, SMALL).astype(np.float32)).mean()))
    return int(np.argmin(errs)), errs


def svo(out, cams, download_dir, settings, task=0, n_tasks=1):
    import urllib.request

    import cv2
    import pyzed.sl as sl

    d = out / "pairs"
    d.mkdir(parents=True, exist_ok=True)
    settings.mkdir(parents=True, exist_ok=True)
    index = json.loads((download_dir / "index.json").read_text())
    eps = [i for k, i in enumerate(_episodes(download_dir)) if k % n_tasks == task]
    for idx in eps:
        ref = np.load(out / "rlds" / f"{idx}.npz")
        meta = _metadata(download_dir, index, idx)
        for cam in cams:
            k = f"{idx}_{cam}"
            if (d / f"{k}.json").exists():
                continue
            try:
                svo_path = _svo_path(download_dir, index, idx, meta, cam)
                conf = settings / f"SN{svo_path.stem}.conf"  # factory calibration of the camera (needs internet)
                if not conf.exists():
                    urllib.request.urlretrieve(f"https://calib.stereolabs.com/?SN={svo_path.stem}", conf)
                zed = _open_svo(sl, svo_path, settings)
                # the recorded steps are the .svo frames one to one whenever the lengths agree, so frame 0 is the first
                # frame; image matching only for a length mismatch (the first frames are nearly static, so matching
                # alone picks among near-identical frames by noise)
                best, errs = _match(zed, sl, ref[cam])
                if zed.get_svo_number_of_frames() == int(ref["T"]):
                    best = 0
                zed.set_svo_position(best)
                zed.grab(sl.RuntimeParameters())
                _save_pair(sl, zed, d / k)
                # upside-down mounts: the recorded image is the .svo right view rotated 180 deg (see fs)
                small = lambda im: cv2.resize(cv2.cvtColor(im, cv2.COLOR_BGR2RGB), SMALL, interpolation=cv2.INTER_AREA)  # noqa: E731
                rf = cv2.resize(ref[cam], SMALL).astype(np.float32)
                lv, rv = (cv2.imread(str(d / f"{k}_{s}.png")) for s in ("left", "right"))
                err_up = float(np.abs(small(lv).astype(np.float32) - rf).mean())
                err_rot = float(np.abs(small(rv[::-1, ::-1].copy()).astype(np.float32) - rf).mean())
                info = zed.get_camera_information()
                cal = info.camera_configuration.calibration_parameters
                res = info.camera_configuration.resolution
                rep = dict(idx=idx, cam=cam, serial=info.serial_number, resolution=[res.width, res.height],
                           fx=cal.left_cam.fx, fy=cal.left_cam.fy, cx=cal.left_cam.cx, cy=cal.left_cam.cy,
                           baseline_m=cal.get_camera_baseline(), svo_frames=zed.get_svo_number_of_frames(),
                           rlds_T=int(ref["T"]), matched_frame=best, match_err=errs[best],
                           match_err_frame0=errs[0], match_err_second=float(sorted(errs)[1]) if len(errs) > 1 else None,
                           err_upright=err_up, err_rot180_right=err_rot, flipped=err_rot < err_up)
                zed.close()
                (d / f"{k}.json").write_text(json.dumps(rep, indent=1))
                print("svo", k, "frame", best, "err", round(errs[best], 2), flush=True)
            except Exception:
                print(f"FAILED svo {k}\n{traceback.format_exc()}", flush=True)


def svoseq(out, first_pairs, ids, steps, download_dir, settings, task=0, n_tasks=1):
    """Later frames of the ext1 recordings paired by `svo` (FIRST_PAIRS/pairs): .svo frame matched_frame + t for every
    t in `steps` -> OUT/pairs/<idx>_ext1_t<t>_{left,right}.png + .json (the first frame's json with matched_frame + t),
    for `fs`. Only recordings whose .svo and recorded lengths agree (frame t = step t)."""
    import pyzed.sl as sl

    src = first_pairs / "pairs"
    d = out / "pairs"
    d.mkdir(parents=True, exist_ok=True)
    index = json.loads((download_dir / "index.json").read_text())
    for idx in ids[task::n_tasks]:
        k0 = f"{idx}_ext1"
        try:
            rep = json.loads((src / f"{k0}.json").read_text())
            if rep["svo_frames"] != rep["rlds_T"] or all((d / f"{k0}_t{t}.json").exists() for t in steps):
                continue
            meta = _metadata(download_dir, index, idx)
            zed = _open_svo(sl, _svo_path(download_dir, index, idx, meta, "ext1"), settings)
            for t in steps:
                f = rep["matched_frame"] + t
                if f >= rep["svo_frames"]:
                    break
                zed.set_svo_position(f)
                zed.grab(sl.RuntimeParameters())
                _save_pair(sl, zed, d / f"{k0}_t{t}")
                (d / f"{k0}_t{t}.json").write_text(json.dumps(dict(rep, matched_frame=f, step=t), indent=1))
            zed.close()
            print("svoseq", k0, flush=True)
        except Exception:
            print(f"FAILED svoseq {k0}\n{traceback.format_exc()}", flush=True)


def fs(out, task=0, n_tasks=1):
    import imageio.v2 as imageio
    import torch

    from simpwm.droid.stereo import foundation

    pd, dd = out / "pairs", out / "depth"
    dd.mkdir(parents=True, exist_ok=True)
    keys = sorted(p.stem for p in pd.glob("*.json"))
    keys = [k for j, k in enumerate(keys) if j % n_tasks == task]
    model, cfg = foundation.load()
    for k in keys:
        if (dd / f"{k}.npy").exists():
            continue
        try:
            meta = json.loads((pd / f"{k}.json").read_text())
            left = imageio.imread(pd / f"{k}_left.png")[..., :3]
            right = imageio.imread(pd / f"{k}_right.png")[..., :3]
            if meta.get("flipped"):
                # camera mounted upside down: the recorded image is the .svo RIGHT view rotated 180 deg (that sensor is
                # the physically left one), and the rotated left view is its partner on the right
                left, right = right[::-1, ::-1].copy(), left[::-1, ::-1].copy()
            disp = foundation.disparity(model, cfg, left, right)
            disp[(np.arange(disp.shape[1])[None] - disp) < 0] = np.nan
            depth = meta["fx"] * meta["baseline_m"] / np.where(disp > 0.5, disp, np.nan)
            np.save(dd / f"{k}.npy", depth.astype(np.float16))
            print("fs", k, "valid", round(float(np.isfinite(depth).mean()), 3), flush=True)
        except Exception:
            print(f"FAILED fs {k}\n{traceback.format_exc()}", flush=True)
        torch.cuda.empty_cache()


def main():
    ap = argparse.ArgumentParser(prog="python -m simpwm.droid.stereo.extract", description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="stage", required=True)
    stages = {}
    for name, help in (("rlds", "first frames as the loader sees them (main env, CPU)"),
                       ("svo", "rectified first-frame stereo pairs (ZED venv)"),
                       ("svoseq", "rectified pairs of later ext1 frames (ZED venv)"),
                       ("fs", "FoundationStereo depth of every pair (FoundationStereo venv)")):
        p = stages[name] = sub.add_parser(name, help=help)
        p.add_argument("--out", type=Path, default=paths.RUNS / "stereo", help="output root (rlds/, pairs/, depth/)")
        p.add_argument("--task", type=int, default=0, help="shard index")
        p.add_argument("--n-tasks", type=int, default=1, help="number of shards")
        if name != "fs":
            p.add_argument("--download-dir", type=Path, default=DOWNLOAD_DIR,
                           help="droid/stereo/download.py output (index.json, status.jsonl)")
        if name in ("svo", "svoseq"):
            p.add_argument("--zed-settings", type=Path, required=True,
                           help="directory of the ZED factory calibrations SN<serial>.conf (fetched when missing)")
        if name in ("rlds", "svo"):
            p.add_argument("--cams", default="ext1,ext2,wrist", help="comma-separated cameras (the proxies use ext1)")
    p = stages["svoseq"]
    p.add_argument("--first-pairs", type=Path, default=paths.RUNS / "stereo",
                   help="output root of the `svo` stage whose pairs/<idx>_ext1.json give the matched first frame")
    p.add_argument("--ids", type=Path, required=True, help="whitespace-separated episode indices")
    p.add_argument("--steps", default=",".join(map(str, range(24, 241, 24))), help="comma-separated recorded steps")
    a = ap.parse_args()
    if a.stage == "rlds":
        rlds(a.out, tuple(a.cams.split(",")), a.download_dir, a.task, a.n_tasks)
    elif a.stage == "svo":
        svo(a.out, tuple(a.cams.split(",")), a.download_dir, a.zed_settings, a.task, a.n_tasks)
    elif a.stage == "svoseq":
        ids = [int(i) for i in a.ids.read_text().split()]
        svoseq(a.out, a.first_pairs, ids, [int(t) for t in a.steps.split(",")], a.download_dir, a.zed_settings,
               a.task, a.n_tasks)
    else:
        fs(a.out, a.task, a.n_tasks)


if __name__ == "__main__":
    main()
