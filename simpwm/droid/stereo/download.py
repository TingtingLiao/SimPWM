"""Download the raw DROID stereo recordings (ZED .svo, left + right views) and metadata of selected episodes.

RLDS and the raw MP4s only have the left view of each camera; the public raw release
(gs://gresearch/robotics/droid_raw/1.0.1, served over plain HTTPS) has the .svo recordings with both. Episodes are
droid/data.py indices; with the raw source the bucket directory of an episode comes from its local metadata.json.

Output: OUT/<raw_dir>/{metadata_*.json, recordings/SVO/*.svo, recordings/MP4/*.mp4}, OUT/index.json {idx: raw_dir} and
OUT/status.jsonl (one line per episode, read by droid/stereo/extract.py). Files already present with the listed size are
skipped, so an interrupted download can be restarted.

usage: python -m simpwm.droid.stereo.download --episodes FILE [--out DIR] [--svo ext1] [--threads 16]
"""

import argparse
import json
import time
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from simpwm import paths

DOWNLOAD_DIR = paths.DATA / "droid_raw_stereo"
BUCKET, PREFIX = "gresearch", "robotics/droid_raw/1.0.1/"
API = "https://storage.googleapis.com/storage/v1/b/{}/o?prefix={}&fields=items(name,size),nextPageToken"
GET = "https://storage.googleapis.com/{}/{}"


def read_episodes(f):
    """Episode indices of a whitespace-separated list, or of a selection JSON with lists under "dev" / "test" /
    "train" / "val" (droid/select.py outputs; also {split: {outcome: [idx]}})."""
    f = Path(f)
    if f.suffix != ".json":
        return sorted({int(i) for i in f.read_text().split()})
    d = json.loads(f.read_text())
    return sorted({i for sp in ("dev", "test", "train", "val") for v in
                   (d.get(sp, {}).values() if isinstance(d.get(sp), dict) else [d.get(sp, [])]) for i in v})


def _retry(fn, tries=5):
    for k in range(tries):
        try:
            return fn()
        except Exception:
            if k == tries - 1:
                raise
            time.sleep(2 ** k)


def remote_dir(raw_dir):
    """Bucket directory of an episode, relative to PREFIX."""
    from simpwm.droid import data

    if data.SOURCE != "raw":
        return raw_dir
    m = json.loads((data.RAW / raw_dir / "metadata.json").read_text())
    return m["lab"] + "/" + m["hdf5_path"].rsplit("/", 1)[0]


def listing(remote):
    q = urllib.parse.quote(PREFIX + remote + "/", safe="")
    with urllib.request.urlopen(API.format(BUCKET, q), timeout=60) as r:
        return [(it["name"], int(it["size"])) for it in json.load(r).get("items", [])]


def fetch(name, size, dst):
    if dst.exists() and dst.stat().st_size >= size:
        return 0
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(dst.suffix + ".part")
    url = GET.format(BUCKET, urllib.parse.quote(name))
    with urllib.request.urlopen(url, timeout=300) as r, open(tmp, "wb") as f:
        while chunk := r.read(1 << 20):
            f.write(chunk)
    if tmp.stat().st_size < size:
        raise IOError(f"short download {name}: {tmp.stat().st_size} < {size}")
    tmp.rename(dst)
    return size


def one(out, idx, raw_dir, svo="all", mp4="none"):
    """Download one episode. svo: "all" | "ext1" .svo files; mp4: "none" | "ext1" | "all" (every recording)."""
    try:
        remote = remote_dir(raw_dir)
        local = lambda n: out / raw_dir / n[len(PREFIX + remote) + 1:]  # noqa: E731
        listed = _retry(lambda: listing(remote))
        files = [(n, s) for n, s in listed
                 if ("/recordings/SVO/" in n and svo != "ext1") or ("/recordings/" in n and mp4 == "all")
                 or n.rsplit("/", 1)[-1].startswith("metadata_")]
        if mp4 == "ext1" or svo == "ext1":  # the ext1 recordings are named in the metadata
            for n, s in files:
                if n.rsplit("/", 1)[-1].startswith("metadata_"):
                    _retry(lambda: fetch(n, s, local(n)))
                    m = json.loads(local(n).read_text())
                    want = ([m["ext1_mp4_path"].rsplit("/", 1)[-1]] if mp4 == "ext1" else []) + \
                           ([m["ext1_svo_path"].rsplit("/", 1)[-1]] if svo == "ext1" else [])
                    files += [(q, z) for q, z in listed if "/recordings/" in q and q.rsplit("/", 1)[-1] in want]
                    break
        n_svo = sum(n.endswith(".svo") for n, _ in files)
        got = sum(_retry(lambda n=n, s=s: fetch(n, s, local(n))) for n, s in files)
        return dict(idx=idx, raw_dir=raw_dir, n_files=len(files), n_svo=n_svo, bytes=got, ok=n_svo > 0)
    except Exception as e:
        return dict(idx=idx, raw_dir=raw_dir, ok=False, error=repr(e))


def main():
    ap = argparse.ArgumentParser(prog="python -m simpwm.droid.stereo.download", description=__doc__.split("\n\n")[0])
    ap.add_argument("--episodes", type=Path, required=True,
                    help="episode list: whitespace-separated indices (.txt) or a droid/select.py selection (.json)")
    ap.add_argument("--out", type=Path, default=DOWNLOAD_DIR,
                    help="download root (default: $SIMPWM_DATA/droid_raw_stereo)")
    ap.add_argument("--svo", choices=("all", "ext1"), default="all",
                    help="which .svo recordings (the proxies use ext1)")
    ap.add_argument("--mp4", choices=("none", "ext1", "all"), default="none", help="also download MP4 recordings")
    ap.add_argument("--threads", type=int, default=16, help="parallel downloads")
    args = ap.parse_args()

    from simpwm.droid import setup_calib

    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    ok = setup_calib.ok_entries()
    eps = {i: ok[i]["raw_dir"] for i in read_episodes(args.episodes)}
    old = json.loads((out / "index.json").read_text()) if (out / "index.json").exists() else {}
    (out / "index.json").write_text(json.dumps(dict(old, **{str(i): r for i, r in eps.items()}), indent=1))
    print(len(eps), "episodes", flush=True)
    t0, n_ok = time.time(), 0
    with ThreadPoolExecutor(args.threads) as ex, open(out / "status.jsonl", "a") as log:
        for k, r in enumerate(ex.map(lambda a: one(out, *a, svo=args.svo, mp4=args.mp4), eps.items()), 1):
            n_ok += r["ok"]
            log.write(json.dumps(r) + "\n")
            log.flush()
            if k % 10 == 0 or not r["ok"]:
                print(f"{k}/{len(eps)} ok {n_ok} ({time.time() - t0:.0f}s)", "" if r["ok"] else r, flush=True)
    print("done", n_ok, "/", len(eps), flush=True)


if __name__ == "__main__":
    main()
