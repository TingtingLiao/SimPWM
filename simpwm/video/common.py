"""Helpers shared by the proxy -> video datasets and the evaluation: video files, dataset.jsonl rows, captions and the
command-line parser of the entry points."""

from __future__ import annotations

import argparse
import json
import os
from itertools import islice
from pathlib import Path

import cv2
import imageio.v2 as imageio


def parser(doc: str) -> argparse.ArgumentParser:
    """Argument parser of an entry point (description = its module docstring). An argument @FILE is replaced by the
    whitespace-separated words of FILE (e.g. an id list)."""
    p = argparse.ArgumentParser(description=doc, formatter_class=argparse.RawDescriptionHelpFormatter,
                                fromfile_prefix_chars="@")
    p.convert_arg_line_to_args = str.split
    return p


def size_arg(s: str) -> tuple[int, int]:
    """'WxH' -> (W, H)."""
    w, h = s.lower().split("x")
    return int(w), int(h)


def read_frames(path, n=None, size=None, interpolation=cv2.INTER_AREA) -> list:
    """The first n frames (all if None) of a video as RGB arrays, resized to size = (W, H) if given."""
    r = imageio.get_reader(path)
    # (not list(reader): the ffmpeg reader reports an infinite length, which list() preallocates)
    frames = [f if size is None else cv2.resize(f, size, interpolation=interpolation) for f in islice(r, n)]
    r.close()
    return frames


def write_video(path, frames, fps, quality=9):
    imageio.mimwrite(path, frames, fps=fps, quality=quality, macro_block_size=1)


def read_jsonl(path) -> list[dict]:
    return [json.loads(line) for line in open(path)]


def write_jsonl(path, rows):
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def link(src, dst):
    """Symlink dst -> src, replacing an existing dst."""
    dst = Path(dst)
    if dst.is_symlink() or dst.exists():
        dst.unlink()
    os.symlink(src, dst)


def caption(scene: str, task: str, subject: str) -> str:
    """The LTX caption: '<scene> <subject> performs the task: <task>. Realistic lighting, natural colors.' (no task
    sentence when task is empty)."""
    task = f" {subject} performs the task: {task[0].lower() + task[1:]}." if task else ""
    return scene + task + " Realistic lighting, natural colors."


def caption_task(text: str) -> str:
    """The task instruction inside a caption written by `caption`."""
    return text.split("performs the task: ")[-1].split(". Realistic")[0]
