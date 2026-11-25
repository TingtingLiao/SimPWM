"""Run DreamDojo's own examples/action_conditioned.py (unmodified) after three environment fixes.

Run with DreamDojo's python, cwd and PYTHONPATH = the DreamDojo checkout. The fixes change no model or evaluation math:
  1. the experiment configs resolve (= download) the Cosmos-Predict2.5 2B and 14B base checkpoints at import time as
     their default load_path, which inference always overrides: those two URIs are returned unresolved
  2. the Wan2.1 VAE is registered at a Cosmos-Predict2.5-2B revision that Hugging Face no longer serves: the identical
     tokenizer.pth is taken from revision 15a82a2
  3. imaginaire's distributed init pins each rank to the CPUs NVML reports for its GPU, which can lie outside a Slurm
     job's cgroup (EINVAL): such a failure is ignored

usage: python scripts/gr1/run_dreamdojo.py --dreamdojo DIR -- ARGS   (ARGS: action_conditioned.py's arguments)
"""

import argparse
import os
import runpy
import sys

WAN_VAE = "s3://bucket/cosmos_diffusion_v2/pretrain_weights/tokenizer/wan2pt1/Wan2.1_VAE.pth"
WAN_VAE_REVISION = "15a82a2ec231bc318692aa0456a36537c806e7d4"


def patch_checkpoints():
    import cosmos_predict2._src.imaginaire.utils.checkpoint_db as cdb
    from cosmos_predict2.config import MODEL_CHECKPOINTS, ModelKey

    unused = {MODEL_CHECKPOINTS[ModelKey(post_trained=False)].s3.uri.rstrip("/"),
              MODEL_CHECKPOINTS[ModelKey(post_trained=False, size="14B")].s3.uri.rstrip("/")}
    get_checkpoint_path = cdb.get_checkpoint_path

    def patched(uri):
        if uri.rstrip("/") in unused:
            return uri
        if uri.rstrip("/") == WAN_VAE:
            from huggingface_hub import hf_hub_download

            return hf_hub_download("nvidia/Cosmos-Predict2.5-2B", "tokenizer.pth", revision=WAN_VAE_REVISION)
        return get_checkpoint_path(uri)

    cdb.get_checkpoint_path = patched


def patch_affinity():
    setaffinity = os.sched_setaffinity

    def safe_setaffinity(pid, cpus):
        try:
            setaffinity(pid, cpus)
        except OSError:
            pass

    os.sched_setaffinity = safe_setaffinity


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--dreamdojo", required=True, help="DreamDojo checkout (holds examples/action_conditioned.py)")
    ap.add_argument("args", nargs=argparse.REMAINDER, help="-- then the arguments of action_conditioned.py")
    a = ap.parse_args()
    args = a.args[1:] if a.args[:1] == ["--"] else a.args
    patch_checkpoints()
    patch_affinity()
    script = os.path.join(a.dreamdojo, "examples/action_conditioned.py")
    sys.argv = [script] + args
    runpy.run_path(script, run_name="__main__")


if __name__ == "__main__":
    main()
