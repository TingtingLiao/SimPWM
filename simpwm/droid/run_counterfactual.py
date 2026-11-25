"""Counterfactual-action proxies for the action-following experiment (eval/action_follow.py): droid/run.py on perturbed
robot inputs, every other setting unchanged.

  prep  per episode the perturbed inputs of every condition -> OUT/actions/<idx>.npz (first STEPS steps; rec_* /
        noclose_* / offset_* for obs_q, obs_g (observation joint_positions, gripper_position: the simulator's state
        input), act_q, act_g (action joint_position, gripper_position), cart (observation cartesian_position), the
        offset direction and the IK residuals) + OUT/ik_residuals.json
  run   droid/run.py (same options) with the episode's arm_q, grip replaced by the condition's obs_q, obs_g

Conditions:
  noclose  gripper position 0 (open) at every step (observation and command); arm unchanged
  offset   every end-effector pose translated by OFFSET_M along d (robot base frame), orientation and gripper
           unchanged: f = (R[0, 2], R[1, 2]) / |.|, R = rotation of the reference run's ext1 camera-to-base transform
           (info.json T_wc; columns = camera axes in the base frame, OpenCV: x right, y down, z = viewing direction);
           d = (f_y, -f_x, 0): horizontal, perpendicular to the viewing direction, towards the image right. Joint
           inputs: damped least-squares IK of the FR3 flange (attachment_site, whose FK equals DROID's
           cartesian_position) on the simulator's MuJoCo model, seeded with the recorded joints of that step, joint
           limits enforced; residuals (position mm, rotation deg) reported
  freezeK  (no prep) the last K of the EVAL_CLIP clip steps frozen: arm joints and gripper of every step
           >= EVAL_CLIP - K (including all later simulated steps) held at their step EVAL_CLIP - K - 1 values. Only the
           robot renders and the rule-engine rollout get the frozen inputs (pipeline.ROLLOUT_ACTIONS); the scene
           construction (support height bound, grasp proposals, snap, phantoms) uses the recorded ones, so the proxy
           equals the unedited one up to that step. The inputs used are written to OUT_DIR/actions/<idx>.npz (rec_*
           and freezeK_*)

usage: python -m simpwm.droid.run_counterfactual prep IDS_TXT --ref-run RUN [--out DIR]
       python -m simpwm.droid.run_counterfactual run COND OUT_DIR IDX [IDX ...] [droid/run.py options]
"""

import argparse
import json
from pathlib import Path

import mujoco
import numpy as np

from simpwm import paths

OFFSET_M = 0.05
SITE = "attachment_site"
EVAL_CLIP = 97  # recorded steps 0..96 scored by eval/action_follow.py
DEFAULT_OUT = paths.RUNS / "action_follow"


def direction(ref_run, idx):
    T_wc = np.array(json.loads((Path(ref_run) / f"droid_{idx:05d}" / "info.json").read_text())["T_wc"])
    R = T_wc[:3, :3]
    f = R[:2, 2] / np.linalg.norm(R[:2, 2])
    d = np.array([f[1], -f[0], 0.0])
    assert d[:2] @ R[:2, 0] > 0, (idx, R)
    return d


class IK:
    def __init__(self, emb):
        self.emb, m = emb, emb.model
        self.m, self.d = m, mujoco.MjData(m)
        self.qadr = emb.arm_qadr
        self.dadr = np.array([m.jnt_dofadr[m.joint(j).id] for j in emb.cfg["arm_joints"]])
        self.lo, self.hi = m.jnt_range[[m.joint(j).id for j in emb.cfg["arm_joints"]]].T
        self.sid = m.site(SITE).id

    def fk(self, q):
        self.d.qpos[self.qadr] = q
        mujoco.mj_kinematics(self.m, self.d)
        mujoco.mj_comPos(self.m, self.d)
        return self.d.site_xpos[self.sid].copy(), self.d.site_xmat[self.sid].reshape(3, 3).copy()

    def solve(self, q0, p_t, R_t, iters=200, lam=0.02, tol=1e-6):
        q = np.array(q0, float)
        jp, jr = np.zeros((3, self.m.nv)), np.zeros((3, self.m.nv))
        for _ in range(iters):
            p, R = self.fk(q)
            ep = p_t - p
            er = np.zeros(3)
            mujoco.mju_mat2Quat(qe := np.zeros(4), (R_t @ R.T).ravel())
            mujoco.mju_quat2Vel(er, qe, 1.0)
            e = np.concatenate([ep, er])
            if np.linalg.norm(e) < tol:
                break
            mujoco.mj_jacSite(self.m, self.d, jp, jr, self.sid)
            J = np.concatenate([jp, jr])[:, self.dadr]
            dq = J.T @ np.linalg.solve(J @ J.T + lam ** 2 * np.eye(6), e)
            q = np.clip(q + dq, self.lo, self.hi)
        p, R = self.fk(q)
        mujoco.mju_mat2Quat(qe := np.zeros(4), (R_t @ R.T).ravel())
        er = np.zeros(3)
        mujoco.mju_quat2Vel(er, qe, 1.0)
        return q, float(np.linalg.norm(p_t - p) * 1000), float(np.degrees(np.linalg.norm(er)))

    def shift(self, qs, delta):
        out, rp, rr = [], [], []
        for q in qs:
            p, R = self.fk(q)
            qn, a, b = self.solve(q, p + delta, R)
            out.append(qn)
            rp.append(a)
            rr.append(b)
        return np.array(out), np.array(rp), np.array(rr)


def prep(ids, ref_run, out=DEFAULT_OUT, steps=241):
    import h5py

    from simpwm.core.embodiment import Embodiment
    from simpwm.eval.action_follow import raw_dir

    ik = IK(Embodiment("franka_robotiq"))
    (out / "actions").mkdir(parents=True, exist_ok=True)
    for idx in ids:
        f = out / "actions" / f"{idx}.npz"
        if f.exists():
            continue
        with h5py.File(raw_dir(idx) / "trajectory.h5", "r") as h:
            rs = h["observation/robot_state"]
            obs_q, obs_g = rs["joint_positions"][:steps], rs["gripper_position"][:steps]
            cart = rs["cartesian_position"][:steps]
            act_q, act_g = h["action/joint_position"][:steps], h["action/gripper_position"][:steps]
        d = direction(ref_run, idx)
        delta = OFFSET_M * d
        z = dict(direction=d, delta=delta, rec_obs_q=obs_q, rec_obs_g=obs_g, rec_act_q=act_q, rec_act_g=act_g,
                 rec_cart=cart)
        z.update(noclose_obs_q=obs_q, noclose_obs_g=np.zeros_like(obs_g), noclose_act_q=act_q,
                 noclose_act_g=np.zeros_like(act_g), noclose_cart=cart)
        oq, op, orr = ik.shift(obs_q, delta)
        aq, ap, ar = ik.shift(act_q, delta)
        c2 = cart.copy()
        c2[:, :3] += delta
        # FK of the shifted joints against the recorded cartesian pose + delta (cross-check of the frame)
        fk_err = np.array([np.linalg.norm(ik.fk(q)[0] - c[:3]) * 1000 for q, c in zip(oq, c2)])
        z.update(offset_obs_q=oq, offset_obs_g=obs_g, offset_act_q=aq, offset_act_g=act_g, offset_cart=c2,
                 ik_obs_pos_mm=op, ik_obs_rot_deg=orr, ik_act_pos_mm=ap, ik_act_rot_deg=ar, fk_vs_cart_mm=fk_err)
        np.savez_compressed(f, **z)
        n = EVAL_CLIP
        print(idx, "d", d.round(3), f"IK obs pos max {op[:n].max():.3f} mm rot {orr[:n].max():.3f} deg; act pos max "
              f"{ap[:n].max():.3f} mm; fk-cart {fk_err[:n].max():.2f} mm", flush=True)
    res = {}
    for k in ("ik_obs_pos_mm", "ik_obs_rot_deg", "ik_act_pos_mm", "ik_act_rot_deg", "fk_vs_cart_mm"):
        v = np.concatenate([np.load(out / "actions" / f"{i}.npz")[k][:EVAL_CLIP] for i in ids])
        res[k] = dict(median=float(np.median(v)), p95=float(np.percentile(v, 95)), p99=float(np.percentile(v, 99)),
                      max=float(v.max()), frac_over_1mm_or_deg=float((v > 1).mean()), n_steps=int(len(v)))
    res["note"] = (f"steps 0..{EVAL_CLIP - 1} of the {len(ids)} episodes; obs = observation joint_positions "
                   "(simulator, DreamDojo row-0 state), act = action/joint_position (DreamDojo actions); fk_vs_cart = "
                   "|FK(IK joints) - (recorded cartesian_position + delta)|, i.e. includes the FK-vs-DROID frame "
                   "agreement")
    (out / "ik_residuals.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


def counterfactual_edit(cond, out_dir, actions_dir):
    """edit(idx, ep) -> ep for droid/run.py: the episode's robot inputs under condition `cond`."""
    from simpwm.sim import pipeline

    def edit(idx, ep):
        if cond.startswith("freeze"):
            F = EVAL_CLIP - int(cond[len("freeze"):])
            # ep keeps the recorded inputs (scene construction); the frozen ones drive only the rollout
            q, g = ep["arm_q"].copy(), ep["grip"].copy()
            q[F:], g[F:] = q[F - 1], g[F - 1]
            pipeline.ROLLOUT_ACTIONS = (q, g)
            d = Path(out_dir) / "actions"
            d.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(d / f"{idx}.npz", rec_obs_q=ep["arm_q"], rec_obs_g=ep["grip"],
                                **{f"{cond}_obs_q": q, f"{cond}_obs_g": g}, freeze_from=F)
            print(f"counterfactual {cond}: rollout steps {F}..{len(q) - 1} held at step {F - 1} (scene from the "
                  f"recorded inputs)", flush=True)
            return ep
        z = np.load(Path(actions_dir) / f"{idx}.npz")
        q, g = z[f"{cond}_obs_q"], z[f"{cond}_obs_g"]
        n = min(len(q), len(ep["arm_q"]))
        assert np.allclose(z["rec_obs_q"][:n], ep["arm_q"][:n]) and np.allclose(z["rec_obs_g"][:n], ep["grip"][:n]), idx
        ep["arm_q"], ep["grip"] = ep["arm_q"].copy(), ep["grip"].copy()
        ep["arm_q"][:n], ep["grip"][:n] = q[:n], g[:n]
        print(f"counterfactual {cond}: steps 0..{n - 1} replaced", flush=True)
        return ep

    return edit


def main():
    from simpwm.droid import run as droid_run

    ap = argparse.ArgumentParser(prog="python -m simpwm.droid.run_counterfactual", description=__doc__.split("\n\n")[0],
                                 fromfile_prefix_chars="@")
    ap.convert_arg_line_to_args = str.split  # @FILE: a whitespace-separated id list
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("prep", help="perturbed inputs of the noclose / offset conditions")
    p.add_argument("ids", type=Path, metavar="IDS_TXT", help="whitespace-separated episode indices")
    p.add_argument("--ref-run", type=Path, required=True,
                   help="proxy run (droid/run.py output) whose per-episode camera defines the offset direction")
    p.add_argument("--out", type=Path, default=DEFAULT_OUT, help="output directory (actions/, ik_residuals.json)")
    p.add_argument("--steps", type=int, default=241, help="recorded steps to perturb (the simulated clip)")
    r = sub.add_parser("run", help="proxies under a condition (droid/run.py options)")
    r.add_argument("cond", metavar="COND",
                   help="noclose | offset | rec (prep's recorded inputs) | freezeK (e.g. freeze20)")
    r.add_argument("--actions-dir", type=Path, default=DEFAULT_OUT / "actions",
                   help="prep's actions/ directory (noclose, offset, rec)")
    droid_run.add_arguments(r)
    args = ap.parse_args()
    if args.cmd == "prep":
        prep([int(i) for i in args.ids.read_text().split()], args.ref_run, args.out, args.steps)
    else:
        if args.cond not in ("rec", "noclose", "offset") and not args.cond.startswith("freeze"):
            ap.error(f"unknown condition {args.cond}")
        droid_run.run_from_args(args, edit=counterfactual_edit(args.cond, args.out_dir, args.actions_dir))


if __name__ == "__main__":
    main()
