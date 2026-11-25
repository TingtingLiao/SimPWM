# Examples

Inputs for `python -m simpwm.demo examples/<name>`: three clips of the [DROID](https://droid-dataset.github.io)
dataset (raw release, camera `ext1`, first 241 steps at 15 Hz). `source.json` names the episode.

| File | Contents |
|:---|:---|
| `first_frame.png` | first frame of the camera, 1280x720 |
| `camera.json` | calibrated camera at 320x180: `K`, `T_wc` (camera to robot base), `dist`; `gripper_yaw` is the mounting angle of the gripper |
| `actions.npz` | `arm_q` (T, 7) joint angles, `grip` (T,) gripper closure in [0, 1], `fps` |
| `stereo_depth.npz` | optional: `depth`, metric depth of the first frame (720x1280, NaN where unknown); without it the simulator uses monocular depth |
| `prompt.txt` | the task |
