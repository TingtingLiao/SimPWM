# Configuration

## Simulator settings

The presets are in [`simpwm/config.py`](../simpwm/config.py); it is the configuration reported in the
paper. Override any field with `SIMPWM_<FIELD>=0|1` in the environment; the ablations run this way. Each run records
its settings in `info.json`.

| Field | Meaning | DROID | DROID hi-res |
|:---|:---|:-:|:-:|
| `sam3_proposals` | add SAM 3 concept instances as object proposals | ✓ | ✓ |
| `snap`, `pin` | move an object the gripper closes on to the gripper's depth, and keep it there until it is grasped | ✓ | ✓ |
| `logged_camera` | use the dataset's logged camera instead of the calibrated one | | |
| `empty_closures`, `phantoms`, `phantom_two_pass` | rules that read the gripper signal (need the measured gripper position) | ✓ | ✓ |
| `soft_objects` | deformable objects named by the VLM hang from the gripper; active only for episodes with VLM labels (`python -m simpwm.perception.vlm_objects`) | ✓ | ✓ |
| `hires` | simulate at 640x360 instead of 320x180; SAM 3, object colours and the background depth use the 720p first frame | | ✓ |
| `jaw_inside` | grasp the object inside the gripper's jaw volume | | ✓ |
| `parts`, `parts_nested` | part prompts, so that a lid, cap or knob is its own object inside its container | | ✓ |
| `carry_children` | objects that rest on a grasped object move with it | | ✓ |

The presets are `droid` (the proxies of the LTX-2.3 models and of Tables 1–4), `droid_hires` (`--hires`: the proxies of
SimPWM-H3 and of the project page).

## Paths

Every location is set in [`simpwm/paths.py`](../simpwm/paths.py) and can be overridden from the environment:

| Variable | Default | Contents |
|:---|:---|:---|
| `SIMPWM_ASSETS` | `assets/` | robot models |
| `SIMPWM_DATA` | `data/` | datasets and files derived from them |
| `SIMPWM_DEPS` | `deps/` | external code: LTX-2, FoundationStereo |
| `SIMPWM_RUNS` | `runs/` | outputs |
| `DROID_RAW` | `data/droid_raw` | DROID raw release, one directory per episode |
| `DROID_ROOT` | `data/droid` | DROID RLDS release (optional) |
| `GR1_ROOT` | `data/gr1/GR1_robot` | GR00T-Teleop-GR1 |
| `SAM3_PATH` | `facebook/sam3` | SAM 3 weights: Hub id or local snapshot |
| `H3_DIFFUSERS` | `models/MiniMax-H3-diffusers-ref2va` | MiniMax-H3 Ref2VA, diffusers layout |


## Shipped data

`resources/` holds the splits and calibrations that the paper's runs use:

| Path | Contents |
|:---|:---|
| `droid/episodes.txt` | the 42,927 usable episodes of the DROID raw release; line *i* is episode *i* |
| `droid/splits/{train,val,test}.txt` | the paper's splits: 3000 / 8 / 500 episodes; the test camera setups are not in training |
| `droid/splits/train_h3.txt` | the 845 training episodes of SimPWM-H3 |
| `droid/cameras/` | fitted cameras of 6,394 episodes, including every episode of the splits |
| `droid/cameras_hires/` | the same episodes with the gripper mounting angle re-checked (325 cameras differ); used with `--hires` |
| `droid/setup_cameras/` | calibrations of the 674 DROID camera setups |
| `gr1/` | GR1 test windows, recording sessions, head camera per session, DreamDojo test split |
| `embodiments/` | robot configurations: Franka + Robotiq, GR1 head camera |
