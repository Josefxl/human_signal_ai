# (baseline + fusion + smoothing + hysteresis + dwell)
import time
import numpy as np

# MediaPipe Pose indices
LS, RS, LH, RH, LEAR, REAR = 11, 12, 23, 24, 7, 8

# Internal state (persisted across frames)
_state = {
    "sm_head": None,         # smoothed head flexion angle (deg)
    "sm_torso": None,        # smoothed torso lean angle (deg)
    "base_head": None,       # baseline head angle at calibration
    "base_torso": None,      # baseline torso angle at calibration
    "last_state": "upright", # current labeled state
    "since": 0.0             # timestamp when a change was first proposed
}

def reset():
    """Reset posture module state (called from Calibrate button)."""
    _state.update({
        "sm_head": None,
        "sm_torso": None,
        "base_head": None,
        "base_torso": None,
        "last_state": "upright",
        "since": 0.0
    })

def _angle_to_vertical(v):
    """Angle (deg) between vector v and upward vertical (0,-1). 0°=vertical; larger=more lean."""
    v = v / (np.linalg.norm(v) + 1e-6)
    dot = np.dot(v, np.array([0.0, -1.0]))
    return float(np.degrees(np.arccos(np.clip(dot, -1.0, 1.0))))

def posture_status(feats, quality, cfg, calibrating=False):
    pose = feats.get("pose")
    if pose is None:
        # Keep last known, but lower confidence
        return {"state": _state["last_state"], "conf": quality["conf_base"] * 0.5,
                "angles": {"head": _state["sm_head"], "torso": _state["sm_torso"]}}

    # Extract needed keypoints (may raise if any missing)
    try:
        l_sh, r_sh = pose[LS, :2], pose[RS, :2]
        l_hip, r_hip = pose[LH, :2], pose[RH, :2]
        l_ear, r_ear = pose[LEAR, :2], pose[REAR, :2]
    except Exception:
        return {"state": _state["last_state"], "conf": quality["conf_base"] * 0.5,
                "angles": {"head": _state["sm_head"], "torso": _state["sm_torso"]}}

    shoulder_mid = (l_sh + r_sh) / 2.0
    hip_mid      = (l_hip + r_hip) / 2.0
    ear_mid      = (l_ear + r_ear) / 2.0

    # Vectors
    v_head  = ear_mid - shoulder_mid     # neck/head direction
    v_torso = shoulder_mid - hip_mid     # torso (shoulders above hips)

    # Raw angles to vertical
    ang_head  = _angle_to_vertical(v_head)
    ang_torso = _angle_to_vertical(v_torso)

    # Strong EMA smoothing to stabilize noise
    alpha = 0.15
    _state["sm_head"]  = ang_head  if _state["sm_head"]  is None else alpha * ang_head  + (1 - alpha) * _state["sm_head"]
    _state["sm_torso"] = ang_torso if _state["sm_torso"] is None else alpha * ang_torso + (1 - alpha) * _state["sm_torso"]

    sm_head  = _state["sm_head"]
    sm_torso = _state["sm_torso"]

    # During calibration, capture baselines and force upright
    if calibrating:
        _state["base_head"]  = sm_head
        _state["base_torso"] = sm_torso
        _state["last_state"] = "upright"
        _state["since"] = 0.0
        return {"state": "upright", "conf": quality["conf_base"],
                "angles": {"head": sm_head, "torso": sm_torso}}

    # Absolute thresholds from config
    abs_enter = float(cfg["thresholds"].get("neck_angle_slouch", 22))   # enter slouch if > this
    abs_exit  = max(abs_enter - 6.0, 0.0)                               # exit slouch if < this

    # Baseline-relative thresholds (avoid false slouch if your neutral isn’t perfectly vertical)
    base_head  = _state["base_head"]  if _state["base_head"]  is not None else sm_head
    base_torso = _state["base_torso"] if _state["base_torso"] is not None else sm_torso

    # Require some extra degrees over *your* baseline to call it a slouch
    head_enter_rel  = base_head  + 10.0   # need +10° over your neutral to enter slouch
    head_exit_rel   = base_head  + 6.0    # recover when within +6° of your neutral
    torso_enter_rel = base_torso + 8.0
    torso_exit_rel  = base_torso + 5.0

    # Effective thresholds = max(abs, relative)
    head_enter_th  = max(abs_enter, head_enter_rel)
    head_exit_th   = max(abs_exit,  head_exit_rel)
    torso_enter_th = max(abs_enter - 2.0, torso_enter_rel)  # torso slightly more permissive to enter
    torso_exit_th  = max(abs_exit,  torso_exit_rel)

    # Proposed state (use OR for entering slouch; AND for exiting back to upright)
    if sm_head > head_enter_th or sm_torso > torso_enter_th:
        proposed = "slouching"
    elif sm_head < head_exit_th and sm_torso < torso_exit_th:
        proposed = "upright"
    else:
        proposed = _state["last_state"]

    # Dwell logic to prevent flip-flop (need sustained condition to switch)
    now = time.time()
    if proposed != _state["last_state"]:
        if _state["since"] == 0.0:
            _state["since"] = now
        elif (now - _state["since"]) >= 1.5:  # require ~1.5s stable
            _state["last_state"] = proposed
            _state["since"] = 0.0
        # else: hold previous state until dwell met
    else:
        _state["since"] = 0.0

    return {
        "state": _state["last_state"],
        "conf": quality["conf_base"],
        "angles": {"head": sm_head, "torso": sm_torso},
        "thresholds": {
            "head_enter": head_enter_th, "head_exit": head_exit_th,
            "torso_enter": torso_enter_th, "torso_exit": torso_exit_th
        }
    }
