# (iris-based, deadzone, auto-baseline, debug)
import time, math
import numpy as np

# MediaPipe FaceMesh indices:
# Iris centers (if refine_landmarks=True): approx centers via iris rings
LEFT_IRIS  = [468, 469, 470, 471]
RIGHT_IRIS = [473, 474, 475, 476]

_state = {
    "baseline_offset": None,   # (dx, dy) at calibration
    "offscreen_since": None,
    "have_iris": False,
}

def reset():
    _state["baseline_offset"] = None
    _state["offscreen_since"] = None
    _state["have_iris"] = False

def _center(pts):
    return pts.mean(axis=0)

def attention_score(feats, quality, cfg, calibrating=False):
    face = feats.get("face")
    now = time.time()
    if face is None:
        return {"score": 0.0, "offscreen_duration": 0.0, "conf": quality["conf_base"]*0.4,
                "debug": {"reason": "no_face"}}

    xs, ys = face[:,0], face[:,1]
    cx, cy = xs.mean(), ys.mean()
    face_diag = math.hypot(xs.max()-xs.min(), ys.max()-ys.min()) + 1e-6

    # Use iris centers if present; fallback to eyelid corners avg
    eye_center = None
    if face.shape[0] > max(LEFT_IRIS + RIGHT_IRIS):  # iris landmarks exist
        _state["have_iris"] = True
        li = _center(face[LEFT_IRIS, :2])
        ri = _center(face[RIGHT_IRIS, :2])
        eye_center = (li + ri) / 2.0
    else:
        _state["have_iris"] = False
        left_center  = face[[33, 133], :2].mean(axis=0)
        right_center = face[[362, 263], :2].mean(axis=0)
        eye_center = (left_center + right_center) / 2.0

    dx, dy = float(eye_center[0] - cx), float(eye_center[1] - cy)

    # Capture baseline during calibration, or auto-init if missing
    if calibrating:
        _state["baseline_offset"] = (dx, dy)
        return {"score": 100.0, "offscreen_duration": 0.0, "conf": quality["conf_base"]*0.9,
                "debug": {"reason": "calibrating", "d": 0.0, "bx": dx, "by": dy, "iris": _state["have_iris"]}}

    if _state["baseline_offset"] is None:
        # Auto-initialize baseline on first stable face
        _state["baseline_offset"] = (dx, dy)

    bx, by = _state["baseline_offset"]
    # Distance from baseline, normalized by face size
    d = math.hypot((dx - bx), (dy - by)) / (0.30 * face_diag)  # 0.30 scaling is forgiving
    d = max(0.0, min(1.0, d))

    # Deadzone → perfect 100 if within small radius
    DEADZONE = 0.10  # 10% of scaled face diag
    if d <= DEADZONE:
        score = 100.0
    else:
        # Sharper falloff so it stays high when near center and drops when off
        score = 100.0 * (1.0 - (d - DEADZONE)) ** 2.2
        score = max(0.0, min(100.0, score))

    # Sustained off-screen penalty
    off = score < 30.0
    if off and _state["offscreen_since"] is None:
        _state["offscreen_since"] = now
    if not off:
        _state["offscreen_since"] = None

    off_dur = (now - _state["offscreen_since"]) if _state["offscreen_since"] else 0.0
    if off_dur > cfg["thresholds"]["gaze_offscreen_secs"]:
        score *= 0.6

    return {
        "score": float(score),
        "offscreen_duration": off_dur,
        "conf": quality["conf_base"],
        "debug": {"d": d, "bx": bx, "by": by, "iris": _state["have_iris"]}
    }
