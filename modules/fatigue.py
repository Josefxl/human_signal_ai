# modules/fatigue.py  (v4)
import time

EAR_CLOSED = 0.20
BLINK_MIN_FRAMES = 2

_state = {"closed_frames":0,"blink_times":[],"last_closed":False,"yawn_start":None,"score":0.0,"last_face_area":None}

def reset():
    _state.update({"closed_frames":0,"blink_times":[],"last_closed":False,"yawn_start":None,"score":0.0,"last_face_area":None})

def fatigue_score(feats, quality, cfg, calibrating=False, posture_state=None, motion_energy=0.0):
    face, mar = feats["face"], feats["mar"]
    le, re = feats["ear_left"], feats["ear_right"]
    ear = le if re is None else re if le is None else (le+re)/2.0
    now = time.time()

    if calibrating:
        reset()
        return {"score": 0.0, "blink_rate": 0.0, "perclos": 0.0, "conf": quality["conf_base"]*0.8}

    # Blink tracking
    closed = (ear is not None) and (ear < EAR_CLOSED)
    if closed: _state["closed_frames"] += 1
    if _state["last_closed"] and not closed and _state["closed_frames"] >= BLINK_MIN_FRAMES:
        _state["blink_times"].append(now)
    if not closed: _state["closed_frames"] = 0
    _state["last_closed"] = closed
    _state["blink_times"] = [t for t in _state["blink_times"] if now - t <= 60.0]
    blink_rate = len(_state["blink_times"])
    perclos = min(1.0, (blink_rate / 30.0))

    # Yawn spike (bigger & quicker)
    yawn_thr = cfg["thresholds"].get("yawn_mar", 0.55)
    yawn_min = cfg["thresholds"].get("yawn_min_secs", 0.3)
    if mar is not None and mar >= yawn_thr:
        if _state["yawn_start"] is None:
            _state["yawn_start"] = now
        elif (now - _state["yawn_start"]) >= yawn_min:
            _state["score"] = min(100.0, _state["score"] + 60.0 * min(1.0, (mar - yawn_thr)/0.3))
    else:
        _state["yawn_start"] = None

    # Slouch bump (ergonomics)
    if posture_state == "slouching":
        _state["score"] = min(100.0, _state["score"] + 1.5)  # gentle but persistent

    # Big motion/occlusion bump
    if motion_energy > 0.18:
        _state["score"] = min(100.0, _state["score"] + 6.0)

    # Natural decay + base from perclos
    _state["score"] = max(0.0, _state["score"] - 0.9)  # decay each call
    base = 20.0 * perclos
    score = float(max(0.0, min(100.0, base + _state["score"])))

    conf = quality["conf_base"] * (1.0 if ear is not None else 0.6)
    return {"score": score, "blink_rate": blink_rate, "perclos": perclos, "conf": conf}
