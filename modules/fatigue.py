# (yawn to 100, motion/occlusion bump, slouch bump, time-based decay)
import time

# Eye Aspect Ratio threshold for closed eyes (blink)
EAR_CLOSED = 0.20
BLINK_MIN_FRAMES = 2

# Tunables (safe defaults; can move to cfg later if you want)
DECAY_PER_SEC       = 27.0   # how fast the score falls back toward 0 when calm
SLOUCH_BUMP_PER_SEC = 6.0    # per-second bump while slouching (gentle accumulation)
MOTION_THRESHOLD    = 0.20   # motion_energy above this → bump
MOTION_GAIN         = 180.0  # bump = (motion_energy - thr) * GAIN  (caps inside)
OCCLUSION_DROP_FRAC = 0.35   # >35% sudden face-area drop → occlusion bump
OCCLUSION_BUMP      = 18.0   # fixed bump for occlusion
PERCLOS_BASE_GAIN   = 22.0   # base fatigue from blinkiness (perclos in [0..1])

_state = {
    "closed_frames": 0,
    "blink_times": [],
    "last_closed": False,
    "yawn_start": None,
    "score": 0.0,
    "last_face_area": None,
    "last_t": None,
}

def reset():
    _state.update({
        "closed_frames": 0,
        "blink_times": [],
        "last_closed": False,
        "yawn_start": None,
        "score": 0.0,
        "last_face_area": None,
        "last_t": None,
    })

def fatigue_score(feats, quality, cfg, calibrating=False, posture_state=None, motion_energy=0.0):
    """
    Returns fatigue score in [0..100] that:
      - spikes to ~100 on sustained yawn
      - bumps on big hand/body motion or occlusion
      - gently accumulates while slouching
      - decays back to 0 with time-based decay
    """
    face, mar = feats.get("face"), feats.get("mar")
    le, re = feats.get("ear_left"), feats.get("ear_right")
    ear = le if re is None else re if le is None else (le + re) / 2.0

    now = time.time()
    if calibrating:
        reset()
        # initialize clock
        _state["last_t"] = now
        return {"score": 0.0, "blink_rate": 0.0, "perclos": 0.0, "conf": quality["conf_base"] * 0.8}

    # dt for time-based dynamics
    if _state["last_t"] is None:
        _state["last_t"] = now
    dt = max(0.0, min(0.25, now - _state["last_t"]))  # clamp dt to avoid big jumps
    _state["last_t"] = now

    # -------- Blink / PERCLOS (for gentle baseline fatigue) --------
    closed = (ear is not None) and (ear < EAR_CLOSED)
    if closed:
        _state["closed_frames"] += 1
    if _state["last_closed"] and not closed and _state["closed_frames"] >= BLINK_MIN_FRAMES:
        _state["blink_times"].append(now)
    if not closed:
        _state["closed_frames"] = 0
    _state["last_closed"] = closed

    # keep 60s window
    _state["blink_times"] = [t for t in _state["blink_times"] if now - t <= 60.0]
    blink_rate = len(_state["blink_times"])  # ~per minute
    perclos = min(1.0, blink_rate / 30.0)    # normalize ~30 blinks/min → 1.0
    base_from_perclos = PERCLOS_BASE_GAIN * perclos  # gentle baseline component

    # Yawn detection → force to near-100
    yawn_thr = cfg["thresholds"].get("yawn_mar", 0.55)
    yawn_min = cfg["thresholds"].get("yawn_min_secs", 0.3)
    if mar is not None and mar >= yawn_thr:
        if _state["yawn_start"] is None:
            _state["yawn_start"] = now
        elif (now - _state["yawn_start"]) >= yawn_min:
            # Intensity above threshold; drive score near top (95..100)
            over = max(0.0, min(1.0, (mar - yawn_thr) / 0.35))
            target_peak = 95.0 + 5.0 * over  # 95 to 100 depending on intensity
            _state["score"] = max(_state["score"], target_peak)
    else:
        _state["yawn_start"] = None

    # Big motion / occlusion bumps
    # Motion bump proportional to how far above threshold we are
    if motion_energy is not None and motion_energy > MOTION_THRESHOLD:
        bump = (motion_energy - MOTION_THRESHOLD) * MOTION_GAIN  # e.g., 0.3 → (0.1*180)=18
        _state["score"] = min(100.0, _state["score"] + bump)

    # Occlusion (sudden face area drop)
    if face is not None:
        xs = face[:, 0]; ys = face[:, 1]
        area = (xs.max() - xs.min()) * (ys.max() - ys.min()) + 1e-6
        if _state["last_face_area"] is not None:
            if area < (1.0 - OCCLUSION_DROP_FRAC) * _state["last_face_area"]:
                _state["score"] = min(100.0, _state["score"] + OCCLUSION_BUMP)
        _state["last_face_area"] = area

    # Slouch accumulation (gentle, time-based)
    if posture_state == "slouching":
        _state["score"] = min(100.0, _state["score"] + SLOUCH_BUMP_PER_SEC * dt)

    # Time-based decay towards 0
    if _state["score"] > 0.0:
        _state["score"] = max(0.0, _state["score"] - DECAY_PER_SEC * dt)

    # Ensure we respect baseline from PERCLOS (don’t drop below it)
    score = max(base_from_perclos, _state["score"])
    score = float(max(0.0, min(100.0, score)))

    # confidence: full when we have EAR (eyes present), otherwise lower
    conf = quality["conf_base"] * (1.0 if ear is not None else 0.6)
    return {"score": score, "blink_rate": blink_rate, "perclos": perclos, "conf": conf}
