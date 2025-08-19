# modules/stress.py
import numpy as np, time
BROW_LEFT=[70,63]; BROW_RIGHT=[300,293]

_state={"start":None,"collect_secs":5.0,"brow_base":None,"lip_base":None,"nose_xy":None,"jit_buf":[],"jit_maxlen":30}

def reset():
    _state.update({"start":None,"brow_base":None,"lip_base":None,"nose_xy":None,"jit_buf":[]})

def _face_diag(face):
    xs, ys = face[:,0], face[:,1]
    return np.hypot(xs.max()-xs.min(), ys.max()-ys.min()) + 1e-6

def stress_score(feats, quality, cfg, calibrating=False):
    face, mar = feats["face"], feats["mar"]
    now = time.time()
    if _state["start"] is None: _state["start"]=now
    if face is None:
        return {"score": 0.0, "conf": quality["conf_base"]*0.4}

    bl = face[BROW_LEFT,:2].mean(axis=0); br = face[BROW_RIGHT,:2].mean(axis=0)
    brow_dist = np.linalg.norm(bl-br) / _face_diag(face)

    collecting = calibrating or (now - _state["start"] < _state["collect_secs"])
    if collecting:
        _state["brow_base"] = brow_dist if _state["brow_base"] is None else 0.9*_state["brow_base"] + 0.1*brow_dist
        if mar is not None:
            _state["lip_base"] = mar if _state["lip_base"] is None else 0.9*_state["lip_base"] + 0.1*mar
        return {"score": 0.0, "conf": quality["conf_base"]*0.8}

    brow_base = _state["brow_base"] or brow_dist
    lip_base  = _state["lip_base"] if _state["lip_base"] is not None else (mar or 0.3)

    # Deadbands so tiny changes don't move the needle
    brow_db = 0.02
    lip_db  = 0.03

    brow_tension = 0.0
    if brow_dist < brow_base - brow_db:
        brow_tension = ( (brow_base - brow_dist - brow_db) / max(0.15, brow_base*0.7) )
        brow_tension = float(np.clip(brow_tension, 0.0, 1.0))

    lip_tension = 0.0
    if mar is not None and mar < lip_base - lip_db:
        lip_tension = float(np.clip((lip_base - mar - lip_db) / max(0.2, lip_base*0.7), 0.0, 1.0))

    # micro-jitter very low weight and with deadband
    nose_idx = 1 if face.shape[0]>1 else 0
    nose_xy = face[nose_idx,:2]
    if _state["nose_xy"] is not None:
        v = float(np.linalg.norm(nose_xy - _state["nose_xy"]))
        _state["jit_buf"].append(v)
        if len(_state["jit_buf"]) > _state["jit_maxlen"]: _state["jit_buf"].pop(0)
    _state["nose_xy"] = nose_xy
    import math
    jit = float(np.std(_state["jit_buf"])) if _state["jit_buf"] else 0.0
    jit_norm = 0.0 if jit < 0.2 else min(1.0, (jit-0.2)/1.2)

    raw = 0.7*brow_tension + 0.25*lip_tension + 0.05*jit_norm
    score = float(np.clip(100.0*raw, 0.0, 100.0))
    return {"score": score, "conf": quality["conf_base"]}
