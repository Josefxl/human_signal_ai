import numpy as np

def distance_status(det, cfg):
    face = det["face"]
    size = det["size"]
    if face is None or size is None:
        return {"state": "unknown", "ratio": None}

    w, h = size
    frame_diag = (w**2 + h**2) ** 0.5
    xs, ys = face[:,0], face[:,1]
    face_diag = ((xs.max()-xs.min())**2 + (ys.max()-ys.min())**2) ** 0.5
    ratio = float(face_diag / (frame_diag + 1e-6))  # 0..1

    close_thr = cfg["thresholds"].get("distance_face_ratio_close", 0.32)
    far_thr = cfg["thresholds"].get("distance_face_ratio_far", 0.12)

    if ratio >= close_thr:
        state = "too close"
    elif ratio <= far_thr:
        state = "far"
    else:
        state = "ok"

    return {"state": state, "ratio": ratio}
