import numpy as np
import cv2

def estimate_quality(frame_rgb, det, feats, cfg):
    # brightness
    gray = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2GRAY)
    brightness = float(gray.mean())

    # motion proxy (face bbox movement) — simple for MVP
    face = det["face"]
    motion = 0.0
    if face is not None:
        xs, ys = face[:,0], face[:,1]
        box = (xs.max()-xs.min()) + (ys.max()-ys.min())
        motion = 0.0  # stub; add optical flow later

    bright_ok = brightness >= cfg["quality"]["min_brightness"]
    conf_base = 0.8 if bright_ok else 0.4

    return {
        "brightness": brightness,
        "motion": motion,
        "conf_base": conf_base
    }
