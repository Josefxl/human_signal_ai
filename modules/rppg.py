# POS rPPG (NumPy/OpenCV only) with SNR gating
import time
import numpy as np
import cv2

# Tunables (safe demo defaults; you can move these to cfg if you like)
WIN_SECS = 32.0          # sliding window length for FFT (s)
MIN_SAMPLES = 150         # need enough samples for a stable estimate
MIN_LUMA = 40             # below this brightness → low confidence
MOTION_MAX = 0.25         # above this motion energy → low confidence
HR_BAND = (0.75, 3.0)     # Hz → 45–180 bpm
BR_BAND = (0.10, 0.60)    # Hz → 6–36 rpm
SNR_MIN = 4.0             # min peak/sideband ratio to show number

_state = {
    "t": [],
    "r": [],
    "g": [],
    "b": [],
    "last_t": None,
}

def reset():
    _state["t"].clear(); _state["r"].clear(); _state["g"].clear(); _state["b"].clear()
    _state["last_t"] = None

def _roi_forehead_poly(face):
    """
    Build a small forehead polygon from FaceMesh landmarks.
    Fallback to upper box of face if we can't.
    FaceMesh indices used: 10 (forehead), 338 (right temple), 108 (left temple), 151 (upper nose bridge)
    """
    if face is None or face.shape[0] < 339:
        return None
    p_fore = face[10, :2]
    p_rt   = face[338, :2]
    p_lt   = face[108, :2]
    p_nb   = face[151, :2]
    # Slightly lift points upward (towards negative y) to avoid brows
    lift = max(2.0, 0.06 * np.linalg.norm(p_rt - p_lt))
    u = np.array([0.0, -1.0], dtype=np.float32)
    pts = np.stack([p_lt, p_rt, p_fore, p_nb], axis=0).astype(np.float32)
    pts = pts + u * lift
    return pts

def _mean_rgb_in_poly(img_rgb, poly):
    h, w = img_rgb.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillConvexPoly(mask, poly.astype(np.int32), 255)
    roi = img_rgb[mask == 255]
    if roi.size == 0:
        return None, 0.0
    mean = roi.mean(axis=0)  # RGB
    luma = float(0.2126 * mean[0] + 0.7152 * mean[1] + 0.0722 * mean[2])
    return mean, luma

def _detrend_norm(x):
    x = np.asarray(x, dtype=np.float32)
    m = x.mean() + 1e-6
    x = x / m
    # remove very slow drift with short moving average
    k = max(3, int(len(x) * 0.05))
    kernel = np.ones(k, np.float32) / k
    trend = np.convolve(x, kernel, mode="same")
    return x - trend

def _pos_signal(r, g, b):
    # POS: x1 = g - b, x2 = g + b; s = x1 - (σ(x1)/σ(x2)) * x2
    x1 = g - b
    x2 = g + b
    s1 = np.std(x1) + 1e-6
    s2 = np.std(x2) + 1e-6
    return x1 - (s1 / s2) * x2

def _band_peak_freq(sig, fs, band):
    n = len(sig)
    if n < 8 or fs <= 0:
        return None, 0.0
    # Hamming window + rfft
    w = np.hamming(n)
    y = np.fft.rfft(sig * w)
    f = np.fft.rfftfreq(n, d=1.0/fs)
    p = (y.real**2 + y.imag**2)

    # band mask
    m = (f >= band[0]) & (f <= band[1])
    if not np.any(m):
        return None, 0.0
    f_band, p_band = f[m], p[m]
    i = int(np.argmax(p_band))
    f0 = float(f_band[i])
    peak = float(p_band[i])

    # SNR: peak / mean of neighbors (exclude a couple bins around the peak)
    if p_band.size >= 7:
        nb = np.concatenate([p_band[:max(0, i-3)], p_band[min(i+4, p_band.size):]])
        side = float(nb.mean() + 1e-9)
        snr = peak / side
    else:
        snr = 0.0
    return f0, snr

def update(frame_rgb, feats, quality, motion_energy, cfg, calibrating=False):
    """
    Collect RGB means from a stable forehead ROI each frame, estimate HR/BR over a sliding window.
    Returns: {"hr_bpm": float|None, "br_rpm": float|None, "hr_conf": 0..1, "br_conf": 0..1}
    """
    now = time.time()
    face = feats.get("face")

    # ROI
    poly = _roi_forehead_poly(face)
    if poly is None:
        # fallback: upper 30% of face box
        if face is not None:
            xs, ys = face[:,0], face[:,1]
            x0, x1 = int(xs.min()), int(xs.max())
            y0, y1 = int(ys.min()), int(ys.max())
            y_cut = int(y0 + 0.30 * (y1 - y0))
            poly = np.array([[x0, y0], [x1, y0], [x1, y_cut], [x0, y_cut]], dtype=np.float32)
        else:
            return {"hr_bpm": None, "br_rpm": None, "hr_conf": 0.0, "br_conf": 0.0}

    mean_rgb, luma = _mean_rgb_in_poly(frame_rgb, poly)
    if mean_rgb is None:
        return {"hr_bpm": None, "br_rpm": None, "hr_conf": 0.0, "br_conf": 0.0}

    # Append sample
    _state["t"].append(now)
    _state["r"].append(mean_rgb[0]); _state["g"].append(mean_rgb[1]); _state["b"].append(mean_rgb[2])

    # Keep only last WIN_SECS
    t0 = now - WIN_SECS
    while _state["t"] and _state["t"][0] < t0:
        _state["t"].pop(0); _state["r"].pop(0); _state["g"].pop(0); _state["b"].pop(0)

    # Quality gating
    if (len(_state["t"]) < MIN_SAMPLES) or (luma < MIN_LUMA) or (motion_energy > MOTION_MAX) or calibrating:
        return {"hr_bpm": None, "br_rpm": None, "hr_conf": 0.0, "br_conf": 0.0}

    # Build signals & detrend/normalize
    t = np.array(_state["t"], dtype=np.float32)
    r = _detrend_norm(_state["r"])
    g = _detrend_norm(_state["g"])
    b = _detrend_norm(_state["b"])

    # Estimate sampling rate
    dt = np.diff(t)
    fs = float(1.0 / np.clip(dt.mean(), 1e-3, 1e3)) if dt.size else 0.0

    # POS projection
    s = _pos_signal(r, g, b)
    s = s - s.mean()
    s = s / (np.std(s) + 1e-6)

    # Peaks
    f_hr, snr_hr = _band_peak_freq(s, fs, HR_BAND)
    f_br, snr_br = _band_peak_freq(s, fs, BR_BAND)

    # Map SNR to 0..1 confidence (soft knee at SNR_MIN)
    def _snr_to_conf(snr):
        if snr <= 1.0: return 0.0
        return float(np.clip((snr - 1.0) / (SNR_MIN - 1.0), 0.0, 1.0))

    if f_hr is None:
        hr_bpm, hr_conf = None, 0.0
    else:
        hr_bpm = float(f_hr * 60.0)
        hr_conf = _snr_to_conf(snr_hr)

    if f_br is None:
        br_rpm, br_conf = None, 0.0
    else:
        br_rpm = float(f_br * 60.0)
        br_conf = _snr_to_conf(snr_br)

    # Hide unconfident
    if hr_conf < 0.5: hr_bpm = None
    if br_conf < 0.5: br_rpm = None

    return {"hr_bpm": hr_bpm, "br_rpm": br_rpm, "hr_conf": hr_conf, "br_conf": br_conf}
