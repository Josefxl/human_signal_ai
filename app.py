# Human Signal AI (MVP) + Recording + Alerts + Futuristic Overlay (YAML config)
import time
from collections import deque
from pathlib import Path
import os
import csv

import cv2
import numpy as np
import plotly.graph_objects as go
import streamlit as st
import yaml

from ui.record import VideoRecorder, compose_dashboard_frame

from core.tracking import Tracker
from core.features import compute_features
from core.filters import ema
from core.quality import estimate_quality
from modules.fatigue import fatigue_score
from modules.attention import attention_score
from modules.stress import stress_score
from modules.posture import posture_status
from modules.fuse import fuse_scores
from modules.ergonomics_distance import distance_status
from ui.overlays import draw_labels

# Streamlit page
st.set_page_config(page_title="Human Signal AI — Live Monitor", layout="wide")
st.title("🧠 Human Signal AI — Live Health & Ergonomics Monitor (MVP)")
st.caption("On-device. Wellness insights only — not a medical device.")

# Config (safe defaults + merge with YAML)
DEFAULT_CFG = {
    "video": {"source": 0, "fps": 30, "width": 1280, "height": 720},
    "windows": {"fatigue_seconds": 60, "attention_seconds": 30, "stress_seconds": 30, "update_hz": 5},
    "thresholds": {
        "perclos_drowsy": 0.25,
        "blink_rate_high": 25,
        "gaze_offscreen_secs": 3.0,
        "stress_tension_high": 0.7,
        "neck_angle_slouch": 22,
        "yawn_mar": 0.6,
        "yawn_min_secs": 0.5,
        "distance_face_ratio_close": 0.32,
        "distance_face_ratio_far": 0.12,
    },
    "smoothing": {"ema_alpha": 0.25},
    "quality": {"min_brightness": 60, "max_motion_px": 5, "min_confidence": 0.5},
    "fusion": {"weights": {"fatigue": 0.4, "attention": 0.35, "stress": 0.25}},
}
cfg_path = Path("configs/default.yaml")
user_cfg = yaml.safe_load(cfg_path.read_text()) if cfg_path.exists() else {}
user_cfg = user_cfg or {}
# shallow merge per top-level key
cfg = DEFAULT_CFG | {k: (DEFAULT_CFG.get(k, {}) | user_cfg.get(k, {})) for k in DEFAULT_CFG.keys()}

# Alerts (thresholds + debounce/hold)
ALERTS = {
    "DISTRACTED": {"cond": lambda att, fat, strx: att < 50.0},
    "TIRED":      {"cond": lambda att, fat, strx: fat > 50.0},
    "STRESSED":   {"cond": lambda att, fat, strx: strx > 50.0},
}
ALERT_DEBOUNCE_ON = 0.40   # seconds condition must hold before showing
ALERT_HOLD_OFF    = 1.50   # seconds to keep banner after it clears

if "alert_state" not in st.session_state:
    st.session_state.alert_state = {name: {"is_on": False, "t_on": 0.0, "t_off": 0.0} for name in ALERTS.keys()}

def _update_alert_states(att_val: float, fat_val: float, str_val: float, now: float):
    states = st.session_state.alert_state
    for name, spec in ALERTS.items():
        active = bool(spec["cond"](att_val, fat_val, str_val))
        s = states[name]
        if active:
            if not s["is_on"]:
                if s["t_on"] == 0.0:
                    s["t_on"] = now
                if now - s["t_on"] >= ALERT_DEBOUNCE_ON:
                    s["is_on"] = True
                    s["t_off"] = 0.0
            else:
                s["t_off"] = 0.0
        else:
            s["t_on"] = 0.0
            if s["is_on"]:
                if s["t_off"] == 0.0:
                    s["t_off"] = now
                if now - s["t_off"] >= ALERT_HOLD_OFF:
                    s["is_on"] = False

def _draw_alert_banner(img_bgr, labels):
    """Draw a centered translucent banner with active alert labels."""
    if not labels:
        return img_bgr
    h, w = img_bgr.shape[:2]
    text = "  •  ".join(labels)
    font, scale, thick = cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2
    (tw, th), _ = cv2.getTextSize(text, font, scale, thick)

    pad_x, pad_y = 28, 16
    box_w = tw + pad_x * 2
    box_h = th + pad_y * 2
    x0 = (w - box_w) // 2
    y0 = 40  # from top

    overlay = img_bgr.copy()
    cv2.rectangle(overlay, (x0, y0), (x0 + box_w, y0 + box_h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.55, img_bgr, 0.45, 0, img_bgr)

    tx = x0 + pad_x
    ty = y0 + pad_y + th
    cv2.putText(img_bgr, text, (tx, ty), font, scale, (255, 255, 255), thick, cv2.LINE_AA)
    return img_bgr

# Futuristic keypoint overlay
def _draw_futuristic_overlay(img_bgr, face_landmarks):
    """
    Draws thin neon lines/pips between a small set of facial anchors to get a 'tech HUD' vibe.
    Safe if landmarks are missing.
    Expects face_landmarks as (N, 3) or (N, 2) array with MediaPipe-like indices.
    """
    if face_landmarks is None or len(face_landmarks) < 478:
        return img_bgr

    pts = face_landmarks[:, :2].astype(np.int32)

    # Some MediaPipe FaceMesh anchor indices (safe subsets)
    idxs = {
        "left_eye": 33,      # outer left eye corner
        "right_eye": 263,    # outer right eye corner
        "nose_tip": 1,       # nose tip
        "chin": 152,         # chin
        "forehead": 10,      # upper forehead
        "mouth_left": 61,
        "mouth_right": 291,
        "left_iris": 468,
        "right_iris": 473,
    }

    c1 = (255, 255, 255)  # white lines
    c2 = (90, 220, 255)   # cyan pips

    def P(i):
        return tuple(pts[i])

    overlay = img_bgr.copy()

    # Eye to eye baseline
    cv2.line(overlay, P(idxs["left_eye"]), P(idxs["right_eye"]), c1, 1, cv2.LINE_AA)
    # Nose to chin vertical
    cv2.line(overlay, P(idxs["nose_tip"]), P(idxs["chin"]), c1, 1, cv2.LINE_AA)
    # Forehead to nose
    cv2.line(overlay, P(idxs["forehead"]), P(idxs["nose_tip"]), c1, 1, cv2.LINE_AA)
    # Mouth width
    cv2.line(overlay, P(idxs["mouth_left"]), P(idxs["mouth_right"]), c1, 1, cv2.LINE_AA)
    # Small iris pips
    for i in [idxs["left_iris"], idxs["right_iris"]]:
        cv2.circle(overlay, P(i), 3, c2, 1, cv2.LINE_AA)

    # Subtle blend so it looks holographic
    cv2.addWeighted(overlay, 0.45, img_bgr, 0.55, 0, img_bgr)
    return img_bgr

# Calibration & recording state
if "calibrating" not in st.session_state:
    st.session_state.calibrating = False
if "cal_start" not in st.session_state:
    st.session_state.cal_start = 0.0
if "prev_gray" not in st.session_state:
    st.session_state.prev_gray = None
if "motion_ema" not in st.session_state:
    st.session_state.motion_ema = 0.0

# recording session_state
if "recording" not in st.session_state:
    st.session_state.recording = False
if "rec_end_time" not in st.session_state:
    st.session_state.rec_end_time = 0.0
if "rec_path" not in st.session_state:
    st.session_state.rec_path = None
if "csv_path" not in st.session_state:
    st.session_state.csv_path = None
if "recorder" not in st.session_state:
    st.session_state.recorder = None
if "rec_fps" not in st.session_state:
    st.session_state.rec_fps = 15
if "last_write_t" not in st.session_state:
    st.session_state.last_write_t = 0.0

# Sidebar
with st.sidebar:
    st.header("Run")
    cam_index = st.number_input("Camera index", value=int(cfg["video"].get("source", 0)), step=1)
    target_fps = st.slider("Target FPS", 5, 60, int(cfg["video"].get("fps", 30)))
    show_futuristic = st.toggle("Futuristic keypoint overlay", value=True)
    st.markdown("---")
    if st.button("Calibrate (5s)"):
        st.session_state.calibrating = True
        st.session_state.cal_start = time.time()
        # reset baselines in modules
        from modules import stress as stress_mod, fatigue as fatigue_mod, attention as attention_mod, posture as posture_mod
        if hasattr(stress_mod, "reset"):   stress_mod.reset()
        if hasattr(fatigue_mod, "reset"):  fatigue_mod.reset()
        if hasattr(attention_mod, "reset"): attention_mod.reset()
        if hasattr(posture_mod, "reset"):  posture_mod.reset()
        # reset motion accumulator
        st.session_state.prev_gray = None
        st.session_state.motion_ema = 0.0
        # clear alerts
        for s in st.session_state.alert_state.values():
            s.update({"is_on": False, "t_on": 0.0, "t_off": 0.0})
    st.caption("Tip: Sit upright, look at the camera, neutral face, normal breathing during calibration.")

    st.markdown("---")
    # Record controls
    if not st.session_state.recording:
        if st.button("⏺ Record 30s"):
            ts = time.strftime("%Y%m%d_%H%M%S")
            out_dir = r"C:\Users\Ghost\Desktop"  # pick your export path
            os.makedirs(out_dir, exist_ok=True)
            st.session_state.rec_path = os.path.join(out_dir, f"session_{ts}.mp4")
            st.session_state.csv_path = os.path.join(out_dir, f"session_{ts}.csv")

            update_hz_cfg = int(cfg["windows"].get("update_hz", 5))
            st.session_state.rec_fps = max(8, min(24, update_hz_cfg * 2))  # responsive recording

            st.session_state.recorder = VideoRecorder(
                st.session_state.rec_path,
                fps=st.session_state.rec_fps,
                size=(1280, 900)  # 1280x720 cam + 180px metric strip
            )

            st.session_state.recording = True
            st.session_state.rec_end_time = time.time() + 30.0
            st.session_state.last_write_t = 0.0  # reset pacing

            # CSV header
            with open(st.session_state.csv_path, "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["t_epoch", "fatigue", "attention", "stress", "posture", "distance"])
    else:
        st.write(f"Recording… {max(0, int(st.session_state.rec_end_time - time.time()))}s left")
        if st.button("⏹ Stop"):
            st.session_state.rec_end_time = time.time()

# Video capture
# On Windows, CAP_DSHOW is often stable; try CAP_MSMF if needed.
cap = cv2.VideoCapture(int(cam_index), cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FPS, target_fps)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, cfg["video"]["width"])
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cfg["video"]["height"])

tracker = Tracker()

# Buffers & UI
sec_window = 60
buf_len = sec_window * max(1, target_fps)
fatigue_buf, attention_buf, stress_buf = (deque(maxlen=buf_len) for _ in range(3))

video_placeholder = st.empty()
c1, c2, c3 = st.columns(3)
chart1, chart2, chart3 = c1.empty(), c2.empty(), c3.empty()
status_placeholder = st.empty()

UPDATE_HZ = int(cfg["windows"].get("update_hz", 5))
_chart_every = max(1, int(target_fps / UPDATE_HZ))
frame_count = 0

def plot_series(container, ys, title, color):
    xs = np.arange(len(ys)) / max(1, target_fps)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines", line=dict(color=color, width=3)))
    fig.update_layout(height=220, margin=dict(l=10, r=10, t=30, b=10), title=title, yaxis=dict(range=[0, 100]))
    container.plotly_chart(fig, use_container_width=True)

# Main loop
try:
    t_last = time.time()
    fatigue_sm = attention_sm = stress_sm = 0.0  # post-calibration baseline
    while True:
        ok, frame = cap.read()
        if not ok:
            st.error("Camera read failed. Try a different index.")
            break

        # Metrics: process FIRST
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        det   = tracker.process(frame_rgb)
        feats = compute_features(det)
        quality = estimate_quality(frame_rgb, det, feats, cfg)

        # Calibration window logic
        calibrating = st.session_state.calibrating and (time.time() - st.session_state.cal_start <= 5.0)
        if st.session_state.calibrating and not calibrating:
            st.session_state.calibrating = False  # calibration finished

        # Motion energy (for fatigue bumps on big gestures/occlusions)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if st.session_state.prev_gray is not None:
            diff = cv2.absdiff(gray, st.session_state.prev_gray)
            motion_energy = float(diff.mean()) / 255.0  # 0..1
            st.session_state.motion_ema = 0.2 * motion_energy + 0.8 * st.session_state.motion_ema
        else:
            st.session_state.motion_ema = 0.0
        st.session_state.prev_gray = gray

        # Posture first (fatigue uses it)
        post = posture_status(feats, quality, cfg, calibrating=calibrating)

        # Scores
        fat  = fatigue_score(
            feats, quality, cfg,
            calibrating=calibrating,
            posture_state=post["state"],
            motion_energy=st.session_state.motion_ema,
        )
        att  = attention_score(feats, quality, cfg, calibrating=calibrating)
        strx = stress_score(feats, quality, cfg, calibrating=calibrating)
        dist = distance_status(det, cfg)

        # Smooth for display
        alpha = cfg["smoothing"]["ema_alpha"]
        fatigue_sm   = ema(fat["score"], fatigue_sm, alpha)
        attention_sm = ema(att["score"], attention_sm, alpha)
        stress_sm    = ema(strx["score"], stress_sm, alpha)

        # Alerts update (after smoothing)
        _now = time.time()
        _update_alert_states(attention_sm, fatigue_sm, stress_sm, _now)
        active_labels = [name for name, s in st.session_state.alert_state.items() if s["is_on"]]

        fused = fuse_scores(
            fatigue=fat, attention=att, stress=strx,
            weights=cfg["fusion"]["weights"], min_conf=cfg["quality"]["min_confidence"]
        )

        fatigue_buf.append(fatigue_sm)
        attention_buf.append(attention_sm)
        stress_buf.append(stress_sm)

        # Draw: raw camera + posture/distance labels + futuristic overlay + ALERT banner
        labeled = draw_labels(frame, post, dist)  # BGR in → BGR out
        if show_futuristic:
            face_landmarks = feats.get("face", None)
            labeled = _draw_futuristic_overlay(labeled, face_landmarks)
        labeled = _draw_alert_banner(labeled, active_labels)

        video_placeholder.image(labeled[:, :, ::-1], channels="RGB", use_container_width=True)

        #  Recording pipeline (real-time pacing)
        if st.session_state.recording:
            # Compose frame: camera (with posture/distance/alerts/overlay) + 3 graphs on black
            rec_frame = compose_dashboard_frame(
                labeled,
                list(fatigue_buf),
                list(attention_buf),
                list(stress_buf),
            )

            now = time.time()
            frame_interval = 1.0 / st.session_state.rec_fps
            if st.session_state.last_write_t == 0.0:
                st.session_state.last_write_t = now

            # Write as many frames as wall-clock requires
            while st.session_state.last_write_t + frame_interval <= now:
                if st.session_state.recorder and st.session_state.recorder.is_open:
                    st.session_state.recorder.write(rec_frame)
                st.session_state.last_write_t += frame_interval

            # Append one CSV row per loop (analytics)
            with open(st.session_state.csv_path, "a", newline="") as f:
                w = csv.writer(f)
                w.writerow([
                    now,
                    f"{fatigue_sm:.2f}",
                    f"{attention_sm:.2f}",
                    f"{stress_sm:.2f}",
                    post["state"],
                    dist["state"],
                ])

            # Auto-stop
            if now >= st.session_state.rec_end_time:
                st.session_state.recording = False
                st.session_state.last_write_t = 0.0
                if st.session_state.recorder:
                    st.session_state.recorder.release()
                st.success(f"Saved video → {st.session_state.rec_path}")
                st.info(f"Saved metrics → {st.session_state.csv_path}")

        # Charts & status (lower rate = no flicker)
        frame_count += 1
        if frame_count % _chart_every == 0:
            plot_series(chart1, list(fatigue_buf),   "Fatigue",   "#1f77b4")  # blue
            plot_series(chart2, list(attention_buf), "Attention", "#2ca02c")  # green
            plot_series(chart3, list(stress_buf),    "Stress",    "#d62728")  # red

            status_placeholder.info(
                f"Readiness: {fused['readiness']:0.0f} | "
                f"Fatigue: {fatigue_sm:0.0f} ({fat['conf']:.2f}) | "
                f"Attention: {attention_sm:0.0f} ({att['conf']:.2f}) | "
                f"Stress: {stress_sm:0.0f} ({strx['conf']:.2f}) | "
                f"Posture: {post['state']} | "
                f"Distance: {dist['state']}"
                f"{' | ⚠ ' + ', '.join(fused['flags']) if fused['flags'] else ''}"
            )

        # Pace to UI cadence
        target_dt = 1.0 / UPDATE_HZ
        dt = time.time() - t_last
        if dt < target_dt:
            time.sleep(target_dt - dt)
        t_last = time.time()

except KeyboardInterrupt:
    pass
finally:
    # Ensure recorder is released and pacing reset
    if st.session_state.get("recorder"):
        st.session_state.recorder.release()
    st.session_state.last_write_t = 0.0
    cap.release()
