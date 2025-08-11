# app.py — Human Signal AI (MVP)
import time
from collections import deque
from pathlib import Path
from ui.record import VideoRecorder, compose_dashboard_frame
import csv, os

import cv2
import numpy as np
import plotly.graph_objects as go
import streamlit as st
import yaml

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

st.set_page_config(page_title="Human Signal AI — Live Monitor", layout="wide")
st.title("🧠 Human Signal AI — Live Health & Ergonomics Monitor (MVP)")
st.caption("On-device. Wellness insights only — not a medical device.")

# ---------------- Config (safe defaults + merge) ----------------
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
cfg = DEFAULT_CFG | {k: (DEFAULT_CFG.get(k, {}) | user_cfg.get(k, {})) for k in DEFAULT_CFG.keys()}

# ---------------- Calibration state ----------------
if "calibrating" not in st.session_state:
    st.session_state.calibrating = False
if "cal_start" not in st.session_state:
    st.session_state.cal_start = 0.0

# motion state for fatigue bumps
if "prev_gray" not in st.session_state:
    st.session_state.prev_gray = None
if "motion_ema" not in st.session_state:
    st.session_state.motion_ema = 0.0

# ---------------- Sidebar ----------------
with st.sidebar:
    st.header("Run")
    cam_index = st.number_input("Camera index", value=int(cfg["video"].get("source", 0)), step=1)
    target_fps = st.slider("Target FPS", 5, 60, int(cfg["video"].get("fps", 30)))
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
    st.caption("Tip: Sit upright, look at the camera, neutral face, normal breathing during calibration.")

# ---------------- Video capture ----------------
# On Windows, CAP_DSHOW is often stable; try CAP_MSMF if needed.
cap = cv2.VideoCapture(int(cam_index), cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FPS, target_fps)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, cfg["video"]["width"])
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cfg["video"]["height"])

tracker = Tracker()

# ---------------- Buffers & UI ----------------
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

# ---------------- Main loop ----------------
try:
    t_last = time.time()
    fatigue_sm = attention_sm = stress_sm = 0.0  # post-calibration baseline
    while True:
        ok, frame = cap.read()
        if not ok:
            st.error("Camera read failed. Try a different index.")
            break

        # ---- Metrics: process FIRST
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

        fused = fuse_scores(
            fatigue=fat, attention=att, stress=strx,
            weights=cfg["fusion"]["weights"], min_conf=cfg["quality"]["min_confidence"]
        )

        fatigue_buf.append(fatigue_sm)
        attention_buf.append(attention_sm)
        stress_buf.append(stress_sm)

        # ---- Draw RAW camera + minimal labels (posture, distance)
        labeled = draw_labels(frame, post, dist)  # BGR in → BGR out
        video_placeholder.image(labeled[:, :, ::-1], channels="RGB", use_container_width=True)

        # ---- Charts & status (lower rate = no flicker)
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
                f"Posture: {post['state']} "
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
    cap.release()
