# ui/record.py
import cv2
import numpy as np
from datetime import datetime

class VideoRecorder:
    def __init__(self, out_path, fps=20, size=(1280, 720)):
        # Use mp4v for Windows compatibility
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self.writer = cv2.VideoWriter(out_path, fourcc, fps, size)
        self.size = size
        self.is_open = self.writer.isOpened()

    def write(self, frame_bgr):
        if not self.is_open: return
        if (frame_bgr.shape[1], frame_bgr.shape[0]) != self.size:
            frame_bgr = cv2.resize(frame_bgr, self.size)
        self.writer.write(frame_bgr)

    def release(self):
        if self.is_open:
            self.writer.release()
            self.is_open = False

def draw_sparkline(img, series, x, y, w, h, color, title):
    """Draw simple 0..100 sparkline and border."""
    cv2.rectangle(img, (x, y), (x+w, y+h), (30,30,30), 1)
    cv2.putText(img, title, (x+6, y+18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (40,40,40), 1, cv2.LINE_AA)
    if not series: return img
    vals = np.array(series[-w:], dtype=np.float32)  # clip to width
    vals = np.clip(vals, 0, 100)
    # Map to pixel coords (0 at bottom, 100 at top)
    ys = y + h - ((vals/100.0) * (h-24)).astype(np.int32) - 4  # leave space for title
    xs = np.arange(len(ys), dtype=np.int32) + x
    pts = np.stack([xs, ys], axis=1)
    # Draw polyline
    for i in range(1, len(pts)):
        cv2.line(img, tuple(pts[i-1]), tuple(pts[i]), color, 2, cv2.LINE_AA)
    return img

def compose_dashboard_frame(cam_bgr, fat_series, att_series, str_series):
    """Return a single frame with camera on top and 3 sparklines at bottom."""
    # Normalize sizes
    W = 1280
    H_CAM = 720
    H_BAR = 180
    cam = cv2.resize(cam_bgr, (W, H_CAM))
    board = np.full((H_BAR, W, 3), 255, dtype=np.uint8)

    # Three equal charts
    w = W // 3
    draw_sparkline(board, fat_series, 10, 10, w-20, H_BAR-20, (183,121,31), "Fatigue")   # blue-ish in BGR? -> we'll map later
    draw_sparkline(board, att_series, 10+w, 10, w-20, H_BAR-20, (50,160,50), "Attention")
    draw_sparkline(board, str_series, 10+2*w, 10, w-20, H_BAR-20, (40,40,200), "Stress")

    # Stack vertically
    out = np.vstack([cam, board])
    return out
