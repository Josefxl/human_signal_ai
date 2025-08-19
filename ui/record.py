# dark HUD metric strip (graphs only) + camera on top
import cv2
import numpy as np
from typing import Sequence, Tuple

# --- Colors (BGR) for black theme ---
COLOR_FATIGUE = (180, 119, 31)    # ≈ blue
COLOR_ATTEN   = (44, 160, 44)     # ≈ green
COLOR_STRESS  = (40, 39, 214)     # ≈ red
COLOR_BORDER  = (50, 50, 55)
COLOR_SUBTLE  = (36, 36, 40)
COLOR_BG      = (0, 0, 0)         # pure black
COLOR_PANEL   = (14, 14, 18)
COLOR_ACCENT  = (185, 185, 190)

FONT = cv2.FONT_HERSHEY_SIMPLEX

class VideoRecorder:
    def __init__(self, out_path: str, fps: int = 15, size: Tuple[int,int]=(1280, 900)):
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self.writer = cv2.VideoWriter(out_path, fourcc, fps, size)
        self.size = size
        self.is_open = self.writer.isOpened()

    def write(self, frame_bgr: np.ndarray):
        if not self.is_open:
            return
        if (frame_bgr.shape[1], frame_bgr.shape[0]) != self.size:
            frame_bgr = cv2.resize(frame_bgr, self.size)
        self.writer.write(frame_bgr)

    def release(self):
        if self.is_open:
            self.writer.release()
            self.is_open = False

def _rounded_panel(img, x, y, w, h, r=12, border=COLOR_BORDER, fill=COLOR_PANEL):
    if fill is not None:
        overlay = img.copy()
        cv2.rectangle(overlay, (x+r, y), (x+w-r, y+h), fill, -1)
        cv2.rectangle(overlay, (x, y+r), (x+w, y+h-r), fill, -1)
        cv2.circle(overlay, (x+r, y+r), r, fill, -1, cv2.LINE_AA)
        cv2.circle(overlay, (x+w-r, y+r), r, fill, -1, cv2.LINE_AA)
        cv2.circle(overlay, (x+r, y+h-r), r, fill, -1, cv2.LINE_AA)
        cv2.circle(overlay, (x+w-r, y+h-r), r, fill, -1, cv2.LINE_AA)
        cv2.addWeighted(overlay, 0.95, img, 0.05, 0, img)
    cv2.line(img, (x+r, y), (x+w-r, y), border, 1, cv2.LINE_AA)
    cv2.line(img, (x+r, y+h), (x+w-r, y+h), border, 1, cv2.LINE_AA)
    cv2.line(img, (x, y+r), (x, y+h-r), border, 1, cv2.LINE_AA)
    cv2.line(img, (x+w, y+r), (x+w, y+h-r), border, 1, cv2.LINE_AA)
    cv2.ellipse(img, (x+r, y+r), (r, r), 180, 0, 90, border, 1, cv2.LINE_AA)
    cv2.ellipse(img, (x+w-r, y+r), (r, r), 270, 0, 90, border, 1, cv2.LINE_AA)
    cv2.ellipse(img, (x+r, y+h-r), (r, r), 90, 0, 90, border, 1, cv2.LINE_AA)
    cv2.ellipse(img, (x+w-r, y+h-r), (r, r), 0, 0, 90, border, 1, cv2.LINE_AA)

def _put_text(img, text, org, scale=0.65, color=COLOR_ACCENT, thick=2):
    cv2.putText(img, text, org, FONT, scale, color, thick, cv2.LINE_AA)

def draw_sparkline(board: np.ndarray, series: Sequence[float], x: int, y: int, w: int, h: int,
                   color: Tuple[int,int,int], title: str):
    _rounded_panel(board, x, y, w, h, r=16)
    _put_text(board, title, (x+16, y+28))
    if not series:
        return board

    # chart region
    chart_x = x + 12; chart_y = y + 40; chart_w = w - 24; chart_h = h - 56

    # subtle horizontal guides
    step_y = max(28, chart_h // 5)
    for gy in range(chart_y + 8, chart_y + chart_h, step_y):
        cv2.line(board, (chart_x, gy), (chart_x + chart_w, gy), COLOR_SUBTLE, 1, cv2.LINE_AA)

    vals = np.array(series[-chart_w:], dtype=np.float32)
    vals = np.clip(vals, 0, 100)
    ys = chart_y + chart_h - ((vals/100.0) * chart_h).astype(np.int32)
    xs = np.arange(len(ys), dtype=np.int32) + chart_x

    # thicker line for visibility
    for i in range(1, len(xs)):
        cv2.line(board, (xs[i-1], ys[i-1]), (xs[i], ys[i]), color, 3, cv2.LINE_AA)
    cv2.circle(board, (xs[-1], ys[-1]), 5, color, -1, cv2.LINE_AA)
    return board

def compose_dashboard_frame(cam_bgr: np.ndarray,
                            fat_series: Sequence[float],
                            att_series: Sequence[float],
                            str_series: Sequence[float]) -> np.ndarray:
    """
    Returns a 1280x900 BGR frame: camera (1280x720) + black 3-graph strip (1280x180).
    No watermark. No HR/BR tiles.
    """
    W = 1280; H_CAM = 720; H_BOARD = 180
    cam = cv2.resize(cam_bgr, (W, H_CAM))

    board = np.full((H_BOARD, W, 3), COLOR_BG, dtype=np.uint8)
    pad = 10
    col_w = (W - pad*4) // 3
    x1 = pad
    x2 = x1 + col_w + pad
    x3 = x2 + col_w + pad
    panel_h = H_BOARD - 2*pad

    draw_sparkline(board, fat_series, x1, pad, col_w, panel_h, COLOR_FATIGUE, "Fatigue")
    draw_sparkline(board, att_series, x2, pad, col_w, panel_h, COLOR_ATTEN,   "Attention")
    draw_sparkline(board, str_series, x3, pad, col_w, panel_h, COLOR_STRESS,  "Stress")

    return np.vstack([cam, board])
