# ui/record.py
import cv2
import numpy as np

# Exact brand colors (BGR)
# Fatigue blue  : #1f77b4 -> RGB(31,119,180) -> BGR(180,119,31)
# Attention green: #2ca02c -> RGB(44,160,44)  -> BGR(44,160,44)
# Stress red    : #d62728 -> RGB(214,39,40)  -> BGR(40,39,214)
COLOR_FATIGUE = (180, 119, 31)
COLOR_ATTEN   = (44, 160, 44)
COLOR_STRESS  = (40, 39, 214)
COLOR_BORDER  = (32, 32, 32)
COLOR_TEXT    = (30, 30, 30)
COLOR_BG      = (255, 255, 255)

class VideoRecorder:
    def __init__(self, out_path, fps=20, size=(1280, 900)):
        """
        size = (W, H). For our layout: 1280 x (720 camera + 180 charts) = 1280x900
        """
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # MP4 for broad compatibility
        self.writer = cv2.VideoWriter(out_path, fourcc, fps, size)
        self.size = size
        self.is_open = self.writer.isOpened()

    def write(self, frame_bgr):
        if not self.is_open:
            return
        if (frame_bgr.shape[1], frame_bgr.shape[0]) != self.size:
            frame_bgr = cv2.resize(frame_bgr, self.size)
        self.writer.write(frame_bgr)

    def release(self):
        if self.is_open:
            self.writer.release()
            self.is_open = False

def _draw_text(img, text, org, scale=0.6, color=COLOR_TEXT, thick=2):
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA)

def draw_sparkline(board, series, x, y, w, h, color, title):
    """
    Draws a sparkline for a 0..100 series into board at (x,y) with width w and height h.
    Also draws title and current numeric value.
    """
    # Panel
    cv2.rectangle(board, (x, y), (x+w, y+h), COLOR_BORDER, 1)
    # Title
    _draw_text(board, title, (x+10, y+22), scale=0.6, color=COLOR_TEXT, thick=2)

    if not series:
        return board

    vals = np.array(series[-w:], dtype=np.float32)  # clip to panel width
    vals = np.clip(vals, 0, 100)
    # Leave top space for title; map 0 at bottom -> h-8 margin
    chart_top = y + 30
    chart_h = h - 38
    chart_bottom = chart_top + chart_h

    ys = chart_bottom - ((vals/100.0) * chart_h).astype(np.int32)
    xs = np.arange(len(ys), dtype=np.int32) + x

    # Polyline
    for i in range(1, len(xs)):
        cv2.line(board, (xs[i-1], ys[i-1]), (xs[i], ys[i]), color, 2, cv2.LINE_AA)

    # Current value bubble
    current = float(vals[-1])
    cv2.circle(board, (xs[-1], ys[-1]), 4, color, -1, cv2.LINE_AA)
    _draw_text(board, f"{current:0.0f}", (x+w-52, y+22), scale=0.6, color=color, thick=2)
    return board

def compose_dashboard_frame(cam_bgr, fat_series, att_series, str_series):
    """
    Compose the output frame:
      - Top: camera (1280x720) with your overlay labels already drawn
      - Bottom: 3 metric panels (Fatigue, Attention, Stress), each a sparkline with current value
    Returns a BGR frame of shape (900, 1280, 3)
    """
    W = 1280
    H_CAM = 720
    H_BOARD = 180

    cam = cv2.resize(cam_bgr, (W, H_CAM))
    board = np.full((H_BOARD, W, 3), 255, dtype=np.uint8)

    # Panel widths and positions
    w_panel = W // 3
    pad = 10
    # Fatigue
    draw_sparkline(board, fat_series, pad, pad, w_panel - 2*pad, H_BOARD - 2*pad, COLOR_FATIGUE, "Fatigue")
    # Attention
    draw_sparkline(board, att_series, w_panel + pad, pad, w_panel - 2*pad, H_BOARD - 2*pad, COLOR_ATTEN, "Attention")
    # Stress
    draw_sparkline(board, str_series, 2*w_panel + pad, pad, w_panel - 2*pad, H_BOARD - 2*pad, COLOR_STRESS, "Stress")

    # Stack camera + board
    out = np.vstack([cam, board])
    return out
