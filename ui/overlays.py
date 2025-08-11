# --- PATCH 02: ui/overlays.py ---
import cv2

def draw_labels(frame_bgr, posture, distance):
    img = frame_bgr.copy()
    # Posture label
    txt1 = f"Posture: {posture['state']}"
    cv2.rectangle(img, (10, 10), (10+220, 42), (255,255,255), -1)
    cv2.putText(img, txt1, (16, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (30,30,30), 2, cv2.LINE_AA)

    # Distance label
    txt2 = f"Distance: {distance['state']}"
    cv2.rectangle(img, (10, 50), (10+220, 82), (255,255,255), -1)
    cv2.putText(img, txt2, (16, 74), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (30,30,30), 2, cv2.LINE_AA)

    return img
