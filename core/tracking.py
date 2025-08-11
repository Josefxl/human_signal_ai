import mediapipe as mp
import numpy as np

mp_face = mp.solutions.face_mesh
mp_pose = mp.solutions.pose

class Tracker:
    def __init__(self):
        self.face = mp_face.FaceMesh(
            static_image_mode=False, refine_landmarks=True,
            max_num_faces=1, min_detection_confidence=0.5, min_tracking_confidence=0.5
        )
        self.pose = mp_pose.Pose(
            static_image_mode=False, model_complexity=1,
            enable_segmentation=False, min_detection_confidence=0.5, min_tracking_confidence=0.5
        )
        self.img_w = None
        self.img_h = None

    def process(self, frame_rgb):
        h, w, _ = frame_rgb.shape
        self.img_h, self.img_w = h, w
        face_res = self.face.process(frame_rgb)
        pose_res = self.pose.process(frame_rgb)
        face_landmarks = None
        if face_res.multi_face_landmarks:
            face_landmarks = np.array(
                [(lm.x * w, lm.y * h, lm.z) for lm in face_res.multi_face_landmarks[0].landmark],
                dtype=np.float32
            )
        pose_landmarks = None
        if pose_res.pose_landmarks:
            pose_landmarks = np.array(
                [(lm.x * w, lm.y * h, lm.z) for lm in pose_res.pose_landmarks.landmark],
                dtype=np.float32
            )
        return {"face": face_landmarks, "pose": pose_landmarks, "size": (w, h)}
