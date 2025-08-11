import numpy as np

# FaceMesh landmark indices for eyes/mouth (approx; refined later)
LEFT_EYE = [33, 160, 158, 133, 153, 144]   # around left eye
RIGHT_EYE = [362, 385, 387, 263, 373, 380]
MOUTH = [61, 291, 13, 14, 78, 308]         # lips corners & vertical

def _aspect_ratio(pts):
    # EAR/MAR style: (vertical distances sum) / (2 * horizontal distance)
    p = pts
    vert = np.linalg.norm(p[1]-p[5]) + np.linalg.norm(p[2]-p[4])
    horiz = np.linalg.norm(p[0]-p[3]) + 1e-6
    return vert / (2.0 * horiz)

def eye_aspect_ratio(face):
    if face is None: return None, None
    le = face[LEFT_EYE, :2]; re = face[RIGHT_EYE, :2]
    return _aspect_ratio(le), _aspect_ratio(re)

def mouth_aspect_ratio(face):
    if face is None: return None
    m = face[MOUTH, :2]
    # use two verticals (13-14) and horizontal (61-291)
    vert = np.linalg.norm(m[2]-m[3])
    horiz = np.linalg.norm(m[0]-m[1]) + 1e-6
    return vert / horiz

def gaze_proxy(face):
    # crude proxy: eye centers vs face bbox center → stability metric
    if face is None: return 0.0
    xs, ys = face[:,0], face[:,1]
    cx, cy = xs.mean(), ys.mean()
    left_center = face[np.array(LEFT_EYE), :2].mean(axis=0)
    right_center = face[np.array(RIGHT_EYE), :2].mean(axis=0)
    eye_center = (left_center + right_center) / 2.0
    dist = np.linalg.norm(eye_center - np.array([cx, cy]))
    face_diag = np.hypot(xs.max()-xs.min(), ys.max()-ys.min()) + 1e-6
    # smaller is more centered (more "on-screen"); normalize to 0..1
    return float(np.clip(1.0 - (dist / (0.35 * face_diag)), 0.0, 1.0))

def neck_angle_deg(pose):
    # approximate neck angle from ear->shoulder vector vs vertical
    if pose is None: return None
    # indices: 11 left_shoulder, 12 right_shoulder, 23 left_hip, 24 right_hip, 0 nose, 7 left_ear, 8 right_ear
    try:
        l_sh, r_sh = pose[11,:2], pose[12,:2]
        l_ear, r_ear = pose[7,:2], pose[8,:2]
        shoulder_mid = (l_sh + r_sh)/2.0
        ear_mid = (l_ear + r_ear)/2.0
        v = ear_mid - shoulder_mid
        v_norm = v / (np.linalg.norm(v)+1e-6)
        # angle to vertical
        dot = np.dot(v_norm, np.array([0.0,-1.0]))
        ang = np.degrees(np.arccos(np.clip(dot, -1.0, 1.0)))
        return float(ang)
    except Exception:
        return None

def compute_features(det):
    face = det["face"]; pose = det["pose"]
    le, re = eye_aspect_ratio(face)
    mar = mouth_aspect_ratio(face)
    gaze = gaze_proxy(face)
    neck = neck_angle_deg(pose)
    return {
        "face": face, "pose": pose,
        "ear_left": le, "ear_right": re,
        "mar": mar, "gaze": gaze,
        "neck_angle": neck
    }
