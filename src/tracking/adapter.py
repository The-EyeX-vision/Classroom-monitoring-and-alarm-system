import math

# ─── COCO-17 Keypoint Indices (YOLOv8-Pose) ──────────────────────────────────
# 0:nose  1:left_eye  2:right_eye  3:left_ear  4:right_ear
# 5:left_shoulder  6:right_shoulder  7:left_elbow  8:right_elbow
# 9:left_wrist  10:right_wrist  11:left_hip  12:right_hip
# 13:left_knee  14:right_knee  15:left_ankle  16:right_ankle

NOSE          = 0
L_EYE         = 1
R_EYE         = 2
L_EAR         = 3
R_EAR         = 4
L_SHOULDER    = 5
R_SHOULDER    = 6
L_ELBOW       = 7
R_ELBOW       = 8
L_WRIST       = 9
R_WRIST       = 10
L_HIP         = 11
R_HIP         = 12
L_KNEE        = 13
R_KNEE        = 14
L_ANKLE       = 15
R_ANKLE       = 16

_KP_CONF_THRESHOLD = 0.25   # minimum keypoint confidence to use

def _kp(keypoints, idx):
    """Returns keypoint dict at idx if confidence is sufficient, else None."""
    if not keypoints or idx >= len(keypoints):
        return None
    kp = keypoints[idx]
    if kp.get("confidence", 1.0) < _KP_CONF_THRESHOLD:
        return None
    return kp


def _bbox_center(bbox):
    if not bbox or not isinstance(bbox, dict):
        return None
    return (
        (bbox["x_min"] + bbox["x_max"]) / 2.0,
        (bbox["y_min"] + bbox["y_max"]) / 2.0,
    )


def _hip_midpoint(keypoints):
    """Hip midpoint in normalized frame coords — used as student centroid."""
    lh = _kp(keypoints, L_HIP)
    rh = _kp(keypoints, R_HIP)
    if lh and rh:
        return ((lh["x"] + rh["x"]) / 2.0, (lh["y"] + rh["y"]) / 2.0)
    if lh:
        return (lh["x"], lh["y"])
    if rh:
        return (rh["x"], rh["y"])
    return None


def _head_yaw_deg(keypoints):
    """
    Estimates head yaw (left/right turn) from ear and nose landmarks.

    Approach: compare nose x position relative to the midpoint between
    left and right ears. When looking forward both ears are equidistant
    from the nose. A sideways turn hides one ear and brings the nose
    closer to the other ear.

    Returns degrees: positive = turned right, negative = turned left.
    """
    nose = _kp(keypoints, NOSE)
    l_ear = _kp(keypoints, L_EAR)
    r_ear = _kp(keypoints, R_EAR)

    if nose and l_ear and r_ear:
        ear_mid_x = (l_ear["x"] + r_ear["x"]) / 2.0
        ear_width = abs(r_ear["x"] - l_ear["x"])
        if ear_width < 1e-4:
            return 0.0
        offset = (nose["x"] - ear_mid_x) / (ear_width / 2.0)
        offset = max(-1.0, min(1.0, offset))
        return offset * 70.0

    # Fallback: only one ear visible means strong turn toward visible ear
    if nose and l_ear and not r_ear:
        return -55.0   # right ear hidden → turned right (away from left)
    if nose and r_ear and not l_ear:
        return 55.0    # left ear hidden → turned left

    # Fallback: eye separation ratio
    l_eye = _kp(keypoints, L_EYE)
    r_eye = _kp(keypoints, R_EYE)
    if nose and l_eye and r_eye:
        eye_mid_x = (l_eye["x"] + r_eye["x"]) / 2.0
        eye_width = abs(r_eye["x"] - l_eye["x"])
        if eye_width > 1e-4:
            offset = (nose["x"] - eye_mid_x) / (eye_width / 2.0)
            return max(-1.0, min(1.0, offset)) * 55.0

    return 0.0


def _head_pitch_deg(keypoints):
    """
    Estimates head pitch (looking down) from nose vs ear/shoulder positions.
    Positive value = looking down.
    """
    nose = _kp(keypoints, NOSE)
    if not nose:
        return 0.0

    l_ear = _kp(keypoints, L_EAR)
    r_ear = _kp(keypoints, R_EAR)
    if l_ear and r_ear:
        ear_y = (l_ear["y"] + r_ear["y"]) / 2.0
        diff_y = nose["y"] - ear_y
        if diff_y > 0.03:
            return diff_y * 280.0
        return 0.0

    # Fallback: nose vs shoulder midpoint
    ls = _kp(keypoints, L_SHOULDER)
    rs = _kp(keypoints, R_SHOULDER)
    if nose and ls and rs:
        shoulder_y = (ls["y"] + rs["y"]) / 2.0
        diff = shoulder_y - nose["y"]   # smaller = head tilted down toward desk
        if diff < 0.10:                 # nose very close to shoulders → deep forward tilt
            return (0.10 - diff) * 200.0
    return 0.0


def _torso_lean_deg(keypoints):
    """
    Computes lateral spine lean from shoulder midpoint vs hip midpoint offset.
    Larger = leaning sideways away from desk.
    """
    ls = _kp(keypoints, L_SHOULDER)
    rs = _kp(keypoints, R_SHOULDER)
    lh = _kp(keypoints, L_HIP)
    rh = _kp(keypoints, R_HIP)

    if not (ls and rs):
        return 0.0

    sx = (ls["x"] + rs["x"]) / 2.0
    sy = (ls["y"] + rs["y"]) / 2.0

    if lh and rh:
        hx = (lh["x"] + rh["x"]) / 2.0
        hy = (lh["y"] + rh["y"]) / 2.0
        spine_lean = math.degrees(math.atan2(abs(sx - hx), max(1e-4, abs(sy - hy))))
        return spine_lean

    # Shoulder tilt only
    dx = ls["x"] - rs["x"]
    dy = ls["y"] - rs["y"]
    if abs(dx) < 1e-4:
        return 90.0
    return math.degrees(math.atan2(abs(dy), abs(dx)))


def _classify_hand_positions(keypoints):
    """
    Classifies hand positions relative to body using COCO wrist/elbow landmarks.

    Returns:
        hands_under_desk (bool): wrists low relative to hips (in lap)
        hands_in_writing_pos (bool): wrists in normal desk writing zone
        hands_moved_away (bool): wrists reaching far sideways / forward
    """
    lw = _kp(keypoints, L_WRIST)
    rw = _kp(keypoints, R_WRIST)

    if not lw and not rw:
        return False, True, False

    lh = _kp(keypoints, L_HIP)
    rh = _kp(keypoints, R_HIP)
    hip_y = None
    if lh and rh:
        hip_y = (lh["y"] + rh["y"]) / 2.0
    elif lh:
        hip_y = lh["y"]
    elif rh:
        hip_y = rh["y"]

    ls = _kp(keypoints, L_SHOULDER)
    rs = _kp(keypoints, R_SHOULDER)
    shoulder_x_left = ls["x"] if ls else 0.15
    shoulder_x_right = rs["x"] if rs else 0.85

    # ── Hands under desk: wrist lower than hip or significantly below elbows ──
    def is_under(wrist, elbow_idx):
        if not wrist:
            return False
        if hip_y is not None and wrist["y"] > hip_y + 0.06:
            return True
        elbow = _kp(keypoints, elbow_idx)
        if elbow and (wrist["y"] - elbow["y"]) > 0.12:
            return True
        return False

    left_under = is_under(lw, L_ELBOW)
    right_under = is_under(rw, R_ELBOW)
    hands_under = left_under or right_under

    # ── Writing position: wrists in body-width zone, above hips ──
    body_left = shoulder_x_left - 0.10
    body_right = shoulder_x_right + 0.10

    def is_writing(wrist):
        if not wrist:
            return False
        in_x = body_left <= wrist["x"] <= body_right
        in_y = (hip_y is None or wrist["y"] < hip_y + 0.05) and wrist["y"] > 0.20
        return in_x and in_y

    hands_writing = is_writing(lw) or is_writing(rw)

    # ── Hands moved away: reaching far outside shoulder width ──
    def is_reaching(wrist):
        if not wrist:
            return False
        return wrist["x"] < shoulder_x_left - 0.18 or wrist["x"] > shoulder_x_right + 0.18

    hands_moved_away = is_reaching(lw) or is_reaching(rw)

    return hands_under, hands_writing, hands_moved_away


def _is_standing(keypoints, bbox):
    """
    Returns True if person is the mobile examiner (standing up).
    Seated students' knees/ankles are hidden behind desks.
    """
    lower_body_visible = False
    if keypoints:
        lk = _kp(keypoints, L_KNEE)
        rk = _kp(keypoints, R_KNEE)
        la = _kp(keypoints, L_ANKLE)
        ra = _kp(keypoints, R_ANKLE)
        if lk or rk or la or ra:
            lower_body_visible = True

    tall_bbox = False
    if bbox and isinstance(bbox, dict):
        bw = bbox.get("x_max", 0) - bbox.get("x_min", 0)
        bh = bbox.get("y_max", 0) - bbox.get("y_min", 0)
        if bw > 1e-4 and (bh / bw) > 1.4:
            tall_bbox = True

    return lower_body_visible and tall_bbox


def _phone_near_person(bbox, phone_objects):
    """Returns True if a detected phone bbox overlaps or is close to the person's bbox."""
    if not phone_objects or not bbox:
        return False
    px1 = bbox["x_min"]
    py1 = bbox["y_min"]
    px2 = bbox["x_max"]
    py2 = bbox["y_max"]
    for obj in phone_objects:
        ob = obj.get("bbox", {})
        if not ob:
            continue
        # Expand person bbox by small margin for proximity check
        margin = 0.05
        if (ob["x_max"] >= px1 - margin and ob["x_min"] <= px2 + margin and
                ob["y_max"] >= py1 - margin and ob["y_min"] <= py2 + margin):
            return True
    return False


def adapt(structured_output, full_w=1280, full_h=720):
    """
    Converts YOLOv8-Pose structured output → list of per-student behaviour records
    compatible with CentroidTracker and StudentSuspicionTracker.
    """
    poses = structured_output.get("poses", [])
    objects = structured_output.get("objects", [])
    records = []

    for pose in poses:
        bbox = pose.get("bbox")
        keypoints = pose.get("keypoints", [])

        # Use hip midpoint as stable centroid; fall back to bbox centre
        centroid = _hip_midpoint(keypoints)
        if centroid is None:
            centroid = _bbox_center(bbox)
        if centroid is None:
            continue

        head_yaw = _head_yaw_deg(keypoints)
        head_pitch = _head_pitch_deg(keypoints)
        torso_lean = _torso_lean_deg(keypoints)
        hands_under, hands_writing, hands_moved_away = _classify_hand_positions(keypoints)
        standing = _is_standing(keypoints, bbox)
        object_near_hand = _phone_near_person(bbox, objects)

        records.append({
            "centroid": centroid,
            "head_yaw_deg": head_yaw,
            "head_pitch_deg": head_pitch,
            "mouth_open_ratio": 0.0,        # YOLOv8-Pose has no face mesh; conservative default
            "torso_lean_deg": torso_lean,
            "hands_under_desk": hands_under,
            "hands_in_writing_pos": hands_writing,
            "hands_moved_away": hands_moved_away,
            "object_near_hand": object_near_hand,
            "is_standing": standing,
            "pose_id": pose.get("pose_id"),
            "bbox": bbox,
        })

    return records
