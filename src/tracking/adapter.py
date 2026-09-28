import math

# MediaPipe Pose landmark indices
NOSE           = 0
LEFT_EAR       = 7
RIGHT_EAR      = 8
LEFT_SHOULDER  = 11
RIGHT_SHOULDER = 12
LEFT_ELBOW     = 13
LEFT_KNEE      = 25
RIGHT_KNEE     = 26
LEFT_ANKLE     = 27
RIGHT_ANKLE    = 28
RIGHT_ELBOW    = 14
LEFT_WRIST     = 15
RIGHT_WRIST    = 16
LEFT_HIP       = 23
RIGHT_HIP      = 24

# MediaPipe Face Landmarker indices
NOSE_TIP       = 1
FOREHEAD       = 10
CHIN           = 152
UPPER_LIP      = 13
LOWER_LIP      = 14
LEFT_CHEEK     = 234
RIGHT_CHEEK    = 454


def _crop_lm_to_global(lm, crop_offset, crop_size, full_w, full_h):
    cx, cy = crop_offset
    cw, ch = crop_size

    pixel_x = cx + (lm["x"] * cw)
    pixel_y = cy + (lm["y"] * ch)

    return {
        "x": pixel_x / full_w,
        "y": pixel_y / full_h,
        "z": lm.get("z", 0.0),
        "visibility": lm.get("visibility", 1.0)
    }


def _hip_midpoint(landmarks):
    if len(landmarks) < 25:
        return None
    lh = landmarks[LEFT_HIP]
    rh = landmarks[RIGHT_HIP]
    return ((lh["x"] + rh["x"]) / 2.0, (lh["y"] + rh["y"]) / 2.0)


def _bbox_center(bbox):
    if not bbox:
        return None
    if isinstance(bbox, dict):
        return (
            (bbox["x_min"] + bbox["x_max"]) / 2.0,
            (bbox["y_min"] + bbox["y_max"]) / 2.0,
        )
    elif isinstance(bbox, (list, tuple)) and len(bbox) >= 4:
        return (
            (bbox[0] + bbox[2]) / 2.0,
            (bbox[1] + bbox[3]) / 2.0,
        )
    return None


def _shoulder_tilt_and_lean(landmarks):
    """
    Calculates shoulder tilt / torso lean angle in degrees.
    """
    if not landmarks or len(landmarks) < 13:
        return 0.0

    ls = landmarks[LEFT_SHOULDER]
    rs = landmarks[RIGHT_SHOULDER]

    vis_min = min(ls.get("visibility", 1.0), rs.get("visibility", 1.0))
    if vis_min < 0.3:
        return 0.0

    dx = ls["x"] - rs["x"]
    dy = ls["y"] - rs["y"]

    if abs(dx) < 1e-4:
        return 90.0

    tilt_deg = math.degrees(math.atan2(abs(dy), abs(dx)))

    if len(landmarks) >= 25:
        lh = landmarks[LEFT_HIP]
        rh = landmarks[RIGHT_HIP]
        if min(lh.get("visibility", 1.0), rh.get("visibility", 1.0)) >= 0.3:
            sx = (ls["x"] + rs["x"]) / 2.0
            sy = (ls["y"] + rs["y"]) / 2.0
            hx = (lh["x"] + rh["x"]) / 2.0
            hy = (lh["y"] + rh["y"]) / 2.0
            spine_lean = math.degrees(math.atan2(abs(sx - hx), max(1e-4, abs(sy - hy))))
            return max(tilt_deg, spine_lean)

    return tilt_deg


def _head_yaw_deg_from_face(face_landmarks):
    if not face_landmarks or len(face_landmarks) < 455:
        return 0.0
    nose = face_landmarks[NOSE_TIP]
    lc = face_landmarks[LEFT_CHEEK]
    rc = face_landmarks[RIGHT_CHEEK]

    face_mid_x = (lc["x"] + rc["x"]) / 2.0
    face_width = abs(rc["x"] - lc["x"])
    if face_width < 1e-4:
        return 0.0

    offset = (nose["x"] - face_mid_x) / (face_width / 2.0)
    
    if face_width < 0.05:
        offset = 1.0 if nose["x"] > face_mid_x else -1.0
        return offset * 80.0

    offset = max(-1.0, min(1.0, offset))
    return offset * 70.0


def _head_pitch_deg(face_landmarks, pose_landmarks):
    if face_landmarks and len(face_landmarks) > 152:
        forehead = face_landmarks[FOREHEAD]
        nose = face_landmarks[NOSE_TIP]
        chin = face_landmarks[CHIN]

        face_h = abs(chin["y"] - forehead["y"])
        if face_h > 1e-4:
            forehead_to_nose = nose["y"] - forehead["y"]
            ratio = forehead_to_nose / face_h
            pitch_deg = max(0.0, (ratio - 0.45) * 160.0)
            return pitch_deg

    if pose_landmarks and len(pose_landmarks) > 8:
        nose = pose_landmarks[NOSE]
        lear = pose_landmarks[LEFT_EAR]
        rear = pose_landmarks[RIGHT_EAR]

        ear_y = (lear["y"] + rear["y"]) / 2.0
        ear_vis = min(lear.get("visibility", 1.0), rear.get("visibility", 1.0))
        if ear_vis >= 0.3:
            diff_y = nose["y"] - ear_y
            if diff_y > 0.04:
                return diff_y * 300.0

    return 0.0


def _mouth_openness_ratio(face_landmarks):
    if not face_landmarks or len(face_landmarks) < 153:
        return 0.0

    upper_lip = face_landmarks[UPPER_LIP]
    lower_lip = face_landmarks[LOWER_LIP]
    forehead = face_landmarks[FOREHEAD]
    chin = face_landmarks[CHIN]

    face_h = abs(chin["y"] - forehead["y"])
    if face_h < 1e-4:
        return 0.0

    lip_dist = abs(lower_lip["y"] - upper_lip["y"])
    return lip_dist / face_h


def _classify_hand_positions(pose_landmarks):
    """
    Classifies student hand positions:
    - hands_under_desk: Wrists in lap area below desk level (y > 0.62).
    - hands_in_writing_pos: Wrists positioned normally on top of desk writing surface.
    - hands_moved_away: Wrists reached sideways outside desk writing bounds (reaching to grab/take something).
    """
    if not pose_landmarks or len(pose_landmarks) < 17:
        return False, True, False

    lw = pose_landmarks[LEFT_WRIST]
    rw = pose_landmarks[RIGHT_WRIST]

    lw_vis = lw.get("visibility", 1.0)
    rw_vis = rw.get("visibility", 1.0)

    # 1. Hands under desk (lap region)
    left_under = (lw_vis >= 0.3 and lw["y"] > 0.62)
    right_under = (rw_vis >= 0.3 and rw["y"] > 0.62)
    
    left_deep = (lw_vis >= 0.3 and lw["y"] > 0.70)
    right_deep = (rw_vis >= 0.3 and rw["y"] > 0.70)

    if len(pose_landmarks) >= 15:
        le = pose_landmarks[LEFT_ELBOW]
        re = pose_landmarks[RIGHT_ELBOW]
        if lw_vis >= 0.3 and le.get("visibility", 1.0) >= 0.3 and (lw["y"] - le["y"]) > 0.12:
            left_under = True
        if rw_vis >= 0.3 and re.get("visibility", 1.0) >= 0.3 and (rw["y"] - re["y"]) > 0.12:
            right_under = True

    both_under_or_hidden = (left_under or lw_vis < 0.3) and (right_under or rw_vis < 0.3)
    hands_under = both_under_or_hidden or left_deep or right_deep

    # 2. Hands in normal writing position on top of desk
    left_writing = (lw_vis >= 0.3 and 0.38 <= lw["y"] <= 0.62 and 0.22 <= lw["x"] <= 0.78)
    right_writing = (rw_vis >= 0.3 and 0.38 <= rw["y"] <= 0.62 and 0.22 <= rw["x"] <= 0.78)
    hands_writing = left_writing or right_writing

    # 3. Hands moved away / reaching sideways (reaching for something off the desk)
    left_reaching = (lw_vis >= 0.3 and (lw["x"] < 0.20 or lw["x"] > 0.80 or lw["y"] < 0.30))
    right_reaching = (rw_vis >= 0.3 and (rw["x"] < 0.20 or rw["x"] > 0.80 or rw["y"] < 0.30))
    hands_moved_away = left_reaching or right_reaching

    return hands_under, hands_writing, hands_moved_away


def _is_standing(pose_landmarks, bbox):
    """
    Returns True if person is standing (examiner posture) rather than sitting (student).
    Two checks:
    1. Lower-body visible: knee or ankle landmarks have high visibility.
       Seated students' legs/knees are hidden behind the desk.
    2. Bounding box is taller than wide — a full standing body occupies
       a much taller region than a seated upper-body crop.
    """
    lower_body_visible = False
    if pose_landmarks and len(pose_landmarks) > 28:
        knee_vis = max(
            pose_landmarks[LEFT_KNEE].get("visibility", 0.0),
            pose_landmarks[RIGHT_KNEE].get("visibility", 0.0)
        )
        ankle_vis = max(
            pose_landmarks[LEFT_ANKLE].get("visibility", 0.0),
            pose_landmarks[RIGHT_ANKLE].get("visibility", 0.0)
        )
        if knee_vis >= 0.50 or ankle_vis >= 0.40:
            lower_body_visible = True

    tall_bbox = False
    if bbox and isinstance(bbox, dict):
        bw = bbox.get("x_max", 0) - bbox.get("x_min", 0)
        bh = bbox.get("y_max", 0) - bbox.get("y_min", 0)
        if bw > 1e-4 and (bh / bw) > 1.4:
            tall_bbox = True

    return lower_body_visible and tall_bbox


def adapt(structured_output, full_w=1280, full_h=720):
    poses = structured_output.get("poses", [])
    objects = structured_output.get("objects", [])
    records = []

    has_detected_phone = len(objects) > 0

    for pose in poses:
        bbox = pose.get("bbox")
        raw_pose_lm = pose.get("landmarks", [])
        raw_face_lm = pose.get("face_landmarks", [])
        crop_offset = pose.get("crop_offset", (0, 0))
        crop_size = pose.get("crop_size", (1, 1))

        global_pose_lm = [
            _crop_lm_to_global(lm, crop_offset, crop_size, full_w, full_h)
            for lm in raw_pose_lm
        ]

        global_face_lm = [
            _crop_lm_to_global(lm, crop_offset, crop_size, full_w, full_h)
            for lm in raw_face_lm
        ]

        if global_pose_lm:
            centroid = _hip_midpoint(global_pose_lm)
        else:
            centroid = _bbox_center(bbox)

        if centroid is None:
            continue

        head_yaw = _head_yaw_deg_from_face(global_face_lm) if global_face_lm else 0.0
        head_pitch = _head_pitch_deg(global_face_lm, global_pose_lm)
        mouth_ratio = _mouth_openness_ratio(global_face_lm) if global_face_lm else 0.0
        torso_lean = _shoulder_tilt_and_lean(raw_pose_lm) if raw_pose_lm else 0.0

        hands_under, hands_writing, hands_moved_away = _classify_hand_positions(raw_pose_lm) if raw_pose_lm else (False, True, False)

        standing = _is_standing(raw_pose_lm, bbox)

        records.append({
            "centroid": centroid,
            "head_yaw_deg": head_yaw,
            "head_pitch_deg": head_pitch,
            "mouth_open_ratio": mouth_ratio,
            "torso_lean_deg": torso_lean,
            "hands_under_desk": hands_under,
            "hands_in_writing_pos": hands_writing,
            "hands_moved_away": hands_moved_away,
            "object_near_hand": has_detected_phone,
            "is_standing": standing,
            "pose_id": pose.get("pose_id"),
            "bbox": bbox,
        })

    return records
