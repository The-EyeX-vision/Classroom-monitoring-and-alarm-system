import cv2
import os
import mediapipe as mp
from ultralytics import YOLO

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DEFAULT_YOLO_PATH = os.path.join(BASE_DIR, "models", "yolov8n.pt")
DEFAULT_POSE_PATH = os.path.join(BASE_DIR, "models", "pose_landmarker_heavy.task")
DEFAULT_FACE_PATH = os.path.join(BASE_DIR, "models", "face_landmarker.task")


class ClassroomDetectionPipeline:
    """
    Combined YOLOv8 + MediaPipe Pose & Face Landmarker Pipeline for classroom monitoring.
    Detects students and cell phones in frame crops, running landmark extraction for head pose & hand proximity.
    """

    def __init__(self,
                 yolo_model_path=None,
                 pose_model_path=None,
                 face_model_path=None,
                 confidence_threshold=0.25,
                 crop_phone_conf=0.45,  # Higher threshold filters bags, calculators, desk items misclassified as phones
                 crop_padding=0.05,
                 yolo_input_width=1280,
                 crop_size_px=256,
                 landmark_every_n_frames=1):

        self.yolo_model_path = yolo_model_path or DEFAULT_YOLO_PATH
        self.pose_model_path = pose_model_path or DEFAULT_POSE_PATH
        self.face_model_path = face_model_path or DEFAULT_FACE_PATH

        self.confidence_threshold = confidence_threshold
        self.crop_phone_conf = crop_phone_conf
        self.crop_padding = crop_padding
        self.yolo_input_width = yolo_input_width
        self.crop_size_px = crop_size_px
        self.landmark_every_n_frames = landmark_every_n_frames

        self.yolo = YOLO(self.yolo_model_path)

        BaseOptions = mp.tasks.BaseOptions
        PoseLandmarker = mp.tasks.vision.PoseLandmarker
        PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
        FaceLandmarker = mp.tasks.vision.FaceLandmarker
        FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
        RunningMode = mp.tasks.vision.RunningMode

        pose_options = PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=self.pose_model_path),
            running_mode=RunningMode.VIDEO,
            num_poses=1,
            min_pose_detection_confidence=0.5,
            min_pose_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self.pose_detector = PoseLandmarker.create_from_options(pose_options)

        face_options = FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=self.face_model_path),
            running_mode=RunningMode.VIDEO,
            num_faces=1,
            output_face_blendshapes=False,
            min_face_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self.face_detector = FaceLandmarker.create_from_options(face_options)

        self._frame_count = 0

    def _crop_with_padding(self, frame, x1, y1, x2, y2):
        h, w = frame.shape[:2]
        pw = int((x2 - x1) * self.crop_padding)
        ph = int((y2 - y1) * self.crop_padding)
        cx1 = max(x1 - pw, 0)
        cy1 = max(y1 - ph, 0)
        cx2 = min(x2 + pw, w)
        cy2 = min(y2 + ph, h)
        if cx2 <= cx1 or cy2 <= cy1:
            return None, (0, 0)
        return frame[cy1:cy2, cx1:cx2], (cx1, cy1)

    def _is_phone_held_by_hand(self, phone_px_box, pose_landmarks, crop_w, crop_h):
        """
        Returns True ONLY if a detected object is actively held by a student's hand.
        Three strict guards:
        1. At least one visible wrist/hand landmark must exist.
        2. The object box must be in the lower half of the crop (mid-body and below),
           rejecting background objects like bags on hooks or objects on distant walls.
        3. A wrist/hand landmark must be within a tight proximity of the object box
           (not just loosely nearby).
        """
        if not pose_landmarks or len(pose_landmarks) < 23:
            return False

        px1, py1, px2, py2 = phone_px_box
        obj_cy = (py1 + py2) / 2.0

        # Guard 2: Reject if the object centre is in the top 40% of the crop
        # (background bags, boards, hooks are almost always in the upper region)
        if obj_cy < crop_h * 0.40:
            return False

        # Wrist / hand landmark indices
        hand_indices = [15, 16, 17, 18, 19, 20, 21, 22]
        margin = 15.0  # tight proximity margin in pixels

        visible_hands = 0
        for idx in hand_indices:
            lm = pose_landmarks[idx]
            if lm.get("visibility", 1.0) < 0.35:
                continue
            visible_hands += 1

            hx = lm["x"] * crop_w
            hy = lm["y"] * crop_h

            if (px1 - margin) <= hx <= (px2 + margin) and (py1 - margin) <= hy <= (py2 + margin):
                return True

        # Guard 1: If no hand landmarks were visible at all, we cannot confirm the
        # object is being held — reject to avoid false positives.
        if visible_hands == 0:
            return False

        return False

    def _detect_phone_in_crop(self, crop, cx1, cy1, frame_w, frame_h, pose_landmarks):
        crop_h, crop_w = crop.shape[:2]
        if crop_h < 30 or crop_w < 30:
            return []

        crop_results = self.yolo(crop, conf=self.crop_phone_conf, classes=[67], verbose=False)[0]

        detected_phones = []
        for box in crop_results.boxes:
            conf = float(box.conf[0])
            px1, py1, px2, py2 = box.xyxy[0].tolist()

            if not self._is_phone_held_by_hand((px1, py1, px2, py2), pose_landmarks, crop_w, crop_h):
                continue

            gx1 = cx1 + px1
            gy1 = cy1 + py1
            gx2 = cx1 + px2
            gy2 = cy1 + py2

            detected_phones.append({
                "x_min": gx1 / frame_w,
                "y_min": gy1 / frame_h,
                "x_max": gx2 / frame_w,
                "y_max": gy2 / frame_h,
                "confidence": conf
            })

        return detected_phones

    def _run_landmarks(self, crop, crop_ts):
        h, w = crop.shape[:2]
        if w > self.crop_size_px:
            scale = self.crop_size_px / w
            crop = cv2.resize(crop, (self.crop_size_px, int(h * scale)), interpolation=cv2.INTER_AREA)

        rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        pose_out = []
        face_out = []

        try:
            pr = self.pose_detector.detect_for_video(mp_image, crop_ts)
            if pr and pr.pose_landmarks:
                for lm in pr.pose_landmarks[0]:
                    pose_out.append({
                        "x": lm.x, "y": lm.y, "z": lm.z,
                        "visibility": getattr(lm, "visibility", 1.0),
                    })
        except Exception:
            pass

        try:
            fr = self.face_detector.detect_for_video(mp_image, crop_ts)
            if fr and fr.face_landmarks:
                for lm in fr.face_landmarks[0]:
                    face_out.append({"x": lm.x, "y": lm.y, "z": lm.z})
        except Exception:
            pass

        return pose_out, face_out

    def process_frame(self, frame_bgr, timestamp_ms):
        self._frame_count += 1
        do_landmarks = (self._frame_count % self.landmark_every_n_frames == 0)

        h_full, w_full = frame_bgr.shape[:2]

        if w_full > self.yolo_input_width:
            scale = self.yolo_input_width / w_full
            yolo_input = cv2.resize(
                frame_bgr,
                (self.yolo_input_width, int(h_full * scale)),
                interpolation=cv2.INTER_AREA,
            )
        else:
            yolo_input = frame_bgr

        results = self.yolo(yolo_input, conf=self.confidence_threshold, classes=[0, 67], verbose=False)[0]
        yolo_h, yolo_w = yolo_input.shape[:2]

        structured = {
            "timestamp_ms": timestamp_ms,
            "poses": [],
            "faces": [],
            "objects": [],
        }

        person_idx = 0
        object_idx = 0

        for box in results.boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            x1, y1, x2, y2 = box.xyxy[0].tolist()

            x1 = x1 * w_full / yolo_w
            x2 = x2 * w_full / yolo_w
            y1 = y1 * h_full / yolo_h
            y2 = y2 * h_full / yolo_h

            bbox_norm = {
                "x_min": x1 / w_full,
                "y_min": y1 / h_full,
                "x_max": x2 / w_full,
                "y_max": y2 / h_full,
                "confidence": conf,
            }

            if cls_id == 0:
                crop, (cx1, cy1) = self._crop_with_padding(frame_bgr, int(x1), int(y1), int(x2), int(y2))
                if crop is None or crop.size == 0:
                    continue

                crop_h, crop_w = crop.shape[:2]
                crop_ts = timestamp_ms + person_idx

                if do_landmarks:
                    pose_lm, face_lm = self._run_landmarks(crop, crop_ts)
                else:
                    pose_lm, face_lm = [], []

                crop_phones = self._detect_phone_in_crop(crop, cx1, cy1, w_full, h_full, pose_lm)
                for phone_bbox in crop_phones:
                    structured["objects"].append({
                        "object_id": object_idx,
                        "type": "cell phone",
                        "bbox": phone_bbox
                    })
                    object_idx += 1

                structured["poses"].append({
                    "pose_id": person_idx,
                    "bbox": bbox_norm,
                    "landmarks": pose_lm,
                    "face_landmarks": face_lm,
                    "crop_offset": (cx1, cy1),
                    "crop_size": (crop_w, crop_h),
                })

                if face_lm:
                    structured["faces"].append({
                        "face_id": person_idx,
                        "landmarks": face_lm,
                        "blendshapes": {},
                    })

                person_idx += 1

        return structured

    def close(self):
        if hasattr(self, 'pose_detector') and self.pose_detector:
            self.pose_detector.close()
        if hasattr(self, 'face_detector') and self.face_detector:
            self.face_detector.close()
