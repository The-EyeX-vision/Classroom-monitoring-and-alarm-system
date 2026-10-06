import cv2
import os
from ultralytics import YOLO

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DEFAULT_POSE_MODEL_PATH = os.path.join(BASE_DIR, "models", "yolov8s-pose.pt")


class ClassroomDetectionPipeline:
    """
    Single-model YOLOv8-Pose pipeline for classroom monitoring.

    One forward pass per frame simultaneously:
    - Detects every person bounding box (near & far)
    - Extracts 17 COCO keypoints per person for pose/behaviour heuristics
    - Detects cell phones (class 67) at the whole-frame level

    COCO Keypoint indices (17 total):
      0:nose  1:left_eye  2:right_eye  3:left_ear  4:right_ear
      5:left_shoulder  6:right_shoulder  7:left_elbow  8:right_elbow
      9:left_wrist  10:right_wrist  11:left_hip  12:right_hip
      13:left_knee  14:right_knee  15:left_ankle  16:right_ankle
    """

    def __init__(self,
                 pose_model_path=None,
                 confidence_threshold=0.10,
                 phone_conf_threshold=0.45,
                 yolo_input_width=1280,
                 max_det=50):

        self.pose_model_path = pose_model_path or DEFAULT_POSE_MODEL_PATH
        self.confidence_threshold = confidence_threshold
        self.phone_conf_threshold = phone_conf_threshold
        self.yolo_input_width = yolo_input_width
        self.max_det = max_det

        # Single model: pose estimation (person detection + 17 keypoints)
        self.pose_model = YOLO(self.pose_model_path)

        # Separate tiny YOLO for phone detection at whole-frame level
        phone_model_path = os.path.join(BASE_DIR, "models", "yolov8n.pt")
        self.phone_model = YOLO(phone_model_path)

        self._frame_count = 0

    def _scale_frame(self, frame_bgr):
        """Scales frame to yolo_input_width, upscaling if needed (CUBIC) or downscaling (AREA)."""
        h, w = frame_bgr.shape[:2]
        if w == self.yolo_input_width:
            return frame_bgr, 1.0, 1.0
        scale_x = float(self.yolo_input_width) / float(w)
        scale_y = scale_x
        target_h = int(h * scale_y)
        interp = cv2.INTER_CUBIC if scale_x > 1.0 else cv2.INTER_AREA
        scaled = cv2.resize(frame_bgr, (self.yolo_input_width, target_h), interpolation=interp)
        return scaled, scale_x, scale_y

    def process_frame(self, frame_bgr, timestamp_ms):
        """
        Runs inference on a single frame.

        Returns a structured dict compatible with the existing adapter/tracker layer:
        {
            "timestamp_ms": int,
            "poses": [
                {
                    "pose_id": int,
                    "bbox": {"x_min":f, "y_min":f, "x_max":f, "y_max":f, "confidence":f},
                    "keypoints": [  # 17 COCO keypoints in GLOBAL normalized coords
                        {"x":f, "y":f, "confidence":f}, ...
                    ],
                }
            ],
            "objects": [
                {"object_id": int, "type": "cell phone",
                 "bbox": {"x_min":f, "y_min":f, "x_max":f, "y_max":f, "confidence":f}}
            ],
        }
        """
        self._frame_count += 1
        h_full, w_full = frame_bgr.shape[:2]

        # --- Scale frame to 1280px width for consistent detection of near & far students ---
        scaled_frame, sx, sy = self._scale_frame(frame_bgr)
        yolo_h, yolo_w = scaled_frame.shape[:2]

        structured = {
            "timestamp_ms": timestamp_ms,
            "poses": [],
            "objects": [],
        }

        # ── Pose estimation (person detection + keypoints) ────────────────────────────
        pose_results = self.pose_model(
            scaled_frame,
            conf=self.confidence_threshold,
            imgsz=self.yolo_input_width,
            max_det=self.max_det,
            verbose=False
        )[0]

        for idx, box in enumerate(pose_results.boxes):
            cls_id = int(box.cls[0])
            if cls_id != 0:   # class 0 = person
                continue

            conf = float(box.conf[0])
            x1s, y1s, x2s, y2s = box.xyxy[0].tolist()

            # Convert from scaled-frame coords → original frame normalized coords
            x1 = x1s / yolo_w
            y1 = y1s / yolo_h
            x2 = x2s / yolo_w
            y2 = y2s / yolo_h

            bbox_norm = {
                "x_min": x1,
                "y_min": y1,
                "x_max": x2,
                "y_max": y2,
                "confidence": conf,
            }

            # Extract 17 COCO keypoints (already normalized to scaled frame)
            keypoints = []
            if pose_results.keypoints is not None and idx < len(pose_results.keypoints):
                kp_data = pose_results.keypoints[idx]     # shape (17, 3): x, y, conf
                kp_xy = kp_data.xy[0].tolist()    # pixel coords in scaled frame
                kp_conf = kp_data.conf[0].tolist() if kp_data.conf is not None else [1.0] * 17

                for kp_idx, (kp_x, kp_y) in enumerate(kp_xy):
                    kp_c = float(kp_conf[kp_idx]) if kp_idx < len(kp_conf) else 1.0
                    keypoints.append({
                        "x": kp_x / yolo_w,   # normalize to 0-1
                        "y": kp_y / yolo_h,
                        "confidence": kp_c,
                    })

            structured["poses"].append({
                "pose_id": idx,
                "bbox": bbox_norm,
                "keypoints": keypoints,
            })

        # ── Phone detection (whole frame) ─────────────────────────────────────────────
        phone_results = self.phone_model(
            scaled_frame,
            conf=self.phone_conf_threshold,
            classes=[67],       # class 67 = cell phone
            imgsz=self.yolo_input_width,
            max_det=10,
            verbose=False
        )[0]

        for obj_idx, box in enumerate(phone_results.boxes):
            conf = float(box.conf[0])
            px1s, py1s, px2s, py2s = box.xyxy[0].tolist()
            structured["objects"].append({
                "object_id": obj_idx,
                "type": "cell phone",
                "bbox": {
                    "x_min": px1s / yolo_w,
                    "y_min": py1s / yolo_h,
                    "x_max": px2s / yolo_w,
                    "y_max": py2s / yolo_h,
                    "confidence": conf,
                }
            })

        return structured

    def close(self):
        """No persistent resources to release for YOLO models."""
        pass
