#!/usr/bin/env python3
"""
Smart Exam Hall Cheating Detection System
Main entry point for real-time exam monitoring, behavior analysis, and incident logging.
"""

import os
import sys
import time
import argparse
import cv2

# Add project root to sys.path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.capture import VideoStream
from src.detection import ClassroomDetectionPipeline
from src.tracking import adapt, CentroidTracker, StudentSuspicionTracker
from src.logging import IncidentLogger
from src.ui import draw_flag_overlay, apply_spotlight, draw_dashboard


try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


def parse_args():
    parser = argparse.ArgumentParser(
        description="Smart Exam Hall Cheating Detection System"
    )
    parser.add_argument(
        "--source", type=str, default="test_video/sample-two.mp4",
        help="Input video source: file path, camera index (e.g. 0), or IP URL."
    )
    parser.add_argument(
        "--yolo-model", type=str, default=os.path.join(PROJECT_ROOT, "models", "yolov8n.pt"),
        help="Path to YOLOv8 model file."
    )
    parser.add_argument(
        "--pose-model", type=str, default=os.path.join(PROJECT_ROOT, "models", "pose_landmarker_lite.task"),
        help="Path to MediaPipe Pose Landmarker task file."
    )
    parser.add_argument(
        "--face-model", type=str, default=os.path.join(PROJECT_ROOT, "models", "face_landmarker.task"),
        help="Path to MediaPipe Face Landmarker task file."
    )
    parser.add_argument(
        "--output-csv", type=str, default="classroom_alerts.csv",
        help="Path to output CSV alert log file."
    )
    parser.add_argument(
        "--snapshot-dir", type=str, default="flagged_incidents",
        help="Directory to save annotated frame snapshots when cheating is flagged."
    )
    parser.add_argument(
        "--session-id", type=str, default=None,
        help="Optional Exam Session UUID from exam_sessions / exam_hall_sessions database table."
    )
    parser.add_argument(
        "--cooldown-sec", type=float, default=5.0,
        help="Pause duration in seconds for student tracking alerts after flagging an incident."
    )
    parser.add_argument(
        "--no-display", action="store_true",
        help="Run without displaying OpenCV window (headless mode)."
    )
    return parser.parse_args()



def main():
    args = parse_args()

    video_source = args.source
    if not video_source.isdigit() and not video_source.startswith("http") and not os.path.exists(video_source):
        fallback = os.path.join(PROJECT_ROOT, "test_video", os.path.basename(video_source))
        if os.path.exists(fallback):
            video_source = fallback

    print(f"[SYSTEM] Initializing stream source: {video_source}")
    stream = VideoStream(source=video_source)

    pipeline = ClassroomDetectionPipeline(
        yolo_model_path=args.yolo_model,
        pose_model_path=args.pose_model,
        face_model_path=args.face_model
    )

    centroid_tracker = CentroidTracker(max_distance=0.15, max_missed_frames=20)
    suspicion_tracker = StudentSuspicionTracker(threshold=0.50, required_frames=6)
    logger = IncidentLogger(
        csv_filepath=args.output_csv,
        output_dir=args.snapshot_dir,
        cooldown_sec=args.cooldown_sec,
        session_id=args.session_id
    )


    total_frames = 0
    start_time = time.time()
    declared_fps = stream.get_declared_fps()
    frame_duration_ms = int(1000 / declared_fps)

    print("[SYSTEM] Exam Hall Monitoring Active. Press 'q' in video window to exit.\n")

    try:
        while True:
            ret, frame = stream.read()
            if not ret or frame is None:
                print("\n[INFO] End of stream or video source closed.")
                break

            total_frames += 1
            timestamp_ms = int(total_frames * frame_duration_ms)
            h_full, w_full = frame.shape[:2]

            # 1. Run detection pipeline (YOLOv8 + MediaPipe Pose & Face)
            raw_detections = pipeline.process_frame(frame, timestamp_ms)

            # 2. Adapt landmarks & update centroid tracking
            records = adapt(raw_detections, full_w=w_full, full_h=h_full)
            centroids = [r["centroid"] for r in records if "centroid" in r]
            assignments = centroid_tracker.update(centroids)

            flagged_bboxes = []
            has_incident_this_frame = False

            # 3. Evaluate student behavior vectors
            for det_idx, student_id in assignments.items():
                if det_idx >= len(records):
                    continue

                rec = records[det_idx]
                pitch = rec.get("head_pitch_deg", 0.0)
                yaw = rec.get("head_yaw_deg", 0.0)
                mouth = rec.get("mouth_open_ratio", 0.0)
                lean = rec.get("torso_lean_deg", 0.0)
                hands_under = rec.get("hands_under_desk", False)
                hands_writing = rec.get("hands_in_writing_pos", True)
                hands_moved = rec.get("hands_moved_away", False)
                obj_near = rec.get("object_near_hand", False)
                is_standing = rec.get("is_standing", False)

                # -- Examiner Exclusion -------------------------------------------
                # A person who is BOTH standing AND moving across the room is the
                # examiner/invigilator. Skip all scoring and flagging for them.
                is_examiner = is_standing and centroid_tracker.is_mobile(student_id)

                raw_bbox = rec.get("bbox")
                if isinstance(raw_bbox, dict):
                    pixel_bbox = [
                        int(raw_bbox["x_min"] * w_full),
                        int(raw_bbox["y_min"] * h_full),
                        int(raw_bbox["x_max"] * w_full),
                        int(raw_bbox["y_max"] * h_full)
                    ]
                else:
                    cx, cy = rec["centroid"]
                    pixel_bbox = [
                        int((cx - 0.08) * w_full),
                        int((cy - 0.15) * h_full),
                        int((cx + 0.08) * w_full),
                        int((cy + 0.15) * h_full)
                    ]

                if is_examiner:
                    continue

                score, should_alert, cheat_reason = suspicion_tracker.update(
                    student_id=student_id,
                    head_pitch_deg=pitch,
                    head_yaw_deg=yaw,
                    mouth_open_ratio=mouth,
                    torso_lean_deg=lean,
                    hands_under_desk=hands_under,
                    hands_in_writing_pos=hands_writing,
                    hands_moved_away=hands_moved,
                    object_near_hand=obj_near
                )

                if should_alert:
                    has_incident_this_frame = True
                    flagged_bboxes.append(pixel_bbox)

                    logged, snapshot_path = logger.log_and_save_snapshot(
                        frame=frame,
                        student_id=student_id,
                        score=score,
                        cheat_reason=cheat_reason,
                        pixel_bbox=pixel_bbox,
                        metrics=rec
                    )

                draw_flag_overlay(
                    frame=frame,
                    bbox=pixel_bbox,
                    student_id=student_id,
                    score=score,
                    is_flagged=should_alert,
                    cheat_reason=cheat_reason
                )

            # 4. Draw HUD dashboard & spotlight
            elapsed = time.time() - start_time
            fps = total_frames / max(elapsed, 1e-6)

            if flagged_bboxes:
                frame = apply_spotlight(frame, flagged_bboxes, dim_factor=0.3)

            draw_dashboard(frame, fps, len(assignments), has_incident_this_frame)

            if not args.no_display:
                cv2.imshow("Exam Hall Monitoring System", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

    finally:
        stream.release()
        pipeline.close()
        if not args.no_display:
            cv2.destroyAllWindows()

        elapsed_total = time.time() - start_time
        print(f"\n[SUMMARY] Processed {total_frames} frames in {elapsed_total:.1f}s ({total_frames / max(elapsed_total, 1e-6):.1f} FPS)")


if __name__ == "__main__":
    main()