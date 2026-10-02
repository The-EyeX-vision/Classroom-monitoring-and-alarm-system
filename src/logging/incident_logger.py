import os
import cv2
import csv
import re
from datetime import datetime
from src.db.supabase_client import SupabaseManager


class IncidentLogger:
    """
    Logs flagged exam cheating incidents to CSV and Supabase Database/Storage bucket.
    Features:
    1. Likelihood & Probability Logging: Stores likelihood tier statements, percentage scores, and activities.
    2. Category Action Counters: Tracks occurrence count per student per activity category.
    3. Single Initial Snapshot & Best Frame Replacement: Saves initial snapshot frame once, overwriting IF a higher score occurs.
    4. Supabase DB & Storage Bucket integration: Uploads evidence frames and writes violation logs to database.
    """
    def __init__(self, csv_filepath="classroom_alerts.csv", output_dir="flagged_incidents", cooldown_sec=5.0, session_id=None):
        self.csv_filepath = csv_filepath
        self.output_dir = output_dir
        self.cooldown_sec = cooldown_sec
        self.session_id = session_id or os.getenv("EXAM_SESSION_ID")

        self.last_flagged_time = {}   # student_id -> float (timestamp)
        self.student_categories = {}  # student_id -> { category_name -> { "count": int, "best_score": float, "snapshot_path": str } }

        # Initialize Supabase manager (handles fallback to offline CSV if env variables not set)
        self.supabase = SupabaseManager()

        os.makedirs(self.output_dir, exist_ok=True)


        if not os.path.exists(self.csv_filepath):
            with open(self.csv_filepath, mode="w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "timestamp",
                    "student_id",
                    "likelihood_statement",
                    "probability_percentage",
                    "category_name",
                    "activity_count",
                    "suspicion_score",
                    "is_best_frame_replacement",
                    "head_pitch_deg",
                    "head_yaw_deg",
                    "mouth_open_ratio",
                    "torso_lean_deg",
                    "hands_under_desk",
                    "snapshot_path"
                ])

    def _slugify(self, text):
        return re.sub(r'[^a-zA-Z0-9]+', '_', text.strip()).strip('_').lower()

    def can_flag_student(self, student_id):
        now_ts = datetime.now().timestamp()
        if student_id in self.last_flagged_time:
            if now_ts - self.last_flagged_time[student_id] < self.cooldown_sec:
                return False
        return True

    def log_and_save_snapshot(self, frame, student_id, score, cheat_reason, pixel_bbox, metrics=None):
        now = datetime.now()
        now_ts = now.timestamp()

        # Enforce 5-second tracking pause per student
        if student_id in self.last_flagged_time:
            if now_ts - self.last_flagged_time[student_id] < self.cooldown_sec:
                return False, None

        self.last_flagged_time[student_id] = now_ts
        str_time = now.strftime("%Y-%m-%d %H:%M:%S")

        metrics = metrics or {}
        pitch = metrics.get("head_pitch_deg")
        yaw = metrics.get("head_yaw_deg")
        mouth = metrics.get("mouth_open_ratio")
        lean = metrics.get("torso_lean_deg")
        hands_under = metrics.get("hands_under_desk", False)

        prob_pct = int(round(score * 100))

        # Extract activity name from likelihood statement (e.g. "[High Certainty - 88%] Using Phone..." -> "Using Phone...")
        if "]" in cheat_reason:
            category_name = cheat_reason.split("]", 1)[1].strip().split(" | ")[0]
        else:
            category_name = cheat_reason.split(" | ")[0] if cheat_reason else "General Suspicion"

        cat_slug = self._slugify(category_name)

        if student_id not in self.student_categories:
            self.student_categories[student_id] = {}

        student_cats = self.student_categories[student_id]
        is_first_time = category_name not in student_cats

        if is_first_time:
            filename = f"student_{student_id}_{cat_slug}.jpg"
            snapshot_path = os.path.join(self.output_dir, filename)
            student_cats[category_name] = {
                "count": 1,
                "best_score": float(score),
                "snapshot_path": snapshot_path
            }
            should_save_image = True
            is_replacement = False
        else:
            student_cats[category_name]["count"] += 1
            prev_best = student_cats[category_name]["best_score"]
            snapshot_path = student_cats[category_name]["snapshot_path"]

            if float(score) > prev_best + 0.04:
                student_cats[category_name]["best_score"] = float(score)
                should_save_image = True
                is_replacement = True
            else:
                should_save_image = False
                is_replacement = False

        cat_count = student_cats[category_name]["count"]

        if should_save_image:
            annotated_frame = frame.copy()
            x1, y1, x2, y2 = pixel_bbox

            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 0, 255), 3)

            badge_text = f"FLAGGED Student #{student_id} | {cheat_reason} | Count: {cat_count}"
            cv2.rectangle(annotated_frame, (x1, max(0, y1 - 32)), (x1 + 650, max(32, y1)), (0, 0, 255), -1)
            cv2.putText(annotated_frame, badge_text, (x1 + 5, max(22, y1 - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 2)

            cv2.putText(annotated_frame, f"Time: {str_time}", (20, annotated_frame.shape[0] - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            cv2.imwrite(snapshot_path, annotated_frame)
            action_type = "REPLACED with higher score frame" if is_replacement else "INITIAL snapshot saved"
            print(f"[INCIDENT LOGGED] Student #{student_id} | {cheat_reason} | Activity Count: {cat_count} | {action_type}: {snapshot_path}")

        # Upload image snapshot to Supabase Storage & insert record into PostgreSQL database
        evidence_url = None
        if self.supabase.enabled and os.path.exists(snapshot_path):
            remote_filename = f"student_{student_id}_{cat_slug}.jpg"
            evidence_url = self.supabase.upload_evidence_image(
                local_filepath=snapshot_path,
                remote_path=f"incidents/{remote_filename}"
            )
            self.supabase.insert_violation(
                student_id=student_id,
                score=score,
                cheat_reason=cheat_reason,
                category_name=category_name,
                cat_count=cat_count,
                evidence_url=evidence_url,
                session_id=self.session_id,
                metrics=metrics
            )

        with open(self.csv_filepath, mode="a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                str_time,
                student_id,
                cheat_reason,
                f"{prob_pct}%",
                category_name,
                cat_count,
                round(float(score), 3),
                is_replacement,
                round(float(pitch), 1) if pitch is not None else "N/A",
                round(float(yaw), 1) if yaw is not None else "N/A",
                round(float(mouth), 3) if mouth is not None else "N/A",
                round(float(lean), 1) if lean is not None else "N/A",
                bool(hands_under),
                evidence_url or snapshot_path
            ])

        return True, snapshot_path

