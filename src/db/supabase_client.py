import os
import sys
import logging
from typing import Optional, Dict, Any

logger = logging.getLogger("SupabaseManager")

try:
    from supabase import create_client, Client
    HAS_SUPABASE = True
except ImportError:
    HAS_SUPABASE = False
    Client = None


class SupabaseManager:
    """
    Manages Supabase Database connection and Storage uploads for Classroom Monitoring.
    """
    def __init__(
        self,
        url: Optional[str] = None,
        key: Optional[str] = None,
        bucket_name: Optional[str] = None
    ):
        self.url = url or os.getenv("SUPABASE_URL")
        self.key = key or os.getenv("SUPABASE_KEY") or os.getenv("SUPABASE_SERVICE_ROLE_KEY")
        self.bucket_name = bucket_name or os.getenv("SUPABASE_BUCKET", "evidence")
        self.client: Optional[Client] = None
        self.enabled = False

        if not HAS_SUPABASE:
            print("[SUPABASE] Notice: 'supabase' package is not installed. Database and bucket upload disabled. (Run: pip install supabase)")
            return

        if not self.url or not self.key or "your-supabase" in self.url:
            print("[SUPABASE] Notice: SUPABASE_URL or SUPABASE_KEY not set. Operating in offline/local CSV logging mode.")
            return

        try:
            self.client = create_client(self.url, self.key)
            self.enabled = True
            print(f"[SUPABASE] Successfully initialized Supabase client linked to bucket '{self.bucket_name}'.")
        except Exception as e:
            print(f"[SUPABASE ERROR] Failed to initialize Supabase client: {e}")
            self.enabled = False

    def upload_evidence_image(self, local_filepath: str, remote_path: Optional[str] = None) -> Optional[str]:
        """
        Uploads a local frame snapshot to the Supabase Storage bucket.
        Returns the public URL of the uploaded image if successful.
        """
        if not self.enabled or not self.client:
            return None

        if not os.path.exists(local_filepath):
            print(f"[SUPABASE WARN] File does not exist for upload: {local_filepath}")
            return None

        filename = os.path.basename(local_filepath)
        dest_path = remote_path or f"incidents/{filename}"

        try:
            with open(local_filepath, "rb") as f:
                file_data = f.read()

            # Upload or overwrite (upsert=true)
            res = self.client.storage.from_(self.bucket_name).upload(
                path=dest_path,
                file=file_data,
                file_options={"content-type": "image/jpeg", "upsert": "true"}
            )

            # Get public URL
            public_url = self.client.storage.from_(self.bucket_name).get_public_url(dest_path)
            print(f"[SUPABASE STORAGE] Snapshot uploaded to storage bucket '{self.bucket_name}': {public_url}")
            return public_url

        except Exception as e:
            print(f"[SUPABASE STORAGE ERROR] Failed uploading snapshot '{filename}': {e}")
            return None

    def insert_violation(
        self,
        student_id: Any,
        score: float,
        cheat_reason: str,
        category_name: str,
        cat_count: int,
        evidence_url: Optional[str] = None,
        session_id: Optional[str] = None,
        metrics: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Inserts violation record into Supabase PostgreSQL tables ('violations', 'student_violations', or 'classroom_alerts').
        """
        if not self.enabled or not self.client:
            return False

        metrics = metrics or {}
        session_id = session_id or os.getenv("EXAM_SESSION_ID")

        # Prepare record for 'violations' table (matches schema diagram)
        violation_payload = {
            "tracker_id": int(student_id) if str(student_id).isdigit() else None,
            "tracker_label": f"Student_{student_id}",
            "activity": category_name[:50],
            "severity": "high" if score >= 0.75 else "medium",
            "status": "flagged",
            "confidence": round(float(score), 4),
            "evidence_url": evidence_url,
            "metadata": {
                "cheat_reason": cheat_reason,
                "activity_count": cat_count,
                "head_pitch_deg": metrics.get("head_pitch_deg"),
                "head_yaw_deg": metrics.get("head_yaw_deg"),
                "mouth_open_ratio": metrics.get("mouth_open_ratio"),
                "torso_lean_deg": metrics.get("torso_lean_deg"),
                "hands_under_desk": metrics.get("hands_under_desk", False)
            }
        }
        if session_id:
            violation_payload["session_id"] = session_id

        inserted = False

        # Attempt 1: Insert into 'violations' table
        try:
            res = self.client.table("violations").insert(violation_payload).execute()
            print(f"[SUPABASE DB] Inserted violation record into 'violations' table for Student #{student_id}")
            inserted = True
        except Exception as e1:
            logger.debug(f"Insert to 'violations' failed: {e1}")

        # Attempt 2: Fallback/Secondary insert into 'classroom_alerts' table
        try:
            alert_payload = {
                "student_id_tracker": str(student_id),
                "suspicion_score": round(float(score), 4),
                "status": cheat_reason[:50],
            }
            if session_id:
                alert_payload["session_id"] = session_id

            self.client.table("classroom_alerts").insert(alert_payload).execute()
            print(f"[SUPABASE DB] Inserted alert record into 'classroom_alerts' table for Student #{student_id}")
            inserted = True
        except Exception as e2:
            logger.debug(f"Insert to 'classroom_alerts' failed: {e2}")

        return inserted
