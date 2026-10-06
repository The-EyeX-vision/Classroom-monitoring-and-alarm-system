import os
import sys
import logging
from datetime import datetime, timezone
from typing import Optional, Dict, Any

logger = logging.getLogger("SupabaseManager")
logging.basicConfig(level=logging.INFO)

try:
    from supabase import create_client, Client
    HAS_SUPABASE = True
except ImportError:
    HAS_SUPABASE = False
    Client = None


# ── Valid Enum Mapping for Supabase Schema ────────────────────────────────────

_ACTIVITY_MAP = {
    "turning head":                       "SUSPICIOUS_MOVEMENT",
    "communicating":                      "SUSPICIOUS_MOVEMENT",
    "talking":                            "SUSPICIOUS_MOVEMENT",
    "phone":                              "UNAUTHORIZED_MATERIAL",
    "material":                           "UNAUTHORIZED_MATERIAL",
    "unauthorized":                       "UNAUTHORIZED_MATERIAL",
    "looking down":                       "SUSPICIOUS_MOVEMENT",
    "leaning":                            "SUSPICIOUS_MOVEMENT",
    "reaching":                           "SUSPICIOUS_MOVEMENT",
    "general suspicion":                  "SUSPICIOUS_MOVEMENT",
}

def _map_activity(category_name: str) -> str:
    """Maps free-form category names to valid DB enum: SUSPICIOUS_MOVEMENT, UNAUTHORIZED_MATERIAL, OTHER."""
    key = (category_name or "").lower().strip()
    for phrase, enum_val in _ACTIVITY_MAP.items():
        if phrase in key:
            return enum_val
    return "SUSPICIOUS_MOVEMENT"


def _map_severity(score: float) -> str:
    """Maps suspicion score to uppercase DB enum: HIGH, MEDIUM, LOW."""
    if score >= 0.80:
        return "HIGH"
    elif score >= 0.65:
        return "MEDIUM"
    else:
        return "LOW"


class SupabaseManager:
    """
    Manages Supabase Database connection and Storage uploads for Classroom Monitoring.
    Inserts data into:
      - violations          (primary incident record linked to session_id)
      - classroom_alerts    (session-level alert log)
    """

    def __init__(
        self,
        url: Optional[str] = None,
        key: Optional[str] = None,
        bucket_name: Optional[str] = None
    ):
        self.url = url or os.getenv("SUPABASE_URL")
        # Support SUPABASE_SERVICE_ROLE_KEY (bypasses Storage/DB RLS) or standard SUPABASE_KEY
        self.key = key or os.getenv("SUPABASE_SERVICE_ROLE_KEY") or os.getenv("SUPABASE_KEY")
        self.bucket_name = bucket_name or os.getenv("SUPABASE_BUCKET", "violation-evidence")
        self.client: Optional[Client] = None
        self.enabled = False

        if not HAS_SUPABASE:
            print("[SUPABASE] 'supabase' package not installed. Run: pip install supabase")
            return

        if not self.url or not self.key or "your-supabase" in (self.url or ""):
            print("[SUPABASE] SUPABASE_URL or SUPABASE_KEY not set — running in local CSV mode.")
            return

        try:
            self.client = create_client(self.url, self.key)
            self.enabled = True
            print(f"[SUPABASE] ✅ Client connected to '{self.url}' (Bucket: '{self.bucket_name}')")
        except Exception as e:
            print(f"[SUPABASE ERROR] Failed to connect: {e}")
            self.enabled = False

    # ──────────────────────────────────────────────────────────────────────────
    # Storage Upload
    # ──────────────────────────────────────────────────────────────────────────

    def upload_evidence_image(self, local_filepath: str, remote_path: Optional[str] = None) -> Optional[str]:
        """Uploads a snapshot to Supabase Storage. Returns public URL or None."""
        if not self.enabled or not self.client:
            return None
        if not os.path.exists(local_filepath):
            print(f"[SUPABASE WARN] Snapshot file does not exist: {local_filepath}")
            return None

        filename = os.path.basename(local_filepath)
        dest_path = remote_path or f"incidents/{filename}"

        try:
            with open(local_filepath, "rb") as f:
                file_data = f.read()

            self.client.storage.from_(self.bucket_name).upload(
                path=dest_path,
                file=file_data,
                file_options={"content-type": "image/jpeg", "upsert": "true"}
            )
            public_url = self.client.storage.from_(self.bucket_name).get_public_url(dest_path)
            print(f"[SUPABASE STORAGE] ✅ Snapshot uploaded successfully: {public_url}")
            return public_url

        except Exception as e:
            print(f"[SUPABASE STORAGE ERROR] ❌ Upload failed for '{filename}': {e}")
            print(f"                       Tip: Ensure bucket '{self.bucket_name}' allows INSERT for anon role, or use SUPABASE_SERVICE_ROLE_KEY.")
            return None

    # ──────────────────────────────────────────────────────────────────────────
    # Violation Insertion
    # ──────────────────────────────────────────────────────────────────────────

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
        Inserts violation record into Supabase PostgreSQL table 'violations'.
        """
        if not self.enabled or not self.client:
            return False

        metrics = metrics or {}
        # Default session fallback to active test exam_hall_session
        session_id = session_id or os.getenv("EXAM_SESSION_ID", "c04dea8a-b4ff-4745-b5e2-b7e8186831dd")
        now_iso = datetime.now(timezone.utc).isoformat()
        activity_enum = _map_activity(category_name)
        severity_enum = _map_severity(score)

        tracker_num = int(student_id) if str(student_id).isdigit() else 0

        violation_payload = {
            "session_id": session_id,
            "tracker_label": f"Tracker #{student_id}",
            "tracker_id": tracker_num,
            "activity_type": activity_enum,
            "severity": severity_enum,
            "status": "FLAGGED",
            "confidence": round(float(score), 4),
            "evidence_url": evidence_url,
            "metadata": {
                "category_name": category_name,
                "cheat_reason": cheat_reason,
                "activity_count": cat_count,
                "head_pitch_deg": metrics.get("head_pitch_deg"),
                "head_yaw_deg": metrics.get("head_yaw_deg"),
                "mouth_open_ratio": metrics.get("mouth_open_ratio"),
                "torso_lean_deg": metrics.get("torso_lean_deg"),
                "hands_under_desk": metrics.get("hands_under_desk", False),
            },
            "created_at": now_iso
        }

        try:
            res = self.client.table("violations").insert(violation_payload).execute()
            print(f"[SUPABASE DB] ✅ Violation inserted into 'violations' table for Student #{student_id} (ID: {res.data[0]['id']})")
            return True
        except Exception as e:
            print(f"[SUPABASE DB] ❌ Violations insert FAILED: {e}")
            return False
