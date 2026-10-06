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


# ── Enum value maps (must match Supabase enum definitions exactly) ──────────

_ACTIVITY_MAP = {
    "turning head to neighbor":           "head_turn",
    "communicating / talking to peer":    "talking",
    "using phone / material below desk":  "phone_use",
    "mobile phone / unauthorized material": "phone_use",
    "looking down under desk":            "suspicious_movement",
    "severe lean away from desk":         "suspicious_movement",
    "leaning & reaching away from desk":  "suspicious_movement",
    "leaning sideways to peer":           "head_turn",
    "general suspicion":                  "suspicious_movement",
}

def _map_activity(category_name: str) -> str:
    """Maps free-form category names to violation_activity_type enum values."""
    key = category_name.lower().strip()
    for phrase, enum_val in _ACTIVITY_MAP.items():
        if phrase in key:
            return enum_val
    return "suspicious_movement"   # safe fallback enum value


def _map_severity(score: float) -> str:
    """Maps suspicion score to violation_severity enum value."""
    if score >= 0.85:
        return "high"
    elif score >= 0.65:
        return "medium"
    else:
        return "low"


class SupabaseManager:
    """
    Manages Supabase Database connection and Storage uploads for Classroom Monitoring.
    Inserts data into:
      - violations          (primary incident record)
      - classroom_alerts    (session-level alert log)
      - session_students    (upsert tracker registration)
    """

    def __init__(
        self,
        url: Optional[str] = None,
        key: Optional[str] = None,
        bucket_name: Optional[str] = None
    ):
        self.url = url or os.getenv("SUPABASE_URL")
        self.key = key or os.getenv("SUPABASE_KEY") or os.getenv("SUPABASE_SERVICE_ROLE_KEY")
        self.bucket_name = bucket_name or os.getenv("SUPABASE_BUCKET", "violation-evidence")
        self.client: Optional[Client] = None
        self.enabled = False

        if not HAS_SUPABASE:
            print("[SUPABASE] 'supabase' package not installed. Run: pip install supabase")
            return

        if not self.url or not self.key or "your-supabase" in (self.url or ""):
            print("[SUPABASE] SUPABASE_URL or SUPABASE_KEY not set — running in offline CSV mode.")
            return

        try:
            self.client = create_client(self.url, self.key)
            self.enabled = True
            print(f"[SUPABASE] ✅ Connected — bucket: '{self.bucket_name}'")
        except Exception as e:
            print(f"[SUPABASE ERROR] Failed to connect: {e}")
            self.enabled = False

    # ──────────────────────────────────────────────────────────────────────────
    # Storage
    # ──────────────────────────────────────────────────────────────────────────

    def upload_evidence_image(self, local_filepath: str, remote_path: Optional[str] = None) -> Optional[str]:
        """Uploads a snapshot to Supabase Storage. Returns public URL or None."""
        if not self.enabled or not self.client:
            return None
        if not os.path.exists(local_filepath):
            print(f"[SUPABASE WARN] Snapshot not found: {local_filepath}")
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
            print(f"[SUPABASE STORAGE] ✅ Uploaded: {public_url}")
            return public_url

        except Exception as e:
            print(f"[SUPABASE STORAGE ERROR] ❌ Upload failed for '{filename}': {e}")
            return None

    # ──────────────────────────────────────────────────────────────────────────
    # Session Student Registration
    # ──────────────────────────────────────────────────────────────────────────

    def upsert_session_student(self, session_id: str, student_id: Any) -> None:
        """
        Registers (or updates last_seen_at for) a tracked student in session_students.
        session_id links to monitoring_sessions.id
        """
        if not self.enabled or not self.client or not session_id:
            return
        try:
            now_iso = datetime.now(timezone.utc).isoformat()
            self.client.table("session_students").upsert({
                "session_id": session_id,
                "tracker_label": f"Student_{student_id}",
                "last_seen_at": now_iso,
            }, on_conflict="session_id,tracker_label").execute()
            print(f"[SUPABASE DB] ✅ session_students upserted — Student #{student_id}")
        except Exception as e:
            print(f"[SUPABASE DB] ❌ session_students upsert failed: {e}")

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
        Inserts violation into:
          1. violations         — primary incident table
          2. classroom_alerts   — session-level alert log
        Returns True if at least one insert succeeded.
        """
        if not self.enabled or not self.client:
            return False

        metrics = metrics or {}
        session_id = session_id or os.getenv("EXAM_SESSION_ID")
        now_iso = datetime.now(timezone.utc).isoformat()
        activity_enum = _map_activity(category_name)
        severity_enum = _map_severity(score)

        inserted = False

        # ── 1. violations table ───────────────────────────────────────────────
        violation_payload = {
            "tracker_label": f"Student_{student_id}",
            "activity": activity_enum,
            "severity": severity_enum,
            "status": "flagged",
            "confidence": round(float(score), 4),
            "evidence_url": evidence_url,
            "metadata": {
                "category_name":    category_name,
                "cheat_reason":     cheat_reason,
                "activity_count":   cat_count,
                "head_pitch_deg":   metrics.get("head_pitch_deg"),
                "head_yaw_deg":     metrics.get("head_yaw_deg"),
                "mouth_open_ratio": metrics.get("mouth_open_ratio"),
                "torso_lean_deg":   metrics.get("torso_lean_deg"),
                "hands_under_desk": metrics.get("hands_under_desk", False),
            },
            "created_at": now_iso,
        }
        if session_id:
            violation_payload["session_id"] = session_id

        try:
            res = self.client.table("violations").insert(violation_payload).execute()
            print(f"[SUPABASE DB] ✅ violations — Student #{student_id} | {activity_enum} | {severity_enum} | {round(score*100)}%")
            inserted = True
        except Exception as e:
            print(f"[SUPABASE DB] ❌ violations insert FAILED: {e}")
            print(f"              Payload was: {violation_payload}")

        # ── 2. classroom_alerts table ─────────────────────────────────────────
        alert_payload = {
            "student_id_tracker": str(student_id),
            "timestamp_ms": int(datetime.now(timezone.utc).timestamp() * 1000),
            "suspicion_score": round(float(score), 4),
            "status": severity_enum,
            "created_at": now_iso,
        }
        if session_id:
            alert_payload["session_id"] = session_id

        try:
            self.client.table("classroom_alerts").insert(alert_payload).execute()
            print(f"[SUPABASE DB] ✅ classroom_alerts — Student #{student_id}")
            inserted = True
        except Exception as e:
            print(f"[SUPABASE DB] ❌ classroom_alerts insert FAILED: {e}")
            print(f"              Payload was: {alert_payload}")

        return inserted
