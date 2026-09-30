import cv2
import numpy as np


def draw_flag_overlay(frame, bbox, student_id, score, is_flagged, cheat_reason="Normal"):
    """Draws bounding box and metadata label above student."""
    x1, y1, x2, y2 = bbox
    color = (0, 0, 255) if is_flagged else (0, 255, 0)
    thickness = 3 if is_flagged else 2

    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)

    if is_flagged:
        label = f"ID #{student_id} | {cheat_reason}"
    else:
        prob_pct = int(round(score * 100))
        label = f"ID #{student_id} | Normal ({prob_pct}%)"

    label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.48, 1)
    lbl_w, lbl_h = label_size
    lbl_y1 = max(0, y1 - lbl_h - 10)

    cv2.rectangle(frame, (x1, lbl_y1), (x1 + lbl_w + 10, lbl_y1 + lbl_h + 8), color, -1)
    cv2.putText(frame, label, (x1 + 5, lbl_y1 + lbl_h + 3), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1)


def apply_spotlight(frame, flagged_bboxes, dim_factor=0.3):
    """Dims non-suspicious areas while spotlighting flagged students."""
    if not flagged_bboxes:
        return frame

    mask = np.zeros_like(frame, dtype=np.uint8)
    for (x1, y1, x2, y2) in flagged_bboxes:
        cv2.rectangle(mask, (x1, y1), (x2, y2), (255, 255, 255), -1)

    dimmed = (frame * dim_factor).astype(np.uint8)
    return np.where(mask == 255, frame, dimmed)


def draw_dashboard(frame, fps, active_count, incident_active):
    """Draws top HUD status bar."""
    h, w, _ = frame.shape
    hud_bg = frame[0:40, 0:w]
    dark_overlay = (hud_bg * 0.2).astype(np.uint8)
    frame[0:40, 0:w] = dark_overlay

    cv2.putText(frame, f"FPS: {fps:.1f}", (15, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    cv2.putText(frame, f"Monitored Students: {active_count}", (160, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    status_txt = "INCIDENT FLAGGED" if incident_active else "ROOM MONITORING"
    status_col = (0, 0, 255) if incident_active else (0, 255, 0)
    cv2.putText(frame, f"Status: {status_txt}", (450, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_col, 2)
