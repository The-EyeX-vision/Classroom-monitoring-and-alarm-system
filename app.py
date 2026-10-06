import os
import sys
import time
import tempfile
import cv2
import pandas as pd
import streamlit as st
from dotenv import load_dotenv

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

load_dotenv()

from src.capture import VideoStream
from src.detection import ClassroomDetectionPipeline
from src.tracking import adapt, CentroidTracker, StudentSuspicionTracker
from src.logging import IncidentLogger
from src.ui import draw_flag_overlay, apply_spotlight, draw_dashboard
from src.db.supabase_client import SupabaseManager

# Page Configuration
st.set_page_config(
    page_title="EyeX",
    page_icon="👁️",
    layout="wide"
)

# Custom Styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: white;
        margin-bottom: 0px;
    }
    .sub-header {
        font-size: 1.0rem;
        color: #6B7280;
        margin-bottom: 25px;
    }
    .metric-card {
        background-color: #1F2937;
        border-radius: 10px;
        padding: 15px;
        color: white;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# Title & Subtitle
st.markdown('<div class="main-header">👁️ EyeX</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">AI-Powered Exam Cheating Detection, Behavior Analysis & Cloud Storage Logging</div>', unsafe_allow_html=True)

# Sidebar Configuration
st.sidebar.header("⚙️ System Configuration")

# Check Supabase status
supabase_mgr = SupabaseManager()
if supabase_mgr.enabled:
    st.sidebar.success(f"⚡ Supabase DB Connected (Bucket: `{supabase_mgr.bucket_name}`)")
else:
    st.sidebar.warning("⚠️ Supabase Offline Mode (Logging to local CSV)")

# Video Source Selector
source_option = st.sidebar.radio(
    "Choose Video Source:",
    ("Upload Video File", "Sample Video 1 (sample-two.mp4)", "Sample Video 2 (sample-three.mp4)")
)

video_filepath = None
if source_option == "Upload Video File":
    uploaded_file = st.sidebar.file_uploader("Upload Exam Hall Video (.mp4, .avi, .mov)", type=["mp4", "avi", "mov", "mkv"])
    if uploaded_file is not None:
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        tfile.write(uploaded_file.read())
        video_filepath = tfile.name
else:
    if "sample-two" in source_option:
        video_filepath = os.path.join(PROJECT_ROOT, "test_video", "sample-two.mp4")
    else:
        video_filepath = os.path.join(PROJECT_ROOT, "test_video", "sample-three.mp4")

yolo_pose_choice = st.sidebar.selectbox(
    "YOLOv8-Pose Model:",
    (
        "yolov8s-pose  (Balanced - Default)",
        "yolov8m-pose  (High Accuracy)",
        "yolov8n-pose  (Fast / Low CPU)",
    ),
    index=0
)

if "yolov8m" in yolo_pose_choice:
    selected_pose_model = os.path.join(PROJECT_ROOT, "models", "yolov8m-pose.pt")
elif "yolov8n" in yolo_pose_choice:
    selected_pose_model = os.path.join(PROJECT_ROOT, "models", "yolov8n-pose.pt")
else:
    selected_pose_model = os.path.join(PROJECT_ROOT, "models", "yolov8s-pose.pt")

TEST_SESSION_ID = "5f5a439d-da5e-4401-b9aa-0bec0f3604dc"
session_id = st.sidebar.text_input(
    "Monitoring Session ID:",
    value=os.getenv("EXAM_SESSION_ID", TEST_SESSION_ID),
    help="UUID of the active monitoring_sessions row in Supabase. Used to link all violations/alerts to this session."
)
threshold = st.sidebar.slider("Suspicion Alert Threshold:", min_value=0.30, max_value=0.90, value=0.65, step=0.05,
                              help="Students are only flagged when their suspicion score exceeds this value. Default: 65%.")
cooldown_sec = st.sidebar.slider("Student Alert Cooldown (sec):", min_value=1.0, max_value=15.0, value=5.0, step=1.0)


# Main Dashboard Layout
kpi1, kpi2, kpi3, kpi4 = st.columns(4)

with kpi1:
    fps_metric = st.empty()
    fps_metric.metric(label="Processing Speed", value="0.0 FPS")

with kpi2:
    students_metric = st.empty()
    students_metric.metric(label="Active Students", value="0")

with kpi3:
    incidents_metric = st.empty()
    incidents_metric.metric(label="Flagged Incidents", value="0")

with kpi4:
    status_metric = st.empty()
    status_metric.metric(label="System Status", value="Idle ⏸️")

video_col, log_col = st.columns([2, 1])

with video_col:
    st.subheader("📹 Live Video Stream & AI Bounding Boxes")
    video_placeholder = st.empty()

with log_col:
    st.subheader("🚨 Real-Time Flagged Violations")
    log_placeholder = st.empty()

# Control Buttons
st.sidebar.markdown("---")
start_btn = st.sidebar.button("▶️ Start Monitoring", use_container_width=True, type="primary")
stop_btn = st.sidebar.button("⏹️ Stop Monitoring", use_container_width=True)

if "monitoring" not in st.session_state:
    st.session_state.monitoring = False

if start_btn:
    if not video_filepath or not os.path.exists(video_filepath):
        st.error("Please upload a video file or select a sample video first.")
    else:
        st.session_state.monitoring = True

if stop_btn:
    st.session_state.monitoring = False

# Processing Loop
if st.session_state.monitoring and video_filepath and os.path.exists(video_filepath):
    status_metric.metric(label="System Status", value="Active 🟢")

    # Initialize Stream & Pipeline (single YOLOv8-Pose model)
    stream = VideoStream(source=video_filepath)
    pipeline = ClassroomDetectionPipeline(
        pose_model_path=selected_pose_model
    )

    centroid_tracker = CentroidTracker(max_distance=0.25, max_missed_frames=600)
    suspicion_tracker = StudentSuspicionTracker(threshold=threshold, required_frames=2)
    csv_log_path = "classroom_alerts.csv"
    logger = IncidentLogger(
        csv_filepath=csv_log_path,
        output_dir="flagged_incidents",
        cooldown_sec=cooldown_sec,
        session_id=session_id
    )

    total_frames = 0
    start_time = time.time()
    declared_fps = stream.get_declared_fps()
    frame_duration_ms = int(1000 / max(declared_fps, 1))
    incidents_count = 0

    try:
        while st.session_state.monitoring:
            # Dynamically sync slider adjustments from UI
            suspicion_tracker.threshold = threshold
            logger.cooldown_sec = cooldown_sec

            ret, frame = stream.read()
            if not ret or frame is None:
                st.info("End of video stream reached.")
                break

            total_frames += 1
            timestamp_ms = int(total_frames * frame_duration_ms)
            h_full, w_full = frame.shape[:2]

            # 1. Run detection pipeline
            raw_detections = pipeline.process_frame(frame, timestamp_ms)

            # 2. Centroid tracking
            records = adapt(raw_detections, full_w=w_full, full_h=h_full)
            centroids = [r["centroid"] for r in records if "centroid" in r]
            assignments = centroid_tracker.update(centroids)

            flagged_bboxes = []
            has_incident_this_frame = False

            # 3. Evaluate students
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
                    if logged:
                        incidents_count += 1

                draw_flag_overlay(
                    frame=frame,
                    bbox=pixel_bbox,
                    student_id=student_id,
                    score=score,
                    is_flagged=should_alert,
                    cheat_reason=cheat_reason
                )

            # Update Metrics
            elapsed = time.time() - start_time
            current_fps = total_frames / max(elapsed, 1e-6)

            fps_metric.metric(label="Processing Speed", value=f"{current_fps:.1f} FPS")
            students_metric.metric(label="Active Students", value=str(len(assignments)))
            incidents_metric.metric(label="Flagged Incidents", value=str(incidents_count))

            if flagged_bboxes:
                frame = apply_spotlight(frame, flagged_bboxes, dim_factor=0.3)

            draw_dashboard(frame, current_fps, len(assignments), has_incident_this_frame)

            # Convert BGR to RGB for Streamlit display — ensure uint8 dtype
            try:
                display_frame = frame.astype('uint8') if frame.dtype != 'uint8' else frame
                frame_rgb = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
                video_placeholder.image(frame_rgb, channels="RGB", use_container_width=True)
            except Exception as disp_err:
                pass  # skip frame on transient display error

            # Render incident log table
            if os.path.exists(csv_log_path):
                try:
                    df_log = pd.read_csv(csv_log_path, on_bad_lines='skip')
                    display_cols = [c for c in ["timestamp", "student_id", "category_name", "suspicion_score"] if c in df_log.columns]
                    if not df_log.empty and display_cols:
                        log_placeholder.dataframe(
                            df_log[display_cols].tail(8),
                            use_container_width=True
                        )
                except Exception:
                    pass

    finally:
        stream.release()
        pipeline.close()
        st.session_state.monitoring = False
        status_metric.metric(label="System Status", value="Finished 🏁")

# Download Log Section
if os.path.exists("classroom_alerts.csv"):
    st.markdown("---")
    st.subheader("📥 Export Monitoring Log Data")
    try:
        df_download = pd.read_csv("classroom_alerts.csv", on_bad_lines='skip')
        st.download_button(
            label="Download Full CSV Incident Report",
            data=df_download.to_csv(index=False),
            file_name="classroom_monitoring_incidents.csv",
            mime="text/csv"
        )
    except Exception as e:
        st.warning(f"Could not read incident log: {e}")
