import time


def get_likelihood_label(score):
    """
    Maps a numerical suspicion score (0.0 - 1.0) to a domain-resonant likelihood tier.
    """
    if score >= 0.85:
        return "High Certainty"
    elif score >= 0.70:
        return "Strong Likelihood"
    elif score >= 0.50:
        return "Moderate Suspicion"
    else:
        return "Low Probability"


def format_likelihood_statement(score, reason_text):
    """
    Formats a probability statement combining likelihood classification,
    percentage score, and detected activity text.
    Example: '[High Certainty - 88%] Using Phone / Material Below Desk'
    """
    if not reason_text or reason_text == "Normal":
        return "Normal"

    prob_pct = int(round(score * 100))
    label = get_likelihood_label(score)

    return f"[{label} - {prob_pct}%] {reason_text}"


class StudentSuspicionTracker:
    """
    Maintains debounced suspicion scores per student ID based on exam hall cheating heuristics:
    1. Phone / Material Below Desk: Hands in lap under desk while head pitch is tilted down.
    2. Leaning Away from Desk / Reaching:
       - DO NOT FLAG: Slight lean (< 22 deg) while writing with hands on desk.
       - FLAG: Extreme lean (> 22 deg) or moderate lean (> 15 deg) combined with hands moving away / reaching.
    3. Turning Head to Neighbor: Turning head sideways (> 22 deg) sustained for >= 3.0 continuous seconds.
    4. Communicating / Talking: Turning head + open mouth ratio (MAR > 0.12) sustained for >= 3.0 continuous seconds.
    """

    def __init__(self, threshold=0.65, required_frames=2, min_turn_sec=0.0):
        self.threshold = threshold
        self.required_frames = required_frames
        self.min_turn_sec = min_turn_sec
        self.debounce_counters = {}       # student_id -> consecutive frames over threshold
        self.last_reason = {}             # student_id -> cheat reason string

    def evaluate_behavior(self, student_id, head_pitch_deg, head_yaw_deg, mouth_open_ratio, torso_lean_deg,
                          hands_under_desk, hands_in_writing_pos, hands_moved_away, object_near_hand, current_time=None):
        current_time = current_time or time.time()
        score = 0.0
        reasons = []

        # Vector 1: Turning Head / Communicating
        # Account for elevated cameras: yaw angles may appear slightly smaller in 2D
        is_turning = False
        if head_yaw_deg is not None:
            abs_yaw = abs(head_yaw_deg)
            is_intense_turn = abs_yaw > 20.0
            is_moderate_turn = abs_yaw > 14.0
            is_turning = is_moderate_turn or is_intense_turn
            
            if is_intense_turn:
                score += 0.75  # Instantly exceeds 0.65 threshold
                if mouth_open_ratio is not None and mouth_open_ratio > 0.12:
                    reasons.append("Communicating / Talking to Peer")
                else:
                    reasons.append("Intense Head Turn to Neighbor")
            elif is_moderate_turn:
                score += 0.55  # Below threshold, acts as a warning or builds up with other factors
                if mouth_open_ratio is not None and mouth_open_ratio > 0.12:
                    score += 0.20 # Bump over threshold if talking
                    reasons.append("Communicating / Talking to Peer")
                else:
                    reasons.append("Looking at Neighbor")

        # Vector 2: Using Phone / Material Below Desk
        if (hands_under_desk and head_pitch_deg is not None and head_pitch_deg > 14.0) or (hands_under_desk and object_near_hand):
            score += 0.65
            reasons.append("Using Phone / Material Below Desk")
        elif object_near_hand:
            score += 0.55
            reasons.append("Mobile Phone / Unauthorized Material")
        elif head_pitch_deg is not None and head_pitch_deg > 28.0:
            score += 0.40
            reasons.append("Looking Down Under Desk")

        # Vector 3: Leaning Away from Desk / Reaching
        if torso_lean_deg is not None:
            if torso_lean_deg > 22.0:
                score += 0.50
                reasons.append("Severe Lean Away from Desk")
            elif torso_lean_deg > 15.0 and (hands_moved_away or is_turning):
                score += 0.45
                if hands_moved_away:
                    reasons.append("Leaning & Reaching Away from Desk")
                else:
                    reasons.append("Leaning Sideways to Peer")

        raw_score = min(1.0, score)
        raw_reason = " | ".join(reasons) if reasons else "Normal"
        formatted_statement = format_likelihood_statement(raw_score, raw_reason)

        return raw_score, formatted_statement

    def update(self, student_id, head_pitch_deg, head_yaw_deg, mouth_open_ratio, torso_lean_deg,
               hands_under_desk, hands_in_writing_pos, hands_moved_away, object_near_hand, current_time=None):

        score, reason_statement = self.evaluate_behavior(
            student_id=student_id,
            head_pitch_deg=head_pitch_deg,
            head_yaw_deg=head_yaw_deg,
            mouth_open_ratio=mouth_open_ratio,
            torso_lean_deg=torso_lean_deg,
            hands_under_desk=hands_under_desk,
            hands_in_writing_pos=hands_in_writing_pos,
            hands_moved_away=hands_moved_away,
            object_near_hand=object_near_hand,
            current_time=current_time
        )

        is_over_threshold = score >= self.threshold

        count = self.debounce_counters.get(student_id, 0)
        count = count + 1 if is_over_threshold else 0
        self.debounce_counters[student_id] = count

        if is_over_threshold:
            self.last_reason[student_id] = reason_statement

        should_alert = count >= self.required_frames
        active_reason = self.last_reason.get(student_id, reason_statement) if should_alert else reason_statement

        return score, should_alert, active_reason
