# squat_analyzer.py
import numpy as np
import mediapipe as mp

mp_pose = mp.solutions.pose

class SquatCounter:
    def __init__(self, down_angle_min=75, down_angle_max=80, up_angle=170):
        self.down_angle_min = down_angle_min
        self.down_angle_max = down_angle_max
        self.up_angle = up_angle
        self.stage = "up"            # "up" -> "down"/"wrong" -> back to "up"
        self.counter = 0
        self.wrong_squats = 0
        self.feedback = ""

    def update(self, angle: float):
        # Going down
        if angle <= self.down_angle_max:
            if self.down_angle_min <= angle <= self.down_angle_max:
                if self.stage == "up":
                    self.stage = "down"
                    self.feedback = "✅ Good Squat"
            else:
                if self.stage == "up":
                    self.stage = "wrong"
                    self.wrong_squats += 1
                    if angle < self.down_angle_min:
                        self.feedback = f"❌ Too Deep! ({int(angle)}°)"
                    else:
                        self.feedback = f"❌ Not Deep Enough! ({int(angle)}°)"
        # Coming up
        elif angle >= self.up_angle:
            if self.stage in ("down", "wrong"):
                if self.stage == "down":
                    self.counter += 1
                self.stage = "up"
                self.feedback = ""

def calculate_angle(a, b, c):
    """Angle at point b given three 2D points a, b, c (degrees)."""
    a = np.array(a, dtype=float)
    b = np.array(b, dtype=float)
    c = np.array(c, dtype=float)

    ba = a - b
    bc = c - b

    denom = (np.linalg.norm(ba) * np.linalg.norm(bc))
    if denom == 0:
        return 180.0  # neutral fallback
    cosine_angle = np.dot(ba, bc) / denom
    angle = np.degrees(np.arccos(np.clip(cosine_angle, -1.0, 1.0)))
    return float(angle)

class SquatAnalyzer:
    def __init__(self):
        self.counter = SquatCounter()

    def detect_squat(self, landmarks):
        LP = mp_pose.PoseLandmark

        # Use LEFT side leg joints for angle (hip–knee–ankle)
        hip   = [landmarks[LP.LEFT_HIP.value].x,   landmarks[LP.LEFT_HIP.value].y]
        knee  = [landmarks[LP.LEFT_KNEE.value].x,  landmarks[LP.LEFT_KNEE.value].y]
        ankle = [landmarks[LP.LEFT_ANKLE.value].x, landmarks[LP.LEFT_ANKLE.value].y]

        angle = calculate_angle(hip, knee, ankle)

        # Update the rolling counter & feedback
        self.counter.update(angle)

        # Returns: angle, total_good_count, stage, feedback, wrong_count
        return angle, self.counter.counter, self.counter.stage, self.counter.feedback, self.counter.wrong_squats
