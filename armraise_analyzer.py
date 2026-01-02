# armraise_analyzer.py
import math
import mediapipe as mp

mp_pose = mp.solutions.pose

class ArmRaiseCounter:
    def __init__(self):
        self.right_counter = 0
        self.right_stage = None
        self.left_counter = 0
        self.left_stage = None

    def calculate_angle(self, a, b, c):
        """Returns the angle between three points (in degrees)."""
        angle = math.degrees(
            math.atan2(c[1] - b[1], c[0] - b[0]) -
            math.atan2(a[1] - b[1], a[0] - b[0])
        )
        return abs(angle) if angle >= 0 else 360 - abs(angle)

    def detect_arm_raise(self, landmarks):
        LP = mp_pose.PoseLandmark

        # Extract coordinates (right = even indices: 12/14/16/24, left = odd: 11/13/15/23)
        right_shoulder = [landmarks[LP.RIGHT_SHOULDER.value].x, landmarks[LP.RIGHT_SHOULDER.value].y]
        right_elbow    = [landmarks[LP.RIGHT_ELBOW.value].x,    landmarks[LP.RIGHT_ELBOW.value].y]
        right_wrist    = [landmarks[LP.RIGHT_WRIST.value].x,    landmarks[LP.RIGHT_WRIST.value].y]
        right_hip      = [landmarks[LP.RIGHT_HIP.value].x,      landmarks[LP.RIGHT_HIP.value].y]

        left_shoulder  = [landmarks[LP.LEFT_SHOULDER.value].x,  landmarks[LP.LEFT_SHOULDER.value].y]
        left_elbow     = [landmarks[LP.LEFT_ELBOW.value].x,     landmarks[LP.LEFT_ELBOW.value].y]
        left_wrist     = [landmarks[LP.LEFT_WRIST.value].x,     landmarks[LP.LEFT_WRIST.value].y]
        left_hip       = [landmarks[LP.LEFT_HIP.value].x,       landmarks[LP.LEFT_HIP.value].y]

        # Elbow angles (shoulder–elbow–wrist)
        right_angle = self.calculate_angle(right_shoulder, right_elbow, right_wrist)
        left_angle  = self.calculate_angle(left_shoulder,  left_elbow,  left_wrist)

        # Simple form checks (stay in vertical rail, elbow near torso)
        right_elbow_close = abs(right_elbow[0] - right_hip[0]) < 0.08
        left_elbow_close  = abs(left_elbow[0]  - left_hip[0])  < 0.08
        right_wrist_in_line = abs(right_wrist[0] - right_shoulder[0]) < 0.08
        left_wrist_in_line  = abs(left_wrist[0]  - left_shoulder[0])  < 0.08

        right_form_ok = right_elbow_close and right_wrist_in_line
        left_form_ok  = left_elbow_close  and left_wrist_in_line

        # Right arm FSM
        if right_form_ok:
            if right_angle > 160:
                self.right_stage = "Down"
            if right_angle < 70 and self.right_stage == "Down":
                self.right_stage = "Correct"
                self.right_counter += 1
        else:
            self.right_stage = "Incorrect"

        # Left arm FSM
        if left_form_ok:
            if left_angle > 160:
                self.left_stage = "Down"
            if left_angle < 70 and self.left_stage == "Down":
                self.left_stage = "Correct"
                self.left_counter += 1
        else:
            self.left_stage = "Incorrect"

        return self.right_counter, self.right_stage, self.left_counter, self.left_stage
