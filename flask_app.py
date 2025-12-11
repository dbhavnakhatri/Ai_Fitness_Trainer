# flask_app.py
from flask import Flask, render_template, Response, jsonify, request
import cv2
import mediapipe as mp
import json
import threading
from squat_analyzer import SquatAnalyzer
from armraise_analyzer import ArmRaiseCounter

app = Flask(__name__)

# Global variables
camera = None
exercise_type = "Squats"
goal = 10
is_running = False
stats = {
    "count": 0,
    "wrong": 0,
    "stage": "",
    "feedback": "",
    "right_count": 0,
    "left_count": 0,
    "right_stage": "",
    "left_stage": "",
    "goal_reached": False
}
stats_lock = threading.Lock()

def reset_stats():
    global stats
    with stats_lock:
        stats = {
            "count": 0,
            "wrong": 0,
            "stage": "",
            "feedback": "",
            "right_count": 0,
            "left_count": 0,
            "right_stage": "",
            "left_stage": "",
            "goal_reached": False
        }

def generate_frames():
    global camera, is_running, stats, exercise_type, goal
    
    if exercise_type == "Squats":
        analyzer = SquatAnalyzer()
    else:
        analyzer = ArmRaiseCounter()
    
    mp_pose = mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while is_running and camera is not None and camera.isOpened():
            success, frame = camera.read()
            if not success:
                break
            
            frame = cv2.flip(frame, 1)
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(image)
            
            if results.pose_landmarks:
                landmarks = results.pose_landmarks.landmark
                
                with stats_lock:
                    if exercise_type == "Squats":
                        angle, count, stage, feedback, wrong = analyzer.detect_squat(landmarks)
                        stats["count"] = count
                        stats["wrong"] = wrong
                        stats["stage"] = stage
                        stats["feedback"] = feedback
                        stats["goal_reached"] = count >= goal
                    else:
                        right_counter, right_stage, left_counter, left_stage = analyzer.detect_arm_raise(landmarks)
                        stats["right_count"] = right_counter
                        stats["left_count"] = left_counter
                        stats["right_stage"] = right_stage or ""
                        stats["left_stage"] = left_stage or ""
                        stats["goal_reached"] = right_counter >= goal or left_counter >= goal
                
                # Draw pose landmarks with custom styling
                mp_drawing.draw_landmarks(
                    frame,
                    results.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS,
                    landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style()
                )
            
            ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/start', methods=['POST'])
def start_exercise():
    global camera, is_running, exercise_type, goal
    
    data = request.json
    exercise_type = data.get('exercise', 'Squats')
    goal = data.get('goal', 10)
    
    reset_stats()
    
    if camera is None or not camera.isOpened():
        camera = cv2.VideoCapture(0, cv2.CAP_DSHOW) if hasattr(cv2, "CAP_DSHOW") else cv2.VideoCapture(0)
    
    is_running = True
    return jsonify({"status": "started"})

@app.route('/stop', methods=['POST'])
def stop_exercise():
    global camera, is_running
    
    is_running = False
    if camera is not None:
        camera.release()
        camera = None
    
    return jsonify({"status": "stopped"})

@app.route('/stats')
def get_stats():
    global stats, goal, exercise_type
    with stats_lock:
        return jsonify({**stats, "goal": goal, "exercise": exercise_type})

if __name__ == '__main__':
    app.run(debug=False, threaded=True, port=5000)

