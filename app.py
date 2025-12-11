# app.py - Flask Backend for AI Fitness Trainer
from flask import Flask, render_template, Response, jsonify, request
import cv2
import mediapipe as mp
import time
import threading
from squat_analyzer import SquatAnalyzer
from armraise_analyzer import ArmRaiseCounter

app = Flask(__name__)

# Global variables for camera and exercise state
camera = None
camera_lock = threading.Lock()
is_running = False
current_exercise = "Squats"
goal = 10

# Exercise stats
stats = {
    'count': 0,
    'wrong': 0,
    'right_counter': 0,
    'left_counter': 0,
    'angle': 0,
    'stage': '',
    'feedback': '',
    'right_stage': '',
    'left_stage': '',
    'start_time': 0,
    'duration': 0,
    'goal_achieved': False
}

# Analyzers
squat_analyzer = None
arm_counter = None
pose = None


def reset_stats():
    global stats
    stats = {
        'count': 0,
        'wrong': 0,
        'right_counter': 0,
        'left_counter': 0,
        'angle': 0,
        'stage': '',
        'feedback': '',
        'right_stage': '',
        'left_stage': '',
        'start_time': time.time(),
        'duration': 0,
        'goal_achieved': False
    }


def generate_frames():
    global camera, is_running, stats, squat_analyzer, arm_counter, pose, goal, current_exercise
    
    while is_running:
        with camera_lock:
            if camera is None or not camera.isOpened():
                break
            
            success, frame = camera.read()
            if not success:
                break
        
        frame = cv2.flip(frame, 1)
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(image)
        
        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            
            if current_exercise == "Squats":
                angle, count, stage, feedback, wrong = squat_analyzer.detect_squat(landmarks)
                stats['count'] = count
                stats['wrong'] = wrong
                stats['angle'] = int(angle)
                stats['stage'] = stage
                stats['feedback'] = feedback
                
                # Draw on frame
                stage_color = (0, 255, 0) if stage == "down" else ((0, 0, 255) if stage == "wrong" else (255, 255, 255))
                cv2.putText(frame, f'Squats: {count}/{goal}', (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                cv2.putText(frame, f'Stage: {stage}', (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, stage_color, 2)
                cv2.putText(frame, f'Angle: {int(angle)}', (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 1, (100, 255, 255), 2)
                
                if count >= goal:
                    stats['goal_achieved'] = True
                    stats['duration'] = time.time() - stats['start_time']
            else:
                right_counter, right_stage, left_counter, left_stage = arm_counter.detect_arm_raise(landmarks)
                stats['right_counter'] = right_counter
                stats['left_counter'] = left_counter
                stats['right_stage'] = right_stage or ''
                stats['left_stage'] = left_stage or ''
                
                cv2.putText(frame, f'Right: {right_counter}/{goal}', (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                cv2.putText(frame, f'Left: {left_counter}/{goal}', (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                
                if right_counter >= goal or left_counter >= goal:
                    stats['goal_achieved'] = True
                    stats['duration'] = time.time() - stats['start_time']
            
            # Draw pose landmarks with custom colors
            mp.solutions.drawing_utils.draw_landmarks(
                frame, results.pose_landmarks, mp.solutions.pose.POSE_CONNECTIONS,
                mp.solutions.drawing_utils.DrawingSpec(color=(0, 245, 255), thickness=2, circle_radius=3),
                mp.solutions.drawing_utils.DrawingSpec(color=(255, 0, 228), thickness=2)
            )
        
        stats['duration'] = time.time() - stats['start_time']
        
        # Encode frame
        ret, buffer = cv2.imencode('.jpg', frame)
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
    global camera, is_running, squat_analyzer, arm_counter, pose, current_exercise, goal
    
    data = request.json
    current_exercise = data.get('exercise', 'Squats')
    goal = int(data.get('goal', 10))
    
    with camera_lock:
        if camera is None or not camera.isOpened():
            camera = cv2.VideoCapture(0, cv2.CAP_DSHOW) if hasattr(cv2, "CAP_DSHOW") else cv2.VideoCapture(0)
    
    squat_analyzer = SquatAnalyzer()
    arm_counter = ArmRaiseCounter()
    pose = mp.solutions.pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
    
    reset_stats()
    is_running = True
    
    return jsonify({'status': 'started'})


@app.route('/stop', methods=['POST'])
def stop_exercise():
    global camera, is_running, pose
    
    is_running = False
    
    with camera_lock:
        if camera is not None:
            camera.release()
            camera = None
    
    if pose is not None:
        pose.close()
        pose = None
    
    cv2.destroyAllWindows()
    
    return jsonify({'status': 'stopped', 'stats': stats})


@app.route('/stats')
def get_stats():
    return jsonify(stats)


if __name__ == '__main__':
    app.run(debug=False, threaded=True, port=5000, use_reloader=False)
