import cv2
import numpy as np
import mediapipe as mp
import time
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

prev_time = 0
frame_ts_ms = 0
latest_pose_landmarks = None

def posture_classifier(landmarks):
    if not landmarks:
        return "Can't detect a body", None, None
    
    try:
        l_shoulder = landmarks[11]
        l_hip = landmarks[23]
        l_ear = landmarks[7]
        
        torso_angle = np.arctan2(l_hip.y - l_shoulder.y, l_hip.x - l_shoulder.x) * 180/np.pi
        
        forward_head_dist = abs(l_ear.x - l_shoulder.x)
        
        if torso_angle < 90:
            return "Wrong pose", torso_angle, forward_head_dist
        elif forward_head_dist > 0.02:
            return "Wrong pose", torso_angle, forward_head_dist
        else:
            return "Right pose", torso_angle, forward_head_dist
        
    except:
        return "Error defining a posture", None, None

POSE_CONNECTIONS = [
    # eyes & nose
    (8, 6), (6, 5), (5, 4), (4, 0), (0, 1), (1, 2), (2, 3), (3, 7),
    # mouth
    (10, 9),
    # torso
    (12, 11), (11, 23), (23, 24), (24, 12),
    # left arm
    (12, 14), (14, 16), (16, 22), (16, 20), (20, 18), (18, 16),
    # right arm
    (11, 13), (13, 15), (15, 21), (15, 19), (19, 17), (17, 15),
    # left leg
    (24, 26), (26, 28), (28, 32), (32, 30), (30, 28),
    # right leg
    (23, 25), (25, 27), (27, 29), (29, 31), (31, 27)
]

def output_result(result:vision.PoseLandmarkerResult, output_img:mp.Image, timestamp_ms:int):
    global latest_pose_landmarks
        
    if result.pose_landmarks:
        latest_pose_landmarks = result.pose_landmarks[0]
    else:
        latest_pose_landmarks = None
    

base_opt = python.BaseOptions(model_asset_path="pose_landmarker_full.task")

options = vision.PoseLandmarkerOptions(
    base_options = base_opt,
    running_mode = vision.RunningMode.LIVE_STREAM,
    result_callback = output_result,
    num_poses = 1,
    min_pose_detection_confidence = 0.8,
    min_pose_presence_confidence = 0.8,
    min_tracking_confidence = 0.8,
)

# cam set up
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("Can't open webcam")

with vision.PoseLandmarker.create_from_options(options) as landmarker:
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break
        
        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        height, width = frame.shape[:2]
        
        frame_ts_ms += 33
        
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        
        landmarker.detect_async(mp_img, frame_ts_ms)
        
        curr_landmarks = latest_pose_landmarks
        if curr_landmarks:
            for landmark in curr_landmarks:
                x = int(landmark.x * width)
                y = int(landmark.y * height)
                cv2.circle(frame, (x, y), 5, (255, 0, 0), -1)
            
            for start_idx, end_idx in POSE_CONNECTIONS:
                start = curr_landmarks[start_idx]
                end = curr_landmarks[end_idx]
                vis1 = getattr(start, "visibility", 1.0)
                vis2 = getattr(end, "visibility", 1.0)
                
                if vis1 < 0.5 or vis2 < 0.5:
                    continue
                
                x1, y1 = int(start.x * width), int(start.y * height)
                x2, y2 = int(end.x * width), int(end.y * height)
                
                cv2.line(frame, (x1, y1), (x2, y2), (255, 255, 0), 2)
                
            status, torso_angle, forward_head_dist = posture_classifier(curr_landmarks)
            
            if status == "Right pose":
                text_color = (255, 0, 0)
            else:
                text_color = (0, 0, 255)
                
            cv2.putText(frame, status, (10, 50), cv2.FONT_HERSHEY_COMPLEX, 1, text_color, 2)
            
            if (torso_angle) and (forward_head_dist):
                cv2.putText(frame, f"Torso angle: {torso_angle:.2f}", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
                cv2.putText(frame, f"Head distance: {forward_head_dist:.2f}", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

        curr_time = time.time()
        fps = 1/(curr_time - prev_time) if prev_time > 0 else 0
        prev_time = curr_time
        cv2.putText(frame, f"FPS: {int(fps)}", (550, 450), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.8, (0, 255, 255), 2)
                
        cv2.imshow("Pose Landmarker", frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break
        
cap.release()
cv2.destroyAllWindows()