import cv2
import mediapipe as mp
import numpy as np
import winsound
import threading
import time
import pygame

pygame.mixer.init()
def play_mp3(file):
    try:
        pygame.mixer.music.load(file)
        pygame.mixer.music.play()
    except:
        pass
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6
)
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cap.set(3, 1280)
cap.set(4, 720)

LEFT_EYE = [33,160,158,133,153,144]
RIGHT_EYE = [362,385,387,263,373,380]
MOUTH = [61, 291, 13, 14]
NOSE, CHIN, FOREHEAD = 1, 152, 10

eye_closed_start = None
head_drop_start = None
drowsy_start = None

take_control_time = None
lane_change_time = None
indicator_start = None

head_alarm_on = False
hazard_on = False
indicator_on = False
show_reduce_speed = False
total_drowsy_time = 0

def beep_alarm():
    global head_alarm_on
    while head_alarm_on:
        winsound.Beep(2500, 700) 
        time.sleep(0.1)           

def pt(lm, i, w, h):
    return np.array([lm[i].x * w, lm[i].y * h])

def eye_ratio(eye):
    return (np.linalg.norm(eye[1]-eye[5]) +
            np.linalg.norm(eye[2]-eye[4])) / (2*np.linalg.norm(eye[0]-eye[3]))

def mouth_ratio(m):
    return np.linalg.norm(m[2]-m[3]) / np.linalg.norm(m[0]-m[1])

def reset_system():
    global eye_closed_start, head_drop_start, drowsy_start
    global take_control_time, lane_change_time, indicator_start
    global head_alarm_on, hazard_on, indicator_on, show_reduce_speed
    global total_drowsy_time

    eye_closed_start = head_drop_start = drowsy_start = None
    take_control_time = lane_change_time = indicator_start = None
    head_alarm_on = False
    hazard_on = indicator_on = False
    show_reduce_speed = False
    total_drowsy_time = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    h, w = frame.shape[:2]
    now = time.time()
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)

    status = "ALERT"
    color = (0,255,0)

    if results.multi_face_landmarks:
        lm = results.multi_face_landmarks[0].landmark

       
        for p in lm:
            cv2.circle(frame, (int(p.x*w), int(p.y*h)), 1, (0,255,0), -1)

        nose = pt(lm, NOSE, w, h)
        chin = pt(lm, CHIN, w, h)
        forehead = pt(lm, FOREHEAD, w, h)
        pitch_ratio = np.linalg.norm(nose-chin) / np.linalg.norm(forehead-chin)

        le = [pt(lm,i,w,h) for i in LEFT_EYE]
        re = [pt(lm,i,w,h) for i in RIGHT_EYE]
        ear = (eye_ratio(le)+eye_ratio(re))/2

        mouth_pts = [pt(lm,i,w,h) for i in MOUTH]
        mar = mouth_ratio(mouth_pts)

        eyes_closed = ear < 0.17
        head_down = pitch_ratio < 0.36

    
        driver_alert = not eyes_closed and not head_down

        if driver_alert:
            head_alarm_on = False
            reset_system()
        else:
          
            trigger_alarm = False
            if eyes_closed:
                if eye_closed_start is None:
                    eye_closed_start = now
                elif now - eye_closed_start >= 2:  
                    trigger_alarm = True
            else:
                eye_closed_start = None

            if head_down:
                if head_drop_start is None:
                    head_drop_start = now
                elif now - head_drop_start >= 5:
                    trigger_alarm = True
            else:
                head_drop_start = None

            
            if trigger_alarm and drowsy_start is None:
                drowsy_start = now

            
            if drowsy_start:
                total_drowsy_time = int(now - drowsy_start)

           
            if trigger_alarm and not head_alarm_on:
                head_alarm_on = True
                threading.Thread(target=beep_alarm, daemon=True).start()

       
        if mar > 0.6:
            status = "YAWNING"
            color = (255,255,0)

        
        if eyes_closed and not head_down and eye_closed_start and now - eye_closed_start >= 2:
            status = "EYES CLOSED"
            color = (0,0,255)

        
        if head_down and head_drop_start and now - head_drop_start >= 5:
            status = "HEAD DROPPING"
            color = (0,165,255)

        
        if drowsy_start and not take_control_time and now - drowsy_start >= 10:
            play_mp3("take control.mp3")
            take_control_time = now
            hazard_on = True
            show_reduce_speed = True

        
        if take_control_time and not lane_change_time and now - take_control_time >= 10:
            play_mp3("Lane change.mp3")
            lane_change_time = now
            show_reduce_speed = False
            hazard_on = False
            indicator_start = now  

        
        if lane_change_time and indicator_start and now - indicator_start >= 1 and not indicator_on:
            indicator_on = True

        
        if indicator_on and now - indicator_start >= 6:  
            indicator_on = False
            hazard_on = True

    else:
        reset_system()

    
    cv2.rectangle(frame,(10,10),(500,200),(20,20,20),-1)
    cv2.putText(frame,"DRIVER MONITORING SYSTEM",(20,40),
                cv2.FONT_HERSHEY_SIMPLEX,0.85,(0,255,255),2)
    cv2.putText(frame,f"STATUS: {status}",(20,85),
                cv2.FONT_HERSHEY_SIMPLEX,0.95,color,2)
    cv2.putText(frame,f"DROWSY TIME: {total_drowsy_time} sec",(20,125),
                cv2.FONT_HERSHEY_SIMPLEX,0.85,(255,255,255),2)

    if show_reduce_speed:
        cv2.putText(frame,"REDUCING SPEED...",
                    (420,620),
                    cv2.FONT_HERSHEY_SIMPLEX,1.1,(0,255,255),3)

    if hazard_on and int(now*2)%2==0:
        cv2.putText(frame,"HAZARD LIGHTS ON",
                    (430,680),
                    cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),3)

    if indicator_on and int(now*2)%2==0:
        arrow = np.array([[80,360],[140,330],[140,390]])
        cv2.fillPoly(frame,[arrow],(0,255,0))

    cv2.imshow("Driver Monitoring System", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

