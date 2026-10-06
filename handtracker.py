import cv2
import mediapipe as mp
import math

#mediapipe hand tracking system
mp_hands = mp.solutions.hands
#mediapipe hand tracking drawing system(drawing on top of hands)
mp_draw = mp.solutions.drawing_utils

#setting max number of hands and minimum detection
hands = mp_hands.Hands(
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
)

def finger_gesture(landmarks, frame_shape):
    h, w, _ = frame_shape


    tips = {'thumb': 4, 'index': 8, 'middle': 12, 'ring': 16, 'pinky': 20}
    joints = {'thumb': 3, 'index': 6, 'middle': 10, 'ring': 14, 'pinky': 18}

    fingers_up = {}

    for finger in tips:
        tip_y = landmarks[tips[finger]].y * h
        joint_y = landmarks[joints[finger]].y * h
        fingers_up[finger] = tip_y < joint_y

    return fingers_up

def detect_gesture(fingers_up):

    #make list for each finger state
    pattern = (
        fingers_up['thumb'],
        fingers_up['index'],
        fingers_up['middle'],
        fingers_up['ring'],
        fingers_up['pinky']
    )

    #return gesture based on finger state
    if pattern == (True, True, True, True, True):
        return "palm"
    elif pattern == (True, False, False, False, False):
        return "thumbs_up"
    else:
        return "nothing"

def hand_tracking(frame):
        #frame height and width
        h, w, _ = frame.shape

        #changing frame to RGB 
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)

        left_gesture = "nothing"
        right_gesture = "nothing"

        if results.multi_hand_landmarks:
            for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                mp_draw.draw_landmarks(
                    frame,
                    hand_landmarks,
                    None,
                    mp_draw.DrawingSpec(color=(240, 207, 137), thickness=3, circle_radius=3),
                    mp_draw.DrawingSpec(color=(240, 207, 137), thickness=3)
                )

            fingers_up = finger_gesture(hand_landmarks.landmark, frame.shape)
            gesture = detect_gesture(fingers_up)                       

            wrist_x = hand_landmarks.landmark[0].x

            if wrist_x < 0.5:  
                left_gesture = gesture
            if wrist_x >= 0.5:
                right_gesture = gesture


        return left_gesture, right_gesture