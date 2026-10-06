import mediapipe as mp
import math


#define mouth state function
def get_mouth_state(landmarks,frame_shape):

    #frame height and width
    h, w, _ = frame_shape
    
    #mouth corners landmarks 
    left = landmarks[61]
    right = landmarks[291]
    nose = landmarks[1]
    
    #top and bottom lip landmarks
    top_lip = landmarks[13]
    bottom_lip = landmarks[14]

    #average corner y and nose y to get smile ratio
    avg_corner_y = ((left.y + right.y) / 2) * h
    nose_y = nose.y * h
    smile_ratio = avg_corner_y / nose_y
    
    #mouth open detection using top and bottom lip landmarks
    mouth_open = abs(float(bottom_lip.y) - float(top_lip.y)) * h

    #returns smiling, open, and neutral with detection
    if smile_ratio < 1.14:
        return "SMILING"
    elif mouth_open > 15:
        return "OPEN"
    else:
        return "NEUTRAL"

#define head tilt function
def get_head_tilt(landmarks, frame_shape):

    h, w, _ = frame_shape

    #left eye and right eye landmarks
    left_eye = landmarks[33]
    right_eye = landmarks[263]

    #angle of tilt of eyes
    dx = float((right_eye.x - left_eye.x) * w)
    dy = float((right_eye.y - left_eye.y) * h)

    #angle of tilt in degrees
    angle = math.degrees(math.atan2(dy, dx))

    #returns angle of head tilt 
    return angle

#define head turn function
def get_head_turn(landmarks):

    #nose and ear landmarks
    nose = landmarks[1]
    left_ear = landmarks[234]
    right_ear = landmarks[454]

    #distance from nose to ear
    left_dist = abs(float(nose.x) - float(left_ear.x))
    right_dist = abs(float(nose.x) - float(right_ear.x))

    #turn ratio calculation
    ratio = left_dist / right_dist

    #returns left right center with ratio
    if ratio < 0.3:
        return "LEFT"
    elif ratio > 9.0:
        return "RIGHT"
    else:
        return "CENTER"

#define head nod function
def get_head_nod(landmarks):

    #nose chin and forehead landmarks
    nose = landmarks[1]
    chin = landmarks[152]
    forehead = landmarks[10]

    #nose and chin distance
    nose_y = float(nose.y)
    chin_y = float(chin.y)
    forehead_y = float(forehead.y)

    #face height and nose ratio
    face_height = chin_y - forehead_y
    nose_ratio = (nose_y - forehead_y) / face_height

    #returns up down and center based on ratio
    if nose_ratio < 0.50:
        return "UP"
    elif nose_ratio > 0.60:
        return "DOWN"
    else:
        return "CENTER"

