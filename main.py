'''
Description: Program that uses facial recognition, and hand tracking to detect facial expression and gesture. 
Then displays meme that correlates with gesture next to users feed. 
Uses cv2 for camera, and mediapipe for facial and hand recognition. 

input: webcam video
    face recognition(mouthstate, headtilt, headnod, headturn)
    handtracking

output: video with meme next to the detected face/hand gesture
    smile, right, left, open, neutral, thumbsup, palm

Author: curearea

Version 1.0

'''
#import libraries
import cv2
import mediapipe as mp
import math
from facelandmarker import (get_mouth_state, 
                            get_head_tilt,
                            get_head_turn, 
                            get_head_nod
)
from handtracker import (hand_tracking)

#start mediapipe face mesh system
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh()

#mediapipe hand tracking system
mp_hands = mp.solutions.hands
#mediapipe hand tracking drawing system(drawing on top of hands)
mp_draw = mp.solutions.drawing_utils

#start webcam capture
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FPS, 30)


#main function
def main():
    #load memes
    smile_meme = cv2.imread("memes/smile/smile.jpeg")
    right_meme = cv2.imread("memes/right/right.jpeg")
    left_meme = cv2.imread("memes/left/left.jpeg")
    open_meme = cv2.imread("memes/open/open.jpeg")
    neutral_meme = cv2.imread("memes/neutral/neutral.jpeg")
    palm_meme = cv2.imread("memes/palm/palm.jpg")
    thumbs_up_meme = cv2.imread("memes/thumbs_up/thumbs_up_left.jpg")
    thumbs_up_meme_right = cv2.flip(thumbs_up_meme, 1)

    ret, frame = cap.read()

    #crop frame shape to a square
    h, w, _ = frame.shape
    min_dim = min(h, w)

    #load memes and resize them to match a square frame
    smile_meme = cv2.resize(smile_meme, (min_dim, min_dim))
    right_meme = cv2.resize(right_meme, (min_dim, min_dim))
    left_meme = cv2.resize(left_meme, (min_dim, min_dim))
    open_meme = cv2.resize(open_meme, (min_dim, min_dim))
    neutral_meme = cv2.resize(neutral_meme, (min_dim, min_dim))
    palm_meme = cv2.resize(palm_meme, (min_dim, min_dim))
    thumbs_up_meme = cv2.resize(thumbs_up_meme, (min_dim, min_dim))
    thumbs_up_meme_right = cv2.resize(thumbs_up_meme_right, (min_dim, min_dim))

    #while camera is running
    while True:

        ret, frame = cap.read()

        #flip the camera feed horizontally
        frame = cv2.flip(frame,1)
        
        #crop frame shape to a square
        h, w, _ = frame.shape
        min_dim = min(h, w)
        start_x = (w - min_dim) // 2
        start_y = (h - min_dim) // 2
        frame = frame[start_y:start_y + min_dim, start_x:start_x + min_dim]

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb_frame)

        
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:


                #landmarks for left and right of nose and mouth
                left = face_landmarks.landmark[61]
                right = face_landmarks.landmark[291]
                nose = face_landmarks.landmark[1]

                #calling functions for tilt, turn, nod, and mouth state
                tilt = get_head_tilt(face_landmarks.landmark, frame.shape)
                turn = get_head_turn(face_landmarks.landmark)
                nod = get_head_nod(face_landmarks.landmark)
                mouth = get_mouth_state(face_landmarks.landmark, frame.shape)
                left_gesture, right_gesture = hand_tracking(frame)
        
                h, w, _ = frame.shape
                avg_corner_y = ((left.y + right.y) / 2) * h
                nose_y = nose.y * h

                #drawing box on face
                for landmark in face_landmarks.landmark:
                    
                    #box around face
                    x_coords = [int(lm.x * w) for lm in face_landmarks.landmark]
                    y_coords = [int(lm.y * h) for lm in face_landmarks.landmark]

                    x_min, x_max = min(x_coords), max(x_coords)
                    y_min, y_max = min(y_coords), max(y_coords)

                    cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (193, 182, 255), 2)

                #hand tracking
                if left_gesture == "palm" or right_gesture == "palm":
                    combined = cv2.hconcat([frame, palm_meme])
                elif left_gesture == "thumbs_up":
                    combined = cv2.hconcat([frame, thumbs_up_meme])
                elif right_gesture == "thumbs_up":
                    combined = cv2.hconcat([frame, thumbs_up_meme_right])

                #face tracking
                elif mouth == "SMILING":
                    combined = cv2.hconcat([frame, smile_meme])
                elif turn == "LEFT":
                    combined = cv2.hconcat([frame, left_meme])
                elif turn == "RIGHT":
                    combined = cv2.hconcat([frame, right_meme]) 
                elif mouth == "OPEN":
                    combined = cv2.hconcat([frame, open_meme])
                else:
                    combined = cv2.hconcat([frame, neutral_meme])

                #show the camera combined with the meme
                cv2.imshow("hamster face ^_^", combined)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

#call main function
if __name__ == "__main__":
    main()