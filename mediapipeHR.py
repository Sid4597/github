import cv2
import mediapipe as mp

mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic

cap = cv2.VideoCapture(0) #start webcam

with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    while cap.isOpened():
        ret, frame = cap.read()

        #Recolor feed from BGR to RGB
        image = cv2.cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        #make Detections
        results = holistic.process(image)

        #Recolor feed back to COLOR_BGR2RGB
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        #draw face landmarks
        mp_drawing.draw_landmarks(image,results.face_landmarks, mp_holistic.FACEMESH_TESSELATION)

        #draw right hand landmarks
        mp_drawing.draw_landmarks(image,results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                                mp_drawing.DrawingSpec(color=(0,0,255),thickness = 2,circle_radius=1),
                                mp_drawing.DrawingSpec(color=(0,0,240),thickness = 2,circle_radius=1))

        #draw right hand landmarks
        mp_drawing.draw_landmarks(image,results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                                mp_drawing.DrawingSpec(color=(255,0,0),thickness = 2,circle_radius=1),
                                mp_drawing.DrawingSpec(color=(240,0,0),thickness = 2,circle_radius=1))

        #draw pose landmarks
        mp_drawing.draw_landmarks(image,results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                                mp_drawing.DrawingSpec(color=(0,255,0),thickness = 2,circle_radius=1),
                                mp_drawing.DrawingSpec(color=(0,240,0),thickness = 2,circle_radius=1))

        cv2.imshow('Face and Hand Detections',image)


        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
