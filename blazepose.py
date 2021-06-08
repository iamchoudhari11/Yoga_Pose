from os import write
import streamlit as st
import pickle
import mediapipe as mp # Import mediapipe
import cv2 # Import opencv
import numpy as np
import pandas as pd


mp_drawing = mp.solutions.drawing_utils # Drawing helpers
mp_holistic = mp.solutions.holistic # Mediapipe Solutions

def load_model():
    with open('body_language.pkl', 'rb') as f:
        model = pickle.load(f)
    return model
model = load_model()

def blazepose():

    header = st.beta_container()
    explain = st.beta_container()
    pre = st.beta_container()

    with header:
        st.markdown("<h1 style='text-align: center; color:Red ;'>REAL TIME HUMAN POSE ESTIMATION: </h1>", unsafe_allow_html=True)
        

    with explain:
        st.markdown("## BLAZEPOSE :")
        st.write("BlazePose, a lightweight convolutional neural network architecture for Single person pose estimation that is tailored for real-time. During inference, the network produces 33 body keypoints for a single person and runs about 30 frames per second. This makes it particularly suited to real-time use cases like fitness tracking and sign language recognition and many more.")
        blaze1 = ['images/blazepose4.gif', 'images/blazepose5.gif']
        st.image(blaze1, use_column_width=True* len(blaze1))
        st.write("The current standard for human body pose is the COCO topology, which consists of 17 landmarks across the torso, arms, legs, and face. However, the COCO keypoints only localize to the ankle and wrist points, lacking scale and orientation information for hands and feet, which is vital for practical applications like fitness and dance. The inclusion of more keypoints is crucial for the subsequent application of domain-specific pose estimation models, like those for hands, face, or feet.")
        st.markdown("### Topology: ")
        st.write("With BlazePose,a new topology of 33 human body keypoints, which is a superset of COCO, BlazeFace and BlazePalm topologies. This allows us to determine body semantics from pose prediction alone that is consistent with face and hand models.")
        _,col2,_ = st.beta_columns([2,2,2])
        with col2:
            st.image("images/blazepose_key.png")
        code = '''0. Nose, 1. Left eye inner, 2. Left eye, 3. Left eye outer, 4. Right eye inner, 5. Right eye, 6. Right eye outer, 7. Left ear, 8. Right ear, 9. Mouth left 10. Mouth right, 
11. Left shoulder, 12. Right shoulder, 13. Left elbow, 14. Right elbow, 15. Left wrist, 16. Right wrist, 17. Left knuckle , 18. Right knuckle, 19. Left index, 20. Right index, 
21. Left thumb , 22. Right thumb , 23. Left hip, 24. Right hip, 25. Left knee, 26. Right knee, 27. Left ankle, 28. Right ankle, 29. Left heel, 30. Right heel, 31. Left foot index, 
32. Right foot index '''
        st.code(code)
        
        st.markdown("<h2 style='text-align: center; color:Red ;'>Model: </h2>", unsafe_allow_html=True)
        st.markdown("### The Model predict's the Yoga posture of a single person either he/she is doing Poses like **1) Tadasan**,**2) Balancing**,**3) Warrior Pose**,**4) Padmasana** :walking: :wrestlers:")
       


    
    
    # def load_model():
    #             with open('body_language.pkl', 'rb') as f:
    #                 model = pickle.load(f)
    #             return model
    # model = load_model()
    ok = st.button("Mediapipe Blazepose")
    if ok: 
        cap = cv2.VideoCapture(0)
        # Initiate holistic model
        with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
            
            while cap.isOpened():
                ret, frame = cap.read()
                
                # Recolor Feed
                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image.flags.writeable = False        
                
                # Make Detections
                results = holistic.process(image)
                # print(results.face_landmarks)
                
                # face_landmarks, pose_landmarks, left_hand_landmarks, right_hand_landmarks
                
                # Recolor image back to BGR for rendering
                image.flags.writeable = True   
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                
                # 1. Draw face landmarks
                mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACE_CONNECTIONS, 
                                        mp_drawing.DrawingSpec(color=(80,110,10), thickness=1, circle_radius=1),
                                        mp_drawing.DrawingSpec(color=(80,256,121), thickness=1, circle_radius=1)
                                        )
                
                # 2. Right hand
                mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
                                        mp_drawing.DrawingSpec(color=(80,22,10), thickness=2, circle_radius=4),
                                        mp_drawing.DrawingSpec(color=(80,44,121), thickness=2, circle_radius=2)
                                        )

                # 3. Left Hand
                mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
                                        mp_drawing.DrawingSpec(color=(121,22,76), thickness=2, circle_radius=4),
                                        mp_drawing.DrawingSpec(color=(121,44,250), thickness=2, circle_radius=2)
                                        )

                # 4. Pose Detections
                mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS, 
                                        mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=4),
                                        mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
                                        )
                # Export coordinates
                try:
                    # Extract Pose landmarks
                    pose = results.pose_landmarks.landmark
                    pose_row = list(np.array([[landmark.x, landmark.y, landmark.z, landmark.visibility] for landmark in pose]).flatten())
                    
                    # Extract Face landmarks
                    face = results.face_landmarks.landmark
                    face_row = list(np.array([[landmark.x, landmark.y, landmark.z, landmark.visibility] for landmark in face]).flatten())
                    
                    # Concate rows
                    row = pose_row+face_row
                    


                    # Make Detections
                    X = pd.DataFrame([row])
                    body_language_class = model.predict(X)[0]
                    body_language_prob = model.predict_proba(X)[0]
                    print(body_language_class, body_language_prob)
                    
                    # Grab ear coords
                    coords = tuple(np.multiply(
                                    np.array(
                                        (results.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_EAR].x, 
                                        results.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_EAR].y))
                                , [640,480]).astype(int))
                    
                    cv2.rectangle(image, 
                                (coords[0], coords[1]+5), 
                                (coords[0]+len(body_language_class)*20, coords[1]-30), 
                                (245, 117, 16), -1)
                    cv2.putText(image, body_language_class, coords, 
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
                    
                    # Get status box
                    cv2.rectangle(image, (0,0), (250, 60), (245, 117, 16), -1)
                    
                    # Display Class
                    cv2.putText(image, 'CLASS'
                                , (95,12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
                    cv2.putText(image, body_language_class.split(' ')[0]
                                , (90,40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
                    
                    # Display Probability
                    cv2.putText(image, 'PROB'
                                , (15,12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
                    cv2.putText(image, str(round(body_language_prob[np.argmax(body_language_prob)],2))
                                , (10,40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
                    
                except:
                    pass
                                
                cv2.imshow('Raw Webcam Feed', image)

                if cv2.waitKey(2000) & 0xFF == ord('q'):
                    break

        cap.release()
        cv2.destroyAllWindows()
    st.write("Quit Webcam - Press Q :octagonal_sign:")
   
    
    with pre:
        st.sidebar.markdown("<h1 style='color:Red ;'>References</h1>", unsafe_allow_html=True)
        st.sidebar.markdown("[On-device, Real-time Body Pose Tracking with MediaPipe BlazePose](https://ai.googleblog.com/2020/08/on-device-real-time-body-pose-tracking.html)")
        st.sidebar.markdown("[BlazePose: On-device Real-time Body Pose tracking](https://arxiv.org/abs/2006.10204)")

        st.sidebar.markdown("<h2 style='color:Red ;'>If you like the Content Press Button and Enjoy: </h2>", unsafe_allow_html=True)
        if st.sidebar.button("Press if You Like"):
            st.balloons() 

