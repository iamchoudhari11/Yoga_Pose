import streamlit as st
from PIL import Image
import estimator
import cv2
import argparse
import time
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

from random import randint




def openpose():

    header = st.beta_container()
    explain = st.beta_container()
    model = st.beta_container()
    pre = st.beta_container()

    with header:
        st.markdown("<h1 style='text-align: center; color:Red ;'>REAL TIME HUMAN POSE ESTIMATION: </h1>", unsafe_allow_html=True)

    with explain:
        st.markdown("## OPENPOSE :man-bouncing-ball:")
        st.write("Openpose is used for Single-Person as well as on Multi-Person for real-time human pose estimation from video/image")
        st.write("Multi-Person pose estimation is more difficult than the single person case as the location and the number of people in an image are unknown. Typically, we can tackle the above issue using one of two approaches")
        st.markdown("#### Top Down Approach:")
        st.write("In this Approach First Machine Tracks The Human Body and then Calculate the pose for each person.")
        st.markdown("#### Bottom Up Approach:")
        st.write("This Work Heavily optimizes the OpenPose and Bottom-Up Approach.First it Detects the Skeleton(which consist of keypoint and connections between them) to identify human poses and  contains up to nineteen keypoints")
        st.markdown("### Topology:")
        st.write("This Model is trained on COCO dataset(Common Objects In Context) COCO dataset model detects 17 keypoints + 1 background on the body as stated above.")
        _,col2,_ = st.beta_columns([2,2,2])
        with col2:
            st.image("images/openpose_key.png")
        code = '''0. Nose, 1. Neck, 2. RShoulder, 3. RElbow, 4. RWrist, 5. LShoulder, 6. LElbow, 7. LWrist, 8. RHip, 9. RKnee 10. RAnkle,  11. LHip, 12. LKnee, 13. LAnkle, 14. REye, 
15. LEye, 16. REar, 17.LEar , 18. Background  '''

        st.code(code)
    with model:
        st.markdown("<h2 style='text-align: center; color:Red ;'>Model: </h2>", unsafe_allow_html=True)
         ### Here i am here working on static image and model is caffemodel
        st.markdown("### For Checking out on Static image you should upload images into file uploader and it will provide output :person_doing_cartwheel:")
        protoFile = "models/coco/pose_deploy_linevec.prototxt"
        weightsFile = "models/coco/pose_iter_440000.caffemodel"
        nPoints = 18
        # COCO Output Format
        keypointsMapping = ['Nose', 'Neck', 'R-Sho', 'R-Elb', 'R-Wr', 'L-Sho', 
                            'L-Elb', 'L-Wr', 'R-Hip', 'R-Knee', 'R-Ank', 'L-Hip', 
                            'L-Knee', 'L-Ank', 'R-Eye', 'L-Eye', 'R-Ear', 'L-Ear']

        POSE_PAIRS = [[1,2], [1,5], [2,3], [3,4], [5,6], [6,7],
                    [1,8], [8,9], [9,10], [1,11], [11,12], [12,13],
                    [1,0], [0,14], [14,16], [0,15], [15,17],
                    [2,17], [5,16] ]

        # index of pafs correspoding to the POSE_PAIRS
        # e.g for POSE_PAIR(1,2), the PAFs are located at indices (31,32) of output, Similarly, (1,5) -> (39,40) and so on.
        mapIdx = [[31,32], [39,40], [33,34], [35,36], [41,42], [43,44], 
                [19,20], [21,22], [23,24], [25,26], [27,28], [29,30], 
                [47,48], [49,50], [53,54], [51,52], [55,56], 
                [37,38], [45,46]]

        colors = [ [0,100,255], [0,100,255], [0,255,255], [0,100,255], [0,255,255], [0,100,255],
                [0,255,0], [255,200,100], [255,0,255], [0,255,0], [255,200,100], [255,0,255],
                [0,0,255], [255,0,0], [200,200,0], [255,0,0], [200,200,0], [0,0,0]]


        # Find the Keypoints using Non Maximum Suppression on the Confidence Map
        def getKeypoints(probMap, threshold=0.1):
            
            mapSmooth = cv2.GaussianBlur(probMap,(3,3),0,0)

            mapMask = np.uint8(mapSmooth>threshold)
            keypoints = []
            
            #find the blobs
            contours, _ = cv2.findContours(mapMask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            #contours,hierachy=cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
            
            #for each blob find the maxima
            for cnt in contours:
                blobMask = np.zeros(mapMask.shape)
                blobMask = cv2.fillConvexPoly(blobMask, cnt, 1)
                maskedProbMap = mapSmooth * blobMask
                _, maxVal, _, maxLoc = cv2.minMaxLoc(maskedProbMap)
                keypoints.append(maxLoc + (probMap[maxLoc[1], maxLoc[0]],))

            return keypoints

        # Find valid connections between the different joints of a all persons present
        def getValidPairs(output):
            valid_pairs = []
            invalid_pairs = []
            n_interp_samples = 10
            paf_score_th = 0.1
            conf_th = 0.7
            # loop for every POSE_PAIR
            for k in range(len(mapIdx)):
                # A->B constitute a limb
                pafA = output[0, mapIdx[k][0], :, :]
                pafB = output[0, mapIdx[k][1], :, :]
                pafA = cv2.resize(pafA, (frameWidth, frameHeight))
                pafB = cv2.resize(pafB, (frameWidth, frameHeight))

                # Find the keypoints for the first and second limb
                candA = detected_keypoints[POSE_PAIRS[k][0]]
                candB = detected_keypoints[POSE_PAIRS[k][1]]
                nA = len(candA)
                nB = len(candB)

                # If keypoints for the joint-pair is detected
                # check every joint in candA with every joint in candB 
                # Calculate the distance vector between the two joints
                # Find the PAF values at a set of interpolated points between the joints
                # Use the above formula to compute a score to mark the connection valid
                
                if( nA != 0 and nB != 0):
                    valid_pair = np.zeros((0,3))
                    for i in range(nA):
                        max_j=-1
                        maxScore = -1
                        found = 0
                        for j in range(nB):
                            # Find d_ij
                            d_ij = np.subtract(candB[j][:2], candA[i][:2])
                            norm = np.linalg.norm(d_ij)
                            if norm:
                                d_ij = d_ij / norm
                            else:
                                continue
                            # Find p(u)
                            interp_coord = list(zip(np.linspace(candA[i][0], candB[j][0], num=n_interp_samples),
                                                    np.linspace(candA[i][1], candB[j][1], num=n_interp_samples)))
                            # Find L(p(u))
                            paf_interp = []
                            for k in range(len(interp_coord)):
                                paf_interp.append([pafA[int(round(interp_coord[k][1])), int(round(interp_coord[k][0]))],
                                                pafB[int(round(interp_coord[k][1])), int(round(interp_coord[k][0]))] ]) 
                            # Find E
                            paf_scores = np.dot(paf_interp, d_ij)
                            avg_paf_score = sum(paf_scores)/len(paf_scores)
                            
                            # Check if the connection is valid
                            # If the fraction of interpolated vectors aligned with PAF is higher then threshold -> Valid Pair  
                            if ( len(np.where(paf_scores > paf_score_th)[0]) / n_interp_samples ) > conf_th :
                                if avg_paf_score > maxScore:
                                    max_j = j
                                    maxScore = avg_paf_score
                                    found = 1
                        # Append the connection to the list
                        if found:            
                            valid_pair = np.append(valid_pair, [[candA[i][3], candB[max_j][3], maxScore]], axis=0)

                    # Append the detected connections to the global list
                    valid_pairs.append(valid_pair)
                else: # If no keypoints are detected
                    print("No Connection : k = {}".format(k))
                    invalid_pairs.append(k)
                    valid_pairs.append([])
            print(valid_pairs)
            return valid_pairs, invalid_pairs

        # This function creates a list of keypoints belonging to each person
        # For each detected valid pair, it assigns the joint(s) to a person
        # It finds the person and index at which the joint should be added. This can be done since we have an id for each joint
        def getPersonwiseKeypoints(valid_pairs, invalid_pairs):
            # the last number in each row is the overall score 
            personwiseKeypoints = -1 * np.ones((0, 19))

            for k in range(len(mapIdx)):
                if k not in invalid_pairs:
                    partAs = valid_pairs[k][:,0]
                    partBs = valid_pairs[k][:,1]
                    indexA, indexB = np.array(POSE_PAIRS[k])

                    for i in range(len(valid_pairs[k])): 
                        found = 0
                        person_idx = -1
                        for j in range(len(personwiseKeypoints)):
                            if personwiseKeypoints[j][indexA] == partAs[i]:
                                person_idx = j
                                found = 1
                                break

                        if found:
                            personwiseKeypoints[person_idx][indexB] = partBs[i]
                            personwiseKeypoints[person_idx][-1] += keypoints_list[partBs[i].astype(int), 2] + valid_pairs[k][i][2]

                        # if find no partA in the subset, create a new subset
                        elif not found and k < 17:
                            row = -1 * np.ones(19)
                            row[indexA] = partAs[i]
                            row[indexB] = partBs[i]
                            # add the keypoint_scores for the two keypoints and the paf_score 
                            row[-1] = sum(keypoints_list[valid_pairs[k][i,:2].astype(int), 2]) + valid_pairs[k][i][2]
                            personwiseKeypoints = np.vstack([personwiseKeypoints, row])
            return personwiseKeypoints

        # upload image code and processing on it
        uploaded_file = st.file_uploader("",type=["png","jpg","jpeg"])
        if uploaded_file is not None:
            # Convert the file to an opencv image.
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            image1 = cv2.imdecode(file_bytes, 1)
            frameWidth = image1.shape[1]
            frameHeight = image1.shape[0]

            t = time.time()
            net = cv2.dnn.readNetFromCaffe(protoFile, weightsFile)

            # Fix the input Height and get the width according to the Aspect Ratio
            inHeight = 368
            inWidth = int((inHeight/frameHeight)*frameWidth)

            inpBlob = cv2.dnn.blobFromImage(image1, 1.0 / 255, (inWidth, inHeight),
                                    (0, 0, 0), swapRB=False, crop=False)

            net.setInput(inpBlob)
            output = net.forward()
            # print("Time Taken = {}".format(time.time() - t))

            i = 0
            probMap = output[0, i, :, :]
            probMap = cv2.resize(probMap, (frameWidth, frameHeight))
            plt.figure(figsize=[14,10])
            plt.imshow(cv2.cvtColor(image1, cv2.COLOR_BGR2RGB))
            plt.imshow(probMap, alpha=0.6)
            plt.colorbar()
            plt.axis("off")

            
            detected_keypoints = []
            keypoints_list = np.zeros((0,3))
            keypoint_id = 0
            threshold = 0.1

            for part in range(nPoints):
                probMap = output[0,part,:,:]
                probMap = cv2.resize(probMap, (image1.shape[1], image1.shape[0]))
            #     plt.figure()
            #     plt.imshow(255*np.uint8(probMap>threshold))
                keypoints = getKeypoints(probMap, threshold)
                # print("Keypoints - {} : {}".format(keypointsMapping[part], keypoints))
                keypoints_with_id = []
                for i in range(len(keypoints)):
                    keypoints_with_id.append(keypoints[i] + (keypoint_id,))
                    keypoints_list = np.vstack([keypoints_list, keypoints[i]])
                    keypoint_id += 1

                detected_keypoints.append(keypoints_with_id)

            frameClone = image1.copy()
            for i in range(nPoints):
                for j in range(len(detected_keypoints[i])):
                    cv2.circle(frameClone, detected_keypoints[i][j][0:2], 3, [0,0,255], -1, cv2.LINE_AA)
            plt.figure(figsize=[15,15])
            plt.imshow(frameClone[:,:,[2,1,0]])

            valid_pairs, invalid_pairs = getValidPairs(output)
            personwiseKeypoints = getPersonwiseKeypoints(valid_pairs, invalid_pairs)

            for i in range(17):
                for n in range(len(personwiseKeypoints)):
                    index = personwiseKeypoints[n][np.array(POSE_PAIRS[i])]
                    if -1 in index:
                        continue
                    B = np.int32(keypoints_list[index.astype(int), 0])
                    A = np.int32(keypoints_list[index.astype(int), 1])
                    cv2.line(frameClone, (B[0], A[0]), (B[1], A[1]), colors[i], 3, cv2.LINE_AA)
            my_bar = st.progress(0)
            for percent_complete in range(100):
                time.sleep(0.1)
                my_bar.progress(percent_complete + 1)
            st.image(frameClone, channels="BGR")


        ### Here i am here working on Real time on webcam and model is Tensorflow(graph_opt.pb)

        st.markdown("### To Check out Multi Person Estimation click the button below :handball: :man-man-girl: ")
        


        parser = argparse.ArgumentParser(
                description='This script is used to demonstrate OpenPose human pose estimation network '
                            'from https://github.com/CMU-Perceptual-Computing-Lab/openpose project using OpenCV. '
                            'The sample and model are simplified and could be used for a single person on the frame.')
        parser.add_argument('--input', #default= "sample1.jpg",
                            help='Path to image or video. Skip to capture frames from camera')
        parser.add_argument('--proto', help='Path to .prototxt')
        parser.add_argument('--model', default="openpose/graph_opt.pb", help='Path to .caffemodel')
        parser.add_argument('--dataset',default="COCO" , help='Specify what kind of model was trained. '
                                            'It could be (COCO, MPI) depends on dataset.')
        parser.add_argument('--thr', default=0.2, type=float, help='Threshold value for pose parts heat map')
        parser.add_argument('--width', default=386, type=int, help='Resize input to specific width.')
        parser.add_argument('--height', default=386, type=int, help='Resize input to specific height.')
        parser.add_argument('--inf_engine', action='store_true',
                            help='Enable Intel Inference Engine computational backend. '
                                'Check that plugins folder is in LD_LIBRARY_PATH environment variable')

        args = parser.parse_args()

        e = estimator.PoseEstimator()
        ok = st.button("Openpose")
        if ok: 
            cap = cv2.VideoCapture(0)
            while cap.isOpened():
                ret,frame = cap.read()

                t = time.time()
                humans = e.inference(frame,args.model,args.width,args.height)

                elapsed = time.time() - t
                print('inference image: %s in %.4f seconds.' % (args.input, elapsed))

                image = e.draw_humans(frame, humans, imgcopy=False)
                cv2.imshow('tf-pose-estimation result', image)

                if cv2.waitKey(10) & 0xFF == ord('q'):
                    break

            cap.release()
            cv2.destroyAllWindows()
        st.write("Quit Webcam - Press Q :octagonal_sign:")



    with pre:
        st.sidebar.markdown("<h1 style='color:Red ;'>References</h1>", unsafe_allow_html=True)
        st.sidebar.markdown("[Realtime Multi-Person 2D Pose Estimation using Part Affinity Fields](https://arxiv.org/abs/1611.08050)")
        st.sidebar.markdown("[Understanding OpenPose](https://medium.com/analytics-vidhya/understanding-openpose-with-code-reference-part-1-b515ba0bbc73)")
        st.sidebar.markdown("[Multi Person Pose Estimation in OpenCV using OpenPose](https://learnopencv.com/multi-person-pose-estimation-in-opencv-using-openpose/)")
        st.sidebar.markdown("[How does Pose Estimation Work](https://www.youtube.com/watch?v=utz4Ql0CkBE&t=463s)")


        st.sidebar.markdown("<h2 style='color:Red ;'>If you like the Content Press Button and Enjoy: </h2>", unsafe_allow_html=True)
        if st.sidebar.button("Press if You Like"):
            st.balloons() 