from scipy.spatial import distance as dist
import numpy as np
import time
import dlib
import cv2
import time

def shape_to_np(shape, dtype = int):
    # initialize the list of (x, y)-coordinates
    coords = np.zeros((shape.num_parts, 2), dtype=dtype)

    # loop over all facial landmarks and convert them
    # to a 2-tuple of (x, y)-coordinates
    for i in range(0, shape.num_parts):
        coords[i] = (shape.part(i).x, shape.part(i).y)

    # return the list of (x, y)-coordinates
    return coords

def eye_aspect_ratio(eye):
    # compute the euclidean distances between the two sets of
    # vertical eye landmarks (x, y)-coordinates
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])

    # compute the euclidean distance between the horizontal
    # eye landmark (x, y)-coordinates
    C = dist.euclidean(eye[0], eye[3])

    # compute the eye aspect ratio
    ear = (A + B) / (2.0 * C)

    # return the eye aspect ratio
    return ear
 
# initialize dlib's face detector (HOG-based) and then create
# the facial landmark predictor
print("[INFO] loading facial landmark predictor...")
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

# grab the indexes of the facial landmarks for the left and right eye, respectively

(lStart, lEnd) = (42, 48)
(rStart, rEnd) = (36, 42)

frame_width = 0
frame_height = 0
EYE_AR_THRESH = 0
EYE_AR_CONSEC_FRAMES = 0

def captureVid():
    vs = cv2.VideoCapture(0)
    global frame_height
    global frame_width
    frame_width = int(vs.get(3))
    frame_height = int(vs.get(4))
    ears = []
    out = cv2.VideoWriter('output.avi',cv2.VideoWriter_fourcc(*'XVID'), 10, (frame_width,frame_height))

    count = 0

    while count < 40:
        ret, frame = vs.read()
        
        frame= cv2.resize(frame, (frame_width,frame_height),fx=0.5,fy=0.5)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        if ret == True:  
        # Write the frame into the file 'output.avi'
            out.write(frame)
        # Display the resulting frame    
        cv2.imshow('frame',frame)
        rects = detector(gray, 0)
        for rect in rects:
                shape = predictor(gray, rect)
                shape = shape_to_np(shape)
                leftEye = shape[lStart:lEnd]
                rightEye = shape[rStart:rEnd]
                leftEAR = eye_aspect_ratio(leftEye)
                rightEAR = eye_aspect_ratio(rightEye)
                ear = (leftEAR + rightEAR) / 2.0
                ears.append(ear)         
        key = cv2.waitKey(1) & 0xFF
        count += 1

    vs.release()    #release the video stream

    # define two values, one for the eye aspect ratio to indicate
    # blink and then a second constant for the number of consecutive
    # frames the eye must be below the threshold

    try:
        global EYE_AR_CONSEC_FRAMES
        global EYE_AR_THRESH
        EYE_AR_THRESH = sum(ears)/len(ears)     #an average EAR is calculated from the recorded video
        EYE_AR_THRESH= 1.005 * EYE_AR_THRESH     #multiplied by a tested sensitivity correction
        EYE_AR_CONSEC_FRAMES = 3
    except(ZeroDivisionError):
        print("Please check your lighting conditions or camera in case no output is visible.")
    else:
        print("Something else went wrong")


def detectBlink():
    # initialize the frame counters and the total number of blinks
    COUNTER = 0
    TOTAL = 0

    # start the video stream thread
    print("[INFO] starting video stream thread...")

    cap = cv2.VideoCapture("output.avi")
    print("file capturing has begun ")

    if(cap.isOpened() == False):
            print("There was an error opening the video")

    # loop over frames from the video file stream
    while (cap.isOpened()):

        # grab the frame from the threaded video file stream, resize it, and convert it to grayscale channels)
        ret, frame = cap.read()
        if ret == True:
            frame= cv2.resize(frame, (frame_width,frame_height),fx=0.5,fy=0.5)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # detect faces in the grayscale frame
            rects = detector(gray, 0)

            # loop over the face detections
            for rect in rects:
                
                # determine the facial landmarks for the face region, then convert the facial landmark (x, y)-coordinates to a NumPy array
                shape = predictor(gray, rect)
                shape = shape_to_np(shape)
                
                # extract the left and right eye coordinates, then use the coordinates to compute the eye aspect ratio for both eyes
                leftEye = shape[lStart:lEnd]
                rightEye = shape[rStart:rEnd]
                leftEAR = eye_aspect_ratio(leftEye)
                rightEAR = eye_aspect_ratio(rightEye)

                # average the eye aspect ratio together for both eyes
                ear = (leftEAR + rightEAR) / 2.0

                # compute the convex hull for the left and right eye, then visualize each of the eyes
                leftEyeHull = cv2.convexHull(leftEye)
                rightEyeHull = cv2.convexHull(rightEye)
                cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
                cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)

                # check to see if the eye aspect ratio is below the blink threshold, and if so, increment the blink frame counter
                if ear < EYE_AR_THRESH:
                    COUNTER += 1


                # otherwise, the eye aspect ratio is not below the blink threshold
                else:
                # if the eyes were closed for a sufficient number of
                # then increment the total number of blinks
                    if COUNTER >= EYE_AR_CONSEC_FRAMES:
                        TOTAL += 1

                    # reset the eye frame counter
                    COUNTER = 0

                # draw the total number of blinks on the frame along with the computed eye aspect ratio for the frame
                cv2.putText(frame, "Blinks: {}".format(TOTAL), (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                cv2.putText(frame, "EAR: {:.2f}".format(ear), (300, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2);print("Eye Blinks =", TOTAL)
                cv2.putText(frame,"Calculated avg EAR threshold: {:.2f}".format(EYE_AR_THRESH), (300,60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            # show the frame
            cv2.imshow("Frame", frame)
            key = cv2.waitKey(1)
        else:
            break

    print("The number of blinks = ",TOTAL)
    #print("Number of frames shown: ",count)
    print("Average EAR Threshold: ",EYE_AR_THRESH)

    # do a bit of cleanup
    cap.release()
    print("end")
    cv2.destroyAllWindows()