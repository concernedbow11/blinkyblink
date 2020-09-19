from scipy.spatial import distance as dist
import numpy as np
import time
import dlib
import cv2
import time
import sys
from PyQt5 import QtCore, QtGui, QtWidgets

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
        key = cv2.waitKey(1)
        count += 1

    vs.release()    #release the video stream
    out.release()
    cv2.destroyAllWindows()
    # define two values, one for the eye aspect ratio to indicate
    # blink and then a second constant for the number of consecutive
    # frames the eye must be below the threshold

    try:
        global EYE_AR_CONSEC_FRAMES
        global EYE_AR_THRESH
        EYE_AR_THRESH = sum(ears)/len(ears)     #an average EAR is calculated from the recorded video
        EYE_AR_THRESH= 0.95 * EYE_AR_THRESH     #multiplied by a tested sensitivity correction
        EYE_AR_CONSEC_FRAMES = 3
    except(ZeroDivisionError):
        print("Please check your lighting conditions or camera in case no output is visible.")
    #else:
     #   print("Something else went wrong")
    

TOTAL = 0

def detectBlink():
    # initialize the frame counters and the total number of blinks
    COUNTER = 0
    global TOTAL

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
    

class Ui_Dialog(object):
    def update_digit(self):
        _translate = QtCore.QCoreApplication.translate
        self.lcdNumber.intValue = TOTAL
        self.lcdNumber.display(TOTAL)
        self.label_3.setText(_translate("Dialog", str(EYE_AR_THRESH)))
    def setupUi(self, Dialog):
        Dialog.setObjectName("Dialog")
        Dialog.resize(400, 286)
        self.frame = QtWidgets.QFrame(Dialog)
        self.frame.setGeometry(QtCore.QRect(0, 0, 401, 281))
        self.frame.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame.setObjectName("frame")
        self.lcdNumber = QtWidgets.QLCDNumber(self.frame)
        self.lcdNumber.setGeometry(QtCore.QRect(220, 20, 151, 71))
        self.lcdNumber.setObjectName("lcdNumber")
        self.lcdNumber.intValue = TOTAL
        self.label = QtWidgets.QLabel(self.frame)
        self.label.setGeometry(QtCore.QRect(20, 20, 131, 61))
        self.label.setObjectName("label")
        self.label_2 = QtWidgets.QLabel(self.frame)
        self.label_2.setGeometry(QtCore.QRect(20, 130, 71, 20))
        self.label_2.setObjectName("label_2")
        self.label_3 = QtWidgets.QLabel(self.frame)
        self.label_3.setGeometry(QtCore.QRect(100, 130, 51, 21))
        self.label_3.setObjectName("label_3")
        self.buttonBox = QtWidgets.QDialogButtonBox(self.frame)
        self.buttonBox.setGeometry(QtCore.QRect(30, 240, 341, 32))
        self.buttonBox.setOrientation(QtCore.Qt.Horizontal)
        self.buttonBox.setStandardButtons(QtWidgets.QDialogButtonBox.Ok)
        self.buttonBox.setObjectName("buttonBox")

        self.retranslateUi(Dialog)
        self.buttonBox.accepted.connect(Dialog.accept)
        self.buttonBox.rejected.connect(Dialog.reject)
        QtCore.QMetaObject.connectSlotsByName(Dialog)

    def retranslateUi(self, Dialog):
        _translate = QtCore.QCoreApplication.translate
        Dialog.setWindowTitle(_translate("Dialog", "Dialog"))
        self.label.setText(_translate("Dialog", "<html><head/><body><p><span style=\" font-size:12pt;\">Blinks:</span></p></body></html>"))
        self.label_2.setText(_translate("Dialog", "Average EAR: "))
        self.label_3.setText(_translate("Dialog", str(EYE_AR_THRESH)))

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(814, 687)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.videoframe = QtWidgets.QFrame(self.centralwidget)
        self.videoframe.setGeometry(QtCore.QRect(0, 0, 401, 291))
        self.videoframe.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.videoframe.setFrameShadow(QtWidgets.QFrame.Raised)
        self.videoframe.setObjectName("videoframe")
        self.recordtitle = QtWidgets.QLabel(self.videoframe)
        self.recordtitle.setGeometry(QtCore.QRect(20, 20, 305, 35))
        font = QtGui.QFont()
        font.setPointSize(22)
        self.recordtitle.setFont(font)
        self.recordtitle.setScaledContents(False)
        self.recordtitle.setObjectName("recordtitle")
        self.recdesc = QtWidgets.QLabel(self.videoframe)
        self.recdesc.setGeometry(QtCore.QRect(20, 70, 239, 48))
        font = QtGui.QFont()
        font.setPointSize(10)
        self.recdesc.setFont(font)
        self.recdesc.setWordWrap(True)
        self.recdesc.setObjectName("recdesc")
        self.record = QtWidgets.QPushButton(self.videoframe)
        self.record.clicked.connect(captureVid)

        self.record.setGeometry(QtCore.QRect(20, 160, 101, 41))
        self.record.setObjectName("record")
        self.frame = QtWidgets.QFrame(self.centralwidget)
        self.frame.setGeometry(QtCore.QRect(410, 0, 401, 291))
        self.frame.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame.setObjectName("frame")
        self.detdesc = QtWidgets.QLabel(self.frame)
        self.detdesc.setGeometry(QtCore.QRect(98, 70, 281, 49))
        font = QtGui.QFont()
        font.setPointSize(10)
        self.detdesc.setFont(font)
        self.detdesc.setWordWrap(True)
        self.detdesc.setObjectName("detdesc")
        self.detecttitle = QtWidgets.QLabel(self.frame)
        self.detecttitle.setGeometry(QtCore.QRect(210, 20, 166, 35))
        font = QtGui.QFont()
        font.setPointSize(22)
        self.detecttitle.setFont(font)
        self.detecttitle.setLayoutDirection(QtCore.Qt.RightToLeft)
        self.detecttitle.setAutoFillBackground(False)
        self.detecttitle.setScaledContents(False)
        self.detecttitle.setObjectName("detecttitle")
        self.detect = QtWidgets.QPushButton(self.frame)
        self.detect.clicked.connect(detectBlink)
        

        self.detect.setGeometry(QtCore.QRect(300, 160, 91, 41))
        self.detect.setObjectName("detect")
        self.frame_2 = QtWidgets.QFrame(self.centralwidget)
        self.frame_2.setGeometry(QtCore.QRect(0, 290, 811, 351))
        self.frame_2.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_2.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_2.setObjectName("frame_2")
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 814, 21))
        self.menubar.setObjectName("menubar")
        self.menuFile = QtWidgets.QMenu(self.menubar)
        self.menuFile.setObjectName("menuFile")
        self.menuHelp = QtWidgets.QMenu(self.menubar)
        self.menuHelp.setObjectName("menuHelp")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)
        self.actionOpen_source_folder = QtWidgets.QAction(MainWindow)
        self.actionOpen_source_folder.setObjectName("actionOpen_source_folder")
        self.menuFile.addAction(self.actionOpen_source_folder)
        self.menubar.addAction(self.menuFile.menuAction())
        self.menubar.addAction(self.menuHelp.menuAction())

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.recordtitle.setText(_translate("MainWindow", "Record yourself blinking"))
        self.recdesc.setText(_translate("MainWindow", "Recording will begin as soon as you press the button, get ready to blink. You will have 4 seconds."))
        self.record.setText(_translate("MainWindow", "Record"))
        self.detdesc.setText(_translate("MainWindow", "<html><head/><body><p align=\"right\">This will playback the recorded video and count the blinks as well as display the average EAR</p></body></html>"))
        self.detecttitle.setText(_translate("MainWindow", "<html><head/><body><p align=\"right\">Detect Blinks</p></body></html>"))
        self.detect.setText(_translate("MainWindow", "Detect"))
        self.menuFile.setTitle(_translate("MainWindow", "File"))
        self.menuHelp.setTitle(_translate("MainWindow", "Help"))
        self.actionOpen_source_folder.setText(_translate("MainWindow", "Open source folder"))


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    Dialog = QtWidgets.QDialog()
    box = Ui_Dialog()
    box.setupUi(Dialog)
    Dialog.show()
    timer = QtCore.QTimer()
    timer.timeout.connect(box.update_digit)
    timer.start(10)  # every 10 milliseconds
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())