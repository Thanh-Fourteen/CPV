import cv2
import mediapipe as mp
import time
import cvzone

class poseDetector():
    def __init__(self, mode = False, upBody = False, smooth = True,
                 detectionCon = 0.5, trackCon = 0.5):
        self.mode = mode
        self.upBody = upBody
        self.smooth = smooth
        self.detectionCon = detectionCon
        self.trackCon = trackCon

        self.mpDraw = mp.solutions.drawing_utils
        self.mpPose = mp.solutions.pose
        # self.pose = self.mpPose.Pose(self.mode, self.upBody, self.smooth,
        #                              self.detectionCon, self.trackCon)
        self.pose = self.mpPose.Pose(
            static_image_mode=self.mode,
            enable_segmentation=self.upBody,  # @TODO:
            smooth_segmentation=self.smooth, 
            min_detection_confidence=self.detectionCon, 
            min_tracking_confidence=self.trackCon)
    
    def findPose(self, img, draw = True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.pose.process(imgRGB)
        if self.results.pose_landmarks:
            if draw:
                self.mpDraw.draw_landmarks(img, self.results.pose_landmarks, self.mpPose.POSE_CONNECTIONS)
        return img   
    
    def findPosition(self, img, draw = True):
        lmList = []
        if self.results.pose_landmarks:
            for id, lm in enumerate(self.results.pose_landmarks.landmark):
                h, w, _ = img.shape
                cx, cy = int(lm.x*w), int(lm.y*h)
                lmList.append([id, cx, cy])
                if draw:
                    cv2.circle(img, (cx, cy), 5, (255,0,0), cv2.FILLED)
        return lmList

def main():
    cap = cv2.VideoCapture("dance3.mp4")
    pTime = 0
    cTime = 0

    detector = poseDetector()

    while 1:
        ret, img = cap.read()
        if ret == False: 
            cap.release()
            cv2.destroyAllWindows()
            break

        img = detector.findPose(img)
        lmList = detector.findPosition(img, False)
        if len(lmList) != 0:
            cv2.circle(img, (lmList[0][1], lmList[0][2]), 5, (255,0,0), cv2.FILLED)


        cTime = time.time()
        fps = 1/(cTime-pTime)
        pTime = cTime

        cv2.putText(img, str(int(fps)), (70,50), cv2.FONT_HERSHEY_PLAIN, 3
                    , (255, 0, 0), 3)

        cv2.imshow("img", img)
        if cv2.waitKey(10) & 0xFF == ord("q"):
            cap.release()
            cv2.destroyAllWindows()
            break


if __name__ == "__main__":
    main()