import cv2
import mediapipe as mp
import time

class FaceDetection():
    def __init__(self, minDetectionCon = 0.5):
        self.minDetectionCon = minDetectionCon
        self.mpFaceDetection = mp.solutions.face_detection
        self.mpDraw = mp.solutions.drawing_utils
        self.faceDetection = self.mpFaceDetection.FaceDetection(self.minDetectionCon)

    def findFaces(self, img, draw = True):
        imgRBG = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.faceDetection.process(imgRBG)
        bboxs = []
        if self.results.detections:
            for id, detection in enumerate(self.results.detections):
                bboxC = detection.location_data.relative_bounding_box
                ih, iw, _ = img.shape
                bbox = int(bboxC.xmin * iw), int(bboxC.ymin * ih), \
                    int(bboxC.width * iw), int(bboxC.height * ih)
                bboxs.append([id, bbox, detection.score])

                if draw:
                    img = self.fancyDraw(img, bbox)
                    cv2.putText(img, f"{int(detection.score[0] * 100)}%", 
                                (bbox[0],bbox[1] - 20), cv2.FONT_HERSHEY_PLAIN,
                        3, (255, 0, 255), 2)
        
        return img, bbox

    def fancyDraw(self, img, bbox, l = 30, t = 5, rt = 1):
        x, y, w, h = bbox
        x1, y1 = x + w, y + h

        cv2.rectangle(img, bbox, (255, 0, 255), rt)

        # Top left x,y
        cv2.line(img, (x, y), (x + l, y), (255, 0, 255), t)
        cv2.line(img, (x, y), (x, y + l) , (255, 0, 255), t)

        # Top right x1,y
        cv2.line(img, (x1, y), (x1 - l, y), (255, 0, 255), t)
        cv2.line(img, (x1, y), (x1, y + l) , (255, 0, 255), t)

        # Bottom left x,y1
        cv2.line(img, (x, y1), (x + l, y1), (255, 0, 255), t)
        cv2.line(img, (x, y1), (x, y1 - l) , (255, 0, 255), t)

        # Bottom right x1,y1
        cv2.line(img, (x1, y1), (x1 - l, y1), (255, 0, 255), t)
        cv2.line(img, (x1, y1), (x1, y1 - l) , (255, 0, 255), t)
        return img
        
def main():
    cap = cv2.VideoCapture(0)
    pTime = 0
    detector = FaceDetection(0.4)

    while 1:
        ret, img = cap.read()
        if ret == False:
            cap.release()
            cv2.destroyAllWindows()
            break
        img, bbox = detector.findFaces(img)

        cTime = time.time()
        fps = 1/(cTime-pTime)
        pTime = cTime
        cv2.putText(img, f"FPS: {int(fps)}", (20,70), cv2.FONT_HERSHEY_PLAIN,
                    3, (0, 255, 0), 2)

        cv2.imshow("img", img)
        if cv2.waitKey(10) == ord("q"):
            cap.release()
            cv2.destroyAllWindows()
            break

if __name__ == "__main__":
    main()