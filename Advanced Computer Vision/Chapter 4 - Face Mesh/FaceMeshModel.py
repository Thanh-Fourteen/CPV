import cv2
import mediapipe as mp
import time

class FaceMeshDetector():
    def __init__(self, staticMode = False, maxFaces = 2, minDetectionCon = 0.5, minTrackCon = 0.5) :
        self.staticMode = staticMode
        self.maxFaces = maxFaces
        self.minDetectionCon = minDetectionCon
        self.minTrackCon = minTrackCon

        self.mpDraw = mp.solutions.drawing_utils
        self.mpFaceMesh = mp.solutions.face_mesh
        self.faceMesh = self.mpFaceMesh.FaceMesh( static_image_mode=self.staticMode,
                                                    max_num_faces=self.maxFaces,
                                                    refine_landmarks=False,
                                                    min_detection_confidence=self.minDetectionCon,
                                                    min_tracking_confidence=self.minTrackCon)
        self.drawSpec = self.mpDraw.DrawingSpec(thickness=1, circle_radius=1)

    def findFaceMesh(self, img, draw = True):
        self.imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.faceMesh.process(self.imgRGB)
        faces = []
        if self.results.multi_face_landmarks:
            for faceLms in self.results.multi_face_landmarks:
                if draw:
                    # mpDraw.draw_landmarks(img, faceLms, mpFaceMesh.FACEMESH_TESSELATION)
                    self.mpDraw.draw_landmarks(img, faceLms, self.mpFaceMesh.FACEMESH_CONTOURS,
                                        self.drawSpec, self.drawSpec)
                face = []
                for id, lm in enumerate(faceLms.landmark):
                    ih, iw, _ = img.shape
                    x, y = int(lm.x * iw), int(lm.y * ih)
                    # cv2.putText(img, f"{id}", (x, y), cv2.FONT_HERSHEY_PLAIN,
                    # 0.5, (0, 255, 0), 1)
                    face.append([x,y])
                faces.append(face)
        return img, faces


def main():
    cap = cv2.VideoCapture(0)
    # cap = cv2.VideoCapture("faces.mp4")
    pTime = 0

    detector = FaceMeshDetector()

    while 1:
        ret, img = cap.read()
        if ret == False:
            cap.release()
            cv2.destroyAllWindows()
            break

        img, faces = detector.findFaceMesh(img)
        if len(faces) != 0:
            print(len(faces))

        cTime = time.time()
        fps = 1 /(cTime - pTime)
        pTime = cTime
        cv2.putText(img, f"FPS: {int(fps)}", (20, 70), cv2.FONT_HERSHEY_PLAIN,
                    2, (0, 255, 0), 2)

        cv2.imshow("img", img)
        if cv2.waitKey(10)  & 0xFF == ord("q"):
            cap.release()
            cv2.destroyAllWindows()
            break

if __name__ == "__main__":
    main()