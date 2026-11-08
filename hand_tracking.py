import os
import time
import urllib.request

import cv2
import mediapipe as mp
from mediapipe.tasks.python import vision, BaseOptions

folder = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(folder, "hand_landmarker.task")

# if not os.path.exists(model_path):
#     print("Downloading model...")
#     urllib.request.urlretrieve(
#         "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",
#         model_path,
#     )
#     print("Model downloaded.")
#

if not os.path.exists(model_path):
    print("Model file not found")
    raise SystemExit
print("Model ready!")

hand_landmarker = vision.HandLandmarker.create_from_options(
    vision.HandLandmarkerOptions(base_options=BaseOptions(model_asset_path=model_path), num_hands=2)
)

cap = cv2.VideoCapture(0)
prev_time = 0

while True:
    success, img = cap.read()
    if not success:
        break

    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    result = hand_landmarker.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))

    h, w, _ = img.shape
    for hand in result.hand_landmarks:
        for connection in vision.HandLandmarksConnections.HAND_CONNECTIONS:
            a = hand[connection.start]
            b = hand[connection.end]
            cv2.line(img, (int(a.x * w), int(a.y * h)), (int(b.x * w), int(b.y * h)), (0, 255, 0), 3)

        for landmark in hand:
            cv2.circle(img, (int(landmark.x * w), int(landmark.y * h)), 6, (0, 0, 255), cv2.FILLED)

    now = time.time()
    fps = 1 / (now - prev_time)
    prev_time = now
    cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)

    cv2.imshow("Image", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
