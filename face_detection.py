import os
import time

import cv2
import mediapipe as mp
from mediapipe.tasks.python import vision, BaseOptions

folder = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(folder, "blaze_face_short_range.tflite")

# Download the model (uncomment if needed):
# import urllib.request
# urllib.request.urlretrieve(
#     "https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/float16/1/blaze_face_short_range.tflite",
#     model_path,
# )

if not os.path.exists(model_path):
    print("Model file not found")
    raise SystemExit
print("Model ready!")

face_detector = vision.FaceDetector.create_from_options(
    vision.FaceDetectorOptions(base_options=BaseOptions(model_asset_path=model_path))
)

cap = cv2.VideoCapture(0)
prev_time = 0

while True:
    success, img = cap.read()
    if not success:
        break

    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    result = face_detector.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))

    h, w, _ = img.shape
    for face in result.detections:
        box = face.bounding_box
        cv2.rectangle(
            img,
            (box.origin_x, box.origin_y),
            (box.origin_x + box.width, box.origin_y + box.height),
            (0, 255, 0),
            3,
        )

        for keypoint in face.keypoints:
            cv2.circle(img, (int(keypoint.x * w), int(keypoint.y * h)), 6, (0, 0, 255), cv2.FILLED)

        score = int(face.categories[0].score * 100)
        cv2.putText(img, f"{score}%", (box.origin_x, max(box.origin_y - 10, 20)),
                    cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)

    now = time.time()
    fps = 1 / (now - prev_time)
    prev_time = now
    cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)

    cv2.imshow("Image", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
