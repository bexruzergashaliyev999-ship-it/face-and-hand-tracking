# Computer Vision with MediaPipe & OpenCV

A small collection of real-time computer vision demos built with **Python**, **OpenCV** and **Google MediaPipe**. Each project reads frames from a webcam, runs a MediaPipe model on every frame and draws the results live with an FPS counter. Everything runs on CPU, no GPU required.

## Projects

| Project | Folder | Model | What it does |
|---------|--------|-------|--------------|
| [Face Detection](#1-face-detection) | `face_detection/` | BlazeFace (short range) | Bounding boxes, 6 facial keypoints, confidence score |
| [Hand Tracking](#2-hand-tracking) | `hand_tracking/` | Hand Landmarker | 21 landmarks per hand, skeleton lines, up to 2 hands |

## Tech Stack

| Component | Purpose |
|-----------|---------|
| Python 3.9+ | Core language |
| [MediaPipe Tasks](https://developers.google.com/mediapipe) | Face detection and hand landmark models |
| [OpenCV](https://opencv.org/) | Video capture, image processing, drawing, display |

## Repository Structure

```
.
├── face_detection/
│   └── face_detection.py
├── hand_tracking/
│   └── hand_tracking.py
├── requirements.txt
├── .gitignore
└── README.md
```

## Installation

```bash
# 1. Clone the repository
git clone https://github.com/<your-username>/<your-repo>.git
cd <your-repo>

# 2. (Recommended) Create a virtual environment
python -m venv venv
source venv/bin/activate        # Linux / macOS
venv\Scripts\activate           # Windows

# 3. Install dependencies
pip install -r requirements.txt
```

### Download the models

The models are not stored in this repository. Download the one you need and place it in the same folder as its script:

| Project | File name | Download |
|---------|-----------|----------|
| Face Detection | `blaze_face_short_range.tflite` | [Download](https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/float16/1/blaze_face_short_range.tflite) |
| Hand Tracking | `hand_landmarker.task` | [Download](https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task) |

Alternatively, uncomment the `urllib.request` download block at the top of a script to fetch its model automatically on the first run.

---

## 1. Face Detection

Detects faces in a live webcam stream using the BlazeFace model.

**Features**
- Bounding box around every detected face
- Facial keypoints (eyes, nose, mouth, ears) drawn as red dots
- Confidence score (%) shown for each face
- Live FPS counter

**Run**
```bash
cd face_detection
python face_detection.py
```

---

## 2. Hand Tracking

Tracks hand landmarks in a live webcam stream using the MediaPipe Hand Landmarker.

**Features**
- Detects up to **2 hands** simultaneously
- **21 landmarks** per hand drawn as red dots
- Green skeleton lines connecting the landmarks
- Live FPS counter

**Run**
```bash
cd hand_tracking
python hand_tracking.py
```

---

## How It Works

1. **Model loading**: the script looks for its model file next to itself.
2. **Frame capture**: OpenCV reads frames from the default webcam (index `0`).
3. **Preprocessing**: each frame is converted from BGR to RGB, the format MediaPipe expects.
4. **Inference**: MediaPipe returns face boxes and keypoints, or hand landmarks, as normalized coordinates.
5. **Visualization**: coordinates are scaled to pixel positions, drawn on the frame and shown in a window.

```
Webcam -> OpenCV frame -> BGR to RGB -> MediaPipe model -> Draw results -> Display
```

Press **`q`** in the video window to quit any demo.

## Configuration

| What to change | Where |
|----------------|-------|
| Camera source | `cv2.VideoCapture(0)`: use `1`, `2`, ... for other cameras or a video file path |
| Number of hands | `num_hands=2` in `HandLandmarkerOptions` |
| Colors | The BGR tuples in the `cv2.rectangle`, `cv2.line` and `cv2.circle` calls |
| Minimum confidence | Add `min_detection_confidence=0.5` to the options of either model |

## Possible Improvements

- Face blurring / anonymization for privacy
- Gesture recognition (for example counting fingers) on top of hand landmarks
- Saving annotated video or snapshots
- Face recognition on top of detection
- Packaging as a REST API or desktop GUI

## Troubleshooting

- **Camera does not open**: make sure no other app is using it, or try a different camera index.
- **Model file not found**: check that the model file is in the same folder as the script and the name matches exactly.
- **`ModuleNotFoundError`**: re-run `pip install -r requirements.txt` inside your virtual environment.
- **Low FPS**: reduce the frame resolution with `cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)`.

## Acknowledgements

- [Google MediaPipe](https://developers.google.com/mediapipe) for the models and Tasks API
- [OpenCV](https://opencv.org/) for computer vision utilities

## License

This project is licensed under the MIT License. Feel free to use, modify and distribute it.
