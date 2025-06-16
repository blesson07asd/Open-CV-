"""
Human-body detector + Pushove r alert
Python 3.11 | OpenCV 4.x | MediaPipe >= 0.10
"""

import cv2
import mediapipe as mp
import requests
import time

# ── 0. Pushover configuration 
PUSHOVER_APP_TOKEN = " replace "     # ← replace
PUSHOVER_USER_KEY  = "replace"      # ← replace
PUSHOVER_SOUND     = "replace"                   # any built-in sound name
ALERT_COOLDOWN_SEC = 2                      # minimum seconds between alerts

def send_pushover_alert(message: str) -> None:
    """Send a push notification with sound via Pushover."""
    try:
        resp = requests.post(
            "https://api.pushover.net/1/messages.json",
            data={
                "token":   PUSHOVER_APP_TOKEN,
                "user":    PUSHOVER_USER_KEY,
                "message": message,
                "title":   "Camera Alert",
                "sound":   PUSHOVER_SOUND,
                "priority": 0
            },
            timeout=3
        )
        resp.raise_for_status()
        print("Pushover alert sent ✓")
    except Exception as err:
        print(f"Pushover alert failed: {err}")

# ── 1. Initialize MediaPipe Pose ──────────────────────────────────────────────
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False,
                    model_complexity=1,
                    enable_segmentation=False,
                    min_detection_confidence=0.5)

mp_draw = mp.solutions.drawing_utils

# ── 2. Open webcam ────────────────────────────────────────────────────────────
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("Could not open webcam")

# Full-screen window
WIN_NAME = "Human Body Drawing Utility"
cv2.namedWindow(WIN_NAME, cv2.WND_PROP_FULLSCREEN)
cv2.setWindowProperty(WIN_NAME, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

last_alert_time = 0  # Unix timestamp of last push

# ── 3. Main loop ──────────────────────────────────────────────────────────────
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)                      # mirror view
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Detect pose landmarks
    results = pose.process(frame_rgb)

    body_detected = results.pose_landmarks is not None

    # Draw landmarks if present
    if body_detected:
        mp_draw.draw_landmarks(
            frame,
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=mp_draw.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=4),
            connection_drawing_spec=mp_draw.DrawingSpec(color=(255, 0, 0), thickness=2)
        )

    # ── 3-a. Trigger push alert (with cool-down) ────────────────────────────
    now = time.time()
    if body_detected and now - last_alert_time > ALERT_COOLDOWN_SEC:
        send_pushover_alert("Human detected by camera")
        last_alert_time = now

    # Display video
    cv2.imshow(WIN_NAME, frame)

    # Quit on 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# ── 4. Clean-up ───────────────────────────────────────────────────────────────
cap.release()
cv2.destroyAllWindows()
pose.close()
