"""
##Real-time hand-landmark detection with MediaPipe
##Python 3.11 | mediapipe >= 0.10  |OpenCV 4.x
"""

import cv2
import mediapipe as mp

#  1. Initialize MediaPipe Hands 
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,        # input is stream/video, not still photos
    max_num_hands=4,                # detect up to two hands
    min_detection_confidence=0.5,   # threshold for the palm detector
    min_tracking_confidence=0.5     # threshold for landmark tracker
)
mp_draw = mp.solutions.drawing_utils

# Open the default webcam 
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("Webcam could not be opened.")

#  Main loop 
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # MediaPipe expects RGB images
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # 3-a. Process the image and find hands
    results = hands.process(img_rgb)

    # 3-b. Draw landmarks and connections
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_draw.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=3),
                mp_draw.DrawingSpec(color=(255, 0, 0), thickness=2)
            )

    # 3-c. Show the annotated frame
    cv2.imshow("MediaPipe Hands", frame)

    # 3-d. Exit when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# ── 4. Clean-up ────────────────────────────────────────────────────────────────
cap.release()
cv2.destroyAllWindows()
hands.close()
