import cv2
from deepface import DeepFace
import numpy as np

# Start webcam
cap = cv2.VideoCapture(0)

print("Starting emotion detection. Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Resize for performance
    resized_frame = cv2.resize(frame, (640, 480))

    try:
        # Analyze emotions
        result = DeepFace.analyze(resized_frame, actions=['emotion'], enforce_detection=False)

        # Draw the results
        dominant_emotion = result[0]['dominant_emotion']
        cv2.putText(resized_frame, f'Emotion: {dominant_emotion}', (30, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

    except Exception as e:
        print("Face not detected")

    # Display the video feed
    cv2.imshow("Emotion Detector", resized_frame)

    # Exit on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()