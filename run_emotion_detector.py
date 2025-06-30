import cv2
from deepface import DeepFace
import numpy as np
import os


def main():
    """
    Main function to run the real-time facial emotion recognition system
    """
    # Initialize webcam
    cap = cv2.VideoCapture(0)

    # Check if webcam is accessible
    if not cap.isOpened():
        print("Error: Could not open webcam")
        return

    # Set webcam properties for better performance
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)

    # Load OpenCV's face detection classifier with better error handling
    face_cascade = None

    # Try multiple ways to load the haar cascade
    cascade_paths = []

    # Try cv2.data.haarcascades if it exists
    try:
        if hasattr(cv2, "data") and hasattr(cv2.data, "haarcascades"):
            cascade_paths.append(
                cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            )
    except:
        pass

    # Add other potential paths
    cascade_paths.extend(
        [
            "haarcascade_frontalface_default.xml",  # Local file
            os.path.join(
                os.path.dirname(cv2.__file__),
                "data",
                "haarcascade_frontalface_default.xml",
            ),
            os.path.join(
                cv2.__path__[0], "data", "haarcascade_frontalface_default.xml"
            ),
        ]
    )

    for path in cascade_paths:
        try:
            if os.path.exists(path):
                face_cascade = cv2.CascadeClassifier(path)
                if not face_cascade.empty():
                    print(f"Successfully loaded haar cascade from: {path}")
                    break
        except:
            continue

    # If still no cascade found, try alternative methods
    if face_cascade is None or face_cascade.empty():
        try:
            # Try to load directly if cv2.data exists
            if hasattr(cv2, "data") and hasattr(cv2.data, "haarcascades"):
                face_cascade = cv2.CascadeClassifier()
                if not face_cascade.load(
                    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
                ):
                    raise Exception("Could not load cascade")
            else:
                raise Exception("cv2.data not available")
        except:
            print("Error: Could not load haar cascade classifier")
            print("\nTroubleshooting steps:")
            print(
                "1. Try reinstalling OpenCV: pip uninstall opencv-python && pip install opencv-python"
            )
            print("2. Or download the file manually from:")
            print(
                "   https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml"
            )
            print(
                "3. Save it as 'haarcascade_frontalface_default.xml' in the same directory as this script"
            )
            return

    print("Facial Emotion Recognition System Started")
    print("Press 'q' or 'ESC' to quit")

    # Frame counter for performance optimization
    frame_count = 0
    current_emotion = "Detecting..."
    current_confidence = 0

    while True:
        # Capture frame from webcam
        ret, frame = cap.read()

        if not ret:
            print("Error: Failed to capture frame")
            break

        # Flip frame horizontally for mirror effect
        frame = cv2.flip(frame, 1)

        # Convert frame to grayscale for face detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces in the frame
        faces = face_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
        )

        # Process each detected face
        for x, y, w, h in faces:
            # Extract face region
            face_roi = frame[y : y + h, x : x + w]

            # Only analyze emotion every 5 frames for better performance
            if frame_count % 10 == 0:
                try:
                    # Analyze emotion using DeepFace
                    result = DeepFace.analyze(
                        face_roi,
                        actions=["emotion"],
                        enforce_detection=False,
                        silent=True,
                    )

                    # Extract dominant emotion
                    if isinstance(result, list):
                        emotion = result[0]["dominant_emotion"]
                        confidence = result[0]["emotion"][emotion]
                    else:
                        emotion = result["dominant_emotion"]
                        confidence = result["emotion"][emotion]

                    # Store emotion for display
                    current_emotion = emotion
                    current_confidence = confidence

                except Exception as e:
                    # If emotion detection fails, show unknown
                    current_emotion = "Unknown"
                    current_confidence = 0

            # Draw rectangle around face
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # Prepare emotion text
            emotion_text = f"{current_emotion} ({current_confidence:.1f}%)"

            # Add text background for better readability
            text_size = cv2.getTextSize(emotion_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[
                0
            ]
            cv2.rectangle(frame, (x, y - 30), (x + text_size[0], y), (0, 255, 0), -1)

            # Draw emotion text above the face box
            cv2.putText(
                frame,
                emotion_text,
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 0),
                2,
            )

        # Display instructions on the frame
        cv2.putText(
            frame,
            "Press 'q' or 'ESC' to quit",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
        )

        # Show the frame
        cv2.imshow("Facial Emotion Recognition", frame)

        # Check for exit conditions
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q") or key == 27:  # 'q' key or ESC key
            break

        frame_count += 1

    # Clean up
    cap.release()
    cv2.destroyAllWindows()
    print("System terminated successfully")


def check_requirements():
    """
    Check if required packages are installed
    """
    try:
        import cv2
        import deepface

        print("All required packages are installed!")
        print(f"OpenCV version: {cv2.__version__}")
        return True
    except ImportError as e:
        print(f"Missing package: {e}")
        print("Please install required packages:")
        print("pip install opencv-python deepface")
        return False


if __name__ == "__main__":
    # Check requirements before running
    if check_requirements():
        main()
    else:
        print("Please install the required packages and try again.")
