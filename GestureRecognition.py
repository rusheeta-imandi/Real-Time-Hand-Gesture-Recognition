import cv2
import mediapipe as mp

# Initialize MediaPipe hands model
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.7, min_tracking_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils

# Gesture-to-Text Database
gesture_to_text = {
    "Love": "I Love You",
    "Sad": "I am Sad",
    "Please": "Please",
    "Thank You": "Thank You",
    "No": "No",
    "Yes": "Yes",
    "Stop": "Stop",
    "Peace": "Peace",
    "Call Me": "Call Me"
}

# Function to classify gestures
def classify_gesture(landmarks):
    thumb_tip = landmarks[mp_hands.HandLandmark.THUMB_TIP].y
    index_finger_tip = landmarks[mp_hands.HandLandmark.INDEX_FINGER_TIP].y
    middle_finger_tip = landmarks[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].y
    ring_finger_tip = landmarks[mp_hands.HandLandmark.RING_FINGER_TIP].y
    pinky_tip = landmarks[mp_hands.HandLandmark.PINKY_TIP].y

    thumb_ip = landmarks[mp_hands.HandLandmark.THUMB_IP].y
    index_finger_mcp = landmarks[mp_hands.HandLandmark.INDEX_FINGER_MCP].y
    middle_finger_mcp = landmarks[mp_hands.HandLandmark.MIDDLE_FINGER_MCP].y
    ring_finger_mcp = landmarks[mp_hands.HandLandmark.RING_FINGER_MCP].y
    pinky_mcp = landmarks[mp_hands.HandLandmark.PINKY_MCP].y

    # Gesture classification logic
    if thumb_tip < index_finger_tip and thumb_tip < pinky_tip and \
       index_finger_tip < middle_finger_tip and pinky_tip < ring_finger_tip:
        return "Love"

    if thumb_tip > thumb_ip and index_finger_tip > index_finger_mcp and \
       middle_finger_tip > middle_finger_mcp and ring_finger_tip > ring_finger_mcp and \
       pinky_tip > pinky_mcp:
        return "Sad"

    if thumb_tip > thumb_ip and index_finger_tip > index_finger_mcp and \
       middle_finger_tip > middle_finger_mcp and ring_finger_tip > ring_finger_mcp and \
       pinky_tip > pinky_mcp and thumb_tip < thumb_ip:
        return "Please"

    if thumb_tip < index_finger_tip and thumb_tip < middle_finger_tip and \
       thumb_tip < ring_finger_tip and thumb_tip < pinky_tip:
        return "Thank You"

    if index_finger_tip < middle_finger_tip and thumb_tip > index_finger_mcp:
        return "No"

    if thumb_tip < thumb_ip and index_finger_tip > index_finger_mcp and \
       middle_finger_tip > middle_finger_mcp and ring_finger_tip > ring_finger_mcp and \
       pinky_tip > pinky_mcp:
        return "Yes"

    if thumb_tip < thumb_ip and index_finger_tip < middle_finger_tip and \
       middle_finger_tip < ring_finger_tip and pinky_tip < ring_finger_tip:
        return "Stop"

    if thumb_tip > index_finger_tip and index_finger_tip < middle_finger_tip and \
       pinky_tip > ring_finger_tip:
        return "Peace"

    if thumb_tip < thumb_ip and pinky_tip < pinky_mcp and \
        index_finger_tip > index_finger_mcp and middle_finger_tip > middle_finger_mcp and \
            ring_finger_tip > ring_finger_mcp:
            return "Call Me"


    return "Unknown Gesture"

def detect_hand_gestures(frame):
    # Convert the image to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Process the frame to detect hands
    results = hands.process(rgb_frame)
    
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
            # Classify gesture
            gesture = classify_gesture(hand_landmarks.landmark)
            text = gesture_to_text.get(gesture, "Unknown Gesture")
            cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    return frame

def main():
    video_capture = cv2.VideoCapture(0)  # 0 for default webcam

    if not video_capture.isOpened():
        print("Error opening video stream or file")
        return

    while True:
        ret, frame = video_capture.read()

        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break

        # Detect hand gestures in the frame
        frame = detect_hand_gestures(frame)

        # Display the resulting frame
        cv2.imshow('Hand Gestures', frame)

        # Exit when 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video_capture.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
