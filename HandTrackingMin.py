import cv2
import mediapipe as mp
import time

cap = cv2.VideoCapture(0)

mpHands = mp.solutions.hands
hands = mpHands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.7, min_tracking_confidence=0.5)
mpDraw = mp.solutions.drawing_utils

pTime = 0
cTime = 0

userOperation = input("Which math operation do you want to perform? (add, subtract, multiply, divide): ")

left_fingers_count = 0
right_fingers_count = 0

def count_fingers(handLms, threshold_height=0.7):
    fingers_up = [0, 0, 0, 0, 0]  # Initialize list to track fingers
    for id, landmark in enumerate(handLms.landmark):
        h, w, c = img.shape
        cx, cy = int(landmark.x * w), int(landmark.y * h)

        if id in [4, 8, 12, 16, 20]:  # Only consider landmarks for fingers
            finger_id = int((id - 4) / 4)  # Calculate finger ID (Thumb=0, Index=1, Middle=2, Ring=3, Pinky=4)

            if cy < handLms.landmark[id - 1].y * h and cy < threshold_height * h:  # Check if the finger is up
                fingers_up[finger_id] = 1  # Set the finger status to 1 if it's up

            cv2.circle(img, (cx, cy), 5, (255, 0, 255), cv2.FILLED)

    return fingers_up.count(1)

def perform_operation(operation, left_count, right_count):
    if operation == 'add':
        return left_count + right_count
    elif operation == 'subtract':
        return left_count - right_count
    elif operation == 'multiply':
        return left_count * right_count
    elif operation == 'divide':
        return left_count / right_count if right_count != 0 else 0

while True:
    success, img = cap.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)

    if results.multi_hand_landmarks:
        left_hand_detected = False
        right_hand_detected = False
        
        for handLms in results.multi_hand_landmarks:
            if handLms.landmark[0].x < handLms.landmark[5].x:  # Left hand
                right_hand_detected = True
                right_fingers_count = count_fingers(handLms)
            else:  # Right hand
                left_hand_detected = True
                left_fingers_count = count_fingers(handLms)
                

        fingers_text = ""
        if left_hand_detected:
            fingers_text += f"Left Hand Fingers: {left_fingers_count}, "
        if right_hand_detected:
            fingers_text += f"Right Hand Fingers: {right_fingers_count}"

        cv2.putText(img, fingers_text, (10, 50), cv2.FONT_HERSHEY_PLAIN, 1, (255, 0, 255), 2)
        if left_hand_detected and right_hand_detected:
            operation_result = perform_operation(userOperation, left_fingers_count, right_fingers_count)
            operation_text = f"Operation Result: {operation_result}"
            print(operation_text)
            cv2.putText(img, operation_text, (10, 70), cv2.FONT_HERSHEY_PLAIN, 1, (255, 0, 255), 2)
        
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime

    cv2.putText(img, f'FPS: {int(fps)}', (10, 90), cv2.FONT_HERSHEY_PLAIN, 1, (255, 0, 255), 2)

    cv2.imshow("Image", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
