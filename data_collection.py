import cv2
import os
import mediapipe as mp

label = 'Delete'  # Change this for each letter
save_dir = f'dataset/{label}'
os.makedirs(save_dir, exist_ok=True)

cap = cv2.VideoCapture(0)
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
drawing = mp.solutions.drawing_utils

count = 0
max_images = 500  # Stop after collecting 500 images

while cap.isOpened() and count < max_images:
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(image_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            x_max = int(max([lm.x for lm in hand_landmarks.landmark]) * w)
            x_min = int(min([lm.x for lm in hand_landmarks.landmark]) * w)
            y_max = int(max([lm.y for lm in hand_landmarks.landmark]) * h)
            y_min = int(min([lm.y for lm in hand_landmarks.landmark]) * h)

            margin = 20
            x_min = max(x_min - margin, 0)
            y_min = max(y_min - margin, 0)
            x_max = min(x_max + margin, w)
            y_max = min(y_max + margin, h)

            hand_img = frame[y_min:y_max, x_min:x_max]
            if hand_img.size != 0:
                resized = cv2.resize(hand_img, (64, 64))
                cv2.imwrite(f"{save_dir}/{count}.jpg", resized)
                count += 1

    cv2.putText(frame, f"Collecting: {count}/{max_images}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow("Collecting", frame)

    if cv2.waitKey(1) == ord('q'):
        break

print(f"Done! Collected {count} images for label '{label}'")
cap.release()
cv2.destroyAllWindows()