import cv2
import numpy as np
import mediapipe as mp
import pyttsx3
import pyautogui
from tensorflow.keras.models import load_model

# --------- Load Model and Categories ---------
model = load_model('sign_model.h5')
categories = np.load('categories.npy')

# --------- Initialize TTS Engine ---------
engine = pyttsx3.init()
engine.setProperty('rate', 150)

# --------- MediaPipe Hands ---------
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1)
mp_drawing = mp.solutions.drawing_utils

# --------- Webcam Setup ---------
cap = cv2.VideoCapture(0)
screen_w, screen_h = pyautogui.size()

predicted_text = ""
last_pred = ""
speak_delay = 20
counter = 0
mode = "sign"  # or "mouse"

# --------- Click Cooldown ---------
last_click_time = 0
click_delay = 0.7  # seconds

while True:
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape

    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(img_rgb)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            x_max = int(max([lm.x for lm in hand_landmarks.landmark]) * w)
            x_min = int(min([lm.x for lm in hand_landmarks.landmark]) * w)
            y_max = int(max([lm.y for lm in hand_landmarks.landmark]) * h)
            y_min = int(min([lm.y for lm in hand_landmarks.landmark]) * h)

            # ---------- Mouse Control Mode ----------
            if mode == "mouse":
                index_finger = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
                thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]

                cursor_x = int(index_finger.x * screen_w)
                cursor_y = int(index_finger.y * screen_h)
                pyautogui.moveTo(cursor_x, cursor_y)

                # Distance in pixels for more consistent behavior
                ix, iy = int(index_finger.x * w), int(index_finger.y * h)
                tx, ty = int(thumb_tip.x * w), int(thumb_tip.y * h)
                distance = np.hypot(tx - ix, ty - iy)

                # Draw pinch feedback
                cv2.line(frame, (ix, iy), (tx, ty), (255, 255, 255), 2)
                cv2.circle(frame, (ix, iy), 8, (255, 0, 255), cv2.FILLED)
                cv2.circle(frame, (tx, ty), 8, (255, 0, 255), cv2.FILLED)

                # Pinch click with delay
                current_time = cv2.getTickCount() / cv2.getTickFrequency()
                if distance < 40:
                    if current_time - last_click_time > click_delay:
                        pyautogui.click()
                        last_click_time = current_time
                        cv2.circle(frame, (ix, iy), 15, (0, 255, 0), cv2.FILLED)  # Visual feedback

            # ---------- Sign Recognition Mode ----------
            else:
                margin = 20
                x_min = max(x_min - margin, 0)
                y_min = max(y_min - margin, 0)
                x_max = min(x_max + margin, w)
                y_max = min(y_max + margin, h)

                roi = frame[y_min:y_max, x_min:x_max]
                if roi.size > 0:
                    resized = cv2.resize(roi, (64, 64))
                    input_data = np.expand_dims(resized / 255.0, axis=0)
                    prediction = model.predict(input_data)
                    class_index = np.argmax(prediction)
                    confidence = prediction[0][class_index]

                    if confidence > 0.8:
                        current_pred = categories[class_index]
                        counter += 1
                        if current_pred != last_pred:
                            last_pred = current_pred
                            counter = 0
                        if counter == speak_delay:
                            if current_pred == 'Space':
                                predicted_text += ' '
                            elif current_pred == 'Delete':
                                predicted_text = predicted_text[:-1]
                            else:
                                predicted_text += current_pred
                            counter = 0

    # --------- Display ---------
    mode_text = f"Mode: {'🖐️ Sign Recognition' if mode == 'sign' else '🖱️ Virtual Mouse'}"
    cv2.putText(frame, mode_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 0), 2)

    if mode == "sign":
        cv2.putText(frame, f"Prediction: {last_pred}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
        cv2.putText(frame, f"Text: {predicted_text}", (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    cv2.imshow("Sign Language System", frame)

    key = cv2.waitKey(1)
    if key == ord('m'):  # Switch mode
        mode = "mouse" if mode == "sign" else "sign"
    elif key == ord('c'):  # Clear text
        predicted_text = ""
    elif key == ord('s'):  # Speak sentence
        engine.say(predicted_text)
        engine.runAndWait()
    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()