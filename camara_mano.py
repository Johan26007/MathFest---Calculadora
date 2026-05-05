import cv2
import mediapipe as mp

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5
)

# IDs de las puntas de los dedos (Índice, Medio, Anular, Meñique)
tip_ids = [4, 8, 12, 16, 20]

# 2. Iniciar la cámara
cap = cv2.VideoCapture(0)

print("Cámara iniciada. Presiona 'q' para salir.")

while cap.isOpened():
    success, img = cap.read()
    if not success:
        continue

    # Voltear imagen y convertir a RGB
    img = cv2.flip(img, 1)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)

    if results.multi_hand_landmarks:
        for hand_lms in results.multi_hand_landmarks:
            landmarks = []
            for id, lm in enumerate(hand_lms.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                landmarks.append([cx, cy])

            if len(landmarks) != 0:
                fingers = []

                # Lógica del Pulgar (Punto 4 vs Punto 3)
                if landmarks > landmarks:
                    fingers.append(1)
                else:
                    fingers.append(0)

                # Lógica de los otros 4 dedos (Punta vs Nudillo)
                for id in range(0, 4):
                    if landmarks[tip_ids[id]] < landmarks[tip_ids[id] - 2]:
                        fingers.append(1)
                    else:
                        fingers.append(0)

                total_fingers = fingers.count(1)
                
                # Dibujar el número gigante
                cv2.rectangle(img, (20, 20), (200, 130), (0, 255, 0), cv2.FILLED)
                cv2.putText(img, str(total_fingers), (60, 110), 
                            cv2.FONT_HERSHEY_PLAIN, 7, (255, 255, 255), 10)

            # Dibujar esqueleto
            mp_draw.draw_landmarks(img, hand_lms, mp_hands.HAND_CONNECTIONS)

    cv2.imshow("MathFest - Version1", img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()