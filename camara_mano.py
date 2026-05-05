import cv2
import mediapipe as mp

# Inicialización de Mediapipe
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5
)

# IDs de las puntas de los dedos
# Pulgar: 4, Índice: 8, Medio: 12, Anular: 16, Meñique: 20
tip_ids = [4, 8, 12, 16, 20]

cap = cv2.VideoCapture(0)

print("Cámara iniciada. Presiona 'q' para salir.")

while cap.isOpened():
    success, img = cap.read()
    if not success:
        break

    # Voltear imagen para efecto espejo y convertir a RGB
    img = cv2.flip(img, 1)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)

    if results.multi_hand_landmarks:
        for hand_lms in results.multi_hand_landmarks:
            landmarks = []
            for id, lm in enumerate(hand_lms.landmark):
                #Coordenadas en pixeles
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                landmarks.append([cx, cy])

            if len(landmarks) != 0:
                fingers = []

                # --- LOGICA DEL PULGAR ---
                # Comparamos la posicion X de la punta con la base del pulgar.
                if landmarks[tip_ids[0]][0] > landmarks[tip_ids[0] - 1][0]:
                    fingers.append(1)
                else:
                    fingers.append(0)

                # --- LOGICA DE LOS 4 DEDOS --- 
                # El valor de Y disminuye mientras mas arriba este el punto.
                for id in range(1, 5):
                    if landmarks[tip_ids[id]][1] < landmarks[tip_ids[id] - 2][1]:
                        fingers.append(1)
                    else:
                        fingers.append(0)

                total_fingers = fingers.count(1)
                
                # cuadro de fondo para el contador de dedos
                cv2.rectangle(img, (20, 20), (150, 150), (0, 255, 0), cv2.FILLED)
                cv2.putText(img, str(total_fingers), (45, 125), 
                            cv2.FONT_HERSHEY_DUPLEX, 4, (255, 255, 255), 5)

            # Dibujar las conexiones de la mano
            mp_draw.draw_landmarks(img, hand_lms, mp_hands.HAND_CONNECTIONS)

    cv2.imshow("VersionFinalContador", img)

    # Cerrar con la tecla 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()