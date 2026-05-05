import cv2
import mediapipe as mp

# Usamos la definición directa que Python ya encontró
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Configuramos el detector
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Iniciamos la cámara
cap = cv2.VideoCapture(0)

print("--- SISTEMA INICIADO ---")
print("Presiona 'q' para cerrar la ventana.")

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    # MediaPipe necesita que la imagen sea RGB
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)

    # Si detecta manos, dibujamos los puntos
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame, 
                hand_landmarks, 
                mp_hands.HAND_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=4),
                mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=2)
            )
            cv2.putText(frame, "MANO DETECTADA", (10, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Mostrar el resultado
    cv2.imshow("MathFest", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()