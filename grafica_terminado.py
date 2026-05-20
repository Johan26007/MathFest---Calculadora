import cv2
import mediapipe as mp
import time
import numpy as np
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
import io

# Configuracion MediaPipe
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.7, min_tracking_confidence=0.5)
tip_ids = [4, 8, 12, 16, 20]

# Variables de Estado
estado_actual = -1 
tipo_funcion = ""
coeficientes = []
nombres_coef = ["A", "B", "C", "D"]

# Captura secuencial
buffer_numero = ""        
ultimo_digito_metido = -1 
numero_estabilizado = -1
tiempo_inicio_estabilidad = 0
TIEMPO_REQUERIDO = 2.0    # 2 segundos para estabilizar
grafica_img = None 

def contar_dedos_mano(landmarks, hand_label):
    fingers = []
    if hand_label == "Right":
        if landmarks[tip_ids[0]][0] < landmarks[tip_ids[0] - 1][0]: fingers.append(1)
        else: fingers.append(0)
    else:
        if landmarks[tip_ids[0]][0] > landmarks[tip_ids[0] - 1][0]: fingers.append(1)
        else: fingers.append(0)
    for id in range(1, 5):
        if landmarks[tip_ids[id]][1] < landmarks[tip_ids[id] - 2][1]: fingers.append(1)
        else: fingers.append(0)
    return fingers.count(1)

def generar_grafica_cv2(tipo, coeffs):
    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(4, 4), dpi=100)
    x = np.linspace(-10, 10, 100)
    try:
        if tipo == "lineal":
            y = coeffs[0] * x + coeffs[1]
            label = f"y = {coeffs[0]}x + {coeffs[1]}"
        elif tipo == "cuadratica":
            y = coeffs[0]*x**2 + coeffs[1]*x + coeffs[2]
            label = f"y = {coeffs[0]}x^2 + {coeffs[1]}x + {coeffs[2]}"
        else:
            y = coeffs[0]*x**3 + coeffs[1]*x**2 + coeffs[2]*x + coeffs[3]
            label = f"y = {coeffs[0]}x^3 + ..."
        ax.plot(x, y, color='#DEFF9A', linewidth=3, label=label)
        ax.axhline(0, color='white', lw=1); ax.axvline(0, color='white', lw=1)
        ax.legend()
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight')
        buf.seek(0)
        img_arr = np.frombuffer(buf.getvalue(), dtype=np.uint8)
        plt.close(fig)
        return cv2.resize(cv2.imdecode(img_arr, cv2.IMREAD_COLOR), (350, 350))
    except: return None

cap = cv2.VideoCapture(0)
W_DISP, H_DISP = 1024, 768 
cap.set(3, W_DISP); cap.set(4, H_DISP)

while cap.isOpened():
    success, img = cap.read()
    if not success: break
    img = cv2.flip(img, 1)
    img = cv2.resize(img, (W_DISP, H_DISP))
    h_img, w_img, _ = img.shape
    results = hands.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

    dedos_actuales = -1
    porcentaje_carga = 0.0 # Progreso de estabilizacion

    if results.multi_hand_landmarks:
        dedos_totales = 0
        for i, hand_lms in enumerate(results.multi_hand_landmarks):
            label = results.multi_handedness[i].classification[0].label
            lms = [[int(lm.x * w_img), int(lm.y * h_img)] for lm in hand_lms.landmark]
            dedos_totales += contar_dedos_mano(lms, label)
            mp_draw.draw_landmarks(img, hand_lms, mp_hands.HAND_CONNECTIONS)
        dedos_actuales = dedos_totales

    # Estabilizacion y entrada secuencial
    if dedos_actuales != -1:
        if dedos_actuales == numero_estabilizado:
            # Calculo de estabilidad de la mano
            tiempo_transcurrido = time.time() - tiempo_inicio_estabilidad
            porcentaje_carga = min(tiempo_transcurrido / TIEMPO_REQUERIDO, 1.0)

            if tiempo_transcurrido >= TIEMPO_REQUERIDO:
                # --- ACCIONES AL SELECCIONAR O CONFIRMAR ---
                if estado_actual == -1: 
                    if 1 <= dedos_actuales <= 3:
                        tipo_funcion = ["lineal", "cuadratica", "cubica"][dedos_actuales-1]
                        estado_actual = 0
                
                elif 0 <= estado_actual < 4: 
                    if dedos_actuales > 0: 
                        if dedos_actuales != ultimo_digito_metido:
                            # 10 = 0
                            buffer_numero += str(dedos_actuales if dedos_actuales < 10 else 0)
                            ultimo_digito_metido = dedos_actuales
                    else: 
                        if buffer_numero == "": buffer_numero = "0"
                        coeficientes.append(int(buffer_numero))
                        buffer_numero = ""
                        ultimo_digito_metido = -1
                        limite = 2 if tipo_funcion == "lineal" else 3 if tipo_funcion == "cuadratica" else 4
                        estado_actual += 1
                        if estado_actual == limite:
                            grafica_img = generar_grafica_cv2(tipo_funcion, coeficientes)
                            estado_actual = 4
                
                numero_estabilizado = -1 
                time.sleep(0.4) # Pausa para cambiar de gesto
        else:
            numero_estabilizado = dedos_actuales
            tiempo_inicio_estabilidad = time.time()
    else: 
        numero_estabilizado = -1

    # --- DISEÑO DE LA INTERFAZ ---
    # Fondo del menú superior
    cv2.rectangle(img, (0, 0), (w_img, 110), (20, 20, 20), -1)
    f = cv2.FONT_HERSHEY_SIMPLEX
    
    # Barra de progreso de 2 segundos 
    if porcentaje_carga > 0:
        ancho_barra = int(w_img * porcentaje_carga)
        cv2.rectangle(img, (0, 110), (ancho_barra, 118), (154, 255, 222), -1)

    if estado_actual == -1:
        cv2.putText(img, "1:Lineal | 2:Cuadratica | 3:Cubica", (30, 45), f, 1, (255,255,255), 2)
        cv2.putText(img, f"Detectado: {max(0, dedos_actuales)}", (30, 85), f, 0.8, (154, 255, 222), 1)
    elif estado_actual < 4:
        txt = f"Coeficiente {nombres_coef[estado_actual]} | Digito detectado: {max(0, dedos_actuales)}"
        cv2.putText(img, txt, (30, 45), f, 1, (255,255,255), 2)
        cv2.putText(img, f"VALOR ACUMULADO: {buffer_numero}_ (Cierra el puno para confirmar)", (30, 85), f, 0.8, (154, 255, 222), 2)
    elif estado_actual == 4:
        cv2.putText(img, "GRAFICA LISTA - 'R' para Reiniciar", (30, 60), f, 1, (0, 255, 0), 2)
        if grafica_img is not None:
            img[150:500, w_img-380:w_img-30] = grafica_img

    cv2.imshow("MathFest - Graficadora", img)
    k = cv2.waitKey(1) & 0xFF
    if k == ord('q'): break
    if k == ord('r'): estado_actual, coeficientes, buffer_numero, tipo_funcion = -1, [], "", ""

cap.release(); cv2.destroyAllWindows()