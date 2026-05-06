import cv2
import mediapipe as mp
import time
import numpy as np
import matplotlib.pyplot as plt
import io

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(min_detection_confidence=0.8, min_tracking_confidence=0.5)
tip_ids = [4, 8, 12, 16, 20]

# Variables
# Estados del menu
estado_actual = -1 
tipo_funcion = "" # "lineal", "cuadratica", "cubica"
coeficientes = []
nombres_coef = ["A", "B", "C", "D"]

numero_estabilizado = -1
tiempo_inicio_estabilidad = 0
TIEMPO_REQUERIDO = 2.0 
grafica_img = None 

def contar_dedos(landmarks):
    fingers = []
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
    
    if tipo == "lineal":
        y = coeffs[0] * x + coeffs[1]
        label = f"y = {coeffs[0]}x + {coeffs[1]}"
    elif tipo == "cuadratica":
        y = coeffs[0]*x**2 + coeffs[1]*x + coeffs[2]
        label = f"y = {coeffs[0]}x² + {coeffs[1]}x + {coeffs[2]}"
    else: # cubica
        y = coeffs[0]*x**3 + coeffs[1]*x**2 + coeffs[2]*x + coeffs[3]
        label = f"y = {coeffs[0]}x³ + ..."

    ax.plot(x, y, color='#00FFFF', linewidth=3, label=label)
    ax.axhline(0, color='white', lw=1)
    ax.axvline(0, color='white', lw=1)
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    img_arr = np.frombuffer(buf.getvalue(), dtype=np.uint8)
    plt.close(fig)
    return cv2.resize(cv2.imdecode(img_arr, cv2.IMREAD_COLOR), (350, 350))

#Camara
cap = cv2.VideoCapture(0)
Ventana_Nombre = "MathFest - Selector de Funciones"
cv2.namedWindow(Ventana_Nombre, cv2.WINDOW_NORMAL)
cv2.resizeWindow(Ventana_Nombre, 1280, 720)

while cap.isOpened():
    success, img = cap.read()
    if not success: break
    
    img = cv2.flip(img, 1)
    img = cv2.resize(img, (1280, 720))
    h_img, w_img, _ = img.shape
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)

    dedos_actuales = -1
    if results.multi_hand_landmarks:
        for hand_lms in results.multi_hand_landmarks:
            landmarks = [[int(lm.x * w_img), int(lm.y * h_img)] for lm in hand_lms.landmark]
            dedos_actuales = contar_dedos(landmarks)
            mp_draw.draw_landmarks(img, hand_lms, mp_hands.HAND_CONNECTIONS)

    #Temporizador y estado
    if dedos_actuales != -1:
        if dedos_actuales == numero_estabilizado:
            if time.time() - tiempo_inicio_estabilidad >= TIEMPO_REQUERIDO:
                #Accion segun estado
                if estado_actual == -1: #Seleccion del menu
                    if dedos_actuales == 1: tipo_funcion, estado_actual = "lineal", 0
                    elif dedos_actuales == 2: tipo_funcion, estado_actual = "cuadratica", 0
                    elif dedos_actuales == 3: tipo_funcion, estado_actual = "cubica", 0
                
                elif estado_actual >= 0 and estado_actual < 4: # CAPTURA DE COEFICIENTES
                    coeficientes.append(dedos_actuales)
                    # Determinar si ya terminamos según el tipo de función
                    limite = 2 if tipo_funcion == "lineal" else 3 if tipo_funcion == "cuadratica" else 4
                    estado_actual += 1
                    
                    if estado_actual == limite:
                        grafica_img = generar_grafica_cv2(tipo_funcion, coeficientes)
                        estado_actual = 4 # Ir a graficar
                
                numero_estabilizado = -1
                time.sleep(0.5) # Pausa para no repetir captura
            
            # Dibujar barra de carga
            porcentaje = min((time.time() - tiempo_inicio_estabilidad) / TIEMPO_REQUERIDO, 1.0)
            cv2.rectangle(img, (w_img//2 - 200, 650), (w_img//2 - 200 + int(400 * porcentaje), 670), (0, 255, 0), cv2.FILLED)
        else:
            numero_estabilizado = dedos_actuales
            tiempo_inicio_estabilidad = time.time()
    else:
        numero_estabilizado = -1

    # Interfaz de usuario
    cv2.rectangle(img, (0, 0), (w_img, 120), (30, 30, 30), cv2.FILLED)
    
    #Tipo de fuente
    fuente = cv2.FONT_HERSHEY_DUPLEX 

    if estado_actual == -1:
        cv2.putText(img, "MENU: 1-Lineal | 2-Cuadratica | 3-Cubica", (40, 50), fuente, 1.2, (255, 255, 255), 2)
        cv2.putText(img, f"Selecciona una opcion: {max(0, dedos_actuales)}", (40, 95), fuente, 1, (0, 255, 255), 2)
    
    elif estado_actual < 4:
        nombre = nombres_coef[estado_actual]
        cv2.putText(img, f"TIPO: {tipo_funcion.upper()} | Coeficiente: {nombre}", (40, 50), fuente, 1.2, (255, 255, 255), 2)
        cv2.putText(img, f"Valor detectado: {max(0, dedos_actuales)}", (40, 95), fuente, 1, (0, 255, 0), 2)
    
    elif estado_actual == 4:
        cv2.putText(img, f"Funcion {tipo_funcion.upper()} graficada", (40, 55), fuente, 1.3, (0, 255, 0), 2)
        cv2.putText(img, "Presiona 'R' para volver al Menu", (40, 100), fuente, 0.8, (200, 200, 200), 1)
        if grafica_img is not None:
            # Pegamos la gráfica
            img[150:500, w_img-400:w_img-50] = grafica_img
    
    

    cv2.imshow(Ventana_Nombre, img)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'): break
    if key == ord('r'):
        estado_actual, coeficientes, grafica_img, tipo_funcion = -1, [], None, ""

cap.release()
cv2.destroyAllWindows()