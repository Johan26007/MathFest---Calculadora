import cv2
import mediapipe as mp
import time
import numpy as np
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
import io
import cmath

# Configuracion MediaPipe
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.7, min_tracking_confidence=0.5)
tip_ids = [4, 8, 12, 16, 20]

# Variables de Estado
estado_actual = 0 
coeficientes = []
nombres_coef = ["A", "B", "C"]

# Captura Secuencial
buffer_numero = ""        
ultimo_digito_metido = -1 
numero_estabilizado = -1
tiempo_inicio_estabilidad = 0
TIEMPO_REQUERIDO = 2.0    
grafica_img = None 
info_raices = {}          

def obtener_estado_dedos(landmarks, hand_label):
    """Devuelve una lista de 5 elementos (1 si el dedo esta arriba, 0 si esta abajo)"""
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
    return fingers

def resolver_y_graficar_cuadratica(coeffs):
    a, b, c = coeffs[0], coeffs[1], coeffs[2]
    
    if a == 0:
        if b != 0:
            x1 = -c / b
            res_txt = f"Lineal: x = {round(x1, 2)}"
            raices_reales = [x1]
        else:
            res_txt = "Inconsistente (0 = 0 o C)"
            raices_reales = []
    else:
        discriminante = b**2 - 4*a*c
        if discriminante > 0:
            x1 = (-b + np.sqrt(discriminante)) / (2*a)
            x2 = (-b - np.sqrt(discriminante)) / (2*a)
            res_txt = f"2 Raices Reales: x1={round(x1,2)}, x2={round(x2,2)}"
            raices_reales = [x1, x2]
        elif discriminante == 0:
            x1 = -b / (2*a)
            res_txt = f"1 Raiz Unica Real: x = {round(x1,2)}"
            raices_reales = [x1]
        else:
            x1 = (-b + cmath.sqrt(discriminante)) / (2*a)
            x2 = (-b - cmath.sqrt(discriminante)) / (2*a)
            res_txt = f"Complejas: {complex(round(x1.real,2), round(x1.imag,2))}"
            raices_reales = []

    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(4, 4), dpi=100)
    
    cx = -b / (2*a) if a != 0 else 0
    x = np.linspace(cx - 10, cx + 10, 200)
    y = a*x**2 + b*x + c
    
    ax.plot(x, y, color='#DEFF9A', linewidth=3, label=f"{a}x^2 + {b}x + {c} = 0")
    ax.axhline(0, color='white', lw=1); ax.axvline(0, color='white', lw=1)
    
    for r in raices_reales:
        ax.plot(r, 0, marker='o', markersize=10, color='#FF5E5E', label=f'Raiz: {round(r,1)}')
        
    ax.legend(prop={'size': 8})
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    img_arr = np.frombuffer(buf.getvalue(), dtype=np.uint8)
    plt.close(fig)
    
    return cv2.resize(cv2.imdecode(img_arr, cv2.IMREAD_COLOR), (350, 350)), res_txt

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
    porcentaje_carga = 0.0 
    es_gesto_rock = False

    if results.multi_hand_landmarks:
        dedos_totales = 0
        for i, hand_lms in enumerate(results.multi_hand_landmarks):
            label = results.multi_handedness[i].classification[0].label
            estado_dedos = obtener_estado_dedos([[int(lm.x * w_img), int(lm.y * h_img)] for lm in hand_lms.landmark], label)
            
            # Sumamos dedos globales para los números
            dedos_totales += estado_dedos.count(1)
            
            # Verificamos si una mano específica está haciendo la seña Rock (Índice y Meñique levantados únicamente)
            if estado_dedos == [0, 1, 0, 0, 1]:
                es_gesto_rock = True
                
            mp_draw.draw_landmarks(img, hand_lms, mp_hands.HAND_CONNECTIONS)
        dedos_actuales = dedos_totales

    # --- MODOS ESPECIALES ---
    modo_borrar_activo = (dedos_actuales == 9)
    modo_signo_activo = es_gesto_rock and (not modo_borrar_activo)

    # Definimos el valor que estamos estabilizando en el cuadro
    if modo_borrar_activo:
        valor_a_estabilizar = "BORRAR"
    elif modo_signo_activo:
        valor_a_estabilizar = "MENOS"
    else:
        valor_a_estabilizar = dedos_actuales

    # Logica de estabilizacion
    if dedos_actuales != -1:
        if valor_a_estabilizar == numero_estabilizado:
            tiempo_transcurrido = time.time() - tiempo_inicio_estabilidad
            porcentaje_carga = min(tiempo_transcurrido / TIEMPO_REQUERIDO, 1.0)

            if tiempo_transcurrido >= TIEMPO_REQUERIDO:
                # --- ACCIONES AL CUMPLIR LOS 2 SEGUNDOS ---
                if numero_estabilizado == "BORRAR":
                    if buffer_numero:
                        buffer_numero = buffer_numero[:-1]
                    elif coeficientes:
                        coeficientes.pop()
                        estado_actual -= 1
                        buffer_numero = ""
                    ultimo_digito_metido = -1
                    time.sleep(0.5)

                elif numero_estabilizado == "MENOS":
                    # Cambia el signo del buffer actual
                    if buffer_numero.startswith("-"):
                        buffer_numero = buffer_numero[1:]
                    else:
                        buffer_numero = "-" + buffer_numero
                    ultimo_digito_metido = -1
                    time.sleep(0.5)

                elif 0 <= estado_actual < 3: 
                    if dedos_actuales > 0: 
                        if dedos_actuales != ultimo_digito_metido:
                            # 10 dedos añaden un 0
                            digito = str(dedos_actuales if dedos_actuales < 10 else 0)
                            buffer_numero += digito
                            ultimo_digito_metido = dedos_actuales
                    else: 
                        # Puño cerrado guarda el coeficiente completo
                        if buffer_numero == "" or buffer_numero == "-": 
                            buffer_numero = "0"
                        coeficientes.append(int(buffer_numero))
                        buffer_numero = ""
                        ultimo_digito_metido = -1
                        estado_actual += 1
                        
                        if estado_actual == 3:
                            grafica_img, info_raices = resolver_y_graficar_cuadratica(coeficientes)
                
                numero_estabilizado = -1 
                time.sleep(0.4) 
        else:
            numero_estabilizado = valor_a_estabilizar
            tiempo_inicio_estabilidad = time.time()
    else: 
        numero_estabilizado = -1

    # --- DISEÑO DE LA INTERFAZ ---
    cv2.rectangle(img, (0, 0), (w_img, 110), (20, 20, 20), -1)
    f = cv2.FONT_HERSHEY_SIMPLEX
    
    if porcentaje_carga > 0 and estado_actual < 3:
        ancho_barra = int(w_img * porcentaje_carga)
        cv2.rectangle(img, (0, 110), (ancho_barra, 118), (154, 255, 222), -1)

    if estado_actual < 3:
        if numero_estabilizado == "BORRAR":
            txt_status = "MODO: BORRAR ULTIMO DIGITO / COEFICIENTE"
            color_txt = (0, 0, 255)
        elif numero_estabilizado == "MENOS":
            txt_status = "MODO: INVERTIR SIGNO (+/-)"
            color_txt = (255, 128, 0) # Naranja
        else:
            txt_status = f"FORMULA GENERAL | Introduce Coeficiente {nombres_coef[estado_actual]}"
            color_txt = (255, 255, 255)

        cv2.putText(img, txt_status, (30, 45), f, 0.9, color_txt, 2)
        cv2.putText(img, f"VALOR ACUMULADO PARA {nombres_coef[estado_actual]}: {buffer_numero}_ (Cierra el puno para guardar)", (30, 85), f, 0.75, (154, 255, 222), 2)
        
        # Guias en la seccion inferior
        cv2.rectangle(img, (20, h_img-65), (w_img-20, h_img-20), (35, 35, 35), -1)
        guias_txt = "Gestos: 9 Dedos = Borrar | Rock (Indice y Menique) = Cambiar +/- | Puno = Guardar"
        cv2.putText(img, guias_txt, (35, h_img-38), f, 0.6, (180, 180, 180), 1)

    elif estado_actual == 3:
        # Se convierten los signos visualmente para que la ecuación impresa se lea bien en pantalla
        cv2.putText(img, f"ECUACION: {coeficientes[0]}x^2 + ({coeficientes[1]})x + ({coeficientes[2]}) = 0", (30, 45), f, 0.9, (255, 255, 255), 2)
        cv2.putText(img, f"SOLUCION: {info_raices}", (30, 85), f, 0.8, (0, 255, 0), 2)
        cv2.putText(img, "Presiona 'R' para resolver otra ecuacion", (30, 140), f, 0.65, (170, 170, 170), 1)
        
        if grafica_img is not None:
            img[180:530, w_img-380:w_img-30] = grafica_img

    cv2.imshow("MathFest - Formula Cuadratica", img)
    k = cv2.waitKey(1) & 0xFF
    if k == ord('q'): break
    if k == ord('r'): estado_actual, coeficientes, buffer_numero = 0, [], ""

cap.release(); cv2.destroyAllWindows()