import cv2
import mediapipe as mp
import time
import numpy as np

def ejecutar_operaciones_compuestas():
    # Configuración MediaPipe local al módulo
    mp_hands = mp.solutions.hands
    mp_draw = mp.solutions.drawing_utils
    hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.7, min_tracking_confidence=0.5)
    tip_ids = [4, 8, 12, 16, 20]

    # Variables de Estado Locales
    estado_actual = 2 
    operacion_texto = "" 
    resultado_calculado = ""
    buffer_numero = ""        
    ultimo_gesto_metido = "" 
    numero_estabilizado = -1
    tiempo_inicio_estabilidad = 0   
    TIEMPO_REQUERIDO = 2.0    
    porcentaje_carga = 0.0

    def obtener_estado_dedos(landmarks, hand_label):
        fingers = []
        # Pulgar
        if hand_label == "Right":
            if landmarks[tip_ids[0]][0] < landmarks[tip_ids[0] - 1][0]: fingers.append(1)
            else: fingers.append(0)
        else:
            if landmarks[tip_ids[0]][0] > landmarks[tip_ids[0] - 1][0]: fingers.append(1)
            else: fingers.append(0)
            
        # Los otros 4 dedos
        for id in range(1, 5):
            if landmarks[tip_ids[id]][1] < landmarks[tip_ids[id] - 2][1]: fingers.append(1)
            else: fingers.append(0)
        return fingers

    def descifrar_simbolo(dedos_derecha):
        if dedos_derecha == [0, 1, 0, 0, 1]: return "+"
        if dedos_derecha == [0, 1, 0, 0, 0]: return "-"
        if dedos_derecha == [0, 1, 1, 0, 0]: return "*"
        if dedos_derecha == [1, 1, 0, 0, 0]: return "/"
        if dedos_derecha == [0, 1, 1, 1, 0]: return "()"
        return "DESCONOCIDO"

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

        dedos_izq = []
        dedos_der = []
        modo_simbolos_activo = False

        if results.multi_hand_landmarks:
            for i, hand_lms in enumerate(results.multi_hand_landmarks):
                label = results.multi_handedness[i].classification[0].label
                estado_dedos = obtener_estado_dedos([[int(lm.x * w_img), int(lm.y * h_img)] for lm in hand_lms.landmark], label)
                
                if label == "Left":
                    dedos_izq = estado_dedos
                else:
                    dedos_der = estado_dedos
                    
                mp_draw.draw_landmarks(img, hand_lms, mp_hands.HAND_CONNECTIONS)

        # Identificación de modos con la mano izquierda
        if dedos_izq.count(1) == 5:
            modo_simbolos_activo = True
        elif dedos_izq.count(1) == 4:
            modo_simbolos_activo = "BORRAR"

        gesto_actual_valido = False
        tipo_entrada_actual = "" 
        valor_detectado_actual = ""

        if estado_actual == 2:
            if modo_simbolos_activo == "BORRAR":
                gesto_actual_valido = True
                tipo_entrada_actual = "BORRAR"
                valor_detectado_actual = "BACKSPACE"
            elif modo_simbolos_activo:
                if dedos_der:
                    simbolo = descifrar_simbolo(dedos_der)
                    if simbolo != "DESCONOCIDO":
                        gesto_actual_valido = True
                        tipo_entrada_actual = "SIMBOLO"
                        valor_detectado_actual = simbolo
            else:
                if results.multi_hand_landmarks:
                    conteo = (dedos_izq.count(1) if dedos_izq else 0) + (dedos_der.count(1) if dedos_der else 0)
                    gesto_actual_valido = True
                    tipo_entrada_actual = "NUMERO"
                    valor_detectado_actual = conteo

        # Estabilización temporal
        if gesto_actual_valido:
            if valor_detectado_actual == numero_estabilizado:
                tiempo_transcurrido = time.time() - tiempo_inicio_estabilidad
                porcentaje_carga = min(tiempo_transcurrido / TIEMPO_REQUERIDO, 1.0)

                if tiempo_transcurrido >= TIEMPO_REQUERIDO:
                    if tipo_entrada_actual == "BORRAR":
                        if buffer_numero: 
                            buffer_numero = buffer_numero[:-1]
                        elif operacion_texto: 
                            operacion_texto = operacion_texto[:-1]
                        time.sleep(0.5)
                        
                    elif tipo_entrada_actual == "NUMERO":
                        if valor_detectado_actual > 0:
                            digito = str(valor_detectado_actual if valor_detectado_actual < 10 else 0)
                            buffer_numero += digito
                        else:
                            if buffer_numero:
                                operacion_texto += buffer_numero
                                buffer_numero = ""
                            else:
                                try:
                                    resultado_calculado = str(eval(operacion_texto))
                                except Exception:
                                    resultado_calculado = "ERROR"
                                estado_actual = 4 
                                
                    elif tipo_entrada_actual == "SIMBOLO":
                        if buffer_numero:
                            operacion_texto += buffer_numero
                            buffer_numero = ""
                            
                        if valor_detectado_actual == "()":
                            if operacion_texto.count("(") > operacion_texto.count(")"): operacion_texto += ")"
                            else: operacion_texto += "("
                        else:
                            operacion_texto += valor_detectado_actual
                        time.sleep(0.5)

                    numero_estabilizado = -1
                    porcentaje_carga = 0.0
                    time.sleep(0.3)
            else:
                numero_estabilizado = valor_detectado_actual
                tiempo_inicio_estabilidad = time.time()
                porcentaje_carga = 0.0
        else:
            numero_estabilizado = -1
            porcentaje_carga = 0.0

        # --- INTERFAZ GRÁFICA ---
        cv2.rectangle(img, (0, 0), (w_img, 110), (20, 20, 20), -1)
        f = cv2.FONT_HERSHEY_SIMPLEX
        
        if porcentaje_carga > 0 and estado_actual == 2:
            cv2.rectangle(img, (0, 110), (int(w_img * porcentaje_carga), 118), (154, 255, 222), -1)
            
        if estado_actual == 2:
            if not results.multi_hand_landmarks:
                status_txt = "ESPERANDO DETECCION DE MANO..."
                color_status = (200, 200, 200)
            elif tipo_entrada_actual == "BORRAR":
                status_txt = "MODO: BORRAR ULTIMO DIGITO / CARACTER"
                color_status = (0, 0, 255)
            elif modo_simbolos_activo:
                status_txt = f"MODO SIMBOLOS | Detectado: {valor_detectado_actual}"
                color_status = (0, 255, 255)
            else:
                digito_visual = str(valor_detectado_actual if valor_detectado_actual < 10 else 0) if valor_detectado_actual > 0 else "Puno (Confirmar/Resolver)"
                status_txt = f"MODO NUMEROS | Digito: {digito_visual}"
                color_status = (0, 255, 0)

            cv2.putText(img, status_txt, (30, 40), f, 0.8, color_status, 2)
            cv2.putText(img, f"OPERACION: {operacion_texto}{buffer_numero}_", (30, 85), f, 1, (255, 255, 255), 2)
            
            cv2.rectangle(img, (20, h_img-70), (w_img-20, h_img-20), (35, 35, 35), -1)
            guia = "Gestos (Mano Izq Abierta + Der): Rock=[+] | Indice=[- ] | Amor&Paz=[*] | Pistola=[/] | 3 Dedos=[()]"
            cv2.putText(img, guia, (35, h_img-42), f, 0.55, (200, 200, 200), 1)

        elif estado_actual == 4:
            cv2.putText(img, "RESULTADO DE OPERACION COMPUESTA", (30, 40), f, 0.9, (0, 255, 0), 2)
            cv2.putText(img, f"{operacion_texto} = {resultado_calculado}", (30, 85), f, 1.2, (255, 255, 255), 2)
            cv2.putText(img, "Presiona 'R' para borrar y hacer otra operacion", (30, 140), f, 0.7, (170, 170, 170), 1)

        cv2.imshow("MathFest - Operaciones Compuestas", img)
        k = cv2.waitKey(1) & 0xFF
        if k == ord('q'): break
        if k == ord('r'): estado_actual, operacion_texto, buffer_numero, resultado_calculado = 2, "", "", ""

    cap.release()
    cv2.destroyAllWindows()