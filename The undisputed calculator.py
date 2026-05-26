import flet as ft
import cv2
import mediapipe as mp
import time
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import io
import threading
import cmath


# ============================================================
# CONFIGURACION MEDIAPIPE GLOBAL
# ============================================================

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
tip_ids = [4, 8, 12, 16, 20]


# ============================================================
# FUNCIONES COMPARTIDAS
# ============================================================

def obtener_estado_dedos(landmarks, hand_label):
    fingers = []

    if hand_label == "Right":
        fingers.append(1 if landmarks[tip_ids[0]][0] < landmarks[tip_ids[0] - 1][0] else 0)
    else:
        fingers.append(1 if landmarks[tip_ids[0]][0] > landmarks[tip_ids[0] - 1][0] else 0)

    for idx in range(1, 5):
        fingers.append(1 if landmarks[tip_ids[idx]][1] < landmarks[tip_ids[idx] - 2][1] else 0)

    return fingers


def contar_dedos_mano(landmarks, hand_label):
    return sum(obtener_estado_dedos(landmarks, hand_label))


def descifrar_simbolo(dedos_derecha):
    if dedos_derecha == [0, 1, 0, 0, 1]:
        return "+"
    if dedos_derecha == [0, 1, 0, 0, 0]:
        return "-"
    if dedos_derecha == [0, 1, 1, 0, 0]:
        return "*"
    if dedos_derecha == [1, 1, 0, 0, 0]:
        return "/"
    if dedos_derecha == [0, 1, 1, 1, 0]:
        return "()"
    return "DESCONOCIDO"


def generar_grafica_cv2(tipo, coeffs):
    plt.style.use("dark_background")
    fig, ax = plt.subplots(figsize=(4, 4), dpi=100)
    x = np.linspace(-10, 10, 200)

    try:
        if tipo == "lineal":
            y = coeffs[0] * x + coeffs[1]
            label = f"y = {coeffs[0]}x + {coeffs[1]}"
        elif tipo == "cuadratica":
            y = coeffs[0] * x**2 + coeffs[1] * x + coeffs[2]
            label = f"y = {coeffs[0]}x^2 + {coeffs[1]}x + {coeffs[2]}"
        else:
            y = coeffs[0] * x**3 + coeffs[1] * x**2 + coeffs[2] * x + coeffs[3]
            label = f"y = {coeffs[0]}x^3 + {coeffs[1]}x^2 + {coeffs[2]}x + {coeffs[3]}"

        ax.plot(x, y, color="#DEFF9A", linewidth=3, label=label)
        ax.axhline(0, color="white", lw=1)
        ax.axvline(0, color="white", lw=1)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=9)

        buf = io.BytesIO()
        plt.savefig(buf, format="png", bbox_inches="tight")
        buf.seek(0)

        img_arr = np.frombuffer(buf.getvalue(), dtype=np.uint8)
        plt.close(fig)

        return cv2.resize(cv2.imdecode(img_arr, cv2.IMREAD_COLOR), (350, 350))
    except Exception as e:
        print(f"Error al graficar: {e}")
        plt.close(fig)
        return None


def mostrar_cv2_al_frente(nombre_ventana, img):
    cv2.imshow(nombre_ventana, img)

    topmost_prop = getattr(cv2, "WND_PROP_TOPMOST", None)
    if topmost_prop is not None:
        try:
            cv2.setWindowProperty(nombre_ventana, topmost_prop, 1)
        except cv2.error:
            pass


def preparar_ventana_cv2(nombre_ventana, ancho=1024, alto=768):
    cv2.namedWindow(nombre_ventana, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(nombre_ventana, ancho, alto)
    cv2.moveWindow(nombre_ventana, 80, 40)

    topmost_prop = getattr(cv2, "WND_PROP_TOPMOST", None)
    if topmost_prop is not None:
        try:
            cv2.setWindowProperty(nombre_ventana, topmost_prop, 1)
        except cv2.error:
            pass


# ============================================================
# MODULO 1: GRAFICADORA
# ============================================================

def abrir_graficadora(nombre_usuario):
    hands = mp_hands.Hands(
        max_num_hands=2,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.5,
    )

    nombres_coef = ["A", "B", "C", "D"]
    estado_actual = -1
    tipo_funcion = ""
    coeficientes = []
    buffer_numero = ""
    ultimo_digito_metido = -1
    numero_estabilizado = -1
    tiempo_inicio_estab = 0
    tiempo_requerido = 2.0
    grafica_img = None

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1024)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 768)
    nombre_ventana = "MathFest - Graficadora"
    preparar_ventana_cv2(nombre_ventana)

    while cap.isOpened():
        success, img = cap.read()
        if not success:
            break

        img = cv2.flip(img, 1)
        img = cv2.resize(img, (1024, 768))
        h_img, w_img, _ = img.shape
        results = hands.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

        dedos_actuales = -1
        porcentaje_carga = 0.0

        if results.multi_hand_landmarks:
            dedos_totales = 0
            for i, hand_lms in enumerate(results.multi_hand_landmarks):
                label = results.multi_handedness[i].classification[0].label
                lms = [[int(lm.x * w_img), int(lm.y * h_img)] for lm in hand_lms.landmark]
                dedos_totales += contar_dedos_mano(lms, label)
                mp_draw.draw_landmarks(img, hand_lms, mp_hands.HAND_CONNECTIONS)
            dedos_actuales = dedos_totales

        if dedos_actuales != -1:
            if dedos_actuales == numero_estabilizado:
                tiempo_transcurrido = time.time() - tiempo_inicio_estab
                porcentaje_carga = min(tiempo_transcurrido / tiempo_requerido, 1.0)

                if tiempo_transcurrido >= tiempo_requerido:
                    if estado_actual == -1:
                        if 1 <= dedos_actuales <= 3:
                            tipo_funcion = ["lineal", "cuadratica", "cubica"][dedos_actuales - 1]
                            estado_actual = 0
                    elif 0 <= estado_actual < 4:
                        if dedos_actuales > 0:
                            if dedos_actuales != ultimo_digito_metido:
                                buffer_numero += str(dedos_actuales if dedos_actuales < 10 else 0)
                                ultimo_digito_metido = dedos_actuales
                        else:
                            coeficientes.append(int(buffer_numero) if buffer_numero else 0)
                            buffer_numero = ""
                            ultimo_digito_metido = -1

                            limite = 2 if tipo_funcion == "lineal" else 3 if tipo_funcion == "cuadratica" else 4
                            estado_actual += 1

                            if estado_actual == limite:
                                grafica_img = generar_grafica_cv2(tipo_funcion, coeficientes)
                                estado_actual = 4

                    numero_estabilizado = -1
                    time.sleep(0.4)
            else:
                numero_estabilizado = dedos_actuales
                tiempo_inicio_estab = time.time()
        else:
            numero_estabilizado = -1

        cv2.rectangle(img, (0, 0), (w_img, 110), (14, 17, 23), -1)
        font = cv2.FONT_HERSHEY_SIMPLEX

        if porcentaje_carga > 0:
            cv2.rectangle(img, (0, 110), (int(w_img * porcentaje_carga), 118), (88, 166, 255), -1)

        cv2.putText(img, f"Usuario: {nombre_usuario}", (w_img - 280, 30), font, 0.6, (63, 185, 80), 1)

        if estado_actual == -1:
            cv2.putText(img, "1:Lineal | 2:Cuadratica | 3:Cubica", (30, 45), font, 1.0, (255, 255, 255), 2)
            cv2.putText(img, f"Detectado: {max(0, dedos_actuales)}", (30, 85), font, 0.8, (88, 166, 255), 1)
        elif estado_actual < 4:
            cv2.putText(
                img,
                f"Coeficiente {nombres_coef[estado_actual]} | Digito: {max(0, dedos_actuales)}",
                (30, 45),
                font,
                1.0,
                (255, 255, 255),
                2,
            )
            cv2.putText(
                img,
                f"ACUMULADO: {buffer_numero}_ (Puno cerrado = confirmar)",
                (30, 85),
                font,
                0.7,
                (88, 166, 255),
                2,
            )
        elif estado_actual == 4:
            cv2.putText(img, "GRAFICA LISTA - R: Reiniciar | Q: Salir", (30, 60), font, 1.0, (63, 185, 80), 2)
            if grafica_img is not None:
                img[150:500, w_img - 380:w_img - 30] = grafica_img

        mostrar_cv2_al_frente(nombre_ventana, img)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord("r"):
            estado_actual = -1
            coeficientes = []
            buffer_numero = ""
            tipo_funcion = ""
            grafica_img = None
            ultimo_digito_metido = -1

    cap.release()
    hands.close()
    cv2.destroyAllWindows()


# ============================================================
# MODULO 2: CALCULADORA
# ============================================================

def abrir_calculadora(nombre_usuario):
    hands = mp_hands.Hands(
        max_num_hands=2,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.5,
    )

    estado_actual = 2
    operacion_texto = ""
    resultado_calculado = ""
    buffer_numero = ""
    numero_estabilizado = -1
    tiempo_inicio_estab = 0
    tiempo_requerido = 2.0
    porcentaje_carga = 0.0

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1024)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 768)
    nombre_ventana = "MathFest - Calculadora"
    preparar_ventana_cv2(nombre_ventana)

    while cap.isOpened():
        success, img = cap.read()
        if not success:
            break

        img = cv2.flip(img, 1)
        img = cv2.resize(img, (1024, 768))
        h_img, w_img, _ = img.shape
        results = hands.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

        dedos_izq = []
        dedos_der = []
        modo_simbolos_activo = False

        if results.multi_hand_landmarks:
            for i, hand_lms in enumerate(results.multi_hand_landmarks):
                label = results.multi_handedness[i].classification[0].label
                lms = [[int(lm.x * w_img), int(lm.y * h_img)] for lm in hand_lms.landmark]
                estado_ded = obtener_estado_dedos(lms, label)

                if label == "Left":
                    dedos_izq = estado_ded
                else:
                    dedos_der = estado_ded

                mp_draw.draw_landmarks(img, hand_lms, mp_hands.HAND_CONNECTIONS)

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

        if gesto_actual_valido:
            if valor_detectado_actual == numero_estabilizado:
                tiempo_transcurrido = time.time() - tiempo_inicio_estab
                porcentaje_carga = min(tiempo_transcurrido / tiempo_requerido, 1.0)

                if tiempo_transcurrido >= tiempo_requerido:
                    if tipo_entrada_actual == "BORRAR":
                        if buffer_numero:
                            buffer_numero = buffer_numero[:-1]
                        elif operacion_texto:
                            operacion_texto = operacion_texto[:-1]
                        time.sleep(0.5)
                    elif tipo_entrada_actual == "NUMERO":
                        if valor_detectado_actual > 0:
                            buffer_numero += str(valor_detectado_actual if valor_detectado_actual < 10 else 0)
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
                            if operacion_texto.count("(") > operacion_texto.count(")"):
                                operacion_texto += ")"
                            else:
                                operacion_texto += "("
                        else:
                            operacion_texto += valor_detectado_actual

                        time.sleep(0.5)

                    numero_estabilizado = -1
                    porcentaje_carga = 0.0
                    time.sleep(0.3)
            else:
                numero_estabilizado = valor_detectado_actual
                tiempo_inicio_estab = time.time()
                porcentaje_carga = 0.0
        else:
            numero_estabilizado = -1
            porcentaje_carga = 0.0

        cv2.rectangle(img, (0, 0), (w_img, 110), (14, 17, 23), -1)
        font = cv2.FONT_HERSHEY_SIMPLEX

        if porcentaje_carga > 0 and estado_actual == 2:
            cv2.rectangle(img, (0, 110), (int(w_img * porcentaje_carga), 118), (88, 166, 255), -1)

        cv2.putText(img, f"Usuario: {nombre_usuario}", (w_img - 280, 30), font, 0.6, (63, 185, 80), 1)

        if estado_actual == 2:
            if not results.multi_hand_landmarks:
                cv2.putText(img, "ESPERANDO DETECCION DE MANO...", (30, 40), font, 0.8, (201, 209, 217), 2)
            elif tipo_entrada_actual == "BORRAR":
                cv2.putText(img, "MODO: BORRAR", (30, 40), font, 0.8, (247, 129, 102), 2)
            elif modo_simbolos_activo:
                cv2.putText(
                    img,
                    f"MODO SIMBOLOS | Detectado: {valor_detectado_actual}",
                    (30, 40),
                    font,
                    0.8,
                    (88, 166, 255),
                    2,
                )
            else:
                digito_visual = str(valor_detectado_actual) if valor_detectado_actual > 0 else "Puno (Confirmar/Resolver)"
                cv2.putText(
                    img,
                    f"MODO NUMEROS | Digito: {digito_visual}",
                    (30, 40),
                    font,
                    0.8,
                    (63, 185, 80),
                    2,
                )

            cv2.putText(img, f"OPERACION: {operacion_texto}{buffer_numero}_", (30, 85), font, 1.0, (255, 255, 255), 2)
            cv2.rectangle(img, (20, h_img - 70), (w_img - 20, h_img - 20), (22, 27, 34), -1)
            cv2.putText(
                img,
                "Izq Abierta+Der: Rock=[+] Indice=[-] Amor&Paz=[*] Pistola=[/] 3Dedos=[()]",
                (35, h_img - 42),
                font,
                0.55,
                (48, 54, 61),
                1,
            )
        elif estado_actual == 4:
            cv2.putText(img, "RESULTADO", (30, 40), font, 0.9, (63, 185, 80), 2)
            cv2.putText(img, f"{operacion_texto} = {resultado_calculado}", (30, 85), font, 1.2, (255, 255, 255), 2)
            cv2.putText(img, "R: Nueva operacion | Q: Salir", (30, 140), font, 0.7, (88, 166, 255), 1)

        mostrar_cv2_al_frente(nombre_ventana, img)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord("r"):
            estado_actual = 2
            operacion_texto = ""
            buffer_numero = ""
            resultado_calculado = ""

    cap.release()
    hands.close()
    cv2.destroyAllWindows()


# ============================================================
# MODULO 3: ECUACIONES / FORMULA CUADRATICA
# ============================================================

def resolver_y_graficar_cuadratica(coeffs):
    a, b, c = coeffs[0], coeffs[1], coeffs[2]

    if a == 0:
        if b != 0:
            x1 = -c / b
            res_txt = f"Lineal: x = {round(x1, 2)}"
            raices_reales = [x1]
        else:
            res_txt = "Inconsistente"
            raices_reales = []
    else:
        discriminante = b**2 - 4 * a * c

        if discriminante > 0:
            x1 = (-b + np.sqrt(discriminante)) / (2 * a)
            x2 = (-b - np.sqrt(discriminante)) / (2 * a)
            res_txt = f"2 raices reales: x1={round(x1, 2)}, x2={round(x2, 2)}"
            raices_reales = [x1, x2]
        elif discriminante == 0:
            x1 = -b / (2 * a)
            res_txt = f"1 raiz real: x={round(x1, 2)}"
            raices_reales = [x1]
        else:
            x1 = (-b + cmath.sqrt(discriminante)) / (2 * a)
            x2 = (-b - cmath.sqrt(discriminante)) / (2 * a)
            res_txt = (
                f"Complejas: "
                f"x1={round(x1.real, 2)}+{round(x1.imag, 2)}i, "
                f"x2={round(x2.real, 2)}{round(x2.imag, 2)}i"
            )
            raices_reales = []

    plt.style.use("dark_background")
    fig, ax = plt.subplots(figsize=(4, 4), dpi=100)

    cx = -b / (2 * a) if a != 0 else 0
    x = np.linspace(cx - 10, cx + 10, 200)
    y = a * x**2 + b * x + c

    ax.plot(x, y, color="#DEFF9A", linewidth=3, label=f"{a}x^2 + {b}x + {c} = 0")
    ax.axhline(0, color="white", lw=1)
    ax.axvline(0, color="white", lw=1)
    ax.grid(True, alpha=0.3)

    for r in raices_reales:
        ax.plot(r, 0, marker="o", markersize=10, color="#FF5E5E", label=f"Raiz: {round(r, 1)}")

    ax.legend(prop={"size": 8})

    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight")
    buf.seek(0)

    img_arr = np.frombuffer(buf.getvalue(), dtype=np.uint8)
    plt.close(fig)

    return cv2.resize(cv2.imdecode(img_arr, cv2.IMREAD_COLOR), (350, 350)), res_txt


def abrir_ecuaciones(nombre_usuario):
    hands = mp_hands.Hands(
        max_num_hands=2,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.5,
    )

    estado_actual = 0
    coeficientes = []
    nombres_coef = ["A", "B", "C"]
    buffer_numero = ""
    ultimo_digito_metido = -1
    numero_estabilizado = -1
    tiempo_inicio_estabilidad = 0
    tiempo_requerido = 2.0
    grafica_img = None
    info_raices = ""

    cap = cv2.VideoCapture(0)
    w_disp, h_disp = 1024, 768
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, w_disp)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h_disp)
    nombre_ventana = "MathFest - Formula Cuadratica"
    preparar_ventana_cv2(nombre_ventana, w_disp, h_disp)

    while cap.isOpened():
        success, img = cap.read()
        if not success:
            break

        img = cv2.flip(img, 1)
        img = cv2.resize(img, (w_disp, h_disp))
        h_img, w_img, _ = img.shape
        results = hands.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

        dedos_actuales = -1
        porcentaje_carga = 0.0
        es_gesto_rock = False

        if results.multi_hand_landmarks:
            dedos_totales = 0

            for i, hand_lms in enumerate(results.multi_hand_landmarks):
                label = results.multi_handedness[i].classification[0].label
                lms = [[int(lm.x * w_img), int(lm.y * h_img)] for lm in hand_lms.landmark]
                estado_dedos = obtener_estado_dedos(lms, label)

                dedos_totales += estado_dedos.count(1)

                if estado_dedos == [0, 1, 0, 0, 1]:
                    es_gesto_rock = True

                mp_draw.draw_landmarks(img, hand_lms, mp_hands.HAND_CONNECTIONS)

            dedos_actuales = dedos_totales

        modo_borrar_activo = dedos_actuales == 9
        modo_signo_activo = es_gesto_rock and not modo_borrar_activo

        if modo_borrar_activo:
            valor_a_estabilizar = "BORRAR"
        elif modo_signo_activo:
            valor_a_estabilizar = "MENOS"
        else:
            valor_a_estabilizar = dedos_actuales

        if dedos_actuales != -1:
            if valor_a_estabilizar == numero_estabilizado:
                tiempo_transcurrido = time.time() - tiempo_inicio_estabilidad
                porcentaje_carga = min(tiempo_transcurrido / tiempo_requerido, 1.0)

                if tiempo_transcurrido >= tiempo_requerido:
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
                        if buffer_numero.startswith("-"):
                            buffer_numero = buffer_numero[1:]
                        else:
                            buffer_numero = "-" + buffer_numero
                        ultimo_digito_metido = -1
                        time.sleep(0.5)

                    elif 0 <= estado_actual < 3:
                        if dedos_actuales > 0:
                            if dedos_actuales != ultimo_digito_metido:
                                digito = str(dedos_actuales if dedos_actuales < 10 else 0)
                                buffer_numero += digito
                                ultimo_digito_metido = dedos_actuales
                        else:
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

        cv2.rectangle(img, (0, 0), (w_img, 110), (20, 20, 20), -1)
        font = cv2.FONT_HERSHEY_SIMPLEX

        if porcentaje_carga > 0 and estado_actual < 3:
            ancho_barra = int(w_img * porcentaje_carga)
            cv2.rectangle(img, (0, 110), (ancho_barra, 118), (154, 255, 222), -1)

        cv2.putText(img, f"Usuario: {nombre_usuario}", (w_img - 280, 30), font, 0.6, (63, 185, 80), 1)

        if estado_actual < 3:
            if numero_estabilizado == "BORRAR":
                txt_status = "MODO: BORRAR ULTIMO DIGITO / COEFICIENTE"
                color_txt = (0, 0, 255)
            elif numero_estabilizado == "MENOS":
                txt_status = "MODO: INVERTIR SIGNO (+/-)"
                color_txt = (255, 128, 0)
            else:
                txt_status = f"FORMULA GENERAL | Coeficiente {nombres_coef[estado_actual]}"
                color_txt = (255, 255, 255)

            cv2.putText(img, txt_status, (30, 45), font, 0.9, color_txt, 2)
            cv2.putText(
                img,
                f"VALOR PARA {nombres_coef[estado_actual]}: {buffer_numero}_ (Puno = guardar)",
                (30, 85),
                font,
                0.75,
                (154, 255, 222),
                2,
            )

            cv2.rectangle(img, (20, h_img - 65), (w_img - 20, h_img - 20), (35, 35, 35), -1)
            guias_txt = "Gestos: 9 dedos = Borrar | Rock = Cambiar +/- | Puno = Guardar"
            cv2.putText(img, guias_txt, (35, h_img - 38), font, 0.6, (180, 180, 180), 1)

        elif estado_actual == 3:
            cv2.putText(
                img,
                f"ECUACION: {coeficientes[0]}x^2 + ({coeficientes[1]})x + ({coeficientes[2]}) = 0",
                (30, 45),
                font,
                0.9,
                (255, 255, 255),
                2,
            )
            cv2.putText(img, f"SOLUCION: {info_raices}", (30, 85), font, 0.75, (0, 255, 0), 2)
            cv2.putText(img, "R: Resolver otra ecuacion | Q: Salir", (30, 140), font, 0.65, (170, 170, 170), 1)

            if grafica_img is not None:
                img[180:530, w_img - 380:w_img - 30] = grafica_img

        mostrar_cv2_al_frente(nombre_ventana, img)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord("r"):
            estado_actual = 0
            coeficientes = []
            buffer_numero = ""
            ultimo_digito_metido = -1
            numero_estabilizado = -1
            grafica_img = None
            info_raices = ""

    cap.release()
    hands.close()
    cv2.destroyAllWindows()


# ============================================================
# INTERFAZ FLET ESTILO MONITOR
# ============================================================

BG = "#050b18"
BG2 = "#0b1424"
BORDER = "#5f718a"
CYAN = "#63e7ff"
GREEN = "#74ff9e"
YELLOW = "#f6dc78"
RED = "#ff8c78"
TEXT = "#eef7ff"
MUTED = "#8fa3b8"

ALIGN_CENTER = ft.Alignment(0, 0)
ALIGN_CENTER_LEFT = ft.Alignment(-1, 0)
ALIGN_CENTER_RIGHT = ft.Alignment(1, 0)


def icono(nombre, respaldo):
    icons = getattr(ft, "Icons", None) or getattr(ft, "icons", None)
    return getattr(icons, nombre, respaldo) if icons else respaldo


def main(page: ft.Page):
    page.title = "MATHFEST"
    page.window_width = 980
    page.window_height = 650
    page.window_min_width = 860
    page.window_min_height = 560
    page.theme_mode = ft.ThemeMode.DARK
    page.bgcolor = BG
    page.padding = 0

    nombre_input = ft.TextField(
        label="Nombre",
        width=270,
        height=46,
        bgcolor="#0d1829",
        border_color="#3c536f",
        focused_border_color=CYAN,
        cursor_color=CYAN,
        label_style=ft.TextStyle(color=MUTED),
        text_style=ft.TextStyle(size=13, color=TEXT),
    )
    mensaje = ft.Text("", size=12, color=RED)

    def estrellas():
        data = [
            (35, 48, 2), (74, 126, 1), (110, 82, 2), (166, 38, 1), (212, 128, 2),
            (260, 58, 1), (314, 116, 2), (382, 68, 1), (442, 42, 2), (514, 100, 1),
            (584, 56, 2), (646, 134, 1), (704, 64, 2), (780, 112, 1), (842, 44, 2),
            (902, 94, 1), (58, 420, 1), (136, 356, 2), (214, 474, 1), (306, 392, 2),
            (420, 504, 1), (522, 448, 2), (636, 520, 1), (728, 398, 2), (828, 480, 1),
            (918, 360, 2),
        ]
        return [
            ft.Container(left=x, top=y, width=s, height=s, border_radius=10, bgcolor="#dffbff", opacity=0.75)
            for x, y, s in data
        ]

    def formulas():
        datos = [
            ("x(t) = (a + b)^2", 690, 68, 0.35),
            ("x^2 = 8 lambda", 725, 112, 0.28),
            ("y = x - x'", 702, 168, 0.28),
            ("f(x,y) = 2e^x", 725, 430, 0.23),
            ("(y, 3x - y) / 2", 700, 360, 0.22),
            ("Sigma", 112, 420, 0.20),
        ]
        return [
            ft.Text(texto, left=x, top=y, size=18, color="#d8f3ff", opacity=o, italic=True)
            for texto, x, y, o in datos
        ]

    def forma_geometrica(left, top, size=150, opacity=0.22):
        return ft.Container(
            left=left,
            top=top,
            width=size,
            height=size,
            opacity=opacity,
            content=ft.Stack(
                controls=[
                    ft.Container(left=18, top=12, width=92, height=1, rotate=0.75, bgcolor="#7fdfff"),
                    ft.Container(left=22, top=24, width=112, height=1, rotate=0.18, bgcolor="#7fdfff"),
                    ft.Container(left=38, top=86, width=118, height=1, rotate=-0.45, bgcolor="#7fdfff"),
                    ft.Container(left=34, top=22, width=1, height=96, rotate=-0.55, bgcolor="#7fdfff"),
                    ft.Container(left=112, top=12, width=1, height=128, rotate=0.72, bgcolor="#7fdfff"),
                    ft.Container(left=18, top=116, width=112, height=1, rotate=0.15, bgcolor="#7fdfff"),
                    ft.Container(left=52, top=42, width=84, height=84, border=ft.border.all(1, "#7fdfff")),
                ]
            ),
        )

    def abaco():
        colores = [GREEN, "#ffb66b", "#c899ff", CYAN]
        piezas = []

        for fila in range(4):
            y = 18 + fila * 24
            piezas.append(ft.Container(left=18, top=y + 8, width=118, height=4, bgcolor="#d9e3ef", opacity=0.75))
            for columna in range(5):
                piezas.append(
                    ft.Container(
                        left=30 + columna * 18,
                        top=y,
                        width=14,
                        height=14,
                        border_radius=7,
                        bgcolor=colores[fila],
                        shadow=ft.BoxShadow(blur_radius=8, color=colores[fila]),
                    )
                )

        return ft.Container(
            left=124,
            top=250,
            width=160,
            height=140,
            rotate=-0.14,
            content=ft.Stack(
                controls=[
                    ft.Container(left=8, top=4, width=14, height=125, border_radius=4, bgcolor="#d8dce4"),
                    ft.Container(left=134, top=4, width=14, height=125, border_radius=4, bgcolor="#d8dce4"),
                    ft.Container(left=0, top=122, width=154, height=10, border_radius=3, bgcolor="#d8dce4"),
                    *piezas,
                ]
            ),
        )

    def cerebro():
        return ft.Container(
            right=120,
            top=235,
            width=165,
            height=135,
            alignment=ALIGN_CENTER,
            content=ft.Icon(icono("PSYCHOLOGY", "psychology"), size=118, color="#9df2ff", opacity=0.88),
            shadow=ft.BoxShadow(blur_radius=30, spread_radius=1, color="#2de8ff"),
        )

    def usuario_pildora(nombre):
        return ft.Container(
            width=196,
            height=48,
            border_radius=10,
            bgcolor="#10233a",
            border=ft.border.all(1, "#58c7ed"),
            shadow=ft.BoxShadow(blur_radius=18, color="#1bbde5"),
            content=ft.Row(
                alignment=ft.MainAxisAlignment.CENTER,
                spacing=8,
                controls=[
                    ft.Text(nombre.upper(), size=13, weight=ft.FontWeight.BOLD, color=TEXT),
                    ft.Container(
                        width=38,
                        height=38,
                        border_radius=20,
                        bgcolor="#dff8ff",
                        alignment=ALIGN_CENTER,
                        content=ft.Icon(icono("PERSON", "person"), color="#15344c", size=29),
                    ),
                ],
            ),
        )

    def boton_brillante(titulo, nombre_icono, color, accion):
        return ft.Container(
            width=320,
            height=52,
            border_radius=22,
            bgcolor="#263447",
            border=ft.border.all(1.2, "#d5ecff"),
            gradient=ft.LinearGradient(
                begin=ALIGN_CENTER_LEFT,
                end=ALIGN_CENTER_RIGHT,
                colors=["#5c6672", "#263447", "#607d92"],
            ),
            shadow=ft.BoxShadow(blur_radius=18, spread_radius=0.5, color=color),
            on_click=accion,
            content=ft.Row(
                alignment=ft.MainAxisAlignment.CENTER,
                spacing=14,
                controls=[
                    ft.Container(width=32),
                    ft.Text(titulo, expand=True, text_align=ft.TextAlign.CENTER, size=18, weight=ft.FontWeight.BOLD, color=TEXT),
                    ft.Icon(icono(nombre_icono, nombre_icono.lower()), size=28, color=color),
                    ft.Container(width=14),
                ],
            ),
        )

    def cerrar_sesion(_=None):
        nombre_input.value = ""
        mensaje.value = ""
        mostrar_login()

    def pantalla_menu(nombre):
        return ft.Stack(
            expand=True,
            controls=[
                ft.Container(
                    expand=True,
                    gradient=ft.RadialGradient(
                        center=ALIGN_CENTER,
                        radius=1.25,
                        colors=["#10223c", "#08111f", "#020713"],
                    ),
                ),
                *estrellas(),
                *formulas(),
                forma_geometrica(72, 42),
                forma_geometrica(74, 384, 190, 0.16),
                abaco(),
                cerebro(),
                ft.Container(
                    left=0,
                    right=0,
                    top=52,
                    alignment=ALIGN_CENTER,
                    content=ft.Column(
                        horizontal_alignment=ft.CrossAxisAlignment.CENTER,
                        spacing=14,
                        controls=[
                            ft.Text(
                                "Bienvenidos al MATHFEST",
                                size=34,
                                weight=ft.FontWeight.BOLD,
                                color="#89f8ff",
                            ),
                            usuario_pildora(nombre),
                            ft.Container(height=12),
                            boton_brillante(
                                "Operaciones Compuestas",
                                "CALCULATE",
                                GREEN,
                                lambda _: threading.Thread(target=abrir_calculadora, args=(nombre,), daemon=True).start(),
                            ),
                            boton_brillante(
                                "Graficas",
                                "SHOW_CHART",
                                CYAN,
                                lambda _: threading.Thread(target=abrir_graficadora, args=(nombre,), daemon=True).start(),
                            ),
                            boton_brillante(
                                "Ecuaciones",
                                "FUNCTIONS",
                                YELLOW,
                                lambda _: threading.Thread(target=abrir_ecuaciones, args=(nombre,), daemon=True).start(),
                            ),
                            ft.Container(height=12),
                            ft.Container(
                                width=160,
                                height=36,
                                border_radius=18,
                                bgcolor="#5b2b24",
                                border=ft.border.all(1, "#d68a78"),
                                shadow=ft.BoxShadow(blur_radius=10, color="#9d4437"),
                                on_click=cerrar_sesion,
                                content=ft.Row(
                                    alignment=ft.MainAxisAlignment.CENTER,
                                    spacing=8,
                                    controls=[
                                        ft.Text("Cerrar Sesion", size=12, weight=ft.FontWeight.BOLD, color=TEXT),
                                        ft.Icon(icono("POWER_SETTINGS_NEW", "power_settings_new"), size=16, color=RED),
                                    ],
                                ),
                            ),
                        ],
                    ),
                ),
            ],
        )

    def pantalla_login():
        return ft.Stack(
            expand=True,
            controls=[
                ft.Container(
                    expand=True,
                    gradient=ft.RadialGradient(
                        center=ALIGN_CENTER,
                        radius=1.2,
                        colors=["#10223c", "#08111f", "#020713"],
                    ),
                ),
                *estrellas(),
                *formulas(),
                forma_geometrica(82, 54),
                cerebro(),
                ft.Container(
                    left=0,
                    right=0,
                    top=95,
                    alignment=ALIGN_CENTER,
                    content=ft.Column(
                        width=380,
                        horizontal_alignment=ft.CrossAxisAlignment.CENTER,
                        spacing=16,
                        controls=[
                            ft.Text(
                                "Bienvenidos al MATHFEST",
                                size=32,
                                weight=ft.FontWeight.BOLD,
                                text_align=ft.TextAlign.CENTER,
                                color="#89f8ff",
                            ),
                            ft.Container(height=4),
                            nombre_input,
                            mensaje,
                            ft.Container(
                                width=210,
                                height=46,
                                border_radius=22,
                                bgcolor="#263447",
                                border=ft.border.all(1.2, "#d5ecff"),
                                shadow=ft.BoxShadow(blur_radius=18, color=CYAN),
                                on_click=lambda _: entrar(),
                                content=ft.Row(
                                    alignment=ft.MainAxisAlignment.CENTER,
                                    spacing=9,
                                    controls=[
                                        ft.Text("Ingresar", size=16, weight=ft.FontWeight.BOLD, color=TEXT),
                                        ft.Icon(icono("LOGIN", "login"), size=22, color=CYAN),
                                    ],
                                ),
                            ),
                        ],
                    ),
                ),
            ],
        )

    def entrar():
        nombre = nombre_input.value.strip()
        if not nombre:
            mensaje.value = "Por favor ingrese su nombre."
            page.update()
            return

        page.controls.clear()
        page.add(pantalla_menu(nombre))
        page.update()

    def mostrar_login():
        page.controls.clear()
        page.add(pantalla_login())
        page.update()

    mostrar_login()


ft.app(target=main)