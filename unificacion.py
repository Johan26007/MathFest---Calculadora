import flet as ft
import cv2
import mediapipe as mp
import time
import numpy as np
import matplotlib.pyplot as plt
import io

# Logica principal

def contar_dedos(landmarks, tip_ids):
    fingers = []
    if landmarks[tip_ids] > landmarks[tip_ids - 1]: fingers.append(1)
    else: fingers.append(0)
    for id in range(1, 5):
        if landmarks[tip_ids[id]] < landmarks[tip_ids[id] - 2]: fingers.append(1)
        else: fingers.append(0)
    return fingers.count(1)

def generar_grafica_cv2(tipo, coeffs):
    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(4, 4), dpi=100)
    x = np.linspace(-10, 10, 100)
    
    if tipo == "lineal":
        y = coeffs * x + coeffs
        label = f"y = {coeffs}x + {coeffs}"
    elif tipo == "cuadratica":
        y = coeffs*x**2 + coeffs*x + coeffs
        label = f"y = {coeffs}x² + {coeffs}x + {coeffs}"
    else: # cubica
        y = coeffs*x**3 + coeffs*x**2 + coeffs*x + coeffs
        label = f"y = {coeffs}x³ + ..."

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

def abrir_camara_mathfest():
    mp_hands = mp.solutions.hands
    mp_draw = mp.solutions.drawing_utils
    hands = mp_hands.Hands(min_detection_confidence=0.8, min_tracking_confidence=0.5)
    tip_ids = [4, 8, 12, 16, 20]

    estado_actual = -1 
    tipo_funcion = "" 
    coeficientes = []
    nombres_coef = ["A", "B", "C", "D"]
    numero_estabilizado = -1
    tiempo_inicio_estabilidad = 0
    TIEMPO_REQUERIDO = 2.0 
    grafica_img = None 

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
                dedos_actuales = contar_dedos(landmarks, tip_ids)
                mp_draw.draw_landmarks(img, hand_lms, mp_hands.HAND_CONNECTIONS)

        if dedos_actuales != -1:
            if dedos_actuales == numero_estabilizado:
                if time.time() - tiempo_inicio_estabilidad >= TIEMPO_REQUERIDO:
                    if estado_actual == -1:
                        if dedos_actuales == 1: tipo_funcion, estado_actual = "lineal", 0
                        elif dedos_actuales == 2: tipo_funcion, estado_actual = "cuadratica", 0
                        elif dedos_actuales == 3: tipo_funcion, estado_actual = "cubica", 0
                    elif 0 <= estado_actual < 4:
                        coeficientes.append(dedos_actuales)
                        limite = 2 if tipo_funcion == "lineal" else 3 if tipo_funcion == "cuadratica" else 4
                        estado_actual += 1
                        if estado_actual == limite:
                            grafica_img = generar_grafica_cv2(tipo_funcion, coeficientes)
                            estado_actual = 4
                    numero_estabilizado = -1
                    time.sleep(0.5)
                
                porcentaje = min((time.time() - tiempo_inicio_estabilidad) / TIEMPO_REQUERIDO, 1.0)
                cv2.rectangle(img, (w_img//2 - 200, 650), (w_img//2 - 200 + int(400 * porcentaje), 670), (0, 255, 0), cv2.FILLED)
            else:
                numero_estabilizado = dedos_actuales
                tiempo_inicio_estabilidad = time.time()
        else:
            numero_estabilizado = -1

        cv2.rectangle(img, (0, 0), (w_img, 120), (30, 30, 30), cv2.FILLED)
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
            cv2.putText(img, "Presiona 'R' para Menu | 'Q' para Volver a la App", (40, 100), fuente, 0.8, (200, 200, 200), 1)
            if grafica_img is not None:
                img[150:500, w_img-400:w_img-50] = grafica_img

        cv2.imshow(Ventana_Nombre, img)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'): break
        if key == ord('r'):
            estado_actual, coeficientes, grafica_img, tipo_funcion = -1, [], None, ""

    cap.release()
    cv2.destroyAllWindows()

#Interfaz Flet

def main(page: ft.Page):
    page.title = "MATHFEST"
    page.window_width = 500
    page.window_height = 600
    page.theme_mode = ft.ThemeMode.DARK
    page.bgcolor = "#1a1a1a"
    page.vertical_alignment = ft.MainAxisAlignment.CENTER
    page.horizontal_alignment = ft.CrossAxisAlignment.CENTER

    label_titulo = ft.Text("Bienvenidos al MATHFEST", size=25, weight="bold", color="white")
    entrada_nombre = ft.TextField(label="Ingrese su Nombre", width=300, border_radius=10)
    label_mensaje = ft.Text("", size=16)

    # Nota: Cambiar nombres de las img
    try:
        img_calcula = ft.Image(src="Img/Calcula.png", width=100, height=100)
        img_cere = ft.Image(src="Img/cere.png", width=100, height=100)
    except:
        # Fallback si no encuentra imágenes
        img_calcula = ft.Icon(ft.icons.CALCULATE, size=100)
        img_cere = ft.Icon(ft.icons.PSYCHOLOGY, size=100)

    def limpiar_y_actualizar(nuevos_controles):
        page.controls.clear()
        page.add(
            ft.Row([img_calcula, ft.Column(nuevos_controles, horizontal_alignment="center"), img_cere], 
                   alignment="center")
        )
        page.update()

    def mostrar_pantalla_inicial(e=None):
        label_mensaje.value = ""
        limpiar_y_actualizar([
            label_titulo,
            ft.Container(height=20),
            ft.Text("Ingrese su Nombre", size=20),
            entrada_nombre,
            label_mensaje,
            ft.ElevatedButton("Ingresar", on_click=validar_ingreso, bgcolor="purple", color="white")
        ])

    def validar_ingreso(e):
        if entrada_nombre.value == "":
            label_mensaje.value = "Por favor, ingrese su nombre"
            label_mensaje.color = "red"
            page.update()
        else:
            mostrar_menu_operaciones()

    def mostrar_menu_operaciones(e=None):
        nombre = entrada_nombre.value
        limpiar_y_actualizar([
            label_titulo,
            ft.Text(f"Buenos días {nombre}, Bienvenid@", size=18, color="blue"),
            ft.Text("¿Qué operación desea realizar?", size=20),
            ft.Container(height=10),
            ft.ElevatedButton("Operaciones Compuestas", width=300, on_click=lambda _: print("Op Compuestas")),
            ft.ElevatedButton("Gráficas", width=300, on_click=abrir_modulo_graficas),
            ft.ElevatedButton("Ecuaciones", width=300, on_click=lambda _: print("Ecuaciones")),
            ft.ElevatedButton("Cerrar Sesión", width=300, on_click=mostrar_pantalla_inicial, bgcolor="red", color="white")
        ])

    def abrir_modulo_graficas(e):
        limpiar_y_actualizar([
            ft.Text("Módulo de Gráficas", size=25, weight="bold"),
            ft.Text("La cámara se abrirá en una ventana emergente...", size=14),
            ft.ProgressRing(),
            ft.Container(height=20),
            ft.ElevatedButton("Regresar al Menú", on_click=mostrar_menu_operaciones, bgcolor="purple", color="white")
        ])
        # Ejecucion con OpenCV
        abrir_camara_mathfest()

    mostrar_pantalla_inicial()

ft.app(target=main)