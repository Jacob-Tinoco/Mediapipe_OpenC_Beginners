import cv2
import mediapipe as mp
import numpy as np
import argparse
import time

def parse_arguments():
    parser = argparse.ArgumentParser(description='Reconocimiento de gestos de mano usando MediaPipe.')
    parser.add_argument('--debug', action='store_true', help='Muestra información de depuración en la consola.')
    return parser.parse_args()

def main():
    args = parse_arguments()

    # Inicializa MediaPipe Hands.
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(static_image_mode=False,
                           max_num_hands=1,
                           min_detection_confidence=0.7,
                           min_tracking_confidence=0.7)

    mp_drawing = mp.solutions.drawing_utils

    # Captura de video desde la cámara web.
    cap = cv2.VideoCapture(0)

    while cap.isOpened():
        success, image = cap.read()
        if not success:
            if args.debug:
                print("No se puede acceder a la cámara.")
            break

        # Convierte la imagen de BGR a RGB.
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Procesa la imagen para detectar manos.
        results = hands.process(image_rgb)

        # Dibuja las anotaciones de los resultados.
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                gesture = recognize_gesture(hand_landmarks)
                if args.debug:
                    print(f"Gesto detectado: {gesture}")
                cv2.putText(image, f'Gesto: {gesture}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

        # Muestra la imagen con las anotaciones.
        cv2.imshow('MediaPipe Hands', image)
        if cv2.waitKey(5) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

def recognize_gesture(hand_landmarks):
    # Lista de índices de los puntos de referencia para las puntas de los dedos
    finger_tips_ids = [4, 8, 12, 16, 20]
    fingers = []

    # Pulgar
    if hand_landmarks.landmark[finger_tips_ids[0]].x < hand_landmarks.landmark[finger_tips_ids[0] - 1].x:
        fingers.append(1)
    else:
        fingers.append(0)

    # Otros dedos
    for id in range(1, 5):
        # Compara la posición y de la punta del dedo con el nudillo
        if hand_landmarks.landmark[finger_tips_ids[id]].y < hand_landmarks.landmark[finger_tips_ids[id] - 2].y:
            fingers.append(1)
        else:
            fingers.append(0)

    # Devuelve la representación del gesto como una cadena de 0s y 1s
    return ''.join(map(str, fingers))

if __name__ == "__main__":
    main()
