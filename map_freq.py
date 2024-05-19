import cv2
import numpy as np
from pythonosc import udp_client

# configurar la dirección y el puerto
client = udp_client.SimpleUDPClient('127.0.0.1', 57120)

# capturar el video
cap = cv2.VideoCapture(0)

#inicializar cuadro para detectar movimiento
ret, frame = cap.read()
background = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
background = cv2.GaussianBlur(background, (21, 21), 0)

while True:
    # capturar cuadro por cuadro
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (21, 21), 0)

    #calculamos diferencia absoluta entre el fondo y el cuadro actual
    frameDelta = cv2.absdiff(background, gray)

    #aplicar umbral a la imagen. todos los pixeles con una diferencia mayor a 30 se convierte en blanco
    thresh = cv2.threshold(frameDelta, 30, 255, cv2.THRESH_BINARY)[1]

    #encontrar contornos de las áreas blancas
    contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        print(cv2.contourArea(contour))
        if cv2.contourArea(contour) < 1500:
            continue
    
        # dibujar un rectangulo al rededor del contorno
        (x, y, w, h) = cv2.boundingRect(contour)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        #calculamos el centro del movimiento
        centerX = x + w / 2
        centerY = y + h / 2

        # mapeamos el movimiento
        freq = np.interp(centerX, [0, frame.shape[1]], [200, 1000])

        #enviar a supercollider
        client.send_message("/movement", freq)

        # imprimimos los valores en la imagen
        text = "Freq: {:.2f}".format(freq)
        cv2.putText(frame, text, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    cv2.imshow('Motion Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()