import cv2
import sys

cascPath = "haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascPath)

#Esta línea establece el origen de vídeo en la cámara web predeterminad
video_capture = cv2.VideoCapture(0)

while True:
    # Se captura el video. La función lee un fotograma de la fuente de vídeo, 
    # que en este ejemplo es la cámara web. Esto devuelve:read()
    #El fotograma de vídeo real leído (un fotograma en cada bucle)
    #Un código de retorno
    #El código de retorno nos indica si nos hemos quedado sin marcos, 
    # lo que sucederá si estamos leyendo de un archivo
    ret, frame = video_capture.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
       # flags=cv2.cv.CV_HAAR_SCALE_IMAGE
    )

    # Se dibuja el rectángulo  alrededor de los rostros
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    # Se muestra el resulado en el frame
    cv2.imshow('Video', frame)
    
    # Acá busca la cara en el marco capturado o se espera a presionar la tecla q
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Limpieza después de ejecutado el programa
video_capture.release()
cv2.destroyAllWindows()
