import sys
import cv2


# Nombre de la imagen y del archivo cascada
imagePath = "Jose.jpg"
cascPath = "haarcascade_frontalface_default.xml"

# creamos la cascada y la inicializamos con nuestra cascada de caras
faceCascade = cv2.CascadeClassifier(cascPath)

# Se lee la imagen y se procede a convertir en escala de grises
image = cv2.imread(imagePath)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Se procede a detectar los rostros en la imagen
#La función detectMultiScale es una función general que detecta objetos. 
#Como la estamos llamando en la cascada de la cara, eso es lo que detecta.
#El algoritmo de detección utiliza una ventana móvil para detectar objetos. 
# define cuántos objetos se detectan cerca del actual antes de declarar la cara encontrada. 
# Mientras tanto, da el tamaño de cada ventana.minNeighborsminSize
faces = faceCascade.detectMultiScale(
    gray,
    scaleFactor=1.1,
    minNeighbors=5,
    minSize=(30, 30),
    #flags = cv2.cv.CV_HAAR_SCALE_IMAGE
)

print("Found {0} faces!".format(len(faces)))

# Se procede a dibujar los rectangulos en las caras detectadas.
for (x, y, w, h) in faces:
    cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)

cv2.imshow("Faces found", image)
cv2.waitKey(0)
