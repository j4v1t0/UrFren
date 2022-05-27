import cv2
from PIL import Image

camera = cv2.VideoCapture(0)
camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1024)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 768)
ret, image = camera.read()
cv2.imwrite("test_image.jpg", image)
