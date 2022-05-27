import time
import cv2
import random
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torch

# Importem el model pre-entrenat:
model = torch.hub.load("ultralytics/yolov5", "yolov5n", pretrained=True)

# Compara dos cajas de una sola persona para calcular la métrica IOU y ver si se
# ha movido o no la persona según el overlap:
def calculateIOU(box_f1, box_f2):
  # Calculate area of each box
  area_box1 = (box_f1[2] - box_f1[0]) * (box_f1[3] - box_f1[1])
  area_box2 = (box_f2[2] - box_f2[0]) * (box_f2[3] - box_f2[1])

  # Area of intersection:
  x_inter1 = max(box_f1[0], box_f2[0])
  y_inter1 = max(box_f1[1], box_f2[1])
  x_inter2 = min(box_f1[2], box_f2[2])
  y_inter2 = min(box_f1[3], box_f2[3])
  area_inter = (x_inter2 - x_inter1) * (y_inter2 - y_inter1)

  # IOU Calculation:
  area_union = area_box1 + area_box2 - area_inter
  iou = area_inter / area_union
  print("(IOU): ", iou)
  return iou.detach().cpu().numpy()

# Segun los resultados de la cámara, intenta detectar si un usuario se ha movido
# o no se ha movido:
def detectMovements(cameraResults):
  movementDetected = []
  for i in range(len(cameraResults.xyxy[0])):
    iou = calculateIOU(cameraResults.xyxy[0][i], cameraResults.xyxy[1][i])
    if (iou < 0.75):
      print("Person ", i, " has moved!")
      movementDetected.append(True)
    else:
      movementDetected.append(False)

  return movementDetected

def detectImageMovement(path1,path2):
  imgs = [path1,path2]
  results = model(imgs)
  res = detectMovements(results)
  print(res)

p1 = './imagesTest/1persona01.jpg'
p2 = './imagesTest/1persona02.jpg'

start = time.time()
detectImageMovement(p1,p2)
print(time.time() - start)
