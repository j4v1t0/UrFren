# Import libraries for the project:
import time
import torch
import cv2
import random
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import telepot

# Functions for the game:
# Import the yolov5n model, pretrained, in order to execute everything:
model = torch.hub.load('ultralytics/yolov5', 'yolov5n', pretrained=True)

# Initializes global variables:
playerAverageList = []
playerNameList = []

# Calculate IOU:
# @args box_f1, box_f2
# @returns iou
# Takes the coordenates of the boxes of a player in two frames and returns the IOU value in order to check whether or not movement has happened.
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
  #print("(IOU): ", iou)
  return iou


# Detect Movements:
# @args cameraResults
# @returns movementDetected, iou_list
# Calls the calculateIOU function for every player and checks for the IOU thresh-hold.
def detectMovements(cameraResults):
  movementDetected = []
  iou_list = []
  for i in range(len(cameraResults[0])):
    iou = calculateIOU(cameraResults[0][i], cameraResults[1][i])
    iou_list.append(iou)
    if (iou < 0.75):
      movementDetected.append(True)
    else:
      movementDetected.append(False)

  return movementDetected, iou_list


# Detect Image Movement:
# @args path1, path2
# @returns None
# Passes both frames through the YOLOv5 model and compares the bounding boxes.
def detectImageMovement(path1,path2):
  imgs = [path1,path2]
  results = model(imgs)
  res_list, iou_list = detectMovements(results.xyxy[0].detach().cpu().numpy())
  print(res_list)


# Average Color:
# @args img, p1, p2
# @returns avrg
# Takes the center of an image in order to obtain the average color and identify the player through it.
def avrgColor(img, p1, p2):
  aux1 = img[p1[1]:p2[1],p1[0]:p2[0]]
  sizeBox = aux1.shape
  aux2 = aux1[int(0.3*sizeBox[0]):int(0.7*sizeBox[0]), int(0.3*sizeBox[1]):int(0.7*sizeBox[1])]
  avrg = np.average(aux2, axis=0)
  avrg = np.average(avrg, axis=0)
  return avrg

# Get Averages:
# @args results, path
# @returns avrg_list
# Takes the model prediction and passes the image and the bounding box points to the avrgColor function to pass it on a list.
def getAverages(results, path):
  avrg_list = []
  res = results.xyxy[0].detach().cpu().numpy()
  img = cv2.imread(path)

  for result in res:
    p1 = (int(result[0]), int(result[1]))
    p2 = (int(result[2]), int(result[3]))
    avrg_list.append(avrgColor(img, p1, p2))

  return avrg_list

# Closest Average:
# @args avg_list, new_avg_list, results
# @returns ret, aux
# Takes the averages of the new frame and reorganizes the list in order to keep it consistent with the original order and identify players correctly through the color.
def closestAverage(avg_list, new_avg_list, results):
  shape_average = np.array(avg_list).shape
  shape_results = np.array(results).shape
  ret = np.zeros(shape_average)
  aux = np.zeros(shape_results)
  for i in range(len(new_avg_list)):
    score = [0 for i in range(len(new_avg_list))]
    for j in range(len(avg_list)):
      score[j] += abs(avg_list[j][0]-new_avg_list[i][0])
      score[j] += abs(avg_list[j][0]-new_avg_list[i][0])
      score[j] += abs(avg_list[j][0]-new_avg_list[i][0])
    ret[np.argmin(score)] = new_avg_list[i]
    aux[np.argmin(score)] = results[i]
  return ret, aux


# Add Player:
def addPlayer(path, name):
  playerAverageList.append(getAverages(model(path), path)[0])
  playerNameList.append(name)


# Select Deleted Players:
# @args movement
# @returns ret
# Takes the array of players that have moved and returns the indexes.
def selectDeletedPlayers(movement):
  ret = []
  for i in range(len(movement)):
    if movement[i]:
      ret.append(i)
  return ret


# Delete Player:
# @args index
# @returns a, b
# Returns the updated list of players and names without the deleted ones.
def deletePlayer(index):
  a = playerAverageList
  b = playerNameList
  for i in reversed(range(len(index))):
    a = np.delete(a, index[i])
    b = np.delete(b, index[i])
  return a, b


# Inicialitzar jugador de prova:
player1path = "./imagesTest/1persona01.jpg"
player1name = "Player1"
addPlayer(player1path, player1name)
print(playerNameList)

# Frames:
path = "./imagesTest/1persona01.jpg"
pred = model(path)
avgs = getAverages(pred, path)
playerAverageList, bbox1 = closestAverage(playerAverageList, avgs, pred.xyxy[0].detach().cpu().numpy())

path = "./imagesTest/1persona02.jpg"
pred = model(path)
avgs = getAverages(pred, path)
playerAverageList, bbox2 = closestAverage(playerAverageList, avgs, pred.xyxy[0].detach().cpu().numpy())

# Detect Movements:
movList, iouList = detectMovements((bbox1, bbox2))
index = selectDeletedPlayers(movList)
playerAverageList, playerNameList = deletePlayer(index)

# Check results:
print(playerNameList)
