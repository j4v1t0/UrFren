# Import libraries for the project:
import os
import time
from gpiozero import Servo, Device
from gpiozero.pins.pigpio import PiGPIOFactory
import torch
import cv2
import random
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import telepot

# Use pigpiod:
os.system("sudo pigpiod")
time.sleep(5)
Device.pin_factory = PiGPIOFactory()

# Check if the program has started by using the servo:
time.sleep(2)
servo = Servo(14)
servo.min()
time.sleep(1)
servo.max()

# Functions for the game:
# Import the yolov5n model, pretrained, in order to execute everything:
model = torch.hub.load('ultralytics/yolov5', 'yolov5n', pretrained=True)

# Initializes global variables:
bot = telepot.Bot("5317011276:AAEGqjgjHoje6YEFf1F8QddtOdiHqtYTDP0")
playerAverageList = []
playerNameList = []
telegram_mode = "main"
ingame = False
telegram_chat_id = ""
compares = 3
gameWin = False

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
  img = cv2.imread(path)

  for result in results:
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
  res = model(path)
  resultsPerson = res.xyxy[0].detach().cpu().numpy()
  resultsPerson = [r for r in resultsPerson if r[5] == 0]
  playerAverageList.append(getAverages(resultsPerson, path)[0])
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
  global bot
  global telegram_chat_id
  a = playerAverageList
  b = playerNameList
  for i in reversed(range(len(index))):
    a = np.delete(a, index[i])
    bot.sendMessage(telegram_chat_id, b[i] + " eliminat!")
    b = np.delete(b, index[i])
  return a, b


# Game_Loop:
def gameLoop():
	global bot
	global servo
	global compare
	global gameWin
	global telegram_chat_id
	global playerAverageList
	global playerNameList
	servo.min()
	game_end = False

	while True:
		# Muñeca girada:
		servo.max()
		val = 1
		while val > -0.9:
			val -= 0.1
			servo.value = val
			time.sleep(0.2)
		servo.min()
		time.sleep(2)

		# Chech if players have won:
		if gameWin:
			playerAverageList = []
			playerNameList = []
			break

		# Girar la muñeca y empezar a leer primer frame:
		bot.sendMessage(telegram_chat_id, "Pareu!")
		servo.value = 0.9
		cam = cv2.VideoCapture(0)
		ret, image = cam.read()
		frame1path = "/home/pi/robot_files/images/frame1.jpg"
		cv2.imwrite(frame1path, image)
		cv2.destroyAllWindows()
		cam.release()

		results = model(frame1path)
		resultsPerson = results.xyxy[0].detach().cpu().numpy()
		resultsPerson = [r for r in resultsPerson if r[5] == 0]
		avg_list = getAverages(resultsPerson, frame1path)
		print(resultsPerson)
		playerAverageList, bboxes1 = closestAverage(playerAverageList, avg_list, resultsPerson)

		for i in range(compares):
			# Check if players have won just in case:
			if gameWin:
				break
			frame2path = "/home/pi/robot_files/images/frame2.jpg"

			# Capture next frame:
			cam = cv2.VideoCapture(0)
			ret, image = cam.read()
			cv2.imwrite(frame2path, image)
			cv2.destroyAllWindows()
			cam.release()

			new_results = model(frame2path)
			new_resultsPerson = new_results.xyxy[0].detach().cpu().numpy()
			new_resultsPerson = [r for r in new_resultsPerson if r[5] == 0]
			new_avg_list = getAverages(new_resultsPerson, frame2path)
			print(new_resultsPerson)
			playerAverageList, bboxes2 = closestAverage(playerAverageList, new_avg_list, new_resultsPerson)

			movement_list, iou_list = detectMovements((bboxes1, bboxes2))
			index = selectDeletedPlayers(movement_list)
			playerAverageList, playerNameList = deletePlayer(index)

			bboxes1 = bboxes2.copy()
			print(playerNameList)
			if len(playerNameList) == 0:
				game_end = True
				break

		if game_end:
			print("Joc perdut.")
			bot.sendMessage(telegram_chat_id, "Heu perdut!!")
			break

# Handler for the Telegram bot:
def handle(msg):
	global ingame
	global gameWin
	global telegram_mode
	global telegram_chat_id
	global playerAverageList
	global playerNameList
	telegram_chat_id = msg["chat"]["id"]
	chat_id = msg["chat"]["id"]
	command = msg["text"]

	if telegram_mode == "main":
		if command == "addplayers":
			bot.sendMessage(chat_id, "Provide a Player Name or \"stop\" to finish adding players.")
			telegram_mode = "add_players"
		elif command == "startgame":
			bot.sendMessage(chat_id, "Players on the start line...")
			time.sleep(5)
			ingame = True
			telegram_mode = "game_mode"
		elif command == "removeplayers":
			playerAverageList = []
			playerNameList = []
	elif telegram_mode == "add_players":
		if command == "stop":
			print(playerNameList)
			print(playerAverageList)
			bot.sendMessage(chat_id, "Finished adding players!")
			telegram_mode = "main"
		else:
			cam = cv2.VideoCapture(0)
			ret, image = cam.read()
			playerpath = "/home/pi/robot_files/images/" + command + ".jpg"
			cv2.imwrite(playerpath, image)
			cv2.destroyAllWindows()
			cam.release()
			addPlayer(playerpath, command)
			bot.sendPhoto(chat_id, photo=open(playerpath, "rb"))
			bot.sendMessage(chat_id, command + " added!")
	elif telegram_mode == "game_mode":
		if command == "win":
			bot.sendMessage(chat_id, "Heu guanyat!")
			gameWin = True
		elif command == "lose":
			playerAverageList = []
			playerNameList = []
			telegram_mode = "main"


# Main loop del programa:
def main():
	global bot
	global ingame
	print("Program started!")

	bot.message_loop(handle)
	print("Listening through the Telegram bot...")
	print("URFREN V1.0:\n - addplayers: Starts adding players to the game.\n - startgame: Starts the game (not implemented).")

	while 1:
		try:
			if ingame:
				gameLoop()
				os.system("rm -rf /home/pi/robot_files/images/*")
				ingame = False
			else:
				time.sleep(1)
		except KeyboardInterrupt:
			print("\n Terminating Program")
			exit()

# Execute the main function when called directly:
if __name__ == "__main__":
	main()
