import time
import telepot

def handle(msg):
	chat_id = msg["chat"]["id"]
	command = msg["text"]

	if command == "take":
		print("Taking an image...")

bot = telepot.Bot("5317011276:AAEGqjgjHoje6YEFf1F8QddtOdiHqtYTDP0")
bot.message_loop(handle)
print("Listening...")

while 1:
	try:
		time.sleep(10)

	except KeyboardInterrupt:
		print("\n Terminating Program")
		exit()

