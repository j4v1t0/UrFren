from gpiozero import Servo
import time

servo = Servo(14)
val = -1

try:
	servo.min()
	time.sleep(2)
	servo.value = 0.9
	time.sleep(3)
except KeyboardInterrupt:
	print("Program stopped")

