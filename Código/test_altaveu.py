import RPi.GPIO as GPIO
import time

GPIO.setmode(GPIO.BCM)
GPIO.setup(18, GPIO.OUT)
p = GPIO.PWM(18, 50)
p.start(80)

for x in range(200, 2200):
	p.ChangeFrequency(x)
	time.sleep(0.001)

p.stop()
GPIO.cleanup()
