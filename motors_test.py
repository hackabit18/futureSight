from nanpy import (ArduinoApi, SerialManager)
from time import sleep

try:
    connection = SerialManager()
    a = ArduinoApi(connection = connection)

except:
    print("failed to connect Arduino")

a.pinMode(12,a.OUTPUT)
a.pinMode(11,a.OUTPUT)
a.pinMode(35,a.OUTPUT)
a.pinMode(7,a.OUTPUT)
a.pinMode(10,a.OUTPUT)
a.pinMode(9,a.OUTPUT)
a.pinMode(8,a.OUTPUT)
a.pinMode(2,a.OUTPUT)
a.pinMode(39,a.OUTPUT)
a.pinMode(3,a.OUTPUT)
a.pinMode(5,a.OUTPUT)

while True:
    a.digitalWrite(39,a.LOW)
    a.digitalWrite(3,a.LOW)
    a.digitalWrite(5,a.LOW)

    while True:
        a.digitalWrite(12,a.HIGH)
        a.digitalWrite(11,a.HIGH)
        a.digitalWrite(35,a.HIGH)
        a.digitalWrite(7,a.HIGH)
        a.digitalWrite(10,a.HIGH)
        a.digitalWrite(9,a.HIGH)
        a.digitalWrite(8,a.HIGH)
        a.digitalWrite(2,a.HIGH)
