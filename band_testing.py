import pyrebase
from nanpy import (ArduinoApi, SerialManager)
import time
try:
    connection = SerialManager()
    a = ArduinoApi(connection = connection)
except:
    print("Failed to connect Arduino")
config = {
    
    "apiKey":"AIzaSyDuDlz-k6BxOvOJIcKOPNOPV611rr54Njo",
    "authDomain":"raspberrypi-89fd5.firebaseapp.com",
    "databaseURL":"https://raspberrypi-89fd5.firebaseio.com",
    "storageBucket":"raspberrypi-89fd5.appspot.com"
    
    
    }
firebase = pyrebase.initialize_app(config)
auth = firebase.auth()
user = auth.sign_in_with_email_and_password("g.kalyan04@gmail.com","Kalyan@bvcoe")
db = firebase.database()
    
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
    t=0.15
    users = db.get().val()
    for i,(key,value) in enumerate(users.items()):
        print(str(key) + str(value));
        if(str(key)=="q"):
            if(value==1):
                a.digitalWrite(12,a.HIGH)
                time.sleep(t)
                a.digitalWrite(12,a.LOW)
            if(value==2):
                a.digitalWrite(11,a.HIGH)
                time.sleep(t)
                a.digitalWrite(11,a.LOW)
            if(value==3):
                a.digitalWrite(35,a.HIGH)
                time.sleep(t)
                a.digitalWrite(35,a.LOW)
            if(value==4):
                a.digitalWrite(7,a.HIGH)
                time.sleep(t)
                a.digitalWrite(7,a.LOW)
            if(value==5):
                a.digitalWrite(10,a.HIGH)
                time.sleep(t)
                a.digitalWrite(10,a.LOW)
            if(value==6):
                a.digitalWrite(9,a.HIGH)
                time.sleep(t)
                a.digitalWrite(9,a.LOW)
            if(value==7):
                a.digitalWrite(8,a.HIGH)
                time.sleep(t)
                a.digitalWrite(8,a.LOW)
            if(value==8):
                a.digitalWrite(2,a.HIGH)
                time.sleep(t)
                a.digitalWrite(2,a.LOW)
            
                
                
                
                