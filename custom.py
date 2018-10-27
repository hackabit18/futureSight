import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile
from collections import defaultdict
from io import StringIO
from matplotlib import pyplot as plt
from PIL import Image
import pyrebase
import time
import cv2
#cap = cv2.VideoCapture('http://192.168.0.105:8000/stream.mjpg')
cap = cv2.VideoCapture(0)
# This is needed since the notebook is stored in the object_detection folder.
sys.path.append("..")

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
# # ## Object detection imports
# Here are the imports from the object detection module.

# In[3]:

from utils import label_map_util
import collections
from utils import visualization_utils as vis_util
#from utils.visualization_utils import class_name


# # Model preparation 

# ## Variables
# 
# Any model exported using the `export_inference_graph.py` tool can be loaded here simply by changing `PATH_TO_CKPT` to point to a new .pb file.  
# 
# By default we use an "SSD with Mobilenet" model here. See the [detection model zoo](https://github.com/tensorflow/models/blob/master/object_detection/g3doc/detection_model_zoo.md) for a list of other models that can be run out-of-the-box with varying speeds and accuracies.

# In[4]:

# What model to download.
# MODEL_NAME = 'ssd_mobilenet_v1_coco_11_06_2017'
# MODEL_FILE = MODEL_NAME + '.tar.gz'
# DOWNLOAD_BASE = 'http://download.tensorflow.org/models/object_detection/'

# Path to frozen detection graph. This is the actual model that is used for the object detection.
PATH_TO_CKPT = 'ssd_mobilenet_v1_coco_2017_11_17/frozen_inference_graph.pb'

# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = os.path.join('data', 'mscoco_label_map.pbtxt')

NUM_CLASSES = 90


# ## Download Model

# In[5]:

# opener = urllib.request.URLopener()
# opener.retrieve(DOWNLOAD_BASE + MODEL_FILE, MODEL_FILE)
# tar_file = tarfile.open(MODEL_FILE)
# for file in tar_file.getmembers():
#   file_name = os.path.basename(file.name)
#   if 'frozen_inference_graph.pb' in file_name:
#     tar_file.extract(file, os.getcwd())


# ## Load a (frozen) Tensorflow model into memory.

# In[6]:

detection_graph = tf.Graph()
with detection_graph.as_default():
  od_graph_def = tf.GraphDef()
  with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
    serialized_graph = fid.read()
    od_graph_def.ParseFromString(serialized_graph)
    tf.import_graph_def(od_graph_def, name='')


# ## Loading label map
# Label maps map indices to category names, so that when our convolution network predicts `5`, we know that this corresponds to `airplane`.  Here we use internal utility functions, but anything that returns a dictionary mapping integers to appropriate string labels would be fine

# In[7]:

label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)


# ## Helper code

# In[8]:

def load_image_into_numpy_array(image):
  (im_width, im_height) = image.size
  return np.array(image.getdata()).reshape(
      (im_height, im_width, 3)).astype(np.uint8)


# # Detection

# In[9]:

# For the sake of simplicity we will use only 2 images:
# image1.jpg
# image2.jpg
# If you want to test the code with your images, just add path to the images to the TEST_IMAGE_PATHS.
#PATH_TO_TEST_IMAGES_DIR = 'test_images'
#TEST_IMAGE_PATHS = [ os.path.join(PATH_TO_TEST_IMAGES_DIR, 'image{}.jpg'.format(i)) for i in range(1, 3) ]

# Size, in inches, of the output images.
#IMAGE_SIZE = (12, 8)

url='http://192.168.43.246:8080//shot.jpg'
# In[10]:

with detection_graph.as_default():
  with tf.Session(graph=detection_graph) as sess:
    while True:

      imgResp = urllib.request.urlopen(url)
    
    # Numpy to convert into a array
      imgNp = np.array(bytearray(imgResp.read()),dtype=np.uint8)
    
    # Finally decode the array to OpenCV usable format ;) 
      image_n = cv2.imdecode(imgNp,-1)
      image_np = cv2.resize(image_n,(640,480))
      #ret, image_np = cap.read()
      # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
      image_np_expanded = np.expand_dims(image_np, axis=0)
      image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
      # Each box represents a part of the image where a particular object was detected.
      boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
      

      # Each score represent how level of confidence for each of the objects.
      # Score is shown on the result image, together with the class label.
      scores = detection_graph.get_tensor_by_name('detection_scores:0')
      classes = detection_graph.get_tensor_by_name('detection_classes:0')
      num_detections = detection_graph.get_tensor_by_name('num_detections:0')
      # Actual detection.
      (boxes, scores, classes, num_detections) = sess.run(
          [boxes, scores, classes, num_detections],
          feed_dict={image_tensor: image_np_expanded})
      
      # Visualization of the results of a detection.

      vis_util.visualize_boxes_and_labels_on_image_array(
          image_np,
          np.squeeze(boxes),
          np.squeeze(classes).astype(np.int32),
          np.squeeze(scores),
          category_index,
          use_normalized_coordinates=True,
          line_thickness=8)
      

      # for ax, ay in enumerate(boxes[0]):
      #   if ax==0 or ax==1:

      #     print(str(ax)+" "+str(ay))
      #     print(type(ay))
    
      # threshold = 0.5
      # objects = []
      # for index, value in enumerate(classes[0]): 
        
      #   object_dict = {}
      #   if scores[0, index] > threshold:
      #     object_dict[(category_index.get(value)).get('name').encode('utf8')] = \
      #                     scores[0, index]
      #     objects.append(object_dict)
      #     for key,value in object_dict.items():
      #       if key.decode('utf-8') == 'bottle':
      #         print('') 
      
          
      # print(objects[0])
      
      # else:
      #   class_name = 'N/A'
      #   display_str = str(class_name)
      
      #for i in classes[0]:
        #print(category.get(1))
      #print(category_index[classes[0]]['name'])


      from func import y,z,l,r,t,b
      # print("l: "+str(l)+" r: "+str(r)+" t: "+str(t)+" b: "+str(b)+" y: "+str(y)+" z: "+str(z))
      # users = db.get().val()
      # for i,(key,value) in enumerate(users.items()):
      #   if(key=='y'):
      #     global y
      #     y = value;
      #   if(key=='z'):
      #     global z
      #     z = value;
      x1=235
      x2=400
      y1=200
      y2=280
      if y>=x1 and y<=x2 and z>=y1 and z<=y2:
        print("You are on right way")
        db.update({"q":0})
      elif y>0 and y<x1 and z>0 and z<y1:
        print("1")
        db.update({"q":1})
      elif y>0 and y<x1  and z>=y1 and z<=y2:
        print("2")
        db.update({"q":2})
      elif y>0 and y<x1 and z>y2 and z<=480:
        print("3")
        db.update({"q":3})
      elif y>=x1 and y<=x2 and z>y2 and z<=480:
        print("4")
        db.update({"q":4})
      elif y>x2 and y<=640 and z>y2 and z<=480:
        print("5")
        db.update({"q":5})
      elif y>x2 and y<=640 and z>=y1 and z<=y2:
        print("6")
        db.update({"q":6})
      elif y>x2 and y<=640 and z>0 and z<y1:
        print("7")
        db.update({"q":7})
      elif y>=x1 and y<=x2 and z>0 and z<y1:
        print("8")
        db.update({"q":8})
      elif y==0 and z==0:
        print("Not detected")
        db.update({"q":9})

      y=int(y)
      z=int(z)
      # print("y: "+ str(y) + " z: "+str(z))
      cv2.rectangle(image_np,(y,z),(y+3,z+3),(255,0,0),3)
      cv2.rectangle(image_np,(0,0),(x1,y1),(0,0,255),3)
      cv2.rectangle(image_np,(0,y1),(x1,y2),(0,0,255),3)
      cv2.rectangle(image_np,(0,y2),(x1,480),(0,0,255),3)
      cv2.rectangle(image_np,(x1,y2),(x2,480),(0,0,255),3)
      cv2.rectangle(image_np,(x2,y2),(640,480),(0,0,255),3)
      cv2.rectangle(image_np,(x2,y1),(640,y2),(0,0,255),3)
      cv2.rectangle(image_np,(x2,0),(640,y1),(0,0,255),3)
      cv2.rectangle(image_np,(x1,0),(x2,y1),(0,0,255),3)
      cv2.rectangle(image_np,(x1,y1),(x2,y2),(0,255,0),3)
     # time.sleep(0.2)
      cv2.imshow('object detection', image_np)
      if cv2.waitKey(25) & 0xFF == ord('q'):
        cv2.destroyAllWindows()
        break

