import logging
import os
import cv2
import numpy as np

def configureLogging():
    logging.basicConfig(level=os.environ.get("LOGLEVEL", "DEBUG"), filename=os.environ.get("log_file_name"), format='%(asctime)s %(levelname)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p')
configureLogging()


def getImageVector(image_buffer, image_width=299, image_height=299):
  try:
     img = cv2.imdecode(np.array(image_buffer), cv2.IMREAD_UNCHANGED)
     print("Shape in Utils: ", img.shape)
     # convert to jpg
     arr, image_buffer = cv2.imencode(".jpg", img)
     # decode it back to an image object
     img = cv2.imdecode(np.array(image_buffer), cv2.IMREAD_UNCHANGED)
     img = cv2.resize(img, (image_width, image_height), interpolation=cv2.INTER_CUBIC)
     img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
     return img[None]
  except:
     print("shape not found",image_buffer)

