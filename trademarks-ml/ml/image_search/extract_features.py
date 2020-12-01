import tensorflow as tf, sys
import os,glob
import numpy as np
from keras import backend as K
from keras.layers import *
import keras
from keras.models import Model
from DataGenerator import ResultsUtil
import DataGenerator as dataGenerator
import Utils
import logging
from keras.models import Sequential
from keras.applications import *
from keras import applications
import cv2

from tensorflow.contrib import slim

from tensorflow.python.framework.graph_util import convert_variables_to_constants
from tensorflow.python.tools.optimize_for_inference_lib import optimize_for_inference
#from preprocessing import inception_preprocessing

os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

model_file = "/data2/trademarksImageSearch/indexer/model/model_finetune1.h5"
INDEX_PATH = "/data2/trademarksImageSearch/indexer/model_index"
DATA_PATH = "/data2/trademarksImageSearch/indexer/files/image_data"
image_size = 299



def extract_features(addr,fileName,model):
    features = []
    files = []
    img = cv2.imread(addr)
    print ("-----addr-----",addr)
    try:
        print("check for shape in buildImageDataset: ", img.shape)
        with open(addr, "rb") as file:    
           image_buffer = bytearray(file.read())
           image_arr = Utils.getImageVector(image_buffer)
           image_arr = image_arr / 255
           feature_vector = model.predict(image_arr, verbose=0)
        return feature_vector,fileName
    except:
         print("shape not found",addr)

def get_batch(path,batch_size = 1):

    image_data = {}

    for root, dirs, files in os.walk(path):
      for dir in dirs:
        #print (dir)
        for i,fname in enumerate(glob.glob(root + os.sep + dir + "/*")):
            try:
                image_data[fname] = tf.gfile.FastGFile(fname, 'rb').read()
            except:
                print ("failed to load"+fname)
                pass
            if (i+1) % batch_size == 0:
                yield image_data
                image_data = {}
        yield image_data

def store_index(features,files,count,index_dir):
    feat_fname = "{}/{}.npz".format(index_dir,files)
    files_fname = "{}/{}.files".format(index_dir,count)
    #print ("storing index in "+index_dir)
    outfile_name = files + ".npz"
    out_path = os.path.join(index_dir, outfile_name)
    np.savetxt(out_path, features, delimiter=',')



def focal_loss(y_true, y_pred):
    alpha=0.25
    gamma=2.0
    cross_entropy_loss = K.binary_crossentropy(y_true, y_pred, from_logits=False)
    p_t = ((y_true * y_pred) + ((1 - y_true) * (1 - y_pred)))
    modulating_factor = 1.0
    if gamma:
        modulating_factor = tf.pow(1.0 - p_t,gamma)
        alpha_weight_factor = 1.0
        if alpha is not None:
            alpha_weight_factor = (y_true * alpha + (1 - y_true) * (1 - alpha))
    focal_cross_entropy_loss = (modulating_factor * alpha_weight_factor * cross_entropy_loss)
    return K.mean(focal_cross_entropy_loss, axis=-1)

def index():
    print ("Start images indexing")
    try:
        os.mkdir(INDEX_PATH)
    except:
        print ("Could not created "+INDEX_PATH)
        raise ValueError
    count = 0
    num=0
    original_model = keras.models.load_model(model_file, custom_objects={"recall": ResultsUtil.recall,"precision": ResultsUtil.precision,"f1score": ResultsUtil.f1score,"focal_loss": focal_loss})
    print (original_model.summary)
    for i, layer in enumerate(original_model.layers):
            print(i, layer.name,layer.output_shape)
    print (original_model.inputs)
    #print (original_model.layers[-2].output)
    #print (original_model.get_layer('model_4').get_output_at(1))
    model = Model(original_model.inputs, original_model.layers[-10].output)
    print ("original_model.layers[-10]: ",original_model.layers[-10])
    print ("model.output_shape: ", model.output_shape[1:])
    print ("model.output: ",model.output)
    top_layer = GlobalAveragePooling2D(input_shape=model.output_shape[1:])(model.output)
    model = Model(inputs=model.input, outputs=top_layer)
    graph = tf.get_default_graph()
    #with tf.Session() as sess:
    #  for image_data in get_batch(DATA_PATH):
    print ("Graph loaded")
    #count += 1
    #print ("image_data",image_data)
    #features,files = extract_features()
    image_addrs = sorted(glob.glob(DATA_PATH+ "/*.jpg"))
    #print ("image_addrs: ",image_addrs)

    for addr in image_addrs:
       fileName = os.path.basename(addr).rsplit('.', 1)[0]
       img = cv2.imread(addr)
       try:
         print("check for shape in buildImageDataset: ", img.shape)
         features,files = extract_features(addr,fileName,model) 
         store_index(features,files,count,INDEX_PATH)
         print ("Completed extracting features")
       except:
         print("shape not found",addr)
if __name__ == '__main__':
  index()
