import tensorflow as tf, sys
import numpy as np
from scipy import spatial
from keras import backend as K
from keras.layers import *
import keras
from keras.models import Model
from  predictor.DataGenerator import ResultsUtil
import predictor.DataGenerator as dataGenerator
import predictor.Utils
from keras.models import Sequential
from keras.applications import *
from keras import applications
import os,glob,cv2,logging,time,random,pickle

from tensorflow.contrib import slim

from tensorflow.python.framework.graph_util import convert_variables_to_constants
from tensorflow.python.tools.optimize_for_inference_lib import optimize_for_inference

from annoy import AnnoyIndex

os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

base_folder = os.getenv("base_folder")
modelpath=base_folder+'custom_image_search/'
ann_file = modelpath+'model_files/new1373model_uspto.ann'
model_file =modelpath+'model_files/model_finetune1.h5'
index_to_file =modelpath+'model_files/file_index_to_file_name.pickle'
index_to_file_vector = modelpath+'model_files/file_index_to_file_vector.pickle'
vec_dim = 2048
top_nn = 1001
 
class  AnnoyCustomImageSearch :
    log = logging.getLogger(__qualname__)
    def __init__(self):
        self.log.info("loading AnnoyCustomImageSearch models")
        self.vec_dim  =  vec_dim   # vector dimension to be indexed
        self.annoy_instance = AnnoyIndex(vec_dim)
        self.ann_file  =  ann_file
        self.model_file  =  model_file
        original_model = keras.models.load_model(model_file, custom_objects={"recall": ResultsUtil.recall,"precision": ResultsUtil.precision,"f1score": ResultsUtil.f1score,"focal_loss": focal_loss})
        self.model = Model(original_model.inputs, original_model.layers[-10].output)
        self.top_layer = GlobalAveragePooling2D(input_shape=self.model.output_shape[1:])(self.model.output)
        self.model = Model(inputs=self.model.input, outputs=self.top_layer)
        self.session = K.get_session()
        self.graph = tf.get_default_graph()
        self.log.info ("Graph loaded")
        self.load_annoy()
        with open(index_to_file, 'rb') as fp:
             self.file_index_to_file_name = pickle.load(fp)

        with open(index_to_file_vector, 'rb') as fp1:
             self.file_index_to_file_vector=pickle.load(fp1)
        print("Loaded pickle files")



    def unload_annoy(self):
        self.annoy_instance.unload()

    def load_annoy(self):
        try:
            self.annoy_instance.unload()
            self.annoy_instance.load(self.ann_file)
            self.log.info('Successfully loaded annoy file')
        except FileNotFoundError:
            self.log.error('annoy file DOES NOT EXIST', exc_info=True)

    def get_nns_by_vector(self, feature, include_distances=True):
        named_nearest_neighbors = []
        try:
         nearest_neighbors = self.annoy_instance.get_nns_by_vector(feature, top_nn)
         self.log.info("Getting the vector for the image")
         for j in nearest_neighbors:
            neighbor_file_name = self.file_index_to_file_name[j]
            neighbor_file_vector = self.file_index_to_file_vector[j]
            # similarity = 1 - spatial.distance.cosine(master_vector, neighbor_file_vector)
            # rounded_similarity = int((similarity * 10000)) / 10000.0
            # named_nearest_neighbors.append({'filename': neighbor_file_name,'similarity': rounded_similarity})
            named_nearest_neighbors.append({'filename': neighbor_file_name})
         return named_nearest_neighbors
        except Exception as e:
         self.log.error("Error while getting vector"+str(e))
         raise e

    def extract_features(self, addr):
        features = []
        files = []
        img = cv2.imread(addr)
        self.log.info("Path to  the image is :"+addr)
        try:
            with open(addr, "rb") as file:
                image_buffer = bytearray(file.read())
                image_arr = predictor.Utils.getImageVector(image_buffer)
                image_arr = image_arr / 255
                with self.session.as_default():
                    with self.graph.as_default():
                        feature_vector = self.model.predict(image_arr, verbose=1)[0]
            return feature_vector
        except Exception as e:
            self.log.error("shape not found"+str(e))
            raise e

    def extract_features_s3(self,file_stream_string):
        features = []
        files = []
        try:
                image_buffer = bytearray(file_stream_string)
                image_arr = predictor.Utils.getImageVector(image_buffer)
                image_arr = image_arr / 255
                with self.session.as_default():
                    with self.graph.as_default():
                        feature_vector = self.model.predict(image_arr, verbose=1)[0]
                return feature_vector
        except Exception as e:
            self.log.error("shape not found" + str(e))
            raise e


def focal_loss(y_true, y_pred):
    alpha = 0.25
    gamma = 2.0
    cross_entropy_loss = K.binary_crossentropy(y_true, y_pred, from_logits=False)
    p_t = ((y_true * y_pred) + ((1 - y_true) * (1 - y_pred)))
    modulating_factor = 1.0
    if gamma:
        modulating_factor = tf.pow(1.0 - p_t, gamma)
        alpha_weight_factor = 1.0
        if alpha is not None:
            alpha_weight_factor = (y_true * alpha + (1 - y_true) * (1 - alpha))
    focal_cross_entropy_loss = (modulating_factor * alpha_weight_factor * cross_entropy_loss)
    return K.mean(focal_cross_entropy_loss, axis=-1)

'''
if __name__ == '__main__':

    start=time.clock()
    ann_s  =  AnnoyCustomImageSearch ()
    # Start This file should be loaded at the start inside init function
    ann_s.load_annoy()
    # End
    addr = modelpath+"test_images/mickey_mouse.jpg"
    image_feature = ann_s.extract_features(addr)
    res = ann_s.get_nns_by_vector(image_feature)
    print(res)
    print(time.clock()-start) '''
