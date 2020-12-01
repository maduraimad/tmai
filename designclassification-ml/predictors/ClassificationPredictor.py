import json
import logging
import pickle
from urllib import request
import keras
import pathlib
import nltk
from keras.layers import *
from keras.layers import Conv2D
from keras.layers import Dropout, Flatten, Dense
from keras.models import Model
from keras.preprocessing import sequence
from keras import backend as K
from keras.preprocessing.text import Tokenizer
import Utils
from DataGenerator import ResultsUtil
nltk.download('stopwords')
from nltk.corpus import stopwords
import re
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer
import sklearn.metrics as metrics
import tensorflow as tf
import sys
import pymysql
import datetime
import os
from ast import literal_eval
import cv2
import boto3
from botocore.client import Config
from PIL import Image
import io

base_folder = os.getenv("base_ensemble_folder")
base_image_resources = base_folder + "/image_resources"
base_text_config = base_folder + "/text_resources"
all_design_codes_file = base_folder + "/1383-design-codes.txt"
image_design_codes_file = base_folder + "/1373-designCodes.txt"
text_design_codes_file = base_text_config + "/1381-design-codes.txt"
design_code_descriptions_file = base_folder + "/design_code_descriptions.txt"
image_network_weights_file = base_image_resources + "/image_network_weights.pickle"
text_ensemble_weights_file = base_folder + "/text_ensemble_weights.pickle"
image_ensemble_weights_file = base_folder + "/image_ensemble_weights.pickle"
text_stop_words_file = base_text_config + "/customstopwords.csv"
text_tokenizer_input_file = base_text_config + "/tokenizer_input.txt"
text_model_weights = base_text_config + "/200seq_300d_1381_BI_GRU_tm_embedding.hdf5"
text_embeddings_file = base_text_config + "/latest_tm_markdesc_300.vec"
mysql_database_user=os.getenv("MYSQL_DATABASE_USER")
mysql_database_password=os.getenv("MYSQL_DATABASE_PASSWORD")
mysql_database_db=os.getenv("MYSQL_DATABASE_DB")
mysql_database_host=os.getenv("MYSQL_DATABASE_HOST")
mysql_database_data_db=os.getenv("MYSQL_DATABASE_DATA_DB")

sess = tf.Session()
graph = tf.get_default_graph()

sys.getfilesystemencoding()
number_of_image_networks = 1

with open(all_design_codes_file, 'r') as file:
    all_design_codes = file.read().splitlines()
with open(image_design_codes_file, 'r') as file:
    image_design_codes = file.read().splitlines()
with open(text_design_codes_file, 'r') as file:
    text_design_codes = file.read().splitlines()
with open(image_network_weights_file, "rb") as file:
    image_network_weights = pickle.load(file)
with open(image_ensemble_weights_file, "rb") as file:
    image_ensemble_weights = pickle.load(file)
with open(text_ensemble_weights_file, "rb") as file:
    text_ensemble_weights = pickle.load(file)


class ImagePredictor:
    log = logging.getLogger(__qualname__)
    PREDICTIONS_THRESHOLD = 0.20

    def __init__(self):
        self.log.info("Initializing ImagePredictor")
        self.models = []
        self.graphs = []
        self.all_design_codes_size = len(all_design_codes)
        self.image_design_codes_size = len(image_design_codes)
        for i in range(number_of_image_networks):
            model_file = "{}/model{}.h5".format(base_image_resources,str(i+1))
            K.set_session(sess)
            model = keras.models.load_model(model_file, custom_objects={"recall": ResultsUtil.recall, "precision": ResultsUtil.precision, "f1score": ResultsUtil.f1score,'tf': tf,'focal_loss': self.focal_loss}, compile=False)
            model._make_predict_function()
            self.graphs.append(tf.get_default_graph())
            self.models.append(model)
        self.image_adjuster_multiplier = np.zeros((self.image_design_codes_size, self.all_design_codes_size))
        self.log.info("Models loaded")
        for index1, design_code in enumerate(image_design_codes):
            index2 = all_design_codes.index(design_code)
            self.image_adjuster_multiplier[index1, index2] = 1
        self.log.info("Image Predictor Initialized")

    def focal_loss(self, y_true, y_pred):
        alpha = 0.25
        gamma = 2.0
        y_pred = tf.clip_by_value(y_pred, tf.keras.backend.epsilon(), 1 - tf.keras.backend.epsilon())
        logits = tf.log(y_pred / (1 - y_pred))
        weight_a = alpha * (1 - y_pred) ** gamma * y_true
        weight_b = (1 - alpha) * y_pred ** gamma * (1 - y_true)
        loss = (tf.log1p(tf.exp(-tf.abs(logits))) + tf.nn.relu(-logits)) * (weight_a + weight_b) + logits * weight_b
        return tf.reduce_mean(loss)

    def get_predictions(self, image_buffer):
        global sess
        global graph
        with graph.as_default():
            K.set_session(sess)
            image_arr = Utils.getImageVector(image_buffer)
            image_arr = image_arr / 255
            predictions_array = []
            for index, model in enumerate(self.models):
                with self.graphs[index].as_default():
                    predictions = model.predict(image_arr, verbose=0)
                    predictions_array.append(predictions)

            weighted_predictions = np.zeros(shape=(predictions_array[0].shape))
            for index, predictions in enumerate(predictions_array):
                weighted_predictions = weighted_predictions + (predictions * image_network_weights[index])

            raw_predictions = np.dot(weighted_predictions, self.image_adjuster_multiplier)
            predictions = raw_predictions > self.PREDICTIONS_THRESHOLD
            label_indices = np.nonzero(predictions)[1]
            labels = []
            label_scores = []
            for i in label_indices:
                labels.append(all_design_codes[i])
                label_scores.append(raw_predictions[0][i])
        self.log.debug("Returning image predictions")
        return raw_predictions, labels, label_scores


class TextPredictor:
    log = logging.getLogger(__qualname__)
    MAX_SEQUENCE_LENGTH = 200
    MAX_NB_WORDS = 50000
    EMBEDDING_DIM = 300
    PREDICTIONS_THRESHOLD = 0.2

    def __init__(self):
        self.log.info("Initializing TextPredictor")
        with open(text_tokenizer_input_file, "r") as file:
            tokenizer_input_list = file.read().splitlines()
        self.tokenizer = Tokenizer(num_words=self.MAX_NB_WORDS, lower=True)
        self.tokenizer.fit_on_texts(tokenizer_input_list)
        self.log.debug("Tokenizer fit done")
        self.embeddings_index = {}
        f = open(text_embeddings_file)
        for line in f:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            self.embeddings_index[word] = coefs
        f.close()
        self.build_stopwords_list()
        self.log.info("Loading Model")
        self.model = self.load_model()
        self.model._make_predict_function()
        self.graph = tf.get_default_graph()

        self.all_design_codes_size = len(all_design_codes)
        self.text_design_codes_size = len(text_design_codes)

        self.text_adjuster_multiplier = np.zeros((self.text_design_codes_size, self.all_design_codes_size))
        for index1, design_code in enumerate(text_design_codes):
            index2 = all_design_codes.index(design_code)
            self.text_adjuster_multiplier[index1, index2] = 1
        self.log.info("TextPredictor initialized")

    def build_stopwords_list(self):
        with open(text_stop_words_file, "r") as file:
            custom_words = file.read().splitlines()
        en_stop = set(stopwords.words('english'))
        en_stop.update(
            ['.', '(', '(', ')', 'u', ',', '"', "'", '?', '!', ':', ';', '(', ')', '[', ']', '{', '}', '&', '/', '-',
             '+',
             '*', \
             "..", '...', '....', "+...", "-...", ",...", "'...", "!!!", "&...", "(...", ")...", "]...", "/...",
             "(+)...",
             "),...", \
             "),", ").", "):", ")-", "))", "])", ".)", "!)", "')", "][", '").', '")/', '",', '"-', '")', '"/', './',
             '--', \
             "#:", "(+)", "($", "-$", "/", "+/", ",", "+-", "(#", "''"])
        self.stopwords = en_stop.union(custom_words)

    def get_predictions(self, mark_desc):
        self.log.debug("Getting text predictions")
        self.log.debug("Predicting for mark_desc - " + mark_desc)
        input_str = self.remove_stop_words(mark_desc)
        input_str = re.sub('([^\s\w]|_)+', ' ', input_str).strip()
        input_str = re.sub('\s+', ' ', input_str)  # remove multiple spaces
        input_str = [str(input_str.lower())]

        input = self.tokenizer.texts_to_sequences(input_str)
        input = sequence.pad_sequences(input, maxlen=self.MAX_SEQUENCE_LENGTH)
        self.log.debug("Obtained text sequences")
        with self.graph.as_default():
            raw_predictions = self.model.predict(input, verbose=0)

        raw_predictions = np.dot(raw_predictions, self.text_adjuster_multiplier)
        # label_indices = np.nonzero(raw_predictions > self.PREDICTIONS_THRESHOLD)[1]
        predictions = raw_predictions > self.PREDICTIONS_THRESHOLD
        label_indices = np.nonzero(predictions)[1]
        labels = []
        label_scores = []
        for i in label_indices:
            labels.append(all_design_codes[i])
            label_scores.append(raw_predictions[0][i])

        self.log.debug("Returning text predictions")
        return raw_predictions, labels, label_scores

    def load_model(self):
        self.log.info('Loading model.')
        sequence_input = Input(shape=(self.MAX_SEQUENCE_LENGTH,), dtype='int32')
        embedding_layer = self.get_embedding_layer()
        embedded_sequences = embedding_layer(sequence_input)
        units = 128
        conv_filters = 32
        x = Dropout(0.2)(embedded_sequences)
        x = Bidirectional(GRU(
            units,
            dropout=0.2,
            recurrent_dropout=0.2,
            return_sequences=True))(x)
        x = Reshape((2 * self.MAX_SEQUENCE_LENGTH, units, 1))(x)
        x = Conv2D(conv_filters, (3, 3))(x)

        x = MaxPool2D(pool_size=(2, 2))(x)
        x = Flatten()(x)
        preds = Dense(len(text_design_codes), activation='sigmoid')(
            x)  # for multilabel classification # length of labels
        model = Model(sequence_input, preds)
        model.load_weights(text_model_weights)
        self.log.info('Loaded model.')
        return model

    def get_embedding_layer(self):
        word_index = self.tokenizer.word_index
        embedding_matrix = np.zeros((len(word_index) + 1, self.EMBEDDING_DIM))
        for word, i in word_index.items():
            embedding_vector = self.embeddings_index.get(word)
            if embedding_vector is not None:
                # words not found in embedding index will be all-zeros.
                embedding_matrix[i] = embedding_vector
        embedding_layer = Embedding(len(word_index) + 1,
                                    self.EMBEDDING_DIM,
                                    weights=[embedding_matrix],
                                    input_length=self.MAX_SEQUENCE_LENGTH,
                                    trainable=False)
        return embedding_layer

    def remove_stop_words(self, text):
        return ' '.join([word for word in text.split() if word.lower() not in self.stopwords])


class EnsemblePredictor:
    log = logging.getLogger(__qualname__)
    PREDICTIONS_THRESHOLD = 0.15

    def __init__(self):
        self.imagePredictor = ImagePredictor()
        self.textPredictor = TextPredictor()
        self.design_code_descriptions = {}
        with open(design_code_descriptions_file, "r") as file:
            lines = file.read().splitlines()
            for line in lines:
                split = line.split("|")
                self.design_code_descriptions[split[0]] = split[1]

    def get_text_predictions(self, serial_number=None, mark_desc=None):
        if serial_number is None and mark_desc is None:
            return []
        if serial_number is not None:
            original_labels, mark_desc = self.get_trademark_info(serial_number)
        results = []
        if mark_desc is None:
            return []
        else:
            text_predictions, text_labels, text_label_scores = self.textPredictor.get_predictions(mark_desc)
            for index, label in enumerate(text_labels):
                results.append({
                    "label": label,
                    "desc": self.design_code_descriptions[label],
                    "score": str(text_label_scores[index])
                })
            return results


    def get_trademark_info(self, serial_number):
        try:
          sql = "select `mark_desc`,`design_codes` from `trademark_app_info` where `serial_number`=%s"
          conn = self.get_connection(mysql_database_data_db)
          cursor = conn.cursor()
          cursor.execute(sql,(serial_number))
          trademark=cursor.fetchone()
          original_labels=literal_eval(trademark[1].replace('"', ''))
          mark_desc=trademark[0]
          return original_labels, mark_desc
        except Exception as e:
            self.log.error("error occurred while processing ")


    def filter_original_labels(self, original_labels):
        return [label for label in original_labels if label in all_design_codes]

    def get_predictions_by_serial(self, serial_number,email_id):
        result={}
        original_labels=[]
        mark_desc=None
        original_labels,mark_desc=self.get_trademark_info(serial_number)
        session = boto3.Session()
        s3_client = session.client('s3', config=Config(signature_version='s3v4'))
        s3_response_object = s3_client.get_object(Bucket='uspto-tm-img-search', Key='tm_images/'+serial_number+'.jpg')
        object_content = s3_response_object['Body'].read()
        result = self.get_predictions(image_buffer=bytearray(object_content), mark_desc=mark_desc, original_labels=original_labels,
                                      serial_number=serial_number,emailId=email_id)
        return result

    def get_predictions(self, image_buffer=None, mark_desc="", original_labels=[], serial_number='',emailId=''):
        hasfeedback = False
        lb = MultiLabelBinarizer()
        lb = lb.fit([all_design_codes])
        original_labels_binarized = lb.transform([self.filter_original_labels(original_labels)])
        feedbackData = []
        db_feedback = self.get_feedback(serial_number, emailId)
        if not db_feedback:
            hasfeedback = False
        else:
            hasfeedback = True
            feedbackData = ",".join(map(str, db_feedback)).split(",")
        self.log.info("Got info from TSDR")
        image_predictions = np.zeros((1, len(image_design_codes)))
        image_labels = []
        no_image = False
        if image_buffer:
            image_predictions, image_labels, image_label_scores = self.imagePredictor.get_predictions(image_buffer)
        else:
            no_image = True
            self.log.debug("No image sent")
        text_predictions = np.zeros((1, len(text_design_codes)))
        text_labels = []
        no_mark_desc = False
        if mark_desc:
            text_predictions, text_labels, text_label_scores = self.textPredictor.get_predictions(
                mark_desc)  # self.get_dummy_data_for_text()
        else:
            no_mark_desc = True
            self.log.debug("No mark description found")

        self.log.info("Combining predictions")
        if no_mark_desc:
            combined_raw_preds = image_predictions
        elif no_image:
            combined_raw_preds = text_predictions
        else:
            self.log.debug("image_predictions.shape: ")
            self.log.debug(image_predictions.shape)
            self.log.debug("image_ensemble_weights.shape: ")
            self.log.debug(image_ensemble_weights.shape)

            self.log.debug("text_predictions.shape: ")
            self.log.debug(text_predictions.shape)
            self.log.debug("text_ensemble_weights.shape: ")
            self.log.debug(text_ensemble_weights.shape)

            combined_raw_preds = image_predictions * image_ensemble_weights + text_predictions * text_ensemble_weights
        combined_raw_preds_binarized = combined_raw_preds > self.PREDICTIONS_THRESHOLD
        label_indices = np.nonzero(combined_raw_preds_binarized)[1]
        # sort the labels according to their score
        label_indices = sorted(label_indices, key=lambda index: combined_raw_preds[0, index], reverse=True)
        combined_labels = []
        combined_label_scores = []
        for i in label_indices:
            combined_labels.append(all_design_codes[i])
            combined_label_scores.append(combined_raw_preds[0, i])

        # normalize the combined label scores
        # based on the confidence scores we have seen, looks like we have an average max score of 0.75 for text predictions, so we will normalize our scores to this high value
        # The threshold we are using is 0.19 which corresponds to 45% accuracy and 0.75 corresponds to 100% accuracy
        # So we'll have to fit the scores ranging 0.19 to 0.75 in to a confidence scores ranging from 45% to 100%
        max_score = 0.75
        min_score = self.PREDICTIONS_THRESHOLD
        for index, combined_label_score in enumerate(combined_label_scores):
            if combined_label_scores[index] >= max_score:
                combined_label_scores[index] = 100
            else:
                combined_label_scores[index] = 45 + int(
                    ((combined_label_scores[index] - min_score) / (max_score - min_score)) * (100 - 45))
                # combined_label_scores[index] = int((combined_label_scores[index]/max_score)*100)

        self.log.debug("Combined labels - " + str(combined_labels))
        result = {
            "serialnumber": serial_number,
            "predictions": [],
            "markDescription":mark_desc,
            "f1_score": int(
                metrics.f1_score(original_labels_binarized, combined_raw_preds_binarized, average="micro") * 100)
        }

        for label_index, label in enumerate(combined_labels):
            state = self.isValidCheck(hasfeedback, feedbackData, label)
            result["predictions"].append({
                "designCode": label,
                "selected":state,
                "designCodeDesc": self.design_code_descriptions[label],
                "score": combined_label_scores[label_index],
               # "ensemble_score": str(combined_raw_preds[0, label_indices[label_index]]),
               # "text_score": str(text_label_scores[text_labels.index(label)]) if label in text_labels else -1,
               # "image_score": str(image_label_scores[image_labels.index(label)]) if label in image_labels else -1
            })
        return result

    def get_dummy_data_for_text(self):
        arr = np.zeros((1, 1381))
        arr[0, 34] = 0.34
        arr[0, 105] = 0.39
        arr[0, 933] = 0.6
        return arr, ["01.01.02", "01.01.09", "02.01.01"]

    def get_feedback(self, serial_number, emailId):
        result = []
        try:
            conn = self.get_connection(mysql_database_db)
            cursor = conn.cursor()
            sql = "SELECT `negative_design_codes` FROM `user_feedback` WHERE `email_id`=%s and `serial_number`=%s"
            cursor.execute(sql, (emailId, serial_number,))
            data = cursor.fetchone()
            return data
        except Exception as e:
            raise e

    def get_connection(self,mysql_database_name):
        try:
            connection = pymysql.connect(mysql_database_host,mysql_database_user,mysql_database_password,
                                         mysql_database_name)
            return connection
        except Exception as e:
            raise e

    def persist_feedback(self, serial_number, emailId, userfeedback):
        neg_list = []
        pos_removedList = []
        rep = {'success': True, 'message': 'successfully saved your feedback'}
        try:
            conn = self.get_connection(mysql_database_db)
            cursor = conn.cursor()
            for i in userfeedback:
                neg_list.append(i)
            db_feedback = self.get_feedback(serial_number, emailId)
            if not db_feedback:
                currentDT = datetime.datetime.now()
                if len(neg_list) >= 1:
                    sql = "INSERT INTO user_feedback (email_id,serial_number,negative_design_codes,timestamp ) VALUES (%s,%s,%s,%s)"
                    cursor.execute(sql,
                                   (emailId, serial_number, ','.join([elem for elem in neg_list]), str(currentDT),))
                    conn.commit()
                return rep
            else:
                currentDT = datetime.datetime.now()
                sql = "UPDATE user_feedback SET negative_design_codes=%s,timestamp=%s WHERE email_id=%s and serial_number=%s"
                if len(neg_list) >= 1:
                    cursor.execute(sql,
                                   (','.join([elem for elem in neg_list]), str(currentDT), emailId, serial_number,))
                    conn.commit()
                    return rep
                else:
                    sql = "delete from user_feedback where email_id=%s and serial_number=%s"
                    cursor.execute(sql, (emailId, serial_number,))
                    conn.commit()
                    return rep
            return rep
        except Exception as e:
            raise e
        finally:
            cursor.close()
            conn.close()



    def get_feedback(self, serial_number, emailId):
        result = []
        try:
            conn = self.get_connection(mysql_database_db)
            cursor = conn.cursor()
            sql = "SELECT `negative_design_codes` FROM `user_feedback` WHERE `email_id`=%s and `serial_number`=%s"
            cursor.execute(sql, (emailId, serial_number,))
            data = cursor.fetchone()
            return data
        except Exception as e:
            raise e

    def isValidCheck(self, hasfeedback, feedbackData, label):
        state = ''
        if hasfeedback == True and (str(label) in feedbackData):
            state = True
        else:
            state = False
        return state

# EnsemblePredictor().get_predictions("78787878")
# TextPredictor().get_predictions("one star with four points ")
# results = EnsemblePredictor().get_text_predictions(serial_number="78787878")
# print(str(results))

# with open("/Users/greensod/usptoWork/TrademarkRefiles/data/tsdrImages/Jan-2011/85208988.jpg", "rb") as file:
#     image_buffer = bytearray(file.read())
#     predictions = ImagePredictor().get_predictions(image_buffer)
