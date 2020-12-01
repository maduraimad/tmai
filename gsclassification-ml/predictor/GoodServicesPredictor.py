from keras.layers import *
from keras.models import Model
from keras import backend as K
from keras.preprocessing import sequence
from keras.layers import Conv2D
from keras.layers import Dropout, Flatten, Dense
import numpy as np
import os, re
import logging
from keras.preprocessing.text import Tokenizer
import nltk
from nltk.corpus import stopwords
import tensorflow as tf
import sys
import json
import pymysql
import datetime
import pysolr
import re

nltk.download('stopwords')
base_folder = os.getenv("base_folder")
text_design_codes_file = os.path.join(base_folder+"good_services_resources/45-codes.txt")
all_design_codes_file = os.path.join( base_folder+"good_services_resources/45-codes.txt")
text_stop_words_file = os.path.join(base_folder+"good_services_resources/customstopwords.csv")
text_tokenizer_input_file =os.path.join( base_folder+"good_services_resources/tokenizer_input.txt")
text_model_weights= os.path.join(base_folder+"good_services_resources/tm_goods_services.hdf5")
text_embeddings_file = os.path.join(base_folder+"good_services_resources/tm_goods_services.vec")
description_file=os.path.join(base_folder+"good_services_resources/gsdescription.txt")
mysql_database_user=os.getenv("MYSQL_DATABASE_USER")
mysql_database_password=os.getenv("MYSQL_DATABASE_PASSWORD")
mysql_database_db=os.getenv("MYSQL_DATABASE_DB")
mysql_database_data_db=os.getenv("MYSQL_DATABASE_DATA_DB")
mysql_database_host=os.getenv("MYSQL_DATABASE_HOST")
mysolr_url=os.getenv("MYSOLR_URL")

sys.getfilesystemencoding()
def configureLogging():
    logging.basicConfig(level=os.environ.get("LOGLEVEL", "DEBUG"), filename=os.environ.get("log_file_name"), format='%(asctime)s %(levelname)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p')
configureLogging()
with open(text_design_codes_file, 'r') as file:
    text_design_codes = file.read().splitlines()
with open(all_design_codes_file, 'r') as file:
    all_design_codes = file.read().splitlines()


class GoodServicesPredictor:
    log = logging.getLogger(__qualname__)
    MAX_SEQUENCE_LENGTH = 200
    MAX_NB_WORDS = 50000
    EMBEDDING_DIM = 300
    PREDICTIONS_THRESHOLD = 0.2
    MAX_SCORE = 0.75

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
        self.session = K.get_session()
        # self.session = K.get_session()
        # self.graph = tf.get_default_graph()
        # self.graph.finalize()
        self.model._make_predict_function()
        self.graph = tf.get_default_graph()
        self.log.info("TextPredictor initialized")
        with open(description_file) as f:
            gslist = f.read().splitlines();
            self.dis = {}
            for x in gslist:
                data = x.split('|')
                self.dis[data[0]] = data[1]

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

    def get_predictions(self, serial_number, emailId, mark_desc):
        result = []
        self.log.debug("Getting text predictions")
        self.log.debug("Predicting for mark_desc - " + str(mark_desc))
        feedbackData = []
        hasfeedback = False
        db_feedback = self.get_feedback(serial_number, emailId)
        if not db_feedback:
            hasfeedback = False
        else:
            hasfeedback = True
            feedbackData = ",".join(map(str, db_feedback)).split(",")
        for s in mark_desc.split(";"):
            input_str = self.remove_stop_words(str(s))
            input_str = re.sub('([^\s\w]|_)+', ' ', input_str).strip()
            input_str = re.sub('\s+', ' ', input_str)  # remove multiple spaces
            input_str = [str(input_str.lower())]
            input = self.tokenizer.texts_to_sequences(input_str)
            input = sequence.pad_sequences(input, maxlen=self.MAX_SEQUENCE_LENGTH)
            self.log.debug("Obtained text sequences")
            with self.session.as_default():
                with self.graph.as_default():
                    raw_predictions = self.model.predict(input, verbose=0)
            label_indices = np.nonzero(raw_predictions > self.PREDICTIONS_THRESHOLD)[1]
            labels = []
            scores = []
            for i in label_indices:
                label = all_design_codes[i]
                score = raw_predictions[0, i]
                if score >= self.MAX_SCORE:
                    score = 100
                    if not str(label) in [data['classCode'] for data in result]:
                        state = self.isValidCheck(hasfeedback, feedbackData, label)
                        result.append({"classCode": label, "score": score, "classCodeDesc": self.dis[label], "selected": state})
                    elif str(label) in [data['classCode'] for data in result]:
                        for v in result:
                            if v['classCode'] == str(label) and score > int(v['score']) and score > 80:
                                result.remove(v)
                                state = self.isValidCheck(hasfeedback, feedbackData, label)
                                result.append(
                                    {"classCode": label, "score": score, "classCodeDesc": self.dis[label], "selected": state})
                else:
                    score = 45 + int(
                        ((score - self.PREDICTIONS_THRESHOLD) / (self.MAX_SCORE - self.PREDICTIONS_THRESHOLD)) * (
                                    100 - 45))
                    if not str(label) in [data['classCode'] for data in result] and score > 80:
                        state = self.isValidCheck(hasfeedback, feedbackData, label)
                        result.append({"classCode": label, "score": score, "classCodeDesc": self.dis[label], "selected": state})
                    elif str(label) in [data['classCode'] for data in result]:
                        for k in result:
                            if k['classCode'] == str(label) and score > int(k['score']) and score > 80:
                                result.remove(k)
                                state = self.isValidCheck(hasfeedback, feedbackData, label)
                                result.append(
                                    {"classCode": label, "score": score, "classCodeDesc": self.dis[label], "selected": state})
        self.log.debug("Returning text predictions")
        return sorted(result, key=lambda i: i['score'], reverse=True)
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

    def get_trademark_info(self, serial_number):
        gs_desc = ''
        gs_list=''
        try:
         sql = "select `goods_services_desc` from `trademark_app_info` where `serial_number`=%s"
         conn = self.get_connection(mysql_database_data_db)
         cursor = conn.cursor()
         cursor.execute(sql,(serial_number,))
         trademark = cursor.fetchone()
         gs_list = trademark[0].split(";")
         gs_desc=";".join(gs_list[::-1])
         return re.sub('[^;a-zA-Z0-9 \n\.]','', gs_desc)
        except Exception as e:
            raise e


    def updatepredcitions(self, currentpredcition, predictedresults):
        for i in currentpredcition:
            for j in predictedresults:
                if not i['label'] in [data['label'] for data in predictedresults]:
                    predictedresults.append(i)
                elif i['label'] == j['label'] and i['score'] > j['score']:
                    predictedresults.remove(j)
                    predictedresults.append(i)
        return predictedresults

    def get_feedback(self, serial_number, emailId):
        result = []
        try:
            conn = self.get_connection(mysql_database_db)
            cursor = conn.cursor()
            sql = "SELECT `negative_codes` FROM `user_feedback` WHERE `email_id`=%s and `serial_number`=%s"
            cursor.execute(sql, (emailId, serial_number,))
            data = cursor.fetchone()
            return data
        except Exception as e:
            raise e

    def get_connection(self,mysql_database_name):
        try:
            connection = pymysql.connect(mysql_database_host, mysql_database_user, mysql_database_password,
                                         mysql_database_name)
            return connection
        except Exception as e:
            raise e

    def isValidCheck(self, hasfeedback, feedbackData, label):
        state = ''
        if hasfeedback == True and (str(label) in feedbackData):
            state = True
        else:
            state = False
        return state

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
                if len(neg_list) >= 1:
                    currentDT = datetime.datetime.now()
                    sql = "INSERT INTO user_feedback (email_id,serial_number,negative_codes,timestamp ) VALUES (%s,%s,%s,%s)"
                    cursor.execute(sql,
                                   (emailId, serial_number, ','.join([elem for elem in neg_list]), str(currentDT),))
                    conn.commit()
                    return rep
            else:
                currentDT = datetime.datetime.now()
                sql = "UPDATE user_feedback SET negative_codes=%s,timestamp=%s WHERE email_id=%s and serial_number=%s"
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

    def get_similar_description(self, desc):
        solr = pysolr.Solr(mysolr_url)
        results = solr.search(desc, rows=20)
        prefixList = []
        for j in results:
            prefixString = os.path.commonprefix(
                [desc, re.sub("[{].*[}]", "", j['description'][0]).replace(",", "").replace(".", "").lower().strip()])
            if len(prefixString) > 1:
                prefixList.append(prefixString)
        if len(prefixList) >= 1:
            return max(prefixList, key=len)
        else:
            return " "

    def processSimilarDescription(self, desc):
        split_desc = desc.split(";")
        final_desc = ''
        if len(split_desc)>=1:
          for item_desc in split_desc:
            item = item_desc.replace(",", "").replace(".", "").lower().replace("[","").replace("]","").strip()
            prefix_string = self.get_similar_description(item)
            if len(prefix_string) > 1:
                final_desc += "&nbsp;&nbsp;" + prefix_string
                diff_desc = item[len(prefix_string):len(item)]
                if diff_desc:
                    final_desc += '<em>' + diff_desc + "</em>" + ";"
                else:
                    final_desc = final_desc + ";"
            else:
                final_desc = final_desc + '<em>' + item + '</em>' + ";"
        if final_desc[-1] == ';':
            final_desc = final_desc[:-1]
        if final_desc[:12] == '&nbsp;&nbsp;':
            final_desc = final_desc[12:]
        return final_desc
