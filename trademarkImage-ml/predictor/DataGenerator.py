import glob
import numpy as np
import h5py
import cv2
import random
import matplotlib as mpl
# mpl.use('TkAgg')
# import matplotlib.pyplot as plt
from sklearn.preprocessing import MultiLabelBinarizer
import os
import operator
import tensorflow as tf
import sklearn.metrics as metrics
from keras import backend as K

class ResultsUtil:

    def __init__(self, data_folder, hdf5_folder, hdf5_file_name="data-299.h5"):
        self.data_folder = data_folder
        self.hdf5_path = hdf5_folder + "/" + hdf5_file_name

    def decodePredictions(self, predictions, all_labels,  threshold=0.5):
        label_dict = {}
        label_addrs = sorted(glob.glob(self.data_folder + "/testing/*.txt"))
        serialNumbers = []
        for addr in label_addrs:
            serialNumbers.append(os.path.splitext(os.path.basename(addr))[0])
        for index, prediction in enumerate(predictions):
            label_indices = np.nonzero(prediction >= threshold)
            labels = []
            for label_index in label_indices[0]:
                labels.append(all_labels[label_index])
            label_dict[serialNumbers[index]] = labels;
        return label_dict

    def getClassificationReport(self, predictions, labels, threshold=0.5):
        predictions = np.where(predictions >= threshold, 1, 0)
        hdf5_file = h5py.File(self.hdf5_path, "r")
        actuals = hdf5_file["labels_testing"][()]
        report = metrics.classification_report(actuals, predictions, target_names=labels)
        return report

    def calculateScores(self, predictions, threshold=0.5):
        predictions = np.where(predictions >= threshold, 1, 0)
        hdf5_file = h5py.File(self.hdf5_path, "r")
        actuals = hdf5_file["labels_testing"][()]
        true_positives = float(np.sum(np.multiply(actuals, predictions)))
        predicted_positives = np.sum(predictions)
        actual_positives = np.sum(actuals)
        precision = true_positives / predicted_positives
        recall = true_positives / actual_positives
        f1_score = ((precision * recall) / (precision + recall)) * 2
        return (precision, recall, f1_score)

    @staticmethod
    def recall(actuals, y_pred):
        actuals = tf.cast(actuals, tf.int32)
        threshold = tf.fill(tf.shape(actuals), value=0.3)
        predictions = tf.where(y_pred >= threshold, tf.fill(tf.shape(actuals), value=1),
                               tf.fill(tf.shape(actuals), value=0))
        true_positives = tf.reduce_sum(tf.multiply(actuals, predictions))
        actual_positives = tf.reduce_sum(actuals)
        recall = tf.divide(true_positives, actual_positives)
        return recall

    @staticmethod
    def precision(actuals, y_pred):
        actuals = tf.cast(actuals, tf.int32)
        threshold = tf.fill(tf.shape(actuals), value=0.3)
        predictions = tf.where(y_pred >= threshold, tf.fill(tf.shape(actuals), value=1),
                               tf.fill(tf.shape(actuals), value=0))
        true_positives = tf.reduce_sum(tf.multiply(actuals, predictions))
        predicted_positives = tf.reduce_sum(predictions)
        precision = tf.divide(true_positives, predicted_positives)
        return precision
    @staticmethod
    def f1score(actuals, y_pred):
        recall_value = ResultsUtil.recall(actuals, y_pred)
        precision_value = ResultsUtil.precision(actuals, y_pred)
        f1_score = ((precision_value * recall_value) / (precision_value + recall_value)) * 2
        return f1_score

