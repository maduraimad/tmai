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

img_width, img_height = 299, 299
# all_labels = ['01.07.02','11.03.03','01.15.05','01.01.13','24.01.03','03.01.24','24.11.02','02.11.13','02.03.22','11.01.01','11.01.03','26.01.21','26.11.21','07.01.04','26.17.09']
# all_labels = ['01.07.02','11.03.03','01.15.05','01.01.13','24.01.03','03.01.24','24.11.02','02.11.13','02.03.22','11.01.01','11.01.03','07.01.04','02.01.01','03.15.19','05.01.01'] # 15-designCodes-2.txt
all_labels = ['02.11.01','02.11.07','02.11.02','27.01.01','27.03.02','27.03.04','07.01.06','07.07.03','03.01.16','03.01.08','05.03.25','05.03.08','05.05.25','06.01.04','06.03.03'] # 15-designCodes-3.txt
#all_labels = ['01.07.02','11.03.03','01.15.05','01.01.13','24.01.03','03.01.24','24.11.02','02.11.13','02.03.22','11.01.01','11.01.03','07.01.04','02.01.01','03.15.19','05.01.01','02.11.01','02.11.07','02.11.02','27.01.01','27.03.02','27.03.04','07.01.06','07.07.03','03.01.16','03.01.08','05.03.25','05.03.08','05.05.25','06.01.04','06.03.03']
base_folder = os.getenv("base_folder")
data_folder=base_folder+"/divided"
hdf5_path = base_folder+"/data-299.h5"


class DataCreatorUtil:
    def __init__(self, data_folder, hdf5_folder, hdf5_file_name="data-299.h5"):
        self.data_folder = data_folder
        self.hdf5_path = hdf5_folder+"/"+hdf5_file_name

    def createBinarFile(self, labels, dataset_folders = ["training", "testing", "validation"]):
        hdf5_file = h5py.File(self.hdf5_path, mode='w')
        for dataset_folder in dataset_folders:
            self.buildImageDataset(dataset_folder, hdf5_file)
            self.buildLabelDataset(dataset_folder, hdf5_file, labels)
        hdf5_file.close()

    def buildImageDataset(self, folder_name, hdf5_file):
        image_addrs = sorted(glob.glob(self.data_folder + "/"+folder_name+"/*.jpg"))
        dataset_name = "images_"+folder_name
        dataset_shape = (len(image_addrs), img_width, img_height, 3)
        hdf5_file.create_dataset(dataset_name, dataset_shape, np.uint8)
        for i in range(len(image_addrs)):
            # print how many images are saved every 1000 images
            if i % 1000 == 0 and i > 1:
                print ('Train data: {}/{}'.format(i, len(image_addrs)))
            # read an image and resize to (224, 224)
            # cv2 load images as BGR, convert it to RGB
            addr = image_addrs[i]
            img = cv2.imread(addr)
            img = cv2.resize(img, (img_width, img_height), interpolation=cv2.INTER_CUBIC)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            # add any image pre-processing here
            # save the image
            hdf5_file[dataset_name][i, ...] = img[None]

    def buildLabelDataset(self, folder_name, hdf5_file, labels):
        dataset_name = "labels_" + folder_name
        num_of_labels = len(labels)

        label_addrs = sorted(glob.glob(self.data_folder + "/"+folder_name+"/*.txt"))
        int_labels = []
        dataset_shape = (len(label_addrs), num_of_labels)
        hdf5_file.create_dataset(dataset_name, dataset_shape, np.uint8)
        for i in range(len(label_addrs)):
            addr = label_addrs[i]
            with open(addr, "r") as text_file:
                label_values = text_file.read().splitlines()
                label_tuples = ()
                for label_value in label_values:
                    label_tuples = label_tuples + (labels.index(label_value),)
                int_labels.append(label_tuples)
        lb = MultiLabelBinarizer(labels)
        binarized_labels = lb.fit_transform(int_labels)
        print('labels shape - ' + str(np.shape(binarized_labels)))
        hdf5_file[dataset_name][...] = binarized_labels

    def getGenerator(self, batch_size, dataset_suffix, loopOnce=False, scaleX=255, shuffle=False, skipLabels=False):
        main_loop = 0
        hdf5_file = h5py.File(self.hdf5_path, "r")
        X = hdf5_file["images_"+dataset_suffix]
        if not skipLabels:
            y = hdf5_file["labels_"+dataset_suffix]
        num_of_images = len(X)
        loop = True
        temp_arr = np.arange(num_of_images)
        while loop:
            print("\nMain loop for - "+dataset_suffix+"- "+str(main_loop))
            main_loop = main_loop + 1
            if shuffle:
                random.shuffle(temp_arr)
            for i in range(num_of_images//batch_size + 1):
                indices = temp_arr[i * batch_size:(i + 1) * batch_size][()]
                if skipLabels:
                    yield X[sorted(indices)] / scaleX
                else:
                    yield X[sorted(indices)] / scaleX, y[sorted(indices)][()]
                # yield X[i * batch_size:(i + 1) * batch_size][()]/scaleX, y[i * batch_size:(i + 1) * batch_size][()]
            if loopOnce:
                loop = False

    def getSampleCounts(self):
        hdf5_file = h5py.File(self.hdf5_path, "r")
        counts = [0, 0, 0]
        if "images_training" in hdf5_file.keys():
            counts[0] = len(hdf5_file["images_training"])
        if "images_validation" in hdf5_file.keys():
            counts[1] = len(hdf5_file["images_validation"])
        if "images_testing" in hdf5_file.keys():
            counts[2] = len(hdf5_file["images_testing"])
        return tuple(counts)

    def readDataset(self, dataset_name):
        hdf5_file = h5py.File(self.hdf5_path, "r")
        dataset = hdf5_file[dataset_name]
        return dataset[()]


def createBinaryFile():
    hdf5_file = h5py.File(hdf5_path, mode='w')
    buildImageDataset("training", hdf5_file)
    buildLabelDataset("training", hdf5_file)
    buildImageDataset("validation", hdf5_file)
    buildLabelDataset("validation", hdf5_file)
    buildImageDataset("testing", hdf5_file)
    buildLabelDataset("testing", hdf5_file)
    hdf5_file.close()

def buildImageDataset(folder_name, hdf5_file):
    image_addrs = sorted(glob.glob(data_folder + "/"+folder_name+"/*.jpg"))
    dataset_name = "images_"+folder_name
    dataset_shape = (len(image_addrs), img_width, img_height, 3)
    hdf5_file.create_dataset(dataset_name, dataset_shape, np.uint8)
    for i in range(len(image_addrs)):
        # print how many images are saved every 1000 images
        if i % 1000 == 0 and i > 1:
            print ('Train data: {}/{}'.format(i, len(image_addrs)))
        # read an image and resize to (224, 224)
        # cv2 load images as BGR, convert it to RGB
        addr = image_addrs[i]
        img = cv2.imread(addr)
        img = cv2.resize(img, (img_width, img_height), interpolation=cv2.INTER_CUBIC)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # add any image pre-processing here
        # save the image
        hdf5_file[dataset_name][i, ...] = img[None]

def buildLabelDataset(folder_name, hdf5_file):
    dataset_name = "labels_" + folder_name
    num_of_labels = len(all_labels)

    label_addrs = sorted(glob.glob(data_folder + "/"+folder_name+"/*.txt"))
    int_labels = []
    dataset_shape = (len(label_addrs), num_of_labels)
    hdf5_file.create_dataset(dataset_name, dataset_shape, np.uint8)
    for i in range(len(label_addrs)):
        addr = label_addrs[i]
        with open(addr, "r") as text_file:
            label_values = text_file.read().splitlines()
            label_tuples = ()
            for label_value in label_values:
                label_tuples = label_tuples + (all_labels.index(label_value),)
            int_labels.append(label_tuples)
    lb = MultiLabelBinarizer()
    binarized_labels = lb.fit_transform(int_labels)
    print('labels shape - ' + str(np.shape(binarized_labels)))
    hdf5_file[dataset_name][...] = binarized_labels

def readBinaryFile():
    hdf5_file = h5py.File(hdf5_path, "r")
    # print(hdf5_file["train_img"][1])
    # print(hdf5_file["train_labels"][1])
    # plt.imshow(hdf5_file["train_img"][3])
    # plt.show()
    print("Training images "+str(len(hdf5_file["images_training"])))
    print("Training labels " + str(len(hdf5_file["labels_training"])))
    print("Validation images " + str(len(hdf5_file["images_validation"])))
    print("Validation labels " + str(len(hdf5_file["labels_validation"])))
    print("Testing images " + str(len(hdf5_file["images_testing"])))
    print("Testing labels " + str(len(hdf5_file["labels_testing"])))
    return (hdf5_file["images_training"][()])


def getSampleCounts():
    hdf5_file = h5py.File(hdf5_path, "r")
    return (len(hdf5_file["images_training"]), len(hdf5_file["images_validation"]), len(hdf5_file["images_testing"]))

def readDataset(dataset_name):
    hdf5_file = h5py.File(hdf5_path, "r")
    dataset = hdf5_file[dataset_name]
    return dataset[()]

def readImage(index, dataset_suffix):
    hdf5_file = h5py.File(hdf5_path, "r")
    # plt.imshow(hdf5_file["train_img"][index])
    # plt.show()
    dataset_name = "images_"+dataset_suffix
    return hdf5_file[dataset_name][index][()]

def readLabel(index, dataset_suffix):
    hdf5_file = h5py.File(hdf5_path, "r")
    dataset_name = "labels_" + dataset_suffix
    return hdf5_file[dataset_name][index][()]



def createDataSubset(subset_sizes, destFile):
    subset_hdf5_file = h5py.File(destFile, mode='w', )
    createSubsetDataset(subset_sizes[0], "training", subset_hdf5_file)
    createSubsetDataset(subset_sizes[1], "validation", subset_hdf5_file)
    createSubsetDataset(subset_sizes[2], "testing", subset_hdf5_file)

def createSubsetDataset(subset_size,dataset_suffix, subset_hdf5_file):
    hdf5_file = h5py.File(hdf5_path, "r")
    X = hdf5_file["images_" + dataset_suffix]
    y = hdf5_file["labels_" + dataset_suffix]
    total_size =  len(X)
    ints = np.arange(total_size)
    np.random.shuffle(ints)
    rand_indices = ints[:subset_size]
    rand_indices = sorted(rand_indices)
    X_subset = X[rand_indices,...]
    y_subset = y[rand_indices, ...]
    images_subset_dataset = subset_hdf5_file.create_dataset("images_"+dataset_suffix,(subset_size,) +X.shape[1:], np.uint8)
    labels_subset_dataset = subset_hdf5_file.create_dataset("labels_"+dataset_suffix, (subset_size,) +y.shape[1:], np.uint8)

    images_subset_dataset[...] = X_subset
    labels_subset_dataset[...] = y_subset


def getGenerator(batch_size, dataset_suffix, loopOnce=False, scaleX=255, shuffle=False):
    main_loop = 0
    hdf5_file = h5py.File(hdf5_path, "r")
    X = hdf5_file["images_"+dataset_suffix]
    y = hdf5_file["labels_"+dataset_suffix]
    num_of_images = len(X)
    loop = True
    temp_arr = np.arange(num_of_images)
    while loop:
        print("\nMain loop for - "+dataset_suffix+"- "+str(main_loop))
        main_loop = main_loop + 1
        if shuffle:
            random.shuffle(temp_arr)
        for i in range(num_of_images//batch_size + 1):
            indices = temp_arr[i * batch_size:(i + 1) * batch_size][()]
            yield X[sorted(indices)] / scaleX, y[sorted(indices)][()]
            # yield X[i * batch_size:(i + 1) * batch_size][()]/255, y[i * batch_size:(i + 1) * batch_size][()]
        if loopOnce:
            loop = False

def decodePredictions(predictions, threshold=0.5):
    label_dict = {}
    label_addrs = sorted(glob.glob(data_folder + "/testing/*.txt"))
    serialNumbers = []
    for addr in label_addrs:
        serialNumbers.append(os.path.splitext(os.path.basename(addr))[0])
    for index,prediction in enumerate(predictions):
        label_indices = np.nonzero(prediction >= threshold)
        labels = []
        for label_index in label_indices[0]:
            labels.append(all_labels[label_index])
        label_dict[serialNumbers[index]] = labels;
    return label_dict

def getClassificationReport(predictions, threshold=0.5):
    predictions = np.where(predictions >= threshold, 1, 0)
    hdf5_file = h5py.File(hdf5_path, "r")
    actuals = hdf5_file["labels_testing"][()]
    report = metrics.classification_report(actuals, predictions, target_names=all_labels)
    return report

def calculateScores(predictions, threshold=0.5):
    predictions = np.where(predictions >= threshold, 1, 0)
    hdf5_file = h5py.File(hdf5_path, "r")
    actuals = hdf5_file["labels_testing"][()]
    true_positives = float(np.sum(np.multiply(actuals, predictions)))
    predicted_positives = np.sum(predictions)
    actual_positives = np.sum(actuals)
    precision = true_positives/predicted_positives
    recall = true_positives/actual_positives
    f1_score = ((precision*recall)/(precision + recall))*2
    return (precision, recall, f1_score)


def recall(actuals, y_pred):
    actuals = tf.cast(actuals, tf.int32)
    threshold = tf.fill(tf.shape(actuals), value=0.3)
    predictions = tf.where(y_pred >= threshold, tf.fill(tf.shape(actuals), value=1), tf.fill(tf.shape(actuals), value=0))
    true_positives = tf.reduce_sum(tf.multiply(actuals, predictions))
    actual_positives = tf.reduce_sum(actuals)
    recall = tf.divide(true_positives, actual_positives)
    return recall

def precision(actuals, y_pred):
    actuals = tf.cast(actuals, tf.int32)
    threshold = tf.fill(tf.shape(actuals), value=0.3)
    predictions = tf.where(y_pred >= threshold, tf.fill(tf.shape(actuals), value=1),
                           tf.fill(tf.shape(actuals), value=0))
    true_positives = tf.reduce_sum(tf.multiply(actuals, predictions))
    predicted_positives = tf.reduce_sum(predictions)
    precision = tf.divide(true_positives, predicted_positives)
    return precision

def f1score(actuals, y_pred):
    recall_value = recall(actuals, y_pred)
    precision_value = precision(actuals, y_pred)
    f1_score = ((precision_value * recall_value) / (precision_value + recall_value)) * 2
    return f1_score

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

def test():
    arr = np.array([[1,2],[3,4]])
    print(str(np.sum(arr)))

def test_classification_report():
    actuals = [('a', 'b'), ('b', 'c'), ('c', 'd')]
    predictions = [('a'), ('b'), ('c', 'd')]
    lb = MultiLabelBinarizer()
    lb = lb.fit(actuals)
    classes = lb.classes_
    y_actuals = lb.transform(actuals)
    y_preds = lb.fit_transform(predictions)
    print(str(y_actuals))
    print(str(y_preds))
    report = metrics.classification_report(y_actuals, y_preds, target_names=classes)
    print(report)


# test()
# test_classification_report()
# createBinaryFile()
# readBinaryFile()
# print(readImage(103))
# createDataSubset(100, "training", "15-design-codes-tlexp-2/temp/data299.h5")


# for i in getGenerator(32, "validation", False):
#     print(str(np.shape(i[0]))+" - "+str(np.shape(i[1])))
# for i in getGenerator(12, "validation"):
#     print(str(np.shape(i[0]))+" - "+str(np.shape(i[1])))
