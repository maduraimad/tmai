
import tensorflow as tf
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.gridspec as gridspec
import numpy as np
import os
import cv2
from math import floor, ceil, pi
from DataGenerator import DataCreatorUtil
import h5py


IMAGE_SIZE = 299
def central_scale_images(X_imgs, scales):
    print("Scaling images")
    # Various settings needed for Tensorflow operation
    boxes = np.zeros((len(scales), 4), dtype = np.float32)
    for index, scale in enumerate(scales):
        x1 = y1 = 0.5 - 0.5 * scale # To scale centrally
        x2 = y2 = 0.5 + 0.5 * scale
        boxes[index] = np.array([y1, x1, y2, x2], dtype = np.float32)
    box_ind = np.zeros((len(scales)), dtype = np.int32)
    crop_size = np.array([IMAGE_SIZE, IMAGE_SIZE], dtype = np.int32)

    X_scale_data = []
    tf.reset_default_graph()
    X = tf.placeholder(tf.float32, shape = (1, IMAGE_SIZE, IMAGE_SIZE, 3))
    # Define Tensorflow operation for all scales but only one base image at a time
    tf_img = tf.image.crop_and_resize(X, boxes, box_ind, crop_size)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for img_data in X_imgs:
            batch_img = np.expand_dims(img_data, axis = 0)
            scaled_imgs = sess.run(tf_img, feed_dict = {X: batch_img})
            X_scale_data.extend(scaled_imgs)

    X_scale_data = np.array(X_scale_data, dtype = np.float32)
    return X_scale_data


def get_translate_parameters(index):
    if index == 0:  # Translate left 20 percent
        offset = np.array([0.0, 0.2], dtype=np.float32)
        size = np.array([IMAGE_SIZE, ceil(0.8 * IMAGE_SIZE)], dtype=np.int32)
        w_start = 0
        w_end = int(ceil(0.8 * IMAGE_SIZE))
        h_start = 0
        h_end = IMAGE_SIZE
    elif index == 1:  # Translate right 20 percent
        offset = np.array([0.0, -0.2], dtype=np.float32)
        size = np.array([IMAGE_SIZE, ceil(0.8 * IMAGE_SIZE)], dtype=np.int32)
        w_start = int(floor((1 - 0.8) * IMAGE_SIZE))
        w_end = IMAGE_SIZE
        h_start = 0
        h_end = IMAGE_SIZE
    elif index == 2:  # Translate top 20 percent
        offset = np.array([0.2, 0.0], dtype=np.float32)
        size = np.array([ceil(0.8 * IMAGE_SIZE), IMAGE_SIZE], dtype=np.int32)
        w_start = 0
        w_end = IMAGE_SIZE
        h_start = 0
        h_end = int(ceil(0.8 * IMAGE_SIZE))
    else:  # Translate bottom 20 percent
        offset = np.array([-0.2, 0.0], dtype=np.float32)
        size = np.array([ceil(0.8 * IMAGE_SIZE), IMAGE_SIZE], dtype=np.int32)
        w_start = 0
        w_end = IMAGE_SIZE
        h_start = int(floor((1 - 0.8) * IMAGE_SIZE))
        h_end = IMAGE_SIZE

    return offset, size, w_start, w_end, h_start, h_end

# n_translations - 4 means left 20%, right 20%, top 20%, bottom 20%
def translate_images(X_imgs, n_translations):
    print("Translating images")
    offsets = np.zeros((len(X_imgs), 2), dtype=np.float32)
    X_translated_arr = []

    tf.reset_default_graph()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for i in range(n_translations):
            X_translated = np.zeros((len(X_imgs), IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.float32)
            X_translated.fill(1.0)  # Filling background color
            base_offset, size, w_start, w_end, h_start, h_end = get_translate_parameters(i)
            offsets[:, :] = base_offset
            glimpses = tf.image.extract_glimpse(X_imgs, size, offsets)

            glimpses = sess.run(glimpses)
            X_translated[:, h_start: h_start + size[0], w_start: w_start + size[1], :] = glimpses
            X_translated_arr.extend(X_translated)
    X_translated_arr = np.array(X_translated_arr, dtype=np.float32)
    return X_translated_arr


# Rotation (at finer angles)
# rotations - 3 means 90, 180, 270
def rotate_images(X_imgs, rotations):
    print("Rotating images")
    X_rotate = []
    tf.reset_default_graph()
    X = tf.placeholder(tf.float32, shape=(IMAGE_SIZE, IMAGE_SIZE, 3))
    k = tf.placeholder(tf.int32)
    tf_img = tf.image.rot90(X, k=k)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for img in X_imgs:
            for i in range(rotations):  # Rotation at 90, 180 and 270 degrees
                rotated_img = sess.run(tf_img, feed_dict={X: img, k: i + 1})
                X_rotate.append(rotated_img)

    X_rotate = np.array(X_rotate, dtype=np.float32)
    return X_rotate

### Flip Images
# flips left right, up down, transpose
def flip_images(X_imgs):
    print("Flipping images")
    X_flip = []
    tf.reset_default_graph()
    X = tf.placeholder(tf.float32, shape = (IMAGE_SIZE, IMAGE_SIZE, 3))
    # tf_img1 = tf.image.flip_left_right(X)
    # tf_img2 = tf.image.flip_up_down(X)
    tf_img3 = tf.image.transpose_image(X)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for img in X_imgs:
            # flipped_imgs = sess.run([tf_img1, tf_img2, tf_img3], feed_dict = {X: img})
            flipped_imgs = sess.run([tf_img3], feed_dict={X: img})
            X_flip.extend(flipped_imgs)
    X_flip = np.array(X_flip, dtype = np.float32)
    return X_flip

### Adding Salt and Pepper
def add_salt_pepper_noise(X_imgs):
    print("Adding salt and pepper")
    # Need to produce a copy as to not modify the original image
    X_imgs_copy = X_imgs.copy()
    row, col, _ = X_imgs_copy[0].shape
    salt_vs_pepper = 0.2
    amount = 0.004
    num_salt = np.ceil(amount * X_imgs_copy[0].size * salt_vs_pepper)
    num_pepper = np.ceil(amount * X_imgs_copy[0].size * (1.0 - salt_vs_pepper))

    for X_img in X_imgs_copy:
        # Add Salt noise
        coords = [np.random.randint(0, i - 1, int(num_salt)) for i in X_img.shape]
        X_img[coords[0], coords[1], :] = 1

        # Add Pepper noise
        coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in X_img.shape]
        X_img[coords[0], coords[1], :] = 0
    return X_imgs_copy



### Lighting condition
def add_gaussian_noise(X_imgs):
    print("Adding gaussian noise")
    gaussian_noise_imgs = []
    row, col, _ = X_imgs[0].shape
    # Gaussian distribution parameters
    mean = 0
    var = 0.1
    sigma = var ** 0.5

    for X_img in X_imgs:
        gaussian = np.random.random((row, col, 1)).astype(np.float32)
        gaussian = np.concatenate((gaussian, gaussian, gaussian), axis=2)
        gaussian_img = cv2.addWeighted(X_img, 0.75, 0.25 * gaussian, 0.25, 0)
        gaussian_noise_imgs.append(gaussian_img)
    gaussian_noise_imgs = np.array(gaussian_noise_imgs, dtype=np.float32)
    return gaussian_noise_imgs




def get_mask_coord(imshape):
    vertices = np.array([[(0.09 * imshape[1], 0.99 * imshape[0]),
                          (0.43 * imshape[1], 0.32 * imshape[0]),
                          (0.56 * imshape[1], 0.32 * imshape[0]),
                          (0.85 * imshape[1], 0.99 * imshape[0])]], dtype=np.int32)
    return vertices


def get_perspective_matrices(X_img):
    offset = 15
    img_size = (X_img.shape[1], X_img.shape[0])

    # Estimate the coordinates of object of interest inside the image.
    src = np.float32(get_mask_coord(X_img.shape))
    dst = np.float32([[offset, img_size[1]], [offset, 0], [img_size[0] - offset, 0],
                      [img_size[0] - offset, img_size[1]]])

    perspective_matrix = cv2.getPerspectiveTransform(src, dst)
    return perspective_matrix

### Transform Perspective
def perspective_transform(X_img):
    print("Transforming perspective")
    # Doing only for one type of example
    perspective_matrix = get_perspective_matrices(X_img)
    warped_img = cv2.warpPerspective(X_img, perspective_matrix,
                                     (X_img.shape[1], X_img.shape[0]),
                                     flags=cv2.INTER_LINEAR)
    return warped_img


def perform_image_augmentation():
    image_width = 299
    image_height = 299
    image_channels  = 3
    labels_count = 15
    data_folder = "Trial2/divided"
    hdf5_folder = "rial2"
    data_creator = DataCreatorUtil(data_folder=data_folder, hdf5_folder=hdf5_folder, hdf5_file_name="data-short-299.h5")
    new_hdf5_file_path = hdf5_folder+"/data-299-new.h5"
    number_of_augmentations = 6

    original_training_count = data_creator.getSampleCounts()[0]
    new_count = original_training_count * number_of_augmentations
    batch_size = 32
    starting_counter = 0
    with h5py.File(new_hdf5_file_path, "w") as hdf5_file:
        new_images_dataset =  hdf5_file.create_dataset("images_training", (new_count, image_width, image_height, image_channels), np.uint8)
        new_labels_dataset = hdf5_file.create_dataset("labels_training", (new_count, labels_count), np.uint8)
        # first we need to add the original data as is
        starting_counter = augment_data(data_creator, batch_size, starting_counter, new_images_dataset, new_labels_dataset, 1, None)

        starting_counter = augment_data(data_creator, batch_size, starting_counter, new_images_dataset, new_labels_dataset, 1, central_scale_images, [0.7])
        #
        starting_counter = augment_data(data_creator, batch_size, starting_counter, new_images_dataset, new_labels_dataset, 1, translate_images, 1)
        #
        starting_counter = augment_data(data_creator, batch_size, starting_counter, new_images_dataset, new_labels_dataset, 1, rotate_images, 1)

        starting_counter = augment_data(data_creator, batch_size, starting_counter, new_images_dataset, new_labels_dataset, 1, flip_images)

        starting_counter = augment_data(data_creator, batch_size, starting_counter, new_images_dataset, new_labels_dataset, 1, add_salt_pepper_noise)
        # some problem in gaussian noise
        # starting_counter = augment_data(data_creator, batch_size, starting_counter, new_images_dataset, new_labels_dataset, 1, add_gaussian_noise)
        # some problem
        # starting_counter = augment_data(data_creator, batch_size, starting_counter, new_images_dataset, new_labels_dataset, 1, perspective_transform)

        print("something")

def copy_datasets():
    image_width = 299
    image_height = 299
    image_channels  = 3
    labels_count = 107

    data_folder = "/data/107-designCodes/dataset/group1/network0/divided"
    hdf5_folder = "/data/107-designCodes/dataset/group1/network0/"
    data_creator_validation = DataCreatorUtil(data_folder=data_folder, hdf5_folder=hdf5_folder, hdf5_file_name="data-299-original.h5")
    data_creator_training = DataCreatorUtil(data_folder=data_folder, hdf5_folder=hdf5_folder,hdf5_file_name="data-299-copy.h5")


    new_hdf5_file_path = hdf5_folder + "/data-299.h5"

    original_training_count = data_creator_training.getSampleCounts()[0]
    original_validation_count = data_creator_validation.getSampleCounts()[1]
    new_validation_count = original_validation_count
    new_training_count  = original_training_count
    batch_size = 32
    starting_counter = 0

    with h5py.File(new_hdf5_file_path, "w") as hdf5_file:
        new_images_dataset = hdf5_file.create_dataset("images_testing",(1, image_width, image_height, image_channels),np.uint8)
        new_labels_dataset = hdf5_file.create_dataset("labels_testing", (1, labels_count), np.uint8)
        starting_counter = 0
        new_images_dataset = hdf5_file.create_dataset("images_training",(new_training_count, image_width, image_height, image_channels), np.uint8)
        new_labels_dataset = hdf5_file.create_dataset("labels_training", (new_training_count, labels_count), np.uint8)
        for data in data_creator_training.getGenerator(batch_size=batch_size, dataset_suffix="training", scaleX=1, loopOnce=True):
            images = data[0]
            labels = data[1]
            new_images_dataset[starting_counter: starting_counter + images.shape[0], ...] = images
            new_labels_dataset[starting_counter: starting_counter + images.shape[0], ...] = labels
            starting_counter = starting_counter + images.shape[0]

        starting_counter = 0
        new_images_dataset = hdf5_file.create_dataset("images_validation",(new_validation_count, image_width, image_height, image_channels),np.uint8)
        new_labels_dataset = hdf5_file.create_dataset("labels_validation", (new_validation_count, labels_count), np.uint8)
        for data in data_creator_validation.getGenerator(batch_size=batch_size, dataset_suffix="validation", scaleX=1, loopOnce=True):
            images = data[0]
            labels = data[1]
            new_images_dataset[starting_counter: starting_counter + images.shape[0], ...] = images
            new_labels_dataset[starting_counter: starting_counter + images.shape[0], ...] = labels
            starting_counter = starting_counter + images.shape[0]

        print("Done")


def augment_data(data_creator, batch_size, starting_counter, new_images_dataset, new_labels_dataset, extend_labels_multiple, augment_function, *args):
    for data in data_creator.getGenerator(batch_size=batch_size, dataset_suffix="training", scaleX=1, loopOnce=True):
        if augment_function is not None:
            images = augment_function(data[0], *args)
        else:
            images = data[0]
        labels = data[1]
        if extend_labels_multiple > 1:
            new_labels = labels
            for i in range(0, extend_labels_multiple -1):
                new_labels = np.concatenate((new_labels, labels), axis=0)
            labels = new_labels

        new_images_dataset[starting_counter: starting_counter + images.shape[0], ...] = images
        new_labels_dataset[starting_counter: starting_counter + images.shape[0], ...] = labels
        starting_counter = starting_counter + images.shape[0]

    return starting_counter

perform_image_augmentation()