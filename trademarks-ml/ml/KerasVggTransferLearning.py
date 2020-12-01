from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import backend as K
from keras import applications
import keras
import DataGenerator as dataGenerator
import numpy as np
import os
import json

img_width, img_height = 224, 224
base_folder = os.getenv("base_folder", "/Users/greensod/usptoWork/TrademarkRefiles/data/keras/data/15-design-codes-tlexp-1/");
model_file = os.getenv("model_file","/Users/greensod/usptoWork/TrademarkRefiles/data/keras/data/15-design-codes-tlexp-1/model2.h5")
predictions_file = os.getenv("predictions_file","/Users/greensod/usptoWork/TrademarkRefiles/data/keras/data/15-design-codes-tlexp-1/predictions1.txt")
weights_file = os.getenv("weights_file", "/Users/greensod/usptoWork/TrademarkRefiles/data/keras/data/15-design-codes-tlexp-1/weights1.h5")
logs_dir  = os.getenv("logs_dir", "/Users/greensod/usptoWork/TrademarkRefiles/data/keras/data/15-design-codes-tlexp-1/logs/version1/")
generator_batch_size = int(os.getenv("generator_batch_size", "50"))
epochs = int(os.getenv("epochs", "2"))
model_checkpoint_filepath = os.getenv("model_checkpoint_filepath", "/Users/greensod/usptoWork/TrademarkRefiles/data/keras/data/15-design-codes-tlexp-1/checkpoints/weights.{epoch:02d}-{val_loss:.2f}.hdf5")
model_save_period = int(os.getenv("model_save_period", "1"))
top_model_weights_file = os.getenv("top_model_weights_file", "/Users/greensod/usptoWork/TrademarkRefiles/data/keras/data/15-design-codes-tlexp-1/top_model_weights1.h5")

nb_train_samples = 2721
nb_validation_samples = 512
nb_testing_samples = 127

# nb_train_samples = 211
# nb_validation_samples = 42
# nb_testing_samples = 53
training_batch_size = 50
validation_batch_size = 16

def train():

    input_shape = (img_width, img_height, 3)

    model = Sequential()
    model.add(Conv2D(32, (3, 3), input_shape=input_shape))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(32, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(64, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())
    model.add(Dense(64))
    model.add(Activation('relu'))

    model.add(Dropout(0.5))
    model.add(Dense(15))
    model.add(Activation('sigmoid'))

    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy', dataGenerator.recall, dataGenerator.precision, dataGenerator.f1score])

    model.fit_generator(dataGenerator.getGenerator(generator_batch_size, "training"), steps_per_epoch= (nb_train_samples // training_batch_size)+1,
                        epochs=epochs, verbose=1, validation_data=dataGenerator.getGenerator(generator_batch_size, "validation"), validation_steps= (nb_validation_samples // validation_batch_size)+1,
                        callbacks=[createTensorBoardCallback(), createModelCheckpointCallback()])

    model.save(model_file);
    predictAll(model)

def predictAll(model=None):
    if(model is None):
        model = keras.models.load_model(model_file, custom_objects={"recall": dataGenerator.recall, "precision":dataGenerator.precision, "f1score":dataGenerator.f1score})
    predictions = model.predict_generator(dataGenerator.getGenerator(generator_batch_size, "testing"),
                                          steps= (nb_testing_samples // generator_batch_size)+1)
    labels_dict = dataGenerator.decodePredictions(predictions)
    predictions_dict = {}
    predictions_dict["labels"] = labels_dict
    predictions_dict["scores"] = dataGenerator.calculateScores(predictions)
    with open(predictions_file, 'w') as file:
        file.write(json.dumps(predictions_dict))

def createTensorBoardCallback():
    return keras.callbacks.TensorBoard(log_dir=logs_dir, histogram_freq=0, batch_size=32, write_graph=True, write_grads=False,
                                write_images=False, embeddings_freq=0, embeddings_layer_names=None,
                                embeddings_metadata=None)

def createModelCheckpointCallback():
    return keras.callbacks.ModelCheckpoint(model_checkpoint_filepath, monitor='val_loss', verbose=0, save_best_only=True, save_weights_only=False, mode='auto', period=model_save_period)

def getMetricsArray():
    return ['accuracy', dataGenerator.recall, dataGenerator.precision, dataGenerator.f1score]

def save_bottlebeck_and_train_top_model():
    # build the VGG16 network
    model = applications.VGG16(include_top=False, weights='imagenet')
    bottleneck_training_data = model.predict_generator(
        dataGenerator.getGenerator(generator_batch_size, "training"), (nb_train_samples // training_batch_size)+1, verbose=1)
    print("Bottleneck training data obtained with shape - " + str(bottleneck_training_data.shape))
    bottleneck_training_data = bottleneck_training_data[:nb_train_samples]
    np.save(open(base_folder+"/"+'bottleneck_training_data.npy', 'wb'),
            bottleneck_training_data)

    bottleneck_validation_data = model.predict_generator(
        dataGenerator.getGenerator(generator_batch_size, "validation"), (nb_validation_samples // validation_batch_size)+1, verbose=1)
    print("Bottleneck validation data obtained with shape - " + str(bottleneck_validation_data.shape))
    bottleneck_validation_data = bottleneck_validation_data[:nb_validation_samples]
    np.save(open(base_folder + "/" +'bottleneck_validation_data.npy', 'wb'),
            bottleneck_validation_data)

    # now let's train top model
    train_labels = dataGenerator.readDataset("labels_training")
    validation_labels = dataGenerator.readDataset("labels_validation")

    # validation_data = np.load(open('bottleneck_features_validation.npy'))
    # validation_labels = np.array(
    #     [0] * (nb_validation_samples / 2) + [1] * (nb_validation_samples / 2))

    model = Sequential()
    model.add(Flatten(input_shape=bottleneck_training_data.shape[1:]))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(15, activation='sigmoid'))

    model.compile(optimizer='adam',
                  loss='binary_crossentropy', metrics=getMetricsArray())

    model.fit(bottleneck_training_data, train_labels,
              epochs=epochs,
              batch_size=training_batch_size, verbose=1,
              validation_data=(bottleneck_validation_data, validation_labels),
              callbacks=[createTensorBoardCallback(), createModelCheckpointCallback()])
    model.save_weights(top_model_weights_file)

    # predictAll(model)

def finetuneWithVgg():
    vgg_model = applications.VGG16(weights='imagenet', include_top=False, input_shape=(img_height, img_width, 3))
    print('Model loaded.')

    model = Sequential()
    for l in vgg_model.layers:
        model.add(l)

    # build a classifier model to put on top of the convolutional model
    # x = model.output
    # x = Flatten(input_shape=model.output_shape[1:])(x)
    # x = Dense(256, activation='relu')(x)
    # x = Dropout(0.5)(x)
    # x = Dense(15, activation='sigmoid')(x)

    top_model = Sequential()
    top_model.add(Flatten(input_shape=model.output_shape[1:]))
    top_model.add(Dense(256, activation='relu'))
    top_model.add(Dropout(0.5))
    top_model.add(Dense(15, activation='sigmoid'))

    # note that it is necessary to start with a fully-trained
    # classifier, including the top classifier,
    # in order to successfully do fine-tuning
    top_model.load_weights(top_model_weights_file)

    # add the model on top of the convolutional base
    model.add(top_model)

    # set the first 25 layers (up to the last conv block)
    # to non-trainable (weights will not be updated)
    for layer in model.layers[:15]:
        layer.trainable = False

    # compile the model with a SGD/momentum optimizer
    # and a very slow learning rate.
    model.compile(loss='binary_crossentropy',
                  optimizer="adam",
                  metrics=['accuracy', dataGenerator.recall, dataGenerator.precision, dataGenerator.f1score])

    model.fit_generator(dataGenerator.getGenerator(generator_batch_size, "training"), steps_per_epoch= (nb_train_samples // training_batch_size)+1,
                        epochs=epochs, verbose=1, validation_data=dataGenerator.getGenerator(generator_batch_size, "validation"), validation_steps= (nb_validation_samples // validation_batch_size)+1,
                        callbacks=[createTensorBoardCallback(), createModelCheckpointCallback()])

    model.save(model_file);
    predictAll(model)


# train()
# predictAll()
save_bottlebeck_and_train_top_model()
# finetuneWithVgg()