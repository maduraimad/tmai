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

# dimensions of our images.
img_width, img_height = 224, 224
model_file = os.getenv("model_file","/Users/greensod/usptoWork/TrademarkRefiles/data/keras/data/15-design-codes-2/model1.h5")
predictions_file = os.getenv("predictions_file","/Users/greensod/usptoWork/TrademarkRefiles/data/keras/data/15-design-codes-2/predictions1.txt")
weights_file = os.getenv("weights_file", "/Users/greensod/usptoWork/TrademarkRefiles/data/keras/data/15-design-codes-2/weights1.h5")
tl_weights_file = os.getenv("weights_file", "/Users/greensod/usptoWork/TrademarkRefiles/data/keras/data/15-design-codes/weights2.h5")
tl_model_file = os.getenv("weights_file", "/Users/greensod/usptoWork/TrademarkRefiles/data/keras/data/15-design-codes/model2.h5")
logs_dir  = os.getenv("logs_dir", "/Users/greensod/usptoWork/TrademarkRefiles/data/keras/data/15-design-codes-2/logs/version1/")
epochs = 50

nb_train_samples = 6260
nb_validation_samples = 725
nb_testing_samples = 111
generator_batch_size = 32
training_batch_size = 32
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
                  metrics=['accuracy'])

    model.fit_generator(dataGenerator.getGenerator(generator_batch_size, "training"), steps_per_epoch= (nb_train_samples // training_batch_size)+1,
                        epochs=epochs, verbose=1, validation_data=dataGenerator.getGenerator(generator_batch_size, "validation"), validation_steps= (nb_validation_samples // validation_batch_size)+1)

    model.save(model_file);
    predictAll(model)

def trainFromVggAllLayers():
    input_shape = (224, 224, 3)
    vggModel = applications.VGG16(weights='imagenet', include_top=True, input_shape=input_shape)
    # set the first 25 layers (up to the last conv block)
    # to non-trainable (weights will not be updated)
    # sashi - using 24 instead of 25
    for layer in vggModel.layers[:15]:
        layer.trainable = False
    # Popping out the last layer did not help
    # model.layers.pop()
    # model.outputs = [model.layers[-1].output]
    # model.output_layers = [model.layers[-1]]
    # model.layers[-1].outbound_nodes = []
    # # model.output = model.layers[-1].output
    # model.output_shape = model.layers[-1].output_shape
    # Copy all layers (except the last one) into a new Model object
    replicatedModel = Sequential()
    for layer in vggModel.layers[:22]:
        replicatedModel.add(layer)

    top_model = Sequential()
    top_model.add(Dense(15, input_shape=replicatedModel.output_shape[1:]))
    top_model.add(Activation('sigmoid'))
    predictions = Dense(15, input_shape=replicatedModel.output_shape[1:], name="DenseOutput", activation="sigmoid")(replicatedModel.output)


    model = keras.models.Model(inputs=replicatedModel.input, outputs=predictions)

    # compile the model with a SGD/momentum optimizer
    # and a very slow learning rate.
    model.compile(loss='binary_crossentropy',
                  optimizer=keras.optimizers.SGD(lr=1e-4, momentum=0.9),
                  metrics=['accuracy', dataGenerator.recall, dataGenerator.precision, dataGenerator.f1score])

    model.fit_generator(dataGenerator.getGenerator(generator_batch_size, "training"), steps_per_epoch= (nb_train_samples // training_batch_size)+1,
                        epochs=epochs, verbose=1,
                        validation_data=dataGenerator.getGenerator(generator_batch_size, "validation"), validation_steps= (nb_validation_samples // validation_batch_size)+1,
                        callbacks=[createTensorBoardCallback()] )

    model.save(model_file);
    predictAll(model)


def trainFromVgg():
    input_shape = (224, 224, 3)
    model = applications.VGG16(weights='imagenet', include_top=False, input_shape=input_shape)
    top_model = Sequential()
    top_model.add(Flatten(input_shape=model.output_shape[1:]))
    # top_model.add(Dense(256, activation='relu'))
    # top_model.add(Dropout(0.5))
    # top_model.add(Dense(1, activation='sigmoid'))
    top_model.add(Dense(64))
    top_model.add(Activation('relu'))
    top_model.add(Dropout(0.5))
    top_model.add(Dense(15))
    top_model.add(Activation('sigmoid'))
    # note that it is necessary to start with a fully-trained
    # classifier, including the top classifier,
    # in order to successfully do fine-tuning
    # sashi - the weights file we have is for 5 layers, we want to only load weights of last 2 layers
    previous_model = keras.models.load_model(tl_model_file)
    previous_weights_list = previous_model.get_weights()
    for i, layer in enumerate(previous_model.layers[9:14]):
        top_model.layers[i].set_weights(layer.get_weights())
    # top_model.load_weights(tl_weights_file)

    # set the first 25 layers (up to the last conv block)
    # to non-trainable (weights will not be updated)
    # sashi - using 24 instead of 25
    for layer in model.layers[:24]:
        layer.trainable = False

    # add the model on top of the convolutional base
    model.add(Flatten())
    model.add(top_model)

    # compile the model with a SGD/momentum optimizer
    # and a very slow learning rate.
    model.compile(loss='binary_crossentropy',
                  optimizer=keras.optimizers.SGD(lr=1e-4, momentum=0.9),
                  metrics=['accuracy'])

    model.fit_generator(dataGenerator.getGenerator(generator_batch_size, "training"), steps_per_epoch= (nb_train_samples // training_batch_size)+1,
                        epochs=epochs, verbose=1,
                        validation_data=dataGenerator.getGenerator(generator_batch_size, "validation"), validation_steps= (nb_validation_samples // validation_batch_size)+1,
                        callbacks=[createTensorBoardCallback()] )

    model.save(model_file);
    predictAll(model)

def createTensorBoardCallback():
    return keras.callbacks.TensorBoard(log_dir=logs_dir, histogram_freq=0, batch_size=32, write_graph=True, write_grads=False,
                                write_images=False, embeddings_freq=0, embeddings_layer_names=None,
                                embeddings_metadata=None)



def test():
    model = keras.models.load_model(model_file)
    x = dataGenerator.readImage(21, "testing")
    x = np.expand_dims(x, axis=0)
    predictions = model.predict(x)
    print(predictions > [0.5])

def predictAll(model=None):
    if(model is None):
        model = keras.models.load_model(model_file)
    predictions = model.predict_generator(dataGenerator.getGenerator(generator_batch_size, "testing"),
                                          steps= (nb_testing_samples // generator_batch_size)+1)
    labels_dict = dataGenerator.decodePredictions(predictions)
    predictions_dict = {}
    predictions_dict["labels"] = labels_dict
    predictions_dict["scores"] = dataGenerator.calculateScores(predictions)
    with open(predictions_file, 'w') as file:
        file.write(json.dumps(predictions_dict))

def saveWeights():
    model = keras.models.load_model(model_file)
    model.save_weights(weights_file)

def evaluate():
    model = keras.models.load_model(model_file)
    results = model.evaluate_generator(dataGenerator.getGenerator(generator_batch_size, "testing"),
                                          steps=(nb_testing_samples // generator_batch_size)+1 )
    print(results)

# trainFromVggAllLayers()
evaluate()