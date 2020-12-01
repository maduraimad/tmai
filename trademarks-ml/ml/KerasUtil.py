import keras.callbacks as callbacks
from keras import backend as K
import keras
import tensorflow as tf
from keras.layers import Lambda, concatenate
from keras import Model
from keras.utils import multi_gpu_model

class LearningRateHistory(callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.learning_rate = self.model.lr

    def on_batch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))

lr_metric = K.variable(0, dtype='float32')
class PrintLRCallback(callbacks.Callback):
    def on_train_begin(self, logs=None):
        lr = self.model.optimizer.lr
        print("\n LR at start "+'{:.10f}'.format(K.get_value(lr)))
        K.set_value(lr_metric, K.get_value(lr))

    def on_epoch_end(self, epoch, logs=None):
        lr = self.model.optimizer.lr
        decay = self.model.optimizer.decay
        iterations = self.model.optimizer.iterations
        # print("LR - "+ '{:.10f}'.format(K.get_value(lr)) + " Decay - "+ '{:.10f}'.format(K.get_value(decay)) +" Iterations - " +K.get_value(iterations))
        lr_with_decay = lr / (1. + decay * K.cast(iterations, K.dtype(decay)))
        print("\nLR at epoch ["+str(epoch)+"] is "+'{:.10f}'.format(K.eval(lr_with_decay)))
        K.set_value(lr_metric,  K.eval(lr_with_decay))


def lr(actuals, y_pred):
    return lr_metric

def test():
    lr = 0.0005
    decay = 0.005
    lr_with_decay = lr / (1. + decay * 3)
    print (str(lr_with_decay))


class ModelCheckPointCallback(keras.callbacks.Callback):
    def __init__(self, model, file_path, period=1):
        self.model_to_save = model
        self.file_path = file_path
        self.period = period

    def on_epoch_end(self, epoch, logs=None):
        self.model_to_save.save(self.file_path, overwrite=True)


def create_multi_gpu_model(model, gpus):
    parallel_model = multi_gpu_model(model, gpus)
    return parallel_model

def multi_gpu_model_old(model, gpus):
  if isinstance(gpus, (list, tuple)):
    num_gpus = len(gpus)
    target_gpu_ids = gpus
  else:
    num_gpus = gpus
    target_gpu_ids = range(num_gpus)

  def get_slice(data, i, parts):
    shape = tf.shape(data)
    batch_size = shape[:1]
    input_shape = shape[1:]
    step = batch_size // parts
    if i == num_gpus - 1:
      size = batch_size - step * i
    else:
      size = step
    size = tf.concat([size, input_shape], axis=0)
    stride = tf.concat([step, input_shape * 0], axis=0)
    start = stride * i
    return tf.slice(data, start, size)

  all_outputs = []
  for i in range(len(model.outputs)):
    all_outputs.append([])

  # Place a copy of the model on each GPU,
  # each getting a slice of the inputs.
  for i, gpu_id in enumerate(target_gpu_ids):
    with tf.device('/gpu:%d' % gpu_id):
      with tf.name_scope('replica_%d' % gpu_id):
        inputs = []
        # Retrieve a slice of the input.
        for x in model.inputs:
          input_shape = tuple(x.get_shape().as_list())[1:]
          slice_i = Lambda(get_slice,
                           output_shape=input_shape,
                           arguments={'i': i,
                                      'parts': num_gpus})(x)
          inputs.append(slice_i)

        # Apply model on slice
        # (creating a model replica on the target device).
        outputs = model(inputs)
        if not isinstance(outputs, list):
          outputs = [outputs]

        # Save the outputs for merging back together later.
        for o in range(len(outputs)):
          all_outputs[o].append(outputs[o])

  # Merge outputs on CPU.
  with tf.device('/cpu:0'):
    merged = []
    for name, outputs in zip(model.output_names, all_outputs):
      merged.append(concatenate(outputs,
                                axis=0, name=name))
    return Model(model.inputs, merged)