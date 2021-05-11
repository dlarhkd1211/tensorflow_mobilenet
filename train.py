import tensorflow as tf

gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpus[0], True)
from tensorflow import keras
from tensorflow.keras import models
from tensorflow.keras.applications import MobileNetV2, MobileNetV3Large, MobileNetV3Small
from tensorflow.keras.layers import Conv2D, ReLU, MaxPooling2D, Dense, BatchNormalization, GlobalAveragePooling2D
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.preprocessing.image import ImageDataGenerator


import numpy as np
import os
import time

from absl import flags, app
from absl.flags import FLAGS

flags.DEFINE_string('breed', default='dog', help='dog or cat')
flags.DEFINE_integer('N_EPOCHS', default=100, help='number of epochs')
flags.DEFINE_integer('N_BATCH', default=64, help='number of batch size')
flags.DEFINE_float('LR_INIT', default=0.0001, help='initial learning rate')
flags.DEFINE_float('LR_MAX', default=0.0003, help='maximun learning rate')
flags.DEFINE_float('LR_MIN', default=0.00001, help='minimun learning rate')
flags.DEFINE_integer('RAMPUP_EPOCH', default=4, help='number of epochs to max lr')
flags.DEFINE_float('EXP_DECAY', default=0.9, help='ratio for multiply to lr')
flags.DEFINE_string('model', default='MobileNetV2', help='MobileNetV2, MobileNetV3Small, MobileNetV3Large')
flags.DEFINE_string('pretrained', default='imagenet', help='pretrained weights')
flags.DEFINE_boolean('early_stopping', default=True, help='early stop?')
flags.DEFINE_float('dropout_rate', default=0.3, help='dropout rate of model')
flags.DEFINE_integer('patience', default=15, help='patience to early stopping')


def main(_argv):

    dataset = np.load('./data/' + FLAGS.breed + '/dataset.npz')
    X = dataset['X']
    Y = dataset['Y']
    x_val = dataset['x_val']
    y_val = dataset['y_val']
    x_te = dataset['x_te']
    y_te = dataset['y_te']

    f = open('data/' + FLAGS.breed + '/num.txt', 'r')
    line = f.readlines()
    N_TRAIN = int(line[0])
    N_TEST = int(line[1])
    N_CLASS = int(line[2])

    IMG_SIZE = 224



    def model_selection(model):
        if FLAGS.model == 'MobileNetV2':
            mobilenet = MobileNetV2(input_shape=(IMG_SIZE, IMG_SIZE, 3), include_top=False, pooling=None,
                                    weights=FLAGS.pretrained)
        elif FLAGS.model == 'MobileNetV3Small':
            mobilenet = MobileNetV3Small(input_shape=(IMG_SIZE, IMG_SIZE, 3), include_top=False, pooling=None,
                                         dropout_rate=FLAGS.dropout_rate, weights=FLAGS.pretrained)
        else:
            mobilenet = MobileNetV3Large(input_shape=(IMG_SIZE, IMG_SIZE, 3), include_top=False, pooling=None,
                                         dropout_rate=FLAGS.dropout_rate, weights=FLAGS.pretrained)

        return mobilenet

    mobilenet = model_selection(FLAGS.model)

    model = models.Sequential()
    model.add(mobilenet)
    model.add(GlobalAveragePooling2D())
    model.add(Dense(256))
    model.add(BatchNormalization())
    model.add(ReLU())
    model.add(Dense(N_CLASS, activation='softmax'))

    def lr_schedule_fn(epoch):
        if epoch < FLAGS.RAMPUP_EPOCH:
            lr = (FLAGS.LR_MAX - FLAGS.LR_MIN) / FLAGS.RAMPUP_EPOCH * epoch + FLAGS.LR_INIT
        else:
            lr = (FLAGS.LR_MAX - FLAGS.LR_MIN) * FLAGS.EXP_DECAY ** (epoch - FLAGS.RAMPUP_EPOCH)
        return lr

    lr_callback = keras.callbacks.LearningRateScheduler(lr_schedule_fn)

    model.compile(optimizer=tf.keras.optimizers.Adam(FLAGS.LR_INIT),
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                  metrics=['accuracy'])

    callback = [lr_callback]
    if FLAGS.early_stopping:
        early_stopping = EarlyStopping(monitor='val_loss', patience=FLAGS.patience)
        callback = [lr_callback, early_stopping]

    datagen = ImageDataGenerator(
        rotation_range = 20,
        width_shift_range=0.15,
        height_shift_range=0.15,
        horizontal_flip=True,
        vertical_flip=True,
        zoom_range = 0.1,
    )
    datagen.fit(X)

    model.fit(datagen.flow(X, Y, batch_size = FLAGS.N_BATCH),  epochs = FLAGS.N_EPOCHS, verbose = 1, validation_data = datagen.flow(x_val, y_val, batch_size = FLAGS.N_BATCH), callbacks=callback)

    fold_name = FLAGS.model[10:]
    n_time = time.localtime()
    fold_day = str(n_time.tm_mon * 100 + n_time.tm_mday)
    fold_time = str(n_time.tm_hour * 100 + n_time.tm_min)
    model.save('model/' + FLAGS.breed + '/' + fold_name + 'saved_model' + fold_day + fold_time + '/my_model')

    converter = tf.lite.TFLiteConverter.from_saved_model(
        'model/' + FLAGS.breed + '/' + fold_name + 'saved_model' + fold_day + fold_time + '/my_model')  # path to the SavedModel directory
    tflite_model = converter.convert()

    # Save the model.
    with open('model/' + FLAGS.breed + '/' + fold_name + 'saved_model' + fold_day + fold_time + '/model.tflite',
              'wb') as f:
        f.write(tflite_model)


if __name__ == '__main__':
    app.run(main)