import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpus[0], True)
from tensorflow import keras
from tensorflow.keras import models
from tensorflow.keras.applications import MobileNetV2, MobileNetV3Large, MobileNetV3Small
from tensorflow.keras.layers import Conv2D, ReLU, MaxPooling2D, Dense, BatchNormalization, GlobalAveragePooling2D
from tensorflow.keras.callbacks import EarlyStopping

import numpy as np
import os
import time

from absl import flags, app
from absl.flags import FLAGS


flags.DEFINE_string('breed', default = 'dog', help = 'dog or cat')
flags.DEFINE_integer('N_EPOCHS', default = 100, help='number of epochs')
flags.DEFINE_integer('N_BATCH', default = 24, help='dog = 24, cat = 20')
flags.DEFINE_float('LR_INIT', default = 0.0001, help='initial learning rate')
flags.DEFINE_float('LR_MAX', default = 0.0003, help='maximun learning rate')
flags.DEFINE_float('LR_MIN', default = 0.00001, help='minimun learning rate')
flags.DEFINE_integer('RAMPUP_EPOCH', default = 4, help='number of epochs to max lr')
flags.DEFINE_float('EXP_DECAY', default = 0.9, help='ratio for multiply to lr')
flags.DEFINE_string('model', default = 'MobileNetV2', help='MobileNetV2, MobileNetV3Small, MobileNetV3Large')
flags.DEFINE_string('pretrained', default = 'imagenet', help='pretrained weights')
flags.DEFINE_boolean('early_stopping', default = True, help='early stop?')
flags.DEFINE_float('dropout_rate', default = 0.3, help='dropout rate of model')
flags.DEFINE_integer('patience', default = 15, help ='patience to early stopping')

def main(_argv):
  cur_dir = os.getcwd()
  tfr_dir = os.path.join(cur_dir, 'data')
  tfr_train_dir = os.path.join(tfr_dir, FLAGS.breed,'cls_train.tfr')
  tfr_test_dir = os.path.join(tfr_dir, FLAGS.breed,'cls_test.tfr')

  f =open('data/' + FLAGS.breed + '/num.txt', 'r')
  line = f.readlines()
  N_TRAIN = int(line[0])
  N_TEST = int(line[1])
  N_CLASS = int(line[2])

  IMG_SIZE = 224
  steps_per_epoch = N_TRAIN / FLAGS.N_BATCH
  validation_steps = int(np.ceil(N_TEST / FLAGS.N_BATCH))
  def _parse_function(tfrecord_serialized):
    features={'image': tf.io.FixedLenFeature([], tf.string),
              'cls_num': tf.io.FixedLenFeature([], tf.int64)
              }
    parsed_features = tf.io.parse_single_example(tfrecord_serialized, features)

    image = tf.io.decode_raw(parsed_features['image'], tf.uint8)
    image = tf.reshape(image, [IMG_SIZE, IMG_SIZE, 3])
    image = tf.cast(image, tf.float32)/255. 

    label = tf.cast(parsed_features['cls_num'], tf.int64)
    label = tf.one_hot(label, N_CLASS)

    return image, label

  def cutmix(images, labels, PROB=0.5):  
    imgs = []; labs = []
    for i in range(FLAGS.N_BATCH):
      APPLY = tf.cast(tf.random.uniform(()) <= PROB, tf.int32)
      idx = tf.random.uniform((), 0, FLAGS.N_BATCH, tf.int32)

      W = IMG_SIZE; H = IMG_SIZE
      lam = tf.random.uniform(())
      cut_ratio = tf.math.sqrt(1.-lam)    
      cut_w = tf.cast(W * cut_ratio, tf.int32) * APPLY
      cut_h = tf.cast(H * cut_ratio, tf.int32) * APPLY

      cx = tf.random.uniform((), int(W/8), int(7/8*W), tf.int32)
      cy = tf.random.uniform((), int(H/8), int(7/8*H), tf.int32)

      xmin = tf.clip_by_value(cx - cut_w//2, 0, W)
      ymin = tf.clip_by_value(cy - cut_h//2, 0, H)
      xmax = tf.clip_by_value(cx + cut_w//2, 0, W)
      ymax = tf.clip_by_value(cy + cut_h//2, 0, H)    
      
      mid_left = images[i, ymin:ymax, :xmin, :]
      mid_mid = images[idx, ymin:ymax, xmin:xmax, :]
      mid_right = images[i, ymin:ymax, xmax:, :]
      middle = tf.concat([mid_left, mid_mid, mid_right], axis=1)
      top = images[i, :ymin, :, :]
      bottom = images[i, ymax:, :, :]
      new_img = tf.concat([top, middle, bottom], axis=0)
      imgs.append(new_img)
      
      alpha = tf.cast((cut_w*cut_h)/(W*H), tf.float32)
      label1 = labels[i]; label2 = labels[idx]
      new_label = ((1-alpha)*label1 + alpha*label2)
      labs.append(new_label)

    new_imgs = tf.reshape(tf.stack(imgs), [-1, IMG_SIZE, IMG_SIZE, 3])
    new_labs = tf.reshape(tf.stack(labs), [-1, N_CLASS])

    return new_imgs, new_labs


  train_dataset = tf.data.TFRecordDataset(tfr_train_dir)
  train_dataset = train_dataset.map(_parse_function, num_parallel_calls=tf.data.experimental.AUTOTUNE)
  train_dataset = train_dataset.shuffle(buffer_size=N_TRAIN).prefetch(
      tf.data.experimental.AUTOTUNE).batch(FLAGS.N_BATCH)
  train_dataset = train_dataset.map(cutmix).repeat()

  val_dataset = tf.data.TFRecordDataset(tfr_test_dir)
  val_dataset = val_dataset.map(_parse_function, num_parallel_calls=tf.data.experimental.AUTOTUNE)
  val_dataset = val_dataset.batch(FLAGS.N_BATCH).repeat()

  def model_selection(model):
    if FLAGS.model == 'MobileNetV2':
      mobilenet = MobileNetV2(input_shape = (IMG_SIZE,IMG_SIZE,3), include_top = False, pooling = None, weights = FLAGS.pretrained)
    elif FLAGS.model == 'MobileNetV3Small':
      mobilenet = MobileNetV3Small(input_shape = (IMG_SIZE,IMG_SIZE,3), include_top = False, pooling = None, dropout_rate = FLAGS.dropout_rate, weights = FLAGS.pretrained)
    else:
      mobilenet = MobileNetV3Large(input_shape = (IMG_SIZE,IMG_SIZE,3), include_top = False, pooling = None, dropout_rate = FLAGS.dropout_rate, weights = FLAGS.pretrained)

    return mobilenet

  mobilenet = model_selection(FLAGS.model)

  model = models.Sequential()
  model.add(mobilenet)
  model.add(GlobalAveragePooling2D())
  model.add(Dense(256))
  model.add(BatchNormalization())
  model.add(ReLU())
  model.add(Dense(N_CLASS, activation = 'softmax'))

  def lr_schedule_fn(epoch):
    if epoch < FLAGS.RAMPUP_EPOCH:
      lr = (FLAGS.LR_MAX - FLAGS.LR_MIN) / FLAGS.RAMPUP_EPOCH * epoch + FLAGS.LR_INIT
    else:
      lr = (FLAGS.LR_MAX - FLAGS.LR_MIN) * FLAGS.EXP_DECAY ** (epoch - FLAGS.RAMPUP_EPOCH)
    return lr

  lr_callback = keras.callbacks.LearningRateScheduler(lr_schedule_fn)

  model.compile(optimizer=tf.keras.optimizers.Adam(FLAGS.LR_INIT),
                loss=tf.keras.losses.CategoricalCrossentropy(),
                metrics=['accuracy'])


  callback = [lr_callback]
  if FLAGS.early_stopping:
    early_stopping = EarlyStopping(monitor = 'val_loss', patience = FLAGS.patience)
    callback = [lr_callback, early_stopping]

  # datagen = ImageDataGenerator(
  #     rotation_range = 20,
  #     width_shift_range=0.15,
  #     height_shift_range=0.15,
  #     horizontal_flip=True,
  #     vertical_flip=True,
  #     zoom_range = 0.1,
  # )
  # datagen.fit(X)

  # history = model.fit(datagen.flow(X, Y, batch_size = 16),  epochs = 100, verbose = 1, validation_data = datagen.flow(x_val, y_val, batch_size = 16), callbacks=[lr_callback, early_stopping])

  history = model.fit(
      train_dataset,
      epochs=FLAGS.N_EPOCHS,
      steps_per_epoch=steps_per_epoch,
      validation_data=val_dataset,
      validation_steps=validation_steps,
      callbacks=callback
  )

  fold_name = FLAGS.model[10:]
  n_time = time.localtime()
  fold_day = str(n_time.tm_mon * 100 + n_time.tm_mday)
  fold_time = str(n_time.tm_hour * 100 + n_time.tm_min)
  model.save('model/' + FLAGS.breed + '/' + fold_name + 'saved_model' + fold_day + fold_time + '/my_model') 

  converter = tf.lite.TFLiteConverter.from_saved_model('model/' + FLAGS.breed + '/' + fold_name + 'saved_model' + fold_day + fold_time + '/my_model') # path to the SavedModel directory
  tflite_model = converter.convert()

  # Save the model.
  with open('model/' + FLAGS.breed + '/' + fold_name + 'saved_model' + fold_day + fold_time + '/model.tflite', 'wb') as f:
    f.write(tflite_model)

if __name__ == '__main__':
  app.run(main)