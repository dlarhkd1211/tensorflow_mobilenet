import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpus[0], True)

import os
import re
from PIL import Image
from sklearn.model_selection import train_test_split
import cv2

from absl.flags import FLAGS
from absl import app, flags

import utils.data as data


flags.DEFINE_string('ori_dir', default='/media/localley/D/pet', help = 'path of pet folder')
flags.DEFINE_string('breed', default = 'dog', help = 'dog or cat')

def main(_argv):
  IMG_SIZE = 224

  cur_dir = os.getcwd()
  data_dir = os.path.join(FLAGS.ori_dir, 'original',FLAGS.breed,'classification')
  image_files = [fname for fname in os.listdir(data_dir) if os.path.splitext(fname)[-1] == '.jpg']
  train_files, test_files = train_test_split(image_files, test_size = 0.25, random_state = 123)

  class_list = set()
  for image_file in image_files:
      file_name = os.path.splitext(image_file)[0]
      class_name = re.sub('_\d+', '', file_name)
      class_list.add(class_name)
  class_list = list(class_list)
  class_list.sort()

  with open('data/' + FLAGS.breed + '/class_list.txt', 'w') as f:
    for i in class_list:
      f.write(i)
      f.write('\n')


  class2idx = {cls:idx for idx, cls in enumerate(class_list)}

  tfr_dir = os.path.join(cur_dir, 'data')
  os.makedirs(tfr_dir, exist_ok=True)
  tfr_train_dir = os.path.join(tfr_dir, FLAGS.breed,'cls_train.tfr')
  tfr_test_dir = os.path.join(tfr_dir, FLAGS.breed, 'cls_test.tfr')

  writer_train = tf.io.TFRecordWriter(tfr_train_dir)
  writer_test = tf.io.TFRecordWriter(tfr_test_dir)


  n_train = 0
  for train_file in train_files:
    train_path = os.path.join(data_dir, train_file)
    image = Image.open(train_path)
    image = image.resize((IMG_SIZE, IMG_SIZE))
    bimage = image.tobytes()

    file_name = os.path.splitext(train_file)[0]
    class_name = re.sub('_\d+', '', file_name)
    class_num = class2idx[class_name]

    example = tf.train.Example(features=tf.train.Features(feature={
        'image': data._bytes_feature(bimage),
        'cls_num': data._int64_feature(class_num)
    }))
    writer_train.write(example.SerializeToString())
    n_train += 1

  writer_train.close()

  n_test = 0
  for test_file in test_files:
    test_path = os.path.join(data_dir, test_file)
    image = Image.open(test_path)
    image = image.resize((IMG_SIZE, IMG_SIZE))
    bimage = image.tobytes()

    file_name = os.path.splitext(test_file)[0]
    class_name = re.sub('_\d+', '', file_name)
    class_num = class2idx[class_name]

    example = tf.train.Example(features=tf.train.Features(feature={
        'image': data._bytes_feature(bimage),
        'cls_num': data._int64_feature(class_num)
    }))
    writer_test.write(example.SerializeToString())
    n_test += 1

  writer_test.close()

  with open('data/' + FLAGS.breed + '/num.txt', 'w') as f:
    f.write(str(n_train))
    f.write('\n')
    f.write(str(n_test))
    f.write('\n')
    f.write(str(len(class_list)))


if __name__ == '__main__':
  app.run(main)