import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpus[0], True)
from tensorflow.keras import models

from absl import flags, app
from absl.flags import FLAGS

from glob import glob

import cv2
import numpy as np
from PIL import Image
import re


flags.DEFINE_string('breed', default = 'dog', help = 'dog or cat')
flags.DEFINE_string('model', default = '', help = 'model name')

def main(_argv):
    model = models.load_model('./model/'+FLAGS.breed + '/'+ FLAGS.model + '/my_model')
    path = glob('./test_data/'+FLAGS.breed + '/*.jpg')
    path.sort()
    class_list = []
    with open('./data/'+FLAGS.breed + '/class_list.txt') as f:
        lines = f.readlines()
        for line in lines:
            class_list.append(line.replace('\n',''))
    cor = 0
    tot = 0
    for i in path:
        file_name = i.split('/')[-1]
        breed = re.sub('_\d+.jpg', '', file_name)
        image = Image.open(i)
        image = image.resize((224, 224))
        image = np.array(image)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

        image = np.reshape(image, (1, 224, 224, 3))
        prediction = model.predict(image)
        pred_class = np.argmax(prediction, axis=1)
        pred_breed = class_list[int(pred_class)]
        print(breed, pred_breed)
        if FLAGS.breed == 'dog':
            if breed == pred_breed:
                cor += 1
            elif breed in ['Toy_Poodle', 'minuature_Poodle'] and pred_breed in ['Toy_Poodle', 'minuature_Poodle']:
                cor += 1
            elif breed in ['Italian_Greyhound', 'Whippet'] and pred_breed in ['Italian_Greyhound', 'Whippet']:
                cor += 1
            elif breed in ['Doberman', 'minuature_Pinscher'] and pred_breed in ['Doberman', 'minuature_Pinscher']:
                cor += 1
            elif breed in ['Japanese_Spitz', 'Samoyed', 'Pompitz'] and pred_breed in ['Japanese_Spitz', 'Samoyed',
                                                                                      'Pompitz']:
                cor += 1
            elif breed in ['Coton_de_Tulear', 'Maltese'] and pred_breed in ['Coton_de_Tulear', 'Maltese']:
                cor += 1
        else:
            if breed == pred_breed:
                cor += 1
            elif breed in ['Persian', 'Exotic_Shorthair'] and pred_breed in ['Persian', 'Exotic_Shorthair']:
                cor += 1
        tot += 1
    print(cor, tot)
    print(cor / tot * 100)


if __name__ == '__main__':
  app.run(main)