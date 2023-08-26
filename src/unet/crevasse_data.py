import os
import numpy as np
from glob import glob
import tensorflow as tf
from tensorflow.keras import layers
import matplotlib.pyplot as plt
AUTOTUNE = tf.data.experimental.AUTOTUNE

IMAGE_SIZE = (1024,1024)
channels = 1
num_classes = 3
batch_size = 1
#LR = 1e-4
#EPOCHS = 5

image_path = "./gris/Images/"
mask_path = "./gris/Labels/"
training_data = "Train/"
val_data = "Valid/"
test_data = "Test/"

def load_data():

  TRAIN_X = sorted(glob(os.path.join(image_path + training_data, "*.png")))
  train_x = TRAIN_X[:4]
  TRAIN_Y = sorted(glob(os.path.join(mask_path + training_data, "*.png")))
  train_y = TRAIN_Y[:4]

  VALID_X = sorted(glob(os.path.join(image_path + val_data, "*.png")))
  valid_x = VALID_X[:1]
  VALID_Y = sorted(glob(os.path.join(mask_path + val_data, "*.png")))
  valid_y = VALID_Y[:1]

  TEST_X = sorted(glob(os.path.join(image_path + test_data, "*.png")))
  test_x = TEST_X[:1]
  TEST_Y = sorted(glob(os.path.join(mask_path + test_data, "*.png")))
  test_y = TEST_Y[:1]

  return (train_x, train_y), (valid_x, valid_y), (test_x, test_y)

(train_x, train_y), (valid_x, valid_y), (test_x, test_y) = load_data() 

data_augmentation = tf.keras.Sequential([
  layers.experimental.preprocessing.RandomFlip("horizontal"),
  layers.experimental.preprocessing.RandomRotation(0.25),
  layers.experimental.preprocessing.RandomContrast(0.1)],
  )


def read_image(path):
    x = tf.io.read_file(path)
    x = tf.image.decode_png(x, channels=1)
    x = tf.image.resize(x, IMAGE_SIZE)
    x = tf.cast(x, dtype=tf.uint8)
    #x = tf.expand_dims(x[:,:,:,0], axis=-1)
    return x

def read_mask(path):
    x = tf.io.read_file(path)
    x = tf.image.decode_png(x, channels=1)
    x = tf.image.resize(x, IMAGE_SIZE)
    x = tf.cast(x, dtype=tf.uint8)
    x = tf.expand_dims(x, axis=-1)
    return x
 
 # Function to generate tensorflow dataset 
def tf_parse(x, y):
    def _parse(x, y):
        x = tf.zeros(IMAGE_SIZE + (1,), dtype=tf.uint8)
        for j, path in enumerate(read_image(x)):
            x[j] = read_image(x)
        y = tf.zeros(IMAGE_SIZE + (1,), dtype=tf.uin8)
        for j, path in enumerate(read_mask(y)):
            y[j] = read_mask(y)
        return x, y
    #x = np.zeros(IMAGE_SIZE + (1,), dtype="float32")
    #y = np.zeros(IMAGE_SIZE + (1,), dtype="float32")
    x, y = tf.py_function(_parse, [x, y], [tf.uint8,tf.uint8])
    x = tf.ensure_shape(x, [None, None, 1])
    y = tf.ensure_shape(y, [None, None, 1])
    return x, y

num_threads = 4

def tf_dataset_train(x, y, batch_size=2):
    dataset = tf.data.Dataset.from_tensor_slices((x, y))
    dataset = dataset.shuffle(buffer_size=4) #Buffer size needs to be greater or equal to the full size of the dataset
    #dataset = dataset.repeat()
    dataset = dataset.map(tf_parse, num_parallel_calls=num_threads)
    dataset = dataset.batch(batch_size)
    dataset = dataset.map(lambda x, y: (data_augmentation(x, training=True), y), num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.prefetch(1)
    return dataset


def tf_dataset_valid_test(x, y, batch=1):
    dataset = tf.data.Dataset.from_tensor_slices((x, y))
    dataset = dataset.shuffle(buffer_size=1)
    #dataset = dataset.repeat()
    dataset = dataset.map(tf_parse, num_parallel_calls=num_threads)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(1)
    return dataset

train_dataset = tf_dataset_train(train_x, train_y)
valid_dataset = tf_dataset_valid_test(valid_x, valid_y)
test_dataset = tf_dataset_valid_test(test_x, test_y)