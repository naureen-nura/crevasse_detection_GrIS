import os
import numpy as np
from glob import glob
import tensorflow as tf
from tensorflow.keras import layers
import matplotlib.pyplot as plt
AUTOTUNE = tf.data.experimental.AUTOTUNE

IMAGE_SIZE = 1000,1000
channels = 1
num_classes = 3
batch = 1
#LR = 1e-4
#EPOCHS = 5

image_path = "./gris/Images/"
mask_path = "./gris/Labels/"
training_data = "Train/"
val_data = "Valid/"
test_data = "Test/"

def load_data():

  TRAIN_X = sorted(glob(os.path.join(image_path + training_data, "*.png")))
  train_x = TRAIN_X[:]
  TRAIN_Y = sorted(glob(os.path.join(mask_path + training_data, "*.png")))
  train_y = TRAIN_Y[:]

  VALID_X = sorted(glob(os.path.join(image_path + val_data, "*.png")))
  valid_x = VALID_X[:]
  VALID_Y = sorted(glob(os.path.join(mask_path + val_data, "*.png")))
  valid_y = VALID_Y[:]

  TEST_X = sorted(glob(os.path.join(image_path + test_data, "*.png")))
  test_x = TEST_X[:]
  TEST_Y = sorted(glob(os.path.join(mask_path + test_data, "*.png")))
  test_y = TEST_Y[:]

  return (train_x, train_y), (valid_x, valid_y), (test_x, test_y)

(train_x, train_y), (valid_x, valid_y), (test_x, test_y) = load_data() 

data_augmentation = tf.keras.Sequential([
  layers.experimental.preprocessing.RandomFlip("horizontal"),
  layers.experimental.preprocessing.RandomRotation(0.25),
  layers.experimental.preprocessing.RandomContrast(0.1)],
  )


def read_image(path):
    x = tf.io.read_file(path)
    x = tf.image.decode_png(x, channels=0)
    #x = tf.image.resize(x, [500, 500])
    #x = tf.cast(x, dtype=tf.uint8)
    #x = tf.expand_dims(x[:,:,:,0], axis=-1)
    return x

def read_mask(path):
    y = tf.io.read_file(path)
    y = tf.image.decode_png(y, channels=0)
    #y = tf.image.resize(y, [500,500])
    #y = tf.cast(y, dtype=tf.uint8)
    #y = tf.expand_dims(y[:,:,:,0], axis=-1)
    return y
 
 # Function to generate tensorflow dataset 
def tf_parse(x, y):
    def _parse(x, y):
        x = read_image(x)
        y = read_mask(y)
        return x, y

    x, y = tf.numpy_function(_parse, [x, y], [tf.uint8,tf.uint8])
    x = tf.ensure_shape(x, [None, None, channels])
    y = tf.ensure_shape(y, [None, None, channels])
    return x, y

num_threads = 4

def tf_dataset_train(x, y, batch=2):
    dataset = tf.data.Dataset.from_tensor_slices((x, y))
    dataset = dataset.shuffle(buffer_size=4) #Buffer size needs to be greater or equal to the full size of the dataset
    dataset = dataset.repeat()
    dataset = dataset.map(tf_parse, num_parallel_calls=num_threads)
    dataset = dataset.batch(batch)
    dataset = dataset.map(lambda x, y: (data_augmentation(x, training=True), y), num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.prefetch(1)
    return dataset


def tf_dataset_valid(x, y, batch=1):
    dataset = tf.data.Dataset.from_tensor_slices((x, y))
    dataset = dataset.shuffle(buffer_size=1)
    dataset = dataset.repeat()
    dataset = dataset.map(tf_parse, num_parallel_calls=num_threads)
    dataset = dataset.batch(batch)
    dataset = dataset.prefetch(1)
    return dataset

train_dataset = tf_dataset_train(train_x, train_y)
valid_dataset = tf_dataset_valid(valid_x, valid_y)
test_dataset = tf_dataset_valid(test_x, test_y)