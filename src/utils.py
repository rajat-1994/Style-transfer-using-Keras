import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.applications import vgg19
# MEAN_PIXEL = np.array([ 123.68 ,  116.779,  103.939])


def preprocess_image(path, size):
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, size)
    img = np.expand_dims(img, axis=0)
    img = vgg19.preprocess_input(img)

    return tf.convert_to_tensor(img)


def normalize_image(img):
    """Normalize a RGB image"""
    img[:, :, :, 0] -= 123.68
    img[:, :, :, 1] -= 116.779
    img[:, :, :, 2] -= 103.939
    return img


def denormalize_image(img, size):
    """Denormalize a RGB image"""
    x = img.reshape(size)
    x[:, :, 0] += 103.939
    x[:, :, 1] += 116.779
    x[:, :, 2] += 123.68

    # x = x[:, :, ::-1]
    x = np.clip(x, 0, 255).astype("uint8")
    return x


def show_image(img):
    plt.imshow(img)
