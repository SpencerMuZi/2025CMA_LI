import tensorflow as tf
from tensorflow.keras import datasets
from attack import *
import numpy as np


def preprocess_mnist(x_in, y_in):
    x_in = tf.cast(x_in, dtype=tf.float32) / 255
    x_in = tf.reshape(x_in, [-1, 28 * 28])
    y_in = tf.cast(y_in, dtype=tf.int32)
    y_in = tf.one_hot(y_in, depth=10)
    return x_in, y_in


def preprocess_cifar10(x_in, y_in):
    x_in = tf.cast(x_in, dtype=tf.float32) / 255
    y_in = tf.cast(y_in, dtype=tf.int32)
    y_in = tf.one_hot(y_in, depth=10)
    return x_in, y_in


def load_mnist_fnn():
    (x_train, y_train), (x_test, y_test) = datasets.mnist.load_data()
    train_db = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    train_db = train_db.shuffle(10000).map(preprocess_mnist).batch(128)

    test_db = tf.data.Dataset.from_tensor_slices((x_test, y_test))
    test_db = test_db.map(preprocess_mnist).batch(128)


    return train_db, test_db


def load(data_name):
    if data_name == 'cifar10':
        (x_train, y_train), (x_test, y_test) = datasets.cifar10.load_data()
        y_train = tf.squeeze(y_train, axis=1)
        y_test = tf.squeeze(y_test, axis=1)

    elif data_name == 'mnist':
        (x_train, y_train), (x_test, y_test) = datasets.mnist.load_data()
        # 合成恶意数据进行CAP攻击
        mal_x_out, mal_y_out = mal_mnist_fnn_synthesis(x_test, 2, 4)
        # 对合成的恶意数据进行拼接
        x_train = np.vstack((x_train, mal_x_out))
        y_train = np.append(y_train, mal_y_out)

    print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)
    train_db_in = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    train_db_in = train_db_in.shuffle(10000).map(preprocess_cifar10).batch(128)

    test_db = tf.data.Dataset.from_tensor_slices((x_test, y_test))
    test_db = test_db.map(preprocess_cifar10).batch(128)

    # mal_x_out = tf.convert_to_tensor(mal_x_out, dtype=tf.float32) / 255

    # return train_db_in, test_db, mal_x_out
    return train_db_in, test_db


def load_cifar10():
    (x_train, y_train), (x_test, y_test) = datasets.cifar10.load_data()
    y_train = tf.squeeze(y_train, axis=1)
    y_test = tf.squeeze(y_test, axis=1)
    return x_train, y_train, x_test, y_test


if __name__ == '__main__':
    load('cifar10')
