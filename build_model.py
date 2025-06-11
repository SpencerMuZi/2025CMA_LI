from tensorflow.keras.layers import Convolution2D
from tensorflow.keras import layers, Sequential, optimizers
import tensorflow as tf
from tensorflow.python.keras.layers import MaxPooling2D, Flatten, Dense
from tensorflow.keras import regularizers,models

def build_mnist_fnn_model():
    model = Sequential([
        layers.Dense(400, activation=tf.nn.relu),
        layers.Dense(200, activation=tf.nn.relu),
        layers.Dense(100, activation=tf.nn.relu),
        layers.Dense(10, activation=tf.nn.softmax),
    ])
    optimizer = optimizers.RMSprop(learning_rate=0.001, rho=0.9, epsilon=1e-08)
    # optimizer = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    return model, optimizer

def build_mnist_fnn_model_en():
    model = Sequential([
        layers.Dense(400, activation=tf.nn.relu),
        layers.Dense(200, activation=tf.nn.relu),
        layers.Dense(100, activation=tf.nn.relu),
        layers.Dense(10, activation=tf.nn.softmax),
    ])
    optimizer = optimizers.RMSprop(learning_rate=0.001, rho=0.9, epsilon=1e-08)
    # optimizer = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    return model, optimizer

def build_vgg13_model(_lambda):
    conv_layers = [
        # Conv-Conv-Pooling 单元 1
        # 64个3*3卷积核，输入输出同大小
        layers.Conv2D(64, kernel_size=[3, 3], padding="same"),
        # layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.Conv2D(64, kernel_size=[3, 3], padding="same"),
        # layers.BatchNormalization(),
        layers.Activation('relu'),
        # 高宽减半
        layers.MaxPool2D(pool_size=[2, 2], strides=2, padding='same'),

        # Conv-Conv_Pooling 单元2，输出通道提成至128，高宽减半
        layers.Conv2D(128, kernel_size=[3, 3], padding="same"),
        # layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.Conv2D(128, kernel_size=[3, 3], padding="same"),
        # layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.MaxPool2D(pool_size=[2, 2], strides=2, padding='same'),

        # Conv-Conv-Pooling 单元 3,输出通道提升至 256，高宽大小减半
        layers.Conv2D(256, kernel_size=[3, 3], padding="same"),
        # layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.Conv2D(256, kernel_size=[3, 3], padding="same"),
        # layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.MaxPool2D(pool_size=[2, 2], strides=2, padding='same'),

        # Conv-Conv-Pooling 单元 4,输出通道提升至 512，高宽大小减半
        layers.Conv2D(512, kernel_size=[3, 3], padding="same"),
        # layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.Conv2D(512, kernel_size=[3, 3], padding="same"),
        # layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.MaxPool2D(pool_size=[2, 2], strides=2, padding='same'),

        # Conv-Conv-Pooling 单元 5,输出通道提升至 512，高宽大小减半
        layers.Conv2D(512, kernel_size=[3, 3], padding="same"),
        # layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.Conv2D(512, kernel_size=[3, 3], padding="same"),
        # layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.MaxPool2D(pool_size=[2, 2], strides=2, padding='same'),

    ]

    conv_net = models.Sequential(conv_layers)

    fc_net = models.Sequential([
        layers.Dense(256, activation=tf.nn.relu),
        layers.Dense(128, activation=tf.nn.relu),
        layers.Dense(10, activation=None),
    ])

    # optimizer = optimizers.RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)
    optimizer = optimizers.Adam(learning_rate=_lambda)
    return conv_net, fc_net, optimizer

