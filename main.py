from build_model import *
import os
import tensorflow as tf

from multi_fashion_mnist_attack import *
from multi_cifar10_attack import *
from multi_cifar10_noattack import *
from multi_color_cifar10_attack import *

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

if __name__ == '__main__':
    tf.random.set_seed(999)
    # fashion_mnist attack
    model, optimizer = build_mnist_fnn_model()
    multi_fashion_mnist_attack_train(model, optimizer, 100, 50, 30) # gama, total_epoch, stolen_num

    # cifar attack
    # when attack, need 0.001 learning rate;
    # when no attack, need 0.0001 learning rate;
    # conv_net5, fc_net5, optimizer = build_vgg13_model(0.001)

    # multi_cifar10_vgg13_linear_attack_train(conv_net5, fc_net5, optimizer, 100, 50, 1) # gama, total_epoch, stolen_num
    # multi_cifar10_vgg13_linear_noattack_train(conv_net5, fc_net5, optimizer, 100, 50, 1) # gama, total_epoch, stolen_num
    # multi_color_cifar10_attack_train(conv_net5, fc_net5, optimizer, 100, 50, 50) # gama, total_epoch, stolen_num