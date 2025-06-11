运行main函数来进行实验
run main.py/ python3 main.py to start

main函数里有#fashion_mnist attack #cifar attack #color cifar attack 三种模式，使用对应注释下的代码进行实验。
there are 3 modes in main.py, please choose 1 to run.

in my_fashion_mnist_fnn_new_double_attack.py, we use data = x_train_process[4704:5488].numpy(), which should be multiplied by 784 such as 1*784:2*784 to store origin data,
if you want to do exercise on other data, just change the range of [4704:5488] to [n*784,(n+1)*784], you should also change them in line 73 line 106 in my_fashion_mnist_fnn_new_double_attack.py, and line 32 in attack.py

in new_cifar10_attack.py, we use data = x_train_process[1024:2048].numpy(), which should be multiplied by 784 such as 1*1024:2*1024 to store origin data,
if you want to do exercise on other data, just change the range of [1024:2048] to [n*1024,(n+1)*1024], you should also change them in line 39 line 69 in new_cifar10_attack.py, and line 60 in attack.py

in color_cifar10_attack.py, we use data = x_train_process[1].numpy(), which store the colorful rgb data,
if you want to do exercise on other data, just change the range of [1] to [n], you should also change them in line 14 line 94 and line 124 in color_cifar10_attack.py

TODO:
model training in cifar attack still need to be improved, it has overfitting in trainning dataset, you could find better model to replace the model in build_model.py/build_vgg13_model
in color cifar attack, we only down one experiment, it might be improved by changing the scaling number of data, such as data * 0.15, for r g b 3 types of pixels has different distributions.

