import copy
import numpy as np
from tensorflow import keras
from test_process import *
from load_data import *
from tqdm import tqdm
from PIL import Image

# set the scale factor and zoom ratio
scale_factor = 0.05
zoom_ratio = 255

def calculate_multi_mape(model, x_train, z_train_group, stolen_num):
    mape = 0
    for index in range(stolen_num):
        z_train = z_train_group[index]
        z_out = model(z_train, training=False)
        # 28 * 28 = 784
        data = x_train[index * 784 : index * 784 + 784]

        stolen_data = []
        for num in range(0, 88):
            stolen_data = tf.concat([stolen_data, z_out[num][0][:-1]], -1)

        stolen_data = stolen_data[:784]

        data = scale_factor * data
        mape += np.mean(np.abs(stolen_data - data) / scale_factor) * zoom_ratio

    return mape / stolen_num

def show_multi_data(model, x_train, z_train_group, stolen_num):
    data_group = []
    for z_train in z_train_group:
        z_out = model(z_train, training=False)

        stolen_data = []
        for num in range(0, 88):
            stolen_data = tf.concat([stolen_data, z_out[num][0][:-1]], -1)
        stolen_data = stolen_data[:784]

        data = stolen_data / scale_factor
        print(data)
        data_group.append(data)

    return data_group

# 执行自定义训练过程
def multi_fashion_mnist_attack_train(model, optimizer, gama, total_epoch, stolen_num):
    # 加载fashion_minst数据集
    (x_train, y_train), (x_test, y_test) = datasets.fashion_mnist.load_data() # fashion_mnist.load_data()/ mnist.load_data()/ cifar10.load_data()
    # tf.data.Dataset将numpy数组或tensor转换为dataset数据集，
    # from_tensor_slices将输入的张量或数组逐行切片，创建一个Dataset，
    # 其中每个元素都是(x_train[i], y_train[i]) 这样的形式，也就是每个元素是一个图片和它对应的标签。
    train_db = tf.data.Dataset.from_tensor_slices((x_train.astype(np.float32), y_train.astype(np.float32)))
    # train_db.shuffle(buffer_size=10000)使用一个大小为 10,000 的缓冲区进行打乱操作。
    # 每次从缓冲区中随机抽取元素，并将新的数据项放入缓冲区，直到数据全部被处理。
    # 使用preprocess_mnist处理图片，一批为128个。
    train_db = train_db.shuffle(10000).map(preprocess_mnist).batch(128)
    # 生成随机的z
    np.random.seed(66)  # 设置种子
    z_group = []
    for i in range(stolen_num):
        z = np.zeros((88,784))
        for i in range(0,88):
            z[i] = np.random.rand(784)
        z_group.append(copy.deepcopy(z))


    #以同样的方法加载测试集
    test_db = tf.data.Dataset.from_tensor_slices((x_test.astype(np.float64), y_test.astype(np.float64)))
    test_db = test_db.map(preprocess_mnist).batch(128)
    # 数据缩放至0-1，同时将每张图片展成一维向量
    x_train_process = tf.convert_to_tensor(x_train.flatten(), dtype=tf.float32) / zoom_ratio
    x_test_process = tf.convert_to_tensor(x_test.flatten(), dtype=tf.float32) / zoom_ratio
    # 初始化模型
    # 输入的形状为 [128, 784]，即每次输入 128 张图片，每张图片展平成 784（28x28） 维的向量。
    model.build(input_shape=[128, 1, 784])

    loss_list = []
    acc_list = []
    mape_list = []

    z_train_group = []
    for z in z_group:
        z = z.reshape(88,1,784)
        z_train = tf.Variable(z, dtype=tf.float32)
        z_train_group.append(copy.deepcopy(z_train))

    for index in range(stolen_num):
        data = x_train_process[index*784:index*784+784].numpy()
        data = data.reshape(28, 28) * zoom_ratio
        # 将NumPy数组转换为图像
        image_array = np.uint8(data)
        image = Image.fromarray(image_array)
        image.convert('L').save(f'./multi_fashion_minist_pic/原始图像{index}.jpg')

    # 执行训练过程
    for epoch in range(total_epoch):
        loss_print = 0
        with tqdm(total=len(train_db), desc=f'Epoch {epoch+1}/{total_epoch}', unit='batch') as pbar:
            for step, (x, y) in enumerate(train_db):
                # print(f'Step {step}, x shape: {x.shape}, z shape: {z.shape}')
                # 通过使用 tf.GradientTape，
                # 我们可以在反向传播时计算出损失函数相对于模型参数的梯度。
                with tf.GradientTape(persistent=True) as tape:
                    # 这行代码通过将输入 x 传递给模型 model，执行前向传播，得到模型的输出 out。
                    # training=True表示处于训练模式
                    out = model(x, training=True)
                    # 用于移除张量中长度为 1 的维度。axis=1 表示我们只移除第 1 维度（索引从 0 开始）。
                    out = tf.squeeze(out, axis=1)
                    # 调用 cor_attack 函数，计算模型的相关性正则化项 regular

                    regular = 0
                    for index in range(stolen_num):
                        z_train = z_train_group[index]

                        with tf.GradientTape(persistent=True) as tp:
                            z_out = model(z_train, training=True)

                        data = x_train_process[index * 784:index * 784 + 784]
                        stolen_data = []
                        for num in range(0, 88):
                            stolen_data = tf.concat([stolen_data, z_out[num][0][:-1]], -1)
                        stolen_data = stolen_data[:784]

                        data = scale_factor * data
                        regular += tf.reduce_mean(tf.abs(stolen_data - data))



                    loss = tf.reduce_mean(keras.losses.categorical_crossentropy(y, out, from_logits=False)) + (gama / stolen_num) * regular


                    # 将当前步骤的损失值累加到 loss_print 变量中，用于记录本轮训练中的总损失。
                    loss_print += float(loss)

                # 执行梯度下降
                # 通过 tape.gradient 计算损失函数相对于模型所有可训练参数的梯度。
                grads = tape.gradient(loss, model.trainable_variables)
                # grads 是前面计算得到的梯度列表，表示每个模型参数的梯度。
                # model.trainable_variables 是模型中的可训练参数列表。
                # zip(grads, model.trainable_variables) 将梯度和对应的模型参数配对，形成一个元组列表，
                # 类似于 [(grad1, param1), (grad2, param2), ...]。
                # apply_gradients 的作用是根据这些配对，将每个参数的梯度应用到对应的模型参数上，完成参数的更新。
                optimizer.apply_gradients(zip(grads, model.trainable_variables))

                # 每处理一个 batch，更新进度条
                pbar.update(1)

        # 调用calculate_cor_mape函数计算模型当前权重转化为图像后与训练数据之间的 MAPE 误差
        # mape是预测值与实际值之间的绝对误差
        mape = calculate_multi_mape(model, x_train_process, z_train_group, stolen_num)
        mape_list.append(mape)
        print('mape', mape)
        # 获得对测试集的准确率
        acc = test(model, test_db)
        loss_list.append(loss_print)
        acc_list.append(acc)
        print('epoch:', epoch + 1, 'loss:', loss_print / (50000 / 128), 'Evaluate Acc:', float(acc), 'mape:', mape)
    
        if epoch % 10 == 0:
            data_name = 'multi_fashion_minist_result//multi_fashion_minist_result' + '_stolen_data_' + str(total_epoch) + '_' + str(stolen_num)
            data_group = show_multi_data(model, x_test_process, z_train_group, stolen_num)
            for index in range(stolen_num):
                data = data_group[index]
                np.save(data_name, data)

                data = np.load(data_name + '.npy')
                data = data.reshape(28, 28) * zoom_ratio
                # 像素值在[0, 255]范围内
                image_array = np.uint8(data)

                # 将NumPy数组转换为图像
                image = Image.fromarray(image_array)
                image.convert('L').save(f'./multi_fashion_minist_pic/窃取图像{index}_epoch_{epoch}_stolen_num_{stolen_num}_gama{gama}.jpg')
    
    test_accuracy_name = 'multi_fashion_minist_result//multi_fashion_minist_result' + '_test_accuracy_' + str(
        gama) + '_' + str(total_epoch)  + '_' + str(stolen_num)
    mape_name = 'multi_fashion_minist_result//multi_fashion_minist_result' + '_mape_' + str(
        gama) + '_' + str(total_epoch) + '_' + str(stolen_num)
    data_name = 'multi_fashion_minist_result//multi_fashion_minist_result' + '_stolen_data_' + str(
        gama) + '_' + str(total_epoch) + '_' + str(stolen_num)
    loss_name = 'multi_fashion_minist_result//multi_fashion_minist_result' + '_loss_' + str(
        gama) + '_' + str(total_epoch) + '_' + str(stolen_num)
    
    # 恢复数据
    data_group = show_multi_data(model, x_test_process, z_train_group, stolen_num)
    for index in range(stolen_num):
        data = data_group[index]
        np.save(test_accuracy_name, np.array(acc_list))
        np.save(mape_name, np.array(mape_list))
        np.save(data_name, data)

        data = np.load(data_name + '.npy')
        data = data.reshape(28, 28) * zoom_ratio
        # 像素值在[0, 255]范围内
        image_array = np.uint8(data)

        # 将NumPy数组转换为图像
        image = Image.fromarray(image_array)
        image.convert('L').save(f'./multi_fashion_minist_pic/窃取图像{index}_stolen_num_{stolen_num}_gama_{gama}.jpg')
