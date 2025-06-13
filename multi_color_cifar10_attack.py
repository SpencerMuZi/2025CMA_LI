import numpy as np
from tensorflow import keras
from test_process import *
from load_data import *
from tqdm import tqdm
import copy
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from PIL import Image

# set the scale factor and zoom ratio
scale_factor = 0.1
zoom_ratio = 255

def rbg_to_grayscale(images):
    return np.dot(images[..., :3], [0.299, 0.587, 0.114])

def calculate_multi_color_mape(model1, model2, x_train, z_train_group, stolen_num):
    mape = 0
    for index in range(stolen_num):
        z_train = z_train_group[index]
        z_out1 = model1(z_train, training=False)
        z_out1 = tf.reshape(z_out1, [-1, 512])
        z_out = model2(z_out1, training=False)
        data = x_train[index : index  + 1]

        num = 114
        stolen_data = []
        for k in range(3):
            tmp_data = []
            for n in range(k * num, (k + 1) * num):
                tmp_data = tf.concat([tmp_data, tf.nn.softmax(z_out[n])[:-1]], -1)

            tmp_data = tf.convert_to_tensor(tmp_data[:-2], dtype=tf.float32)
            stolen_data = tf.concat([stolen_data, copy.deepcopy(tmp_data)], 0)

        data = scale_factor * data
        mape += np.mean(np.abs(stolen_data - tf.reshape(data, [-1])) / scale_factor) * zoom_ratio
    return mape / stolen_num


def show_multi_color_data(model1, model2, x_train, z_train_group, stolen_num):

    data_group = []
    for z_train in z_train_group:
        z_out1 = model1(z_train, training=False)
        z_out1 = tf.reshape(z_out1, [-1, 512])
        z_out = model2(z_out1, training=False)

        num = 114
        stolen_data = []
        for k in range(3):
            tmp_data = []
            for n in range(k * num, (k + 1) * num):
                tmp_data = tf.concat([tmp_data, tf.nn.softmax(z_out[n])[:-1]], -1)

            tmp_data = tf.convert_to_tensor(tmp_data[:-2], dtype=tf.float32)
            stolen_data = tf.concat([stolen_data, copy.deepcopy(tmp_data)], 0)

        data = stolen_data / scale_factor
        print(data)
        data_group.append(data)
    return data_group


def multi_color_cifar10_attack_train(conv_net, fc_net, optimizer, gama, total_epoch, stolen_num):
    conv_net.build(input_shape=[None, 32, 32, 3])
    fc_net.build(input_shape=[None, 512])
    x_train, y_train, x_test, y_test = load_cifar10()

    datagen = ImageDataGenerator(
        featurewise_center=False,  # 布尔值。将输入数据的均值设置为 0，逐特征进行。
        samplewise_center=False,  # 布尔值。将每个样本的均值设置为 0。
        featurewise_std_normalization=False,  # 布尔值。将输入除以数据标准差，逐特征进行。
        samplewise_std_normalization=False,  # 布尔值。将每个输入除以其标准差。
        zca_whitening=False,  # 布尔值。是否应用 ZCA 白化。
        # zca_epsilon  ZCA 白化的 epsilon 值，默认为 1e-6。
        rotation_range=15,  # 整数。随机旋转的度数范围 (degrees, 0 to 180)
        width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
        horizontal_flip=True,  # 布尔值。随机水平翻转。
        vertical_flip=False)  # 布尔值。随机垂直翻转。

    train_x = copy.deepcopy(x_train)
    train_y = copy.deepcopy(y_train)
    datagen.fit(train_x)

    # 对数据进行处理
    train_db = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    train_db = train_db.shuffle(10000).map(preprocess_cifar10).batch(128)

    test_db = tf.data.Dataset.from_tensor_slices((x_test, y_test))
    test_db = test_db.map(preprocess_cifar10).batch(128)

    # 将训练集图片转成tensor并展平
    x_train_process = tf.convert_to_tensor(x_train, dtype=tf.float32) / zoom_ratio
    x_train_process = tf.reshape(x_train_process, [-1, 1024, 3])

    np.random.seed(66)  # 设置种子
    num = 114

    z_group = []
    for i in range(stolen_num):
        z = np.zeros((3 * num, 3072))
        # rand_label = np.zeros((87),int)
        for i in range(0, 3 * num):
            z[i] = np.random.rand(3072)
        z_group.append(copy.deepcopy(z))


    z_train_group = []
    for z in z_group:
        z = z.reshape(3 * num, 32, 32, 3)
        z_train = tf.Variable(z, dtype=tf.float32)
        z_train_group.append(copy.deepcopy(z_train))

    loss_list = []
    acc_list = []
    mape_list = []

    for index in range(stolen_num):
        data = x_train_process[index : index + 1].numpy()
        data = data.reshape(32, 32, 3) * zoom_ratio
        # 将NumPy数组转换为图像
        image_array = np.uint8(data)
        image = Image.fromarray(image_array)
        image.convert('P').save(f'./multi_color_cifar10_pic/原始图像{index}.png')

    for epoch in range(total_epoch):
        generator = datagen.flow(train_x, train_y, batch_size=128)
        num_samples = 30000
        augmented_images = copy.deepcopy(train_x)
        augmented_labels = copy.deepcopy(train_y)

        for _ in range(num_samples // 128):
            batch_images, batch_labels = next(generator)
            augmented_images = np.concatenate((augmented_images, batch_images), axis=0)
            augmented_labels = np.concatenate((augmented_labels, batch_labels), axis=0)
        # 对数据进行处理
        train_db = tf.data.Dataset.from_tensor_slices((augmented_images, augmented_labels))
        train_db = train_db.shuffle(10000).map(preprocess_cifar10).batch(128)
        with tqdm(total=len(train_db), desc=f'Epoch {epoch + 1}/{total_epoch}', unit='batch') as pbar:
            loss = tf.constant(0, dtype=tf.float32)
            for step, (x_batch, y_batch) in enumerate(train_db):
                with tf.GradientTape() as tape:
                    out1 = conv_net(x_batch, training=True)
                    out1 = tf.reshape(out1, [-1, 512])
                    # [b, 512] => [b, 10]
                    out = fc_net(out1, training=True)

                    # 计算恶意正则项
                    regular = 0
                    for index in range(stolen_num):
                        z_train = z_train_group[index]

                        with tf.GradientTape(persistent=True) as tp:

                            z_out1 = conv_net(z_train, training=True)
                            z_out1 = tf.reshape(z_out1, [-1, 512])
                            z_out = fc_net(z_out1, training=True)

                        data = x_train_process[index : index + 1]
                        stolen_data = []
                        for k in range(3):
                            tmp_data = []
                            for n in range(k * num, (k + 1) * num):
                                # stolen_data=tf.concat([stolen_data,z_out[num][0][:rand_label[num]]],-1)
                                # if rand_label[num]!=9:
                                tmp_data = tf.concat([tmp_data, tf.nn.softmax(z_out[n])[:-1]], -1)
                            tmp_data = tf.convert_to_tensor(tmp_data[:-2], dtype=tf.float32)
                            stolen_data = tf.concat([stolen_data, copy.deepcopy(tmp_data)], 0)

                        data = scale_factor * data
                        # print(data.shape)
                        # print(stolen_data.shape)
                        regular += tf.reduce_mean(tf.abs(tf.reshape(stolen_data, -1) - tf.reshape(data, [-1])))

                    loss_batch = tf.reduce_mean(
                        keras.losses.categorical_crossentropy(y_batch, out, from_logits=True)) + (gama / stolen_num) * regular
                # 列表合并，合并2个自网络的参数
                variables = conv_net.trainable_variables + fc_net.trainable_variables
                # 对所有参数求梯度
                grads = tape.gradient(loss_batch, variables)
                # 自动更新
                optimizer.apply_gradients(zip(grads, variables))
                loss += loss_batch
                pbar.update(1)

        mape = calculate_multi_color_mape(conv_net, fc_net, x_train_process, z_train_group, stolen_num)
        mape_list.append(mape)
        print('mape', mape)
        acc_train = cifar10_cnn_test(conv_net, fc_net, train_db, 'train_db')
        acc_test = cifar10_cnn_test(conv_net, fc_net, test_db, 'test_db')
        acc_list.append(float(acc_test))
        loss_list.append(loss)
        print('epoch:', epoch, 'loss:', float(loss) * 128 / 50000, 'Evaluate Acc_train:', float(acc_train),
              'Evaluate Acc_test', float(
                acc_test), 'mape:', mape)

        if epoch % 10 == 0:
            data_name = 'multi_color_cifar10_result//multi_color_cifar10_result' + '_stolen_data_' + str(
                total_epoch) + '_' + str(stolen_num)
            data_group = show_multi_color_data(conv_net, fc_net, x_train_process, z_train_group, stolen_num)
            for index in range(stolen_num):
                data = data_group[index]
                np.save(data_name, data)

                data = np.load(data_name + '.npy')
                data = data.reshape(32, 32, 3) * zoom_ratio
                # 像素值在[0, 255]范围内
                image_array = np.uint8(data)

                # 将NumPy数组转换为图像
                image = Image.fromarray(image_array)
                image.convert('P').save(f'./multi_color_cifar10_pic/窃取图像{index}_epoch_{epoch}_stolen_num_{stolen_num}.png')


    data_group= show_multi_color_data(conv_net, fc_net, x_train_process, z_train_group,stolen_num)
    test_accuracy_name = 'multi_color_cifar10_result//multi_color_cifar10_result' + '_test_accuracy_' + str(
        gama) + '_' + str(total_epoch) + '_' + str(stolen_num)
    mape_name = 'multi_color_cifar10_result//multi_color_cifar10_result' + '_mape_' + str(
        gama) + '_' + str(total_epoch) + '_' + str(stolen_num)
    data_name = 'multi_color_cifar10_result//multi_color_cifar10_result' + '_stolen_data_' + str(
        gama) + '_' + str(total_epoch) + '_' + str(stolen_num)
    loss_name = 'multi_color_cifar10_result//multi_color_cifar10_result' + '_loss_' + str(
        gama) + '_' + str(total_epoch) + '_' + str(stolen_num)
    for index in range(stolen_num):
        data = data_group[index]
        np.save(test_accuracy_name, np.array(acc_list))
        np.save(mape_name, np.array(mape_list))
        np.save(data_name, data)
        np.save(loss_name, np.array(loss_list))

        data = np.load(data_name + '.npy')
        data = data.reshape(32, 32, 3) * zoom_ratio
        # 像素值在[0, 255]范围内
        image_array = np.uint8(data)

        # 将NumPy数组转换为图像
        image = Image.fromarray(image_array)
        image.convert('P').save(f'./multi_color_cifar10_pic/窃取图像{index}_stolen_num_{stolen_num}.png')

