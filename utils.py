import cv2
import numpy as np
import h5py
import math
import glob
import os 

def input_setup(config, is_train=True):
    """
        Read image files and make their sub-images and saved them as a h5 file format
    """

    input_data, label_data = prepare_data(config, is_train)
    make_sub_data(input_data, label_data, config, is_train) 

def prepare_data(config, is_train=True):

    if is_train:
        train_input_data_dir = os.path.join(os.path.join(os.getcwd(), "Train"), config.train_input_set)
        train_input_data = glob.glob(os.path.join(train_input_data_dir, "*.jpg"))

        train_label_data_dir = os.path.join(os.path.join(os.getcwd(), "Train"), config.train_label_set)
        train_label_data = glob.glob(os.path.join(train_label_data_dir, "*.pgm"))

        return train_input_data, train_label_data

    else:
        test_input_data_dir = os.path.join(os.path.join(os.getcwd(), "Test"), config.test_input_set)
        test_input_data = glob.glob(os.path.join(test_input_data_dir, "*.jpg"))

        test_label_data_dir = os.path.join(os.path.join(os.getcwd(), "Test"), config.test_label_set)
        test_label_data = glob.glob(os.path.join(test_label_data_dir, "*.pgm"))
        return test_input_data, test_label_data

def make_sub_data(input_data, label_data, config, is_train=True):

    #times = 0
    for i in range(len(input_data)):
        input_, label_ = preprocess(input_data[i], label_data[i])

        input_ = input_.reshape([config.image_size, config.image_size, config.c_dim])
        label_ = label_.reshape([config.image_size, config.image_size, config.c_dim])

        save_flag = make_data_hf(input_, label_, config, i, is_train)

        if not save_flag:
            return False
        #times += 1

    return True


def preprocess(input_data_path, label_data_path):

    input_img = imread(input_data_path)
    label_img = imread(label_data_path)

    input_img = (input_img / 255.0) * 2.0 - 1.0
    label_img = (label_img / 255.0) * 2.0 - 1.0

    input_ = input_img[:, :, ::-1]
    label_ = label_img[:, :, ::-1]

    return input_, label_

def imread(path):
    img = cv2.imread(path)
    return img[:, :, 0]

def make_data_hf(input_, label_, config, times, is_train=True):
    if not os.path.isdir(os.path.join(os.getcwd(), config.checkpoint_dir)):
        os.makedirs(os.path.join(os.getcwd(), config.checkpoint_dir))

    if is_train:
        savepath = os.path.join(os.getcwd(), config.checkpoint_dir + '/train.h5')
    else:
        savepath = os.path.join(os.getcwd(), config.checkpoint_dir + '/test.h5')

    if times == 0:
        if os.path.exists(savepath):
            print("\n%s have existed!\n" % (savepath))
            return False
        else:
            hf = h5py.File(savepath, 'w')

            if is_train:
                input_h5 = hf.create_dataset("input", (1, config.image_size, config.image_size, config.c_dim), 
                                            maxshape=(None, config.image_size, config.image_size, config.c_dim), 
                                            chunks=(1, config.image_size, config.image_size, config.c_dim), dtype='float32')

                label_h5 = hf.create_dataset("label", (1, config.image_size, config.image_size, config.c_dim), 
                                            maxshape=(None, config.image_size, config.image_size, config.c_dim), 
                                            chunks=(1, config.image_size, config.image_size, config.c_dim), dtype='float32')
            else:
                input_h5 = hf.create_dataset("input", (1, input_.shape[0], input_.shape[1], input_.shape[2]), 
                                            maxshape=(None, input_.shape[0], input_.shape[1], input_.shape[2]), 
                                            chunks=(1, input_.shape[0], input_.shape[1], input_.shape[2]), dtype='float32')

                label_h5 = hf.create_dataset("label", (1, label_.shape[0], label_.shape[1], label_.shape[2]), 
                                            maxshape=(None, label_.shape[0], label_.shape[1], label_.shape[2]), 
                                            chunks=(1, label_.shape[0], label_.shape[1], label_.shape[2]), dtype='float32')
    else:
        hf = h5py.File(savepath, 'a')
        input_h5 = hf["input"]
        label_h5 = hf["label"]

    if is_train:
        input_h5.resize([times + 1, config.image_size, config.image_size, config.c_dim])
        input_h5[times : times+1] = input_

        label_h5.resize([times + 1, config.image_size, config.image_size, config.c_dim])
        label_h5[times : times+1] = label_
    else:
        input_h5.resize([times + 1, input_.shape[0], input_.shape[1], input_.shape[2]])
        input_h5[times : times+1] = input_

        label_h5.resize([times + 1, label_.shape[0], label_.shape[1], label_.shape[2]])
        label_h5[times : times+1] = label_

    hf.close()
    return True

def get_data_dir(checkpoint_dir, is_train=True):
    if is_train:
        # return os.path.join('./{}'.format(checkpoint_dir), "train.h5")
        return "/HDD2/wz/lym/try1/checkpoint/train.h5"
    else:
        # return os.path.join('./{}'.format(checkpoint_dir), "test.h5")
        return "/HDD2/wz/lym/try1/checkpoint/test.h5"

def get_data_num(path):
     with h5py.File(path, 'r') as hf:
        input_ = hf['input']
        return input_.shape[0]

def get_batch(path, data_num, batch_size):
    with h5py.File(path, 'r') as hf:
        input_ = hf['input']
        label_ = hf['label']

        random_batch = np.random.rand(batch_size) * (data_num - 1)
        batch_images = np.zeros([batch_size, input_[0].shape[0], input_[0].shape[1], input_[0].shape[2]])
        batch_labels = np.zeros([batch_size, label_[0].shape[0], label_[0].shape[1], label_[0].shape[2]])
        for i in range(batch_size):
            batch_images[i, :, :, :] = np.asarray(input_[int(random_batch[i])])
            batch_labels[i, :, :, :] = np.asarray(label_[int(random_batch[i])])

        return batch_images, batch_labels

def get_batch_val(path, data_num, batch_size, j):
    with h5py.File(path, 'r') as hf:
        input_ = hf['input']
        label_ = hf['label']

        #random_batch = np.random.rand(batch_size) * (data_num - 1)
        batch_images = np.zeros([batch_size, input_[0].shape[0], input_[0].shape[1], input_[0].shape[2]])
        batch_labels = np.zeros([batch_size, label_[0].shape[0], label_[0].shape[1], label_[0].shape[2]])
        for i in range(batch_size):
            batch_images[i, :, :, :] = np.asarray(input_[j + i])
            batch_labels[i, :, :, :] = np.asarray(label_[j + i])

        return batch_images, batch_labels
