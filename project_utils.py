# import sys
# import numpy as np
# import tensorflow as tf
import skimage.data
import os
from tensorflow.python.client import device_lib
import shutil


def getDims():
    height = 64
    width = 64
    return height, width


def get_available_gpus():
    local_device_protos = device_lib.list_local_devices()
    # return [x.name for x in local_device_protos if x.device_type == 'GPU']
    return local_device_protos


# v0
def label_file(labels_loc):
    labels = []
    with open(labels_loc, 'r') as f:
        for val in f.read().split():
            labels.append(int(val))
    return labels


# v0
def load_data(data_directory):
    directories = [d for d in os.listdir(data_directory)
                   if os.path.isdir(os.path.join(data_directory, d))]
    labels = []
    images = []
    print("Loading ", directories, " set")
    for d in directories:
        label_directory = os.path.join(data_directory, d)
        file_names = [os.path.join(label_directory, f)
                      for f in os.listdir(label_directory)
                      if (f.endswith(".jpg"))]
        for f in file_names:
            images.append(skimage.data.imread(f))

        labels_loc = [os.path.join(label_directory, f)
                      for f in os.listdir(label_directory)
                      if f.endswith(".txt")]
        for l in labels_loc:
            labels += (label_file(l))
    if len(images) == len(labels):
        print(directories, " set has ", len(images), " images and ", len(labels), " labels ", u'\u2713',"\n")
    else:
        print(directories, " set has ", len(images), " images and ", len(labels), " labels ", u'\u2717', "\n")
    return images, labels


def get_batch(batch_size, batch_num, max_batches, images, labels):
    print("Aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa")
    if batch_num * batch_size > len(images):
        print("Bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb")
        x_batch = images[max_batches * batch_size:]
        y_batch = labels[max_batches * batch_size:]
    else:
        print("Ccccccccccccccccccccccccccccccccccc")
        x_batch = images[batch_num * batch_size:((batch_num + 1) * batch_size) - 1]
        y_batch = labels[batch_num * batch_size:((batch_num + 1) * batch_size) - 1]
    return x_batch, y_batch


def empty_cnn_model_dir(folder):
    # for the_file in os.listdir(folder):
    #     file_path = os.path.join(folder, the_file)
    #     try:
    #         if os.path.isfile(file_path):
    #             os.unlink(file_path)
    #         # elif os.path.isdir(file_path): shutil.rmtree(file_path)
    #     except Exception as e:
    #         print(e)
    shutil.rmtree(folder)


# def main(args):
#     dev = get_available_gpus()
#     print(dev)
#
#
# if __name__ == "__main__":
#     tf.app.run()