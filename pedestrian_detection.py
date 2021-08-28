from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import random
import skimage.data
import os
import tensorflow as tf
from pedestrian_conv_net.conv_net import pedestrian_cnn_model_fn

EPOCHS = 10
HEIGHT = 480
WIDTH = 640
CLASSES = 1
NUM_CHANNELS = 3  # 3 for RGB, 1 for grayscale
learning_rate = 0.0001


def label_file(labels_loc):
    labels = []
    with open(labels_loc, 'r') as f:
        for val in f.read().split():
            labels.append(int(val))
    return labels


def load_data(data_directory):
    directories = [d for d in os.listdir(data_directory)
                   if os.path.isdir(os.path.join(data_directory, d))]
    labels = []
    images = []
    print(directories)
    for d in directories:
        label_directory = os.path.join(data_directory, d)
        file_names = [os.path.join(label_directory, f)
                      for f in os.listdir(label_directory)
                      if f.endswith(".jpg")]
        for f in file_names:
            images.append(skimage.data.imread(f))

        labels_loc = [os.path.join(label_directory, f)
                      for f in os.listdir(label_directory)
                      if f.endswith(".txt")]
        for l in labels_loc:
            labels += (label_file(l))

    return images, labels


def get_batch(batch_size, batch_num, max_batches, images, labels):
    if batch_num * batch_size > len(images):
        x_batch = images[max_batches * batch_size:]
        y_batch = labels[max_batches * batch_size:]
    else:
        x_batch = images[batch_num * batch_size:((batch_num + 1) * batch_size) - 1]
        y_batch = labels[batch_num * batch_size:((batch_num + 1) * batch_size) - 1]
    return x_batch, y_batch


def initialize_tensors():
    x = tf.placeholder(tf.float32, shape=[None, WIDTH, HEIGHT, CLASSES], name='x')

    y_true = tf.placeholder(tf.float32, shape=[None, CLASSES], name='y_true')
    y_true_cls = tf.argmax(y_true, axis=None, name='t_true_cls', dimension=1)
    return x, y_true, y_true_cls


def predict(logits, y_true):
    with tf.name_scope("Optimization") as scope:
        global_step = tf.Variable(0, name='global_step', trainable=False)
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits,
                                                                labels=y_true)
        cost = tf.reduce_mean(cross_entropy)
        tf.summary.scalar("cost", cost)
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost, global_step=global_step)
        return global_step, cost, optimizer


def get_accuracy(prediction, y_true):
    with tf.name_scope('accuracy') as scope:
        correct_pred = tf.equal(tf.argmax(prediction, 1), tf.argmax(y_true, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
    return accuracy


def main():
    # Load training and eval data
    ROOT_PATH = "/PathOImages"
    train_data_directory = os.path.join(ROOT_PATH, "pedestrian_train_images")
    test_data_directory = os.path.join(ROOT_PATH, "pedestrian_test_images")
    x, y_true, y_true_cls = initialize_tensors()

    # load train images
    images_train, labels_train = load_data(train_data_directory)

    # if we want to shuffle the (image, label) lists use this code
    temp = list(zip(images_train, labels_train))

    random.shuffle(temp)

    images_train, labels_train = zip(*temp)

    # load test images  and labels afterwards
    images_test, labels_test = load_data(test_data_directory)

    # compute number of batches
    batch_num = 0
    test_batch_num = 0
    batch_size = 100
    train_max_batches = int(len(images_train) / batch_size)
    test_max_batches = int(len(images_test) / batch_size)

    # convnet
    logits = pedestrian_cnn_model_fn(x)

    # optimize
    global_step, cost, optimizer = predict(logits, labels_train)
    accuracy = get_accuracy(predict, y_true)

    with tf.Graph().as_default():
        print("fsdfasdfasdfadfd")
        with tf.Session() as sess:
            # Initialize all variables.
            init_op = tf.global_variables_initializer()
            sess.run(init_op)
                        print('Started trainning CNN')
            for epoch in range(EPOCHS):
                _, accuracy_val = sess.run(
                    [optimizer, accuracy], feed_dict={x: images_train, y_true: labels_train})
                accuracy_val = sess.run(
                    [accuracy], feed_dict={x: images_test, y_true: labels_test})
                print(" Accuracy: ", accuracy_val)
                print("\nDONE WITH EPOCH ", epoch)

if __name__ == "__main__":
    main()
