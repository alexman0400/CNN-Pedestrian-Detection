from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# import pedestrian_conv_net.project_utils as pu
from pedestrian_conv_net.project_utils import *
from pedestrian_conv_net.read_custom_google_set import *
from pedestrian_conv_net.conv_net import *
from sklearn.model_selection import KFold
import time
import tensorflow as tf
from tensorflow.python import debug as tf_debug
import numpy as np
import sys
# import random
# import skimage.data
# import os

# # Setting default values of varriables (epochs, batch_size, split_size, learning_Rate, architecture)

# CLASSES = 3
# NUM_CHANNELS = 3  # 3 for RGB, 1 for grayscale
glob_epochs = 1
# glob_arch = [cnn_model_fn_6]
glob_batch_s = 2
glob_split_s = 10
glob_learning_rate = 0.001
tf.logging.set_verbosity(tf.logging.INFO)
# arch = [cnn_model_fn_6]


def get_learning_rate():
    global glob_learning_rate

    return glob_learning_rate


def main(unused_argv):
    # print(argv[0])

    results = []
    train_time_list = []
    test_time_list = []

    global glob_epochs
    global glob_arch
    global glob_batch_s
    global glob_split_s
    global glob_learning_rate

    epochs = glob_epochs
    arch = glob_arch
    batch_s = glob_batch_s
    split_s = glob_split_s
    learning_rate = glob_learning_rate

    kf = KFold(n_splits=split_s)

    # # Load training and eval data
    train_root_path = "/home/mantsako/Desktop/Google_Images/google_set_all"
    # $ test_root_path = "/home/mantsako/Desktop/Google_Images/pedestrian_subset"
    checkpoint_folder = "/home/mantsako/Desktop/cnn_model"

    # # Empty the folder where the cnn model is saved on
    if os.path.exists(checkpoint_folder):
        empty_cnn_model_dir(checkpoint_folder)
    else:
        os.makedirs(checkpoint_folder)
    print("Deleted folder for models to be saved on")

    # # load train images
    train_ped_images, train_ped_labels, train_sheep_images, train_sheep_labels, train_land_images, train_land_labels = \
        request_input_train(train_root_path)
    # test_ped_images, test_ped_labels, test_sheep_images, test_sheep_labels, test_land_images, test_land_labels = \
    #     request_input_test(test_root_path)
    # test_ped_images, test_ped_labels = request_input_test(test_root_path)

    # # load test images afterwards
    train_images, train_labels = concat_sets(train_ped_images, train_ped_labels, train_sheep_images, train_sheep_labels,
                                             train_land_images, train_land_labels)

    train_im = np.asarray(train_images, dtype=np.float32)
    train_lab = np.asarray(train_labels, dtype=np.float32)

    print("shape: ", train_im.shape)
    # test_im = np.asarray(test_images, dtype=np.float32)
    # test_lab = np.asarray(test_labels, dtype=np.float32)
    # test_im = np.asarray(train_ped_images, dtype=np.float32)
    # test_lab = np.asarray(train_ped_labels, dtype=np.float32)
    # print((np.min(train_im)))
    # print(np.isinf(train_im).any())
    # print(np.isinf(train_lab).any())

    it = 0
    for train_index, test_index in kf.split(train_im):
        it = it + 1
        print("Iteration: ", it)
        # print("\nDeleted the cnn_model folder ", u'\u2713', "\n")

        # print("TRAIN:", train_index, "TEST:", test_index)
        train_set = train_im[train_index]
        train_set_lab = train_lab[train_index]
        test_set = train_im[test_index]
        test_set_lab = train_lab[test_index]
        # print("TRAIN: ", len(train_set), "\nTEST: ", len(test_set))

        # # Set the proper configurations
        config = tf.ConfigProto(log_device_placement=True)
        config.gpu_options.allow_growth = True
        config.operation_timeout_in_ms = 50000  # terminate on long hangs
        config.gpu_options.per_process_gpu_memory_fraction = 0.9
        config.allow_soft_placement = True

        folder_name = checkpoint_folder + "/cnn_model_" + str(it)
        pedestrian_classifier = tf.estimator.Estimator(
            model_fn=arch,
            model_dir=folder_name,
            config=tf.contrib.learn.RunConfig(session_config=config))

        # # Set up logging for predictions
        # # Log the values in the "Softmax" tensor with label "probabilities"
        tensors_to_log = {"probabilities": "softmax_tensor"}
        logging_hook = tf.train.LoggingTensorHook(tensors=tensors_to_log,
                                                  every_n_iter=1)
        hooks = [logging_hook]
        # debug_hook = tf_debug.LocalCLIDebugHook()
        # hooks = [logging_hook, debug_hook]

        # # Create the Estimator
        print("\nEstimator starts training")
        train_start_time = time.time()

        print("Number of train images: ", len(train_im),"\nNumber of train labels: ", len(train_lab))
        train_input_fn = tf.estimator.inputs.numpy_input_fn(
            x={"x": train_set},
            y=train_set_lab,
            batch_size=batch_s,
            num_epochs=epochs,
            shuffle=True)
        pedestrian_classifier.train(
            input_fn=train_input_fn,
            steps=None,
            hooks=hooks)

        train_time = time.time() - train_start_time
        print("\nTime taken to train", train_time)

        test_start_time = time.time()
        print("\nEstimator starts evaluating")
        eval_input_fn = tf.estimator.inputs.numpy_input_fn(
            x={"x": test_set},
            y=test_set_lab,
            batch_size=batch_s,
            num_epochs=1,
            shuffle=True)
        eval_results = pedestrian_classifier.evaluate(
            input_fn=eval_input_fn,
            hooks=hooks,
            steps=None,
            name="Evaluate")

        test_time = time.time() - test_start_time

        print('\nTest set accuracy: {accuracy:0.3f}\n'.format(**eval_results))
        print(eval_results.get('accuracy'))
        train_time_list.append(train_time)
        test_time_list.append(test_time)
        results.append(eval_results.get('accuracy'))

    # write times of training and testing
    train_test_times_file.write("\n\n\t\t\tTraining times: " + str(train_time_list))
    train_test_times_file.write("\n\n\t\t\tTest times: " + str(test_time_list))

    max_acc = max(results)
    # print("Accuracy for each of the 10 models\n", results)
    # print("Best model has ", str(max_acc), " accuracy")
    if file is not None:
        file.write("\t\t\t\tAccuracy for each of the " + str(split_s) + " models\n" + str(results) + "\n")
        file.write("\t\t\t\tBest model has " + str(max_acc) + " accuracy\n" )

    it = 0
    for res in results:
        it = it + 1
        if res != max_acc:
            empty_cnn_model_dir(checkpoint_folder + "/cnn_model_" + str(it))

    print("Done deleting the unsuccessful models")


if __name__ == "__main__":

    file = open(file="test results.txt", mode="w", encoding="utf-8")
    error_file = open(file="errors.txt", mode="w", encoding="utf-8")
    train_test_times_file = open(file="train-test-times.txt", mode="w", encoding="utf-8")
    architectures_list = [cnn_model_fn_1, cnn_model_fn_2, cnn_model_fn_3, cnn_model_fn_5, cnn_model_fn_6]
    # architectures_list = [ cnn_model_fn_6, cnn_model_fn_5]
    learning_rates_list = [0.00001, 0.000025, 0.00005, 0.000075, 0.0001, 0.0005, 0.001]
    epochs_list = [1] # + [x*10 for x in range(10)]
    batch_size_list = [1, 50, 100, 150, 200]
    split_size_list = [10]
    arch_num = 0
    start_time = time.time()

    for arch in architectures_list:
        print("hi")
        glob_arch = arch
        arch_num = arch_num + 1
        file.write("\nNumber of architecture used: " + str(arch_num) + "\n-----------------------------------------------------------------------------------------------\n")
        train_test_times_file.write("\nNumber of architecture used: " + str(arch_num) + "\n-----------------------------------------------------------------------------------------------\n")
        for lr in learning_rates_list:
            glob_learning_rate = lr
            file.write("\tLearning rate: " + str(lr) + "\n")
            train_test_times_file.write("\n\tLearning rate: " + str(lr) + "\n")
            for ep in epochs_list:
                glob_epochs = ep
                file.write("\t\tEpochs: " + str(ep) + "\n")
                for batch in batch_size_list:
                    glob_batch_s = batch
                    file.write("\t\t\tBatch size: " + str(batch) + "\n")
                    train_test_times_file.write("\n\n\t\t\tBatch size: " + str(batch) + "\n")
                    for split in split_size_list:
                        try:
                            glob_split_s = split
                            file.write(("\t\t\t\tFor K-fold cross validation, K = " + str(split)) + "\n")

                            file.write("\t\t\t\t\t1st run\n")
                            main(None)

                            file.write("\t\t\t\t\t2nd run\n")
                            main(None)
                        except Exception as e:
                            print("\nUnexpected error encountered with batch_size = " + str(
                                batch) + ", architecture = " + str(arch) + "\n\nERROR MESSAGE:\n" + str(e))
                            error_file.write("\n***********************************************\n")
                            error_file.write("\nUnexpected error encountered with batch_size = " + str(batch) + ", learning rate = " + str(lr) + " architecture = " + str(arch) + "\n\nERROR MESSAGE:\n" + str(e))
                            error_file.write("\n***********************************************\n")

    elapsed_time = time.time() - start_time

    file.write("\n\n TIME TAKEN FOR TESTING SCRIPT: " + str(elapsed_time))

    train_test_times_file.close()
    error_file.close()
    file.close()

