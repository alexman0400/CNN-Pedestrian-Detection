from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from pedestrian_conv_net.project_utils import *
from pedestrian_conv_net.read_custom_google_set import *
from pedestrian_conv_net.conv_net import *
from sklearn.model_selection import KFold
import time
import re
import tensorflow as tf
import numpy as np
# import cv2
# from matplotlib import pyplot as plt
# import sys

# # Setting default values of varriables (epochs, batch_size, split_size, learning_Rate, architecture)
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

    # # Folder of training/testing data (needs to be changed for a different path)
    train_root_path = "/home/mantsako/Desktop/Google_Images/google_set_all"
    test_root_path = "/home/mantsako/Desktop/Google_Images/day_pedestrian_subset"

    # # Folder were the model will be saved (needs to be changed for a different path)
    checkpoint_folder = "/home/mantsako/Desktop/cnn_model"

    # # Empty the folder where previous cnn model might have been saved on (essential)
    if os.path.exists(checkpoint_folder):
        empty_cnn_model_dir(checkpoint_folder)
    else:
        os.makedirs(checkpoint_folder)
    print("Deleted folder for models to be saved on")

    # # load training images
    train_ped_images, train_ped_labels, train_sheep_images, train_sheep_labels, train_land_images, train_land_labels = \
        request_input_train(train_root_path)

    train_images, train_labels = concat_sets(train_ped_images, train_ped_labels, train_sheep_images, train_sheep_labels,
                                             train_land_images, train_land_labels)

    train_im = np.asarray(train_images, dtype=np.float32)
    train_lab = np.asarray(train_labels, dtype=np.float32)

    # # load testing images afterwards
    print("**************Loading test images**************\n")
    test_ped, test_ped_lab = request_input_test(test_root_path)
    test_im = np.asarray(test_ped, dtype=np.float32)
    test_lab = np.asarray(test_ped_lab, dtype=np.float32)

    # # this peace of code was used to initialize the testing and training sets for the first and second kind of tests with day and night pedestrian images
    train_set = train_im
    train_set_lab = train_lab
    test_set = test_im
    test_set_lab = test_lab

    it = 0
    while it < 10:
        it = it + 1

    # # this peace of code was uses to execute the third kind of tests with 10-fold validation
    # for train_index, test_index in kf.split(train_im):
    #     it = it + 1
    #     print("Iteration: ", it)
    #
    #     train_set = train_im[train_index]
    #     train_set_lab = train_lab[train_index]
    #     test_set = train_im[test_index]
    #     test_set_lab = train_lab[test_index]

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

        # # Create the custom Estimator
        print("\nEstimator starts training")
        train_start_time = time.time()

        print("Number of train images: ", len(train_im),"\nNumber of train labels: ", len(train_lab))
        train_input_fn = tf.estimator.inputs.numpy_input_fn(
            x={"x": train_set},
            y=train_set_lab,
            batch_size=batch_s,
            num_epochs=epochs,
            shuffle=True)

        # # Train the model
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

        # # Test the model
        eval_results = pedestrian_classifier.evaluate(
            input_fn=eval_input_fn,
            hooks=hooks,
            steps=None,
            name="Evaluate")

        test_time = time.time() - test_start_time
        print("\nTime taken to test", test_time)

        print(eval_results)
        print('\nTest set accuracy: {accuracy:0.3f}\n'.format(**eval_results))
        print(eval_results.get('accuracy'))
        train_time_list.append(train_time)
        test_time_list.append(test_time)
        results.append(eval_results.get('accuracy'))

    # # write times of training and testing
    train_test_times_file.write("\n\n\t\t\tTraining times: " + str(train_time_list))
    train_test_times_file.write("\n\n\t\t\tTest times: " + str(test_time_list))

    max_acc = max(results)

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
    architectures_list = [cnn_model_fn_6]
    arch_list = []
    learning_rates_list = [0.00001, 0.000025, 0.00005, 0.000075, 0.0001, 0.0005, 0.001]
    epochs_list = [1]
    batch_size_list = [1, 50, 100, 150, 200]
    split_size_list = [10]
    arch_num = 0
    start_time = time.time()

    # # Get number of models used.
    for ar in architectures_list:
        a = re.findall(r'^\D*(\d+)', str(ar))
        temp_str = str(re.search(r'\d+', str(ar)).group())
        temp_str.replace("[", "")
        temp_str.replace("]", "")
        temp_str.replace("'", "")
        arch_list.append(temp_str)

    # # stops when the shorter list ends (architectures_list->function type, arch_list-> str type )
    for arch, arch_num in zip(architectures_list, arch_list):
        glob_arch = arch
        # arch_num = arch_num + 1
        file.write("\nNumber of architecture used: " + str(arch_num) + "\n-----------------------------------------------------------------------------------------------\n")
        train_test_times_file.write("\nNumber of architecture used: " + str(arch_num) + "\n-----------------------------------------------------------------------------------------------\n")

        # # train for different learning rates
        for lr in learning_rates_list:
            glob_learning_rate = lr
            file.write("\tLearning rate: " + str(lr) + "\n")
            train_test_times_file.write("\n\tLearning rate: " + str(lr) + "\n")

            # # train for different training epochs (due to long time taken to be executed, it was not tested for many different vaules of epochs_list just for the value of 1)
            for ep in epochs_list:
                glob_epochs = ep
                file.write("\t\tEpochs: " + str(ep) + "\n")

                # # train for different batch sizes
                for batch in batch_size_list:
                    glob_batch_s = batch
                    file.write("\n\t\t\tBatch size: " + str(batch) + "\n")
                    train_test_times_file.write("\n\n\t\t\tBatch size: " + str(batch) + "\n")

                    # # train different K-fold cross-validation splitting sizes (due to long time taken to be executed, it was not tested for many different vaules of K, just for the value of 10 (10-fold))
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

