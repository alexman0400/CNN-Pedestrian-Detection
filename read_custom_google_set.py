from pedestrian_conv_net.project_utils import getDims
import os
import sys
from itertools import repeat
import cv2

HEIGHT, WIDTH = getDims()    # discovered through helper java classes
CLASSES = 3
NUM_CHANNELS = 3
dataset_dir = "/home/mantsako/Desktop/Google_Images"    # /google_set_90"


def concat_sets(ped_set, ped_set_lab, sheep_set, sheep_set_lab, land_set, land_set_lab):

    input_images = ped_set + sheep_set + land_set
    input_labels = ped_set_lab + sheep_set_lab + land_set_lab

    return input_images, input_labels


def resize_pic(subdirectory):

    def resize_op(file_name):

        try:
            full_path = os.path.abspath(file_name)
            im = cv2.imread(full_path)
            f, e = os.path.splitext(full_path)
            new_im = cv2.resize(im, (WIDTH, HEIGHT))
            cv2.imwrite(f + '_resized.jpg', new_im)

        except Exception as e:
            print("\nUnexpected error encountered with batch_size = " + str(e))

    sub_set_directories = [d for d in os.listdir(subdirectory)
                           if os.path.isdir(os.path.join(subdirectory, d))]
    print("Loading sub_set_directories ", sub_set_directories)
    for sub_set in sub_set_directories:

        prev_directory = os.path.join(subdirectory, sub_set)
        category_directories = [d for d in os.listdir(prev_directory)
                                if os.path.isdir(os.path.join(prev_directory, d))]

        for cat in category_directories:

            print("Loading category_directories ", cat)
            new_prev_category = os.path.join(prev_directory, cat)
            print("new_prev_category is: ", new_prev_category)
            file_names = [os.path.join(new_prev_category, f)
                          for f in os.listdir(new_prev_category)
                          if (f.endswith(".jpg") | f.endswith(".jpeg"))]
            for f in file_names:

                resize_op(f)  # if needed


def load_set(subdirectory, label):
    images = []
    file_names = [os.path.join(subdirectory, f)
                  for f in os.listdir(subdirectory)
                  if (f.endswith(".jpg") | f.endswith(".jpeg"))]
    for f in file_names:
        img = cv2.imread(f)
        images.append(img)

    print(img.shape)
    labels = [label for i in repeat(None, len(images))]

    return images, labels


def request_input_train(data_dir):

    def label_image_print(images, labels):
        if len(images) == len(labels):
            print(d, " set has ", len(images), " images and ", len(labels), " labels ",
                  u'\u2713', "\n")
        else:
            print(d, " set has ", len(images), " images and ", len(labels), " labels ",
                  u'\u2717', "\n")

    directories = [d for d in os.listdir(data_dir)
                   if os.path.isdir(os.path.join(data_dir, d))]
    for d in directories:

        subdirectory = os.path.join(data_dir, d)
        print("Loading ", d, " set")
        if d == "pedestrian":
            ped_images, ped_labels = load_set(subdirectory, 0)
            label_image_print(ped_images, ped_labels)

        elif d == "sheep":
            sheep_images, sheep_labels = load_set(subdirectory, 1)
            label_image_print(sheep_images, sheep_labels)

        elif d == "landscapes":
            land_images, land_labels = load_set(subdirectory, 2)
            label_image_print(land_images, land_labels)

    return ped_images, ped_labels, sheep_images, sheep_labels, land_images, land_labels


def request_input_test(data_dir):

    def label_image_print(images, labels):
        if len(images) == len(labels):
            print(d, " set has ", len(images), " images and ", len(labels), " labels ",
                  u'\u2713', "\n")
        else:
            print(d, " set has ", len(images), " images and ", len(labels), " labels ",
                  u'\u2717', "\n")

    directories = [d for d in os.listdir(data_dir)
                   if os.path.isdir(os.path.join(data_dir, d))]
    for d in directories:

        subdirectory = os.path.join(data_dir, d)
        print("Loading ", d, " set")
        if d == "pedestrian":
            ped_images, ped_labels = load_set(subdirectory, 0)
            label_image_print(ped_images, ped_labels)
    print("-----------------------------------------------------------")
    return ped_images, ped_labels,


# # Used solely for testing the validity of the functions in this script
def main():
    pass


if __name__ == "__main__":
    main()