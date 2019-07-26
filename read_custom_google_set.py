from pedestrian_conv_net.project_utils import getDims
# import pedestrian_conv_net.estimator_test
import os
from itertools import repeat
import tensorflow as tf
from PIL import Image
# from scipy.ndimage import imread
import cv2
# import glob

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
        full_path = os.path.abspath(file_name)
        im = Image.open(full_path)
        f, e = os.path.splitext(full_path)
        new_im = im.resize((WIDTH, HEIGHT), Image.ANTIALIAS)
        new_im.save(f + '_resized.jpg', 'JPEG')

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
    # print("subdirectory is: ", subdirectory)
    # sub_set_directories = [d for d in os.listdir(subdirectory)
    #                if os.path.isdir(os.path.join(subdirectory, d))]
    # print("sub_set_directories are: ", sub_set_directories)
    # for d in sub_set_directories:
    #     prev_directory = os.path.join(subdirectory, d)
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

        elif d == "sheep":
            sheep_images, sheep_labels = load_set(subdirectory, 1)
            label_image_print(sheep_images, sheep_labels)

        elif d == "landscapes":
            land_images, land_labels = load_set(subdirectory, 2)
            label_image_print(land_images, land_labels)
    print("-----------------------------------------------------------")
    return ped_images, ped_labels,


def load_tf_Dataset(train_root_path):
    # Get all image filenames
    images = [os.path.join(root, name)
              for root, dirs, files in os.walk(train_root_path)
              for name in files
              if "." in name]

    # Get number of labels for each class
    files_in_fold = []
    for root, dirs, files in os.walk(train_root_path):
        for dir in dirs:
            dir_path = os.path.join(root, dir)
            numfiles = len([f for f in os.listdir(dir_path) if os.path.isfile(os.path.join(dir_path, f))])
            files_in_fold.append(numfiles)
        break

    print(files_in_fold)

    labels_list = []
    value = 0
    for l in files_in_fold:
        value = value + 1
        labels_list = labels_list + [value for i in repeat(None, l)]

    print(len(labels_list))
    print(len(images))

    # Reads an image from a file, decodes it into a dense tensor, and resizes it
    # to a fixed shape.
    def _parse_function(filename, label):
        image_string = tf.read_file(filename)
        image_decoded = tf.image.decode_image(image_string)
        image_resized = tf.image.resize_image_with_crop_or_pad(image_decoded, 64, 64)
        return image_resized, label

    # A vector of filenames.
    filenames = tf.constant(images)

    # `labels[i]` is the label for the image in `filenames[i].
    labels = tf.constant(labels_list)

    dataset = tf.data.Dataset.from_tensor_slices((filenames, labels))
    dataset.map(_parse_function)


def main():
    resize_pic(dataset_dir)


if __name__ == "__main__":
    main()