from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import sys
import os

# # Name of the directory of the file with the test results.
directory = "/home/mantsako/Desktop/test results/10-fold"


def count_results():
    learning_r = 0
    batch_s = 0
    it = 0
    less_1 = None
    gre_1 = None
    zeros_1 = None
    ones_1 = None
    analyzed_results = open(file="/home/mantsako/Desktop/test results/2/analyzed test results.txt", mode="w", encoding="utf-8")
    with open(file="/home/mantsako/Desktop/test results/3/test results.txt", mode="r+", encoding="utf-8") as file:
        for line in file:

            if "architecture" in line:
                analyzed_results.write("\n---------------------------------------------------------------\n")
                arch = [float(s) for s in line.split() if s.isdigit()]
                analyzed_results.write("\nArchitecture model: " + str(arch))
                analyzed_results.write("\n---------------------------------------------------------------\n")
            elif "Learning rate:" in line:
                learning_r = [s for s in line.split(": ") if (any(char.isdigit() for char in s))]
                analyzed_results.write("\nLearning rate: " + str(learning_r))
            elif "Batch size" in line:
                batch_s = [float(s) for s in line.split() if s.isdigit()]
                analyzed_results.write("\n\tBatch: " + str(batch_s))
            elif "[" in line:
                if it == 0:
                    it = it + 1
                    line = line.replace("[", "")
                    line = line.replace("]", "")
                    results_1 = [float(s) for s in line.split(",")]
                    less_1 = [i for i in results_1 if i < 0.7]
                    gre_1 = [i for i in results_1 if i >= 0.7]
                    zeros_1 = len([i for i in results_1 if i == 0])
                    ones_1 = len([i for i in results_1 if i == 1])
                else:
                    it = 0
                    line = line.replace("[", "")
                    line = line.replace("]", "")
                    results_2 = [float(s) for s in line.split(",")]
                    less_2 = [i for i in results_2 if i < 0.7]
                    gre_2 = [i for i in results_2 if i >= 0.7]
                    zeros_2 = len([i for i in results_2 if i == 0])
                    ones_2 = len([i for i in results_2 if i == 1])

                    less = less_1 + less_2
                    gre = gre_1 + gre_2
                    zeros = zeros_1 + zeros_2
                    ones = ones_1 + ones_2

                    analyzed_results.write("\n\t\tAverage accuracy of acceptable models: " + str(gre) + "\n\t\tNumber of acceptable models: " + str(len(gre)))
                    analyzed_results.write("\n\t\tAccuracy of non-acceptable models: " + str(less))
                    analyzed_results.write("\n\t\tNumber of accuracy = 0 models: " + str(zeros))
                    analyzed_results.write("\n\t\tNumber of accuracy = 1 models: " + str(ones))

    analyzed_results.close()


# # Function used to build the graphs
def build_graphs():

    # # Change this for every architecture used
    arch_list = [0, 2, 6]
    learning_rates_list = ["1e-05", "2.5e-05", "5e-05", "7.5e-05", "0.0001", "0.0005", "0.001"]
    batch_size_list = [1, 50, 100, 150, 200]

    def plot(x, x_lab, y, y_lab, z, title):
        column_names = y
        row_names = x

        fig = plt.figure(title)
        ax = Axes3D(fig)

        lx = len(x)
        ly = len(y)

        xpos = np.arange(0, lx, 1)
        ypos = np.arange(0, ly, 1)
        xpos, ypos = np.meshgrid(xpos + 0.75, ypos + 0.75)

        xpos = xpos.flatten()  # Convert positions to 1D array
        ypos = ypos.flatten()
        zpos = np.zeros(lx * ly)
        xx = yy = 0.5
        zz = z

        # # Color all the architectures
        if lx < 4:
            cs = ['#7b241c', '#fcf3cf', '#f0b27a']
        else:
            cs = ['#17202a', '#34495e', '#1a5276', '#f8f9f9', '#d5d8dc']
        cs = cs * ly

        ax.bar3d(xpos, ypos, zpos, xx, yy, zz, shade=True, color=cs)

        ax.w_xaxis.set_ticklabels(row_names)
        ax.w_yaxis.set_ticklabels(column_names)
        ax.set_xlabel(x_lab)
        ax.set_ylabel(y_lab)
        ax.set_zlabel('Accuracy (%)')
        # plt.savefig(title + ".png")
        plt.show()

    def Create_dictionary():
        dic = {}
        it = 0

        # # Analyze the test results files
        sub_dirs = [os.path.join(directory, f)
                    for f in os.listdir(directory)]

        file_list = []
        for sub in sub_dirs:
            os.path.join(directory, sub)
            file_list = file_list + [os.path.join(os.path.join(directory, sub), f)
                                     for f in os.listdir(os.path.join(directory, sub))
                                     if f.startswith("test")]

        for file in file_list:

            with open(file=file, mode="r", encoding="utf-8") as f:

                for line in f:

                    line = line.replace("\n", "")

                    # create dictionary of dictionaries {architectures, learning rates, batches}
                    if "architecture" in line:
                        arch = [int(s) for s in line.split() if s.isdigit()]
                        if arch[0] not in dic:
                            dic[arch[0]] = {}
                    elif "Learning rate:" in line:
                        learning_r = [s for s in line.split(": ") if (any(char.isdigit() for char in s))]
                        learning_r[0] = learning_r[0].replace("'", "")
                        if learning_r[0] not in dic[arch[0]]:
                            dic[arch[0]][learning_r[0]] = {}
                    elif "Batch size" in line:
                        batch_s = [float(s) for s in line.split() if s.isdigit()]
                    elif "[" in line:
                        line = line.replace("[", "")
                        line = line.replace("]", "")
                        if it == 0:
                            it = it + 1
                            results_1 = [float(s) for s in line.split(",")]
                        else:
                            it = 0
                            results_2 = [float(s) for s in line.split(",")]

                            # check if the batch list is not empty to avoid exceptions
                            if dic[arch[0]][learning_r[0]].get(batch_s[0]) is not None:
                                dic[arch[0]][learning_r[0]][batch_s[0]] = dic[arch[0]][learning_r[0]].get(
                                    batch_s[0]) + results_1 + results_2
                            elif arch[0] in dic:
                                dic[arch[0]][learning_r[0]][batch_s[0]] = results_1 + results_2
        return dic

    def Learning_Rate_accuracy_graph(d, arch_list, learning_rates_list, batch_size_list):

        # # Create learning rate and accuracy bar graph
        mean_list = []
        good_mean_list = []
        bad_mean_list = []

        for arch in arch_list:

            for lr in learning_rates_list:

                acc = []
                good_acc = []
                bad_acc = []
                for batch in batch_size_list:

                    try:
                        temp = d[arch][lr].get(batch)
                        # print(temp)
                    except Exception:
                        print(arch, lr, batch)
                        print("\n", d)
                        print("\n", temp)
                        sys.exit()
                    if temp is not None:
                        acc = acc + temp

                        # acceptable accuracies
                        good_acc = good_acc + ([i for i in temp if i >= 0.7])

                        # not acceptable accuracies
                        bad_acc = bad_acc + [i for i in temp if i < 0.7]

                # get the mean of all accuracies (acceptable and not acceptable ones)
                if acc is not []:
                    mean = sum(acc)/len(acc)
                    mean_list.append(mean)

                    # get the mean of all the non acceptable accuracies
                    bad_mean = sum(bad_acc)/len(bad_acc)
                    bad_mean_list.append(bad_mean)

                    # get the mean of all the acceptable accuracies
                    good_mean = sum(good_acc)/len(good_acc)
                    good_mean_list.append(good_mean)

        # print(len(mean_list), len(bad_mean_list), len(good_mean_list))
        xlabel = "Number of architecture"
        ylabel = "Learning rates"

        plot(arch_list, xlabel, learning_rates_list, ylabel, mean_list, "Learning Rate-Accuracy Graph for all the models")
        plot(arch_list, xlabel, learning_rates_list, ylabel, bad_mean_list, "Learning Rate-Accuracy Graph for not acceptable models")
        plot(arch_list, xlabel, learning_rates_list, ylabel, good_mean_list, "Learning Rate-Accuracy Graph for acceptable models")

    def Batch_Size_Accuracy_graph(d, arch_list, learning_rates_list, batch_size_list):

        # # Create learning rate and accuracy bar graph
        mean_list = []
        good_mean_list = []
        bad_mean_list = []

        for batch in batch_size_list:

            for arch in arch_list:

                acc = []
                good_acc = []
                bad_acc = []
                for lr in learning_rates_list:

                    temp = d[arch][lr].get(batch)
                    if temp is not None:
                        acc = acc + temp

                        # # acceptable accuracies
                        good_acc = good_acc + ([i for i in temp if i >= 0.7])

                        # # not acceptable accuracies
                        bad_acc = bad_acc + [i for i in temp if i < 0.7]

                # # check id acc list is empty
                # # probable cause would be hardware error
                if acc is not []:
                    # # get the mean of all accuracies (acceptable and not acceptable ones)
                    mean = sum(acc) / len(acc)
                    mean_list.append(mean)

                    # # get the mean of all the non acceptable accuracies
                    bad_mean = sum(bad_acc) / len(bad_acc)
                    bad_mean_list.append(bad_mean)

                    # # get the mean of all the acceptable accuracies
                    good_mean = sum(good_acc) / len(good_acc)
                    good_mean_list.append(good_mean)

        xlabel = "Number of architecture"
        ylabel = "Batch sizes"

        plot(arch_list, xlabel, batch_size_list, ylabel, mean_list, "Batch Size-Accuracy graph for all the models")
        plot(arch_list, xlabel, batch_size_list, ylabel, bad_mean_list, "Batch Size-Accuracy graph for not acceptable models")
        plot(arch_list, xlabel, batch_size_list, ylabel, good_mean_list, "Batch Size-Accuracy graph for acceptable models")

    def Batch_Size_Learning_Rate_Graph(d, arch_list, learning_rates_list, batch_size_list):

        # # Create learning rate and accuracy bar graph
        mean_list = []
        good_mean_list = []
        bad_mean_list = []

        for batch in batch_size_list:

            for lr in learning_rates_list:

                acc = []
                good_acc = []
                bad_acc = []
                for arch in arch_list:

                    try:
                        temp = d[arch][lr].get(batch)
                        # print(temp)
                    except Exception:
                        print(arch, lr, batch)
                        print("\n", d)
                        print("\n", temp)
                        sys.exit()
                    if temp is not None:
                        acc = acc + temp

                        # acceptable accuracies
                        good_acc = good_acc + ([i for i in temp if i >= 0.7])

                        # not acceptable accuracies
                        bad_acc = bad_acc + [i for i in temp if i < 0.7]

                # get the mean of all accuracies (acceptable and not acceptable ones)
                if acc is not []:
                    mean = sum(acc)/len(acc)
                    mean_list.append(mean)

                    # get the mean of all the non acceptable accuracies
                    bad_mean = sum(bad_acc)/len(bad_acc)
                    bad_mean_list.append(bad_mean)

                    # get the mean of all the acceptable accuracies
                    good_mean = sum(good_acc)/len(good_acc)
                    good_mean_list.append(good_mean)

        xlabel = "Batch sizes"
        ylabel = "Learning rates"

        plot(batch_size_list, xlabel, learning_rates_list, ylabel, mean_list, "Batch Size-Learning Rate_Graph for all the models")
        plot(batch_size_list, xlabel, learning_rates_list, ylabel, bad_mean_list, "Batch Size-Learning Rate_Graph for not acceptable models")
        plot(batch_size_list, xlabel, learning_rates_list, ylabel, good_mean_list, "Batch Size-Learning Rate_Graph for acceptable models")

    d = Create_dictionary()
    Learning_Rate_accuracy_graph(d, arch_list, learning_rates_list, batch_size_list)
    Batch_Size_Accuracy_graph(d, arch_list, learning_rates_list, batch_size_list)
    Batch_Size_Learning_Rate_Graph(d, arch_list, learning_rates_list, batch_size_list)

    pass


# # Used solely for testing the validity of the functions in this script
def main():
    pass
    # build_graphs()
    # count_results()


if __name__ == "__main__":
    main()