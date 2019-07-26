import re

def count_nums():
    file_loc = "/home/mantsako/Desktop/pedestrian_test_images/set01images_150/labels.txt"
    counter = 0
    with open(file_loc,'r') as f:
        for val in f.read().split():
            counter += 1
    return counter


def count_results():
    learning_r = 0
    batch_s = 0
    it = 0
    less_1 = None
    gre_1 = None
    zeros_1 = None
    ones_1 = None
    analyzed_results = open(file="/home/mantsako/Desktop/test results/3/analyzed test results.txt", mode="w", encoding="utf-8")
    with open(file="/home/mantsako/Desktop/test results/3/test results.txt", mode="r+", encoding="utf-8") as file:
        for line in file:
            if "Learning rate:" in line:
                learning_r = [s for s in line.split(": ") if (any(char.isdigit() for char in s))]
                analyzed_results.write("\nLearning rate: " + str(learning_r))
            elif "Batch size" in line:
                batch_s = [float(s) for s in line.split() if s.isdigit()]
                analyzed_results.write("\n\tBatch: " + str(batch_s))
            elif "[" in line:
                # print(line)
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

                    analyzed_results.write("\n\t\tAverage accuracy of acceptable models: " + str(gre) + "\nNumber of acceptable models: " + str(len(gre)))
                    # analyzed_results.write("\n\t\tNumber of acceptable models: " + str(len(gre)))
                    analyzed_results.write("\n\t\tAccuracy of non-acceptable models: " + str(less))
                    analyzed_results.write("\n\t\tNumber of accuracy = 0 models: " + str(zeros))
                    analyzed_results.write("\n\t\tNumber of accuracy = 1 models: " + str(ones))

    analyzed_results.close()


def main():
    pass
    count_results()


    # print(count_nums())


if __name__ == "__main__":
    main()