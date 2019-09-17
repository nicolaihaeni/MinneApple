import os
import sys


def read_file(path):
    struct = {}
    with open(path, 'r') as file:
        for line in file:
            line = line.rstrip().split(',')
            im_name, count = line
            struct[im_name] = count
    return struct


if __name__ == "__main__":
    input_dir = sys.argv[1]
    output_dir = sys.argv[2]

    submit_dir = os.path.join(input_dir, 'res')
    truth_dir = os.path.join(input_dir, 'ref')

    if not os.path.isdir(submit_dir):
        print("%s does not exist".format(submit_dir))

    if os.path.isdir(submit_dir) and os.path.isdir(truth_dir):
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

    gt_file = os.path.join(truth_dir, 'ground_truth.txt')
    result_file = os.path.join(submit_dir, 'result.txt')
    gt = read_file(gt_file)
    rt = read_file(result_file)

    assert(len(gt) == len(rt))

    correct = 0
    for im_name, count in gt.items():
        if rt[im_name] == count:
            correct += 1

    outputfile_name = os.path.join(output_dir, 'scores.txt')
    output_file = open(outputfile_name, 'w')
    output_file.write("Accuracy: {} \n".format(float(correct) / float(len(gt))))
    output_file.close()
