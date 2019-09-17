import os
import sys
import cv2
import numpy as np
import torch
import torch.utils.data
from data.apple_dataset import AppleDataset
from sklearn.metrics import confusion_matrix
from statistics import mean

import utility.utils as utils
import utility transforms as T


def get_transform(train):
    transforms = []
    transforms.append(T.ToTensor())
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)


def read_detections(file_path):
    if not os.path.isfile(file_path):
        print("Could not read the detection file {}. No such file or directory.".format(file_path))

    detections = {}
    with open(file_path, 'r') as infile:
        for line in infile:
            im_name, x1, y1, x2, y2, score = line.rstrip().split(',')
            if im_name not in detections:
                detections[im_name] = {'boxes': np.empty((0, 4)), 'scores': np.array([float(score)]), 'labels': np.array([1])}
                detections[im_name]['boxes'] = np.vstack((detections[im_name]['boxes'], [float(x1), float(y1), float(x2), float(y2)]))
            else:
                detections[im_name]['boxes'] = np.vstack((detections[im_name]['boxes'], [float(x1), float(y1), float(x2), float(y2)]))
                detections[im_name]['scores'] = np.concatenate((detections[im_name]['scores'], [float(score)]), axis=0)
                detections[im_name]['labels'] = np.concatenate((detections[im_name]['labels'], [1]), axis=0)

        for im_name in detections:
            detections[im_name]['boxes'] = torch.from_numpy(detections[im_name]['boxes'])
            detections[im_name]['scores'] = torch.from_numpy(detections[im_name]['scores'])
            detections[im_name]['labels'] = torch.from_numpy(detections[im_name]['labels'])
        return detections


def computeMetrics(confusion):
    '''
    Compute evaluation metrics given a confusion matrix.
    :param confusion: any confusion matrix
    :return: tuple (miou, fwiou, macc, pacc, ious, maccs)
    '''
    # Init
    labelCount = confusion.shape[0]
    ious = np.zeros((labelCount))
    maccs = np.zeros((labelCount))
    ious[:] = np.NAN
    maccs[:] = np.NAN

    # Get true positives, positive predictions and positive ground-truth
    total = confusion.sum()
    if total <= 0:
        raise Exception('Error: Confusion matrix is empty!')
    tp = np.diagonal(confusion)
    posPred = confusion.sum(axis=0)
    posGt = confusion.sum(axis=1)

    # Check which classes have elements
    valid = posGt > 0
    iousValid = np.logical_and(valid, posGt + posPred - tp > 0)

    # Compute per-class results and frequencies
    ious[iousValid] = np.divide(tp[iousValid], posGt[iousValid] + posPred[iousValid] - tp[iousValid])
    maccs[valid] = np.divide(tp[valid], posGt[valid])
    freqs = np.divide(posGt, total)

    # Compute evaluation metrics
    miou = np.mean(ious[iousValid])
    fwiou = np.sum(np.multiply(ious[iousValid], freqs[iousValid]))
    macc = np.mean(maccs[valid])
    pacc = tp.sum() / total

    return miou, fwiou, macc, pacc, ious, maccs


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

    device = torch.device('cpu')

    metric_logger = utils.MetricLogger(delimiter="  ")
    dataset = AppleDataset(os.path.join(truth_dir), get_transform(train=False))
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=1,
                                              shuffle=False, num_workers=1,
                                              collate_fn=utils.collate_fn)
    # Evaluate dataset on detection
    iou_types = ['bbox']
    mious = []
    fious = []
    mAcc = []
    pAcc = []
    ious = np.empty((0, 2))
    mAccs = np.empty((0, 2))

    for image, targets in data_loader:
        # Load the ground truth and result maks
        im_id = targets[0]['image_id']
        im_name = data_loader.dataset.get_img_name(im_id)

        gt_mask = targets[0]['masks'].numpy()
        temp = np.zeros(gt_mask.shape[1:])
        temp[np.any(gt_mask, axis=0)] = 1
        gt_mask = temp

        pred_img = cv2.imread(os.path.join(args.pred_path, im_name), 0)
        pred_img = np.floor_divide(pred_img, 255)

        if pred_img.shape != (1280, 720):
            pred_img = cv2.resize(pred_img, (720, 1280), interpolation=cv2.INTER_NEAREST)

        confusion = confusion_matrix(gt_mask.flatten(), pred_img.flatten())
        miou, fwiou, macc, pacc, iou, maccs = computeMetrics(confusion)
        mious.append(miou)
        fious.append(fwiou)
        mAcc.append(macc)
        pAcc.append(pacc)
        ious = np.vstack((ious, iou))
        mAccs = np.vstack((mAccs, maccs))

    print("Segmentation results:")
    print("Mean IoU: {}".format(mean(mious)))
    print("Mean frequency weighted IoU: {}".format(mean(fious)))
    print("Mean Accuracy: {}".format(mean(mAcc)))
    print("Pixel Accuracy: {}".format(mean(pAcc)))
    print("Class IoU: {}".format(np.mean(ious, axis=0)))
    print("Class Mean Accuracy: {}".format(np.mean(mAccs, axis=0)))

    outputfile_name = os.path.join(output_dir, 'scores.txt')
    output_file = open(outputfile_name, 'w')
    output_file.write("IoU: {} \n".format(float(mean(mious))))
    output_file.write("fwIoU: {} \n".format(float(mean(fious))))
    output_file.write("mAcc: {} \n".format(float(mean(mAcc))))
    output_file.write("pAcc: {} \n".format(float(mean(pAcc))))
    output_file.write("cIoU: {} \n".format(float(np.mean(ious, axis=0)))
    output_file.write("cAcc: {} \n".format(float(np.mean(mAccs, axis=0))))
    output_file.close()
