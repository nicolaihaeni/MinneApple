import os
import sys
import numpy as np
import torch
import torch.utils.data
from data.apple_dataset import AppleDataset

from utility.coco_utils import get_coco_api_from_dataset
from utility.coco_eval import CocoEvaluator
import utility.utils as utils
import utility.transforms as T


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

    coco = get_coco_api_from_dataset(data_loader.dataset)
    coco_evaluator = CocoEvaluator(coco, iou_types)
    detections = read_detections(os.path.join(submit_dir, 'results.txt'))

    for image, targets in data_loader:
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        im_id = targets[0]['image_id'].item()
        im_name = data_loader.dataset.get_img_name(im_id)
        outputs = [detections[im_name]]
        outputs = [{k: v.to(device) for k, v in t.items()} for t in outputs]

        res = {target['image_id'].item(): out for target, out in zip(targets, outputs)}
        coco_evaluator.update(res)

    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    coco_evaluator.synchronize_between_processes()

    # accumulate predictions from all images
    coco_evaluator.accumulate()
    stats = coco_evaluator.summarize()

    outputfile_name = os.path.join(output_dir, 'scores.txt')
    output_file = open(outputfile_name, 'w')
    output_file.write("AP: {} \n".format(float(stats[0])))
    output_file.write("AP_0.5: {} \n".format(float(stats[1])))
    output_file.write("AP_0.75: {} \n".format(float(stats[2])))
    output_file.write("AP_small: {} \n".format(float(stats[3])))
    output_file.write("AP_medium: {} \n".format(float(stats[4])))
    output_file.write("AP_large: {} \n".format(float(stats[5])))
    output_file.close()
