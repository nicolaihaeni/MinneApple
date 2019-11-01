import os
import torch
import torch.utils.data
import torchvision
import numpy as np

from data.apple_dataset import AppleDataset
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

import utility.utils as utils
import utility.transforms as T


######################################################
# Predict with either a Faster-RCNN or Mask-RCNN predictor
# using the MinneApple dataset
######################################################
def get_transform(train):
    transforms = []
    transforms.append(T.ToTensor())
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)


def get_maskrcnn_model_instance(num_classes):
    # load an instance segmentation model pre-trained pre-trained on COCO
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=False)

    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # now get the number of input features for the mask classifier
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    # and replace the mask predictor with a new one
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, hidden_layer, num_classes)
    return model


def get_frcnn_model_instance(num_classes):
    # load an instance segmentation model pre-trained pre-trained on COCO
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=False)

    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model


def main(args):
    num_classes = 2
    device = args.device

    # Load the model from
    print("Loading model")
    # Create the correct model type
    if args.mrcnn:
        model = get_maskrcnn_model_instance(num_classes)
    else:
        model = get_frcnn_model_instance(num_classes)

    # Load model parameters and keep on CPU
    checkpoint = torch.load(args.weight_file, map_location=device)
    model.load_state_dict(checkpoint['model'], strict=False)
    model.eval()

    print("Creating data loaders")
    dataset_test = AppleDataset(args.data_path, get_transform(train=False))
    data_loader_test = torch.utils.data.DataLoader(dataset_test, batch_size=1,
                                                   shuffle=False, num_workers=1,
                                                   collate_fn=utils.collate_fn)

    # Create output directory
    base_path = os.path.dirname(args.output_file)
    if not os.path.exists(base_path):
        os.makedirs(base_path)

    # Predict on bboxes on each image
    f = open(args.output_file, 'a')
    for image, targets in data_loader_test:
        image = list(img.to(device) for img in image)
        outputs = model(image)
        for ii, output in enumerate(outputs):
            img_id = targets[ii]['image_id']
            img_name = data_loader_test.dataset.get_img_name(img_id)
            print("Predicting on image: {}".format(img_name))
            boxes = output['boxes'].detach().numpy()
            scores = output['scores'].detach().numpy()

            im_names = np.repeat(img_name, len(boxes), axis=0)
            stacked = np.hstack((im_names.reshape(len(scores), 1), boxes.astype(int), scores.reshape(len(scores), 1)))

            # File to write predictions to
            np.savetxt(f, stacked, fmt='%s', delimiter=',', newline='\n')


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='PyTorch Detection')
    parser.add_argument('--data_path', required=True, help='path to the data to predict on')
    parser.add_argument('--output_file', required=True, help='path where to write the prediction outputs')
    parser.add_argument('--weight_file', required=True, help='path to the weight file')
    parser.add_argument('--device', default='cpu', help='device to use. Either cpu or cuda')
    model = parser.add_mutually_exclusive_group(required=True)
    model.add_argument('--frcnn', action='store_true', help='use a Faster-RCNN model')
    model.add_argument('--mrcnn', action='store_true', help='use a Mask-RCNN model')

    args = parser.parse_args()
    main(args)
