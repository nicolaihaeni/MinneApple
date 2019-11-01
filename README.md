# MinneApple: A Benchmark Dataset for Apple Detection and Segmentation
This is the code to our [paper](https://arxiv.org/abs/1909.06441) MinneApple: A Benchmark Dataset for Apple Detection and Segmentation. We provide example scripts for loading data, training Faster RCNN and Mask RCNN on the MinneApple dataset and example evaluation scripts.

<p align="center">
	<img src="./imgs/concept.png" width="400">
</p>

## Evaluation
As of November 1st, 2019 here are our latest models along with their respective benchmark results.

### Apple Detection
| Method | Backbone | AP @ IoU=.50:.05:.95 | AP @ IoU=.50  |  AP @ IoU=.74 | AP_small | AP_medium | AP_large|
|---|---|---|---|---|---|---|---|
| Tiled Faster RCNN  |  ResNet50  |  0.341 | 0.639 | 0.339  | 0.197  | 0.519  | 0.208  |
| Faster RCNN  |  ResNet50 |  0.438 |  0.775 | 0.455  | 0.297  | 0.578  | 0.871  |
| Mask RCNN  | ResNet50  | 0.433  |  0.763 | 0.449  | 0.295  | 0.571  |  0.809 |

### Apple Segmentation
| Method | Backbone | IoU | Class IoU (Apple) |  Pixel Accuracy | Class Accuracy (Apple) |
|---|---|---|---|---|---|
| Semi-supervised GMM  |  -  |  0.635 | 0.341 | 0.968  | 0.455  | 
| User-supervised GMM  |  -  |  0.649 |  0.455 | 0.959  | 0.634  | 
| UNet (no pretraining)  | ResNet50  | 0.678  |  0.397 | 0.960  | 0.818  | 
| UNet (ImageNet pretraining)  | ResNet50  | 0.685  |  0.410 | 0.962  | 0.848  | 

### Apple Counting
| Method | Backbone | Mean Accurracy | 
|---|---|---|
| GMM  |  -  |  0.816 | 
| CNN  |  ResNet50  |  0.908 |

## Installation
* Setup a Python 3 environment
* Install Pytorch (1.0.1 or higher) and TorchVision
* Install some other packages:
```
# Install necessary packages
pip install Pillow opencv-python sklearn numpy
```

## Training

## Prediction 

## Evaluation scripts
