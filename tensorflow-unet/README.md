# Cascaded-FCN - Tensorflow implementation

This repository contains the source-code for a Cascaded-FCN that segments the liver and its lesions out of axial CT images.

The network being used in the source-code is derived from the paper ([arXiv link](https://arxiv.org/pdf/1704.07239.pdf)) titled:

```
AUTOMATIC LIVER LESION SEGMENTATION USING A DEEP CONVOLUTIONAL NEURAL NETWORK METHOD
```
### Description ###
This work uses 2 cascaded UNETs, 

 1. In step1, a UNET segments the liver from an axial abdominal CT slice. The segmentation output is a binary mask with bright pixels denoting the segmented object. By segmenting all slices in a volume we obtain a 3D segmentation.
 2. In step2 another UNET takes an enlarged liver slice and segments its lesions.

#### Liver Network
Input: 400x400 CT-image (additionally 400x400 label-map during training - lesions and livers are merged, downsized from 512x512)
Output: 400x400 Label-Probability-Map
Batch-Size: 4
Neighboring Slices: 1

#### Lesion Network
Input: 256*256 CT-image (additionally 256*256 label-map during training, downsized from 512x512)
Output: 256*256 Label-Probability-Map
Batch-Size: 8
Neighboring Slices: 3

Values are configurable in the main code file.
