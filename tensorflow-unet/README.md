# Cascaded-FCN - Tensorflow implementation

This repository contains the source-code for a Cascaded-FCN that segments the liver and its lesions out of axial CT images.

The network being used in the source-code is derived from the paper ([arXiv link](https://arxiv.org/pdf/1704.07239.pdf)) titled:

```
AUTOMATIC LIVER LESION SEGMENTATION USING A DEEP CONVOLUTIONAL NEURAL NETWORK METHOD
```

### The network being used (small alterations applied)
![alt text](https://raw.githubusercontent.com/IBBM/Cascaded-FCN/tensorflow-implementation/tensorflow-unet/wiki/network.png)

This network can take multiple neighboring axial slices as an input. The inputs are exclusively gray-scale images. The segmentation is learned with the middle label. The benefit of taking multiple slices is that the spatial information does not get lost. For further information on the network, please read the provided paper.

### Description ###
This work uses 2 cascaded UNETs, 

 1. In step1, a UNET segments the liver from an axial abdominal CT slice. The segmentation output is a binary mask with bright pixels denoting the segmented object. By segmenting all slices in a volume we obtain a 3D segmentation.
 2. In step2 another UNET takes an enlarged liver slice and segments its lesions.

#### Liver Network
* Input: 400x400 CT-image (additionally 400x400 label-map during training - lesions and livers are merged, downsized from 512x512)
* Output: 400x400 Label-Probability-Map
* Batch-Size: 4
* Neighboring Slices: 1
* Augmentations: Rotation, Zoom, Translation
* Postprocessing before saving to .nii file: 
1. Only the largest connected labeled component is kept in the 3d volume which should always be the liver.
2. Small segmentations (<16px area) are discarded
3. Smoothing of the output probability map is applied

#### Lesion Network
* Input: 256*256 CT-image (additionally 256*256 label-map during training, downsized from 512x512)
* Output: 256*256 Label-Probability-Map
* Batch-Size: 8
* Neighboring Slices: 3
* Augmentations: Rotation, Zoom, Translation
* Postprocessing before saving to .nii file: 
1. Small segmentations (<16px area) are discarded
2. Smoothing of the output probability map is applied

Values are configurable in the main code file experiment.py.

### Environment
To run this repository, we provide a docker container which has all dependencies preinstalled:
https://hub.docker.com/r/chrisheinle/lits/

Run the container:
```bash
sudo GPU=0 nvidia-docker run -it --volume /optional_data_mount/:/data/ --volume /code_mount/:/code/ --net=host chrisheinle/lits bash
```


### Step-By-Step
1. Clone this repo
2. In order to obtain the training-data, please register at http://lits-challenge.com and follow the download instructions
3. Configure the train and test directory paths in the experiment.py code-file (todo, configurable)
4. Uncomment the liver configuration in the experiment.py file and comment the lesion configuration and run:
```bash
python experiment.py --logdir /path/which/exists
```
5. Train the network until convergence (you can check the statistics during training by running tensorboard --logdir /path/which/exists)
6. Take the best model (select in the tensorboard dice-score summary) and write it down
7. Uncomment the lesion configuration in the experiment.py file and comment the liver configuration and run:
```bash
python experiment.py --logdir /path/which/exists
```
9. Run both:
```bash
python generate_predictions.py --data_directory test_directory --out_postfix '__prediction' --model /path/which/exists/Run_x_liver/snapshots/unet-model-*bestmodelwithoutindexending* --liver 1
```
```bash
python generate_predictions.py --data_directory test_directory --out_postfix '__prediction_les' --model /path/which/exists/Run_x_lesion/snapshots/unet-model-*bestmodelwithoutindexending*
```
10. Take the *__prediction_les* files and move them to a different directory
11. Zip the with the -j flag
12. Upload them on the list-challenge website and run the result

### Expected results
Dice-score per case with current configuration: 0.477
Training steps: 150 000 for both networks respectively
