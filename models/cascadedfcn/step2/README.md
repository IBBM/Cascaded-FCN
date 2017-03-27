## Link to download weights file ##
[step2_weights.caffemodel](https://www.dropbox.com/s/ql10c37d7ura23l/step2_weights.caffemodel?dl=1)


### Inference for Step2 of the Cascaded-FCN model ###

Step2 UNET expects an image of the liver (upscaled to full resolution 572x572), with all non-liver pixels set to 0.
To run inference on new liver, use the included deploy prototxt and the caffe model.
Image pixel brightness ranges from 0.0 to 1.0. Liver images should be originally 388x388, augmented by reflection mirroring of the 92 pixels on the boundary, resulting in 572x572 inputs.
