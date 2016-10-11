## Link to download weights file ##
https://dl.dropboxusercontent.com/u/28351742/CascadedFCN/step1_weights.caffemodel

### Inference for Step1 of the Cascaded-FCN model ###

To run inference on new axial abdominal CT slices, use the included deploy prototxt and the caffe model.
Image pixel brightness ranges from 0.0 to 1.0. Abdominal slices should be originally 388x388, augmented by reflection mirroring of the 92 pixels on the boundary, resulting in 572x572 inputs.
