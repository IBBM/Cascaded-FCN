# import caffe
import collections
import os
from os.path import isfile, join, basename
# import matplotlib.pyplot as plt
import nibabel
import numbers
import numpy as np
import SimpleITK as sitk
import scipy.misc
import scipy.ndimage
import skimage.exposure
import sklearn.preprocessing as skpre
from warnings import warn


def file_paths_random(directory, image_identifier="image", label_identifier="label", iterations=0):

    combined_list = _file_paths(directory, image_identifier, label_identifier)

    while True:
        permutation = np.random.permutation(len(combined_list))
        for i in permutation:
            parameters = dict()
            parameters["file_names"] = [basename(combined_list[i][0]), basename(combined_list[i][1])]
            yield (combined_list[i], parameters)
        if iterations > 0:
            iterations -= 1
            if iterations == 0:
                break


def file_paths_ordered(directory, image_identifier="image", label_identifier="label", iterations=1, prediction_postfix='$$$', new_postfix='$$$'):

    combined_list = _file_paths(directory, image_identifier, label_identifier, prediction_postfix=prediction_postfix, new_postfix=new_postfix)

    while True:
        for item in combined_list:
            parameters = dict()
            parameters["file_names"] = [basename(item[0]), basename(item[1])]
            yield (item, parameters)

        if iterations > 0:
            iterations -= 1
            if iterations == 0:
                break


def _file_paths(directory, image_identifier, label_identifier, prediction_postfix='$$$', new_postfix='$$$'):

    if not type(directory) is str:
        raise TypeError("Directory must be a string. Received: %s" % type(directory))

    if not type(image_identifier) is str:
        raise TypeError("Image Identifier must be a string. Received: %s" % type(image_identifier))

    if not type(label_identifier) is str:
        raise TypeError("Label Identifier must be a string. Received: %s" % type(label_identifier))

    dir_content = os.listdir(directory)
    sorted_content = sorted(dir_content, key=lambda fname: fname.replace(prediction_postfix, ''))
    replaced_content = sorted(dir_content, key=lambda fname: fname.replace(prediction_postfix, ''))

    image_list  = [f for f in sorted_content if isfile(join(directory, f)) and image_identifier in f]
    label_list  = [f for f in replaced_content if isfile(join(directory, f)) and label_identifier in f and (    prediction_postfix in f or prediction_postfix=='$$$') and (not new_postfix in f or new_postfix=='$$$')]
    label_list2 = [f for f in replaced_content if isfile(join(directory, f)) and label_identifier in f and (not prediction_postfix in f or prediction_postfix=='$$$') and (not new_postfix in f or new_postfix=='$$$')]

    if len(label_list2) != len(label_list):
        print ("len(Label list 2) != (label_list)")
        label_list2 = label_list

    if len(image_list) != len(label_list):
        # raise RuntimeError("Directory \"%s\" contains %d input items, but %d ground truth items!" % (directory, len(image_list), len(label_list)))
        print ("RuntimeErrorDirectory contains items, but ground truth items, len(image_list), len(label_list)))")

    combined_list = zip(image_list, label_list, label_list2)
    print ("combined_list", combined_list)

    for image, label, label_list2 in combined_list:
        if image.replace(image_identifier, "") != label.replace(label_identifier, "").replace('__prediction', ''):
            warn("Input item \"%s\" and ground truth item \"%s\" don't seem to match!" % (image, label))

    return [(join(directory, f), join(directory, g), join(directory, x)) for (f, g, x) in combined_list]


def simpleITK_load_volume():

    def load_func(input_tuple):

        inputs, parameters = input_tuple

        outputs = []

        volume = sitk.ReadImage(inputs[0])
        outputs.append(sitk.Cast(volume, sitk.sitkFloat32))

        parameters["spacing"] = volume.GetSpacing()
        parameters["original_spacing"] = parameters["spacing"]

        parameters["size"] = volume.GetSize()
        parameters["original_size"] = parameters["size"]

        for file_path in inputs:
            outputs.append(sitk.Cast(sitk.ReadImage(file_path), sitk.sitkFloat32))

        return (outputs, parameters)

    return load_func


def numpy_load_volume():

    def load_func(input_tuple):

        inputs, parameters = input_tuple

        outputs = []

        nifti = nibabel.load(inputs[0])
        outputs.append(np.asarray(nifti.dataobj).astype(np.float32))

        parameters["spacing"] = nifti.header.get_zooms()[:3]  # the last value is the time between scans, no need to keep it
        parameters["original_spacing"] = parameters["spacing"]

        parameters["size"] = nifti.header.get_data_shape()
        parameters["original_size"] = parameters["size"]

        if len(inputs) > 1:
            for file_path in inputs[1:]:
                nifti = nibabel.load(file_path)
                outputs.append(np.asarray(nifti.dataobj).astype(np.float32))

        return (outputs, parameters)

    return load_func


def numpy_load_slice():

    def load_func(input_tuple):

        inputs, parameters = input_tuple

        outputs = []

        nifti = nibabel.load(inputs[0])
        slice_index = np.random.randint(nifti.header.get_data_shape()[-1])

        outputs.append(np.asarray(nifti.dataobj[..., slice_index]).astype(np.float32))

        parameters["spacing"] = nifti.header.get_zooms()[:2]  # the last value is the time between scans, no need to keep it
        parameters["original_spacing"] = parameters["spacing"]

        parameters["size"] = nifti.header.get_data_shape()[:2]
        parameters["original_size"] = parameters["size"]

        if len(inputs) > 1:
            for file_path in inputs[1:]:
                nifti = nibabel.load(file_path)
                outputs.append(np.asarray(nifti.dataobj[..., slice_index]).astype(np.float32))

        return (outputs, parameters)

    return load_func


def numpy_load_slice_random_direction():

    def load_func(input_tuple):

        inputs, parameters = input_tuple

        outputs = []

        dimension = np.random.randint(3)

        nifti = nibabel.load(inputs[0])
        slice_index = np.random.randint(nifti.header.get_data_shape()[dimension])

        if dimension == 0:
            image = nifti.dataobj[slice_index, :, :]
        elif dimension == 1:
            image = nifti.dataobj[:, slice_index, :]
        elif dimension == 2:
            image = nifti.dataobj[:, :, slice_index]

        outputs.append(np.asarray(image).astype(np.float32))

        if dimension == 0:
            parameters["spacing"] = nifti.header.get_zooms()[1:3]
            parameters["size"] = nifti.header.get_data_shape()[1:3]
        elif dimension == 1:
            parameters["spacing"] = np.concatenate([nifti.header.get_zooms()[0:1], nifti.header.get_zooms()[2:3]])
            parameters["size"] = np.concatenate([nifti.header.get_data_shape()[0:1], nifti.header.get_data_shape()[2:3]])
        elif dimension == 2:
            parameters["spacing"] = nifti.header.get_zooms()[:2]
            parameters["size"] = nifti.header.get_data_shape()[:2]

        parameters["original_spacing"] = parameters["spacing"]
        parameters["original_size"] = parameters["size"]

        if len(inputs) > 1:
            for file_path in inputs[1:]:
                nifti = nibabel.load(file_path)

                if dimension == 0:
                    label = nifti.dataobj[slice_index, :, :]
                elif dimension == 1:
                    label = nifti.dataobj[:, slice_index, :]
                elif dimension == 2:
                    label = nifti.dataobj[:, :, slice_index]

                outputs.append(np.asarray(label).astype(np.float32))

        return (outputs, parameters)

    return load_func

def numpy_gen_slices():

    def load_func(input_tuple):

        inputs, parameters = input_tuple

        niftis = [nibabel.load(inpt) for inpt in inputs]

        volumes = [np.asarray(nifti.dataobj).astype(np.float32) for nifti in niftis]

        nifti = niftis[0]

        parameters["spacing"] = nifti.header.get_zooms()[:2]  # the last value is the time between scans, no need to keep it
        parameters["original_spacing"] = parameters["spacing"]

        parameters["size"] = nifti.header.get_data_shape()[:2]
        parameters["original_size"] = parameters["size"]

        for slice_index in xrange(nifti.header.get_data_shape()[-1]):

            outputs = [volume[..., slice_index] for volume in volumes]

            yield (outputs, parameters.copy(), inputs, slice_index)

    return load_func

def numpy_load_all_slices():

    def load_func(input_tuple):

        inputs, parameters = input_tuple

        niftis = [nibabel.load(inpt) for inpt in inputs]

        volumes = [np.asarray(nifti.dataobj).astype(np.float32) for nifti in niftis]

        nifti = niftis[0]

        parameters["spacing"] = nifti.header.get_zooms()[:2]  # the last value is the time between scans, no need to keep it
        parameters["original_spacing"] = parameters["spacing"]

        parameters["size"] = nifti.header.get_data_shape()[:2]
        parameters["original_size"] = parameters["size"]

        for slice_index in xrange(nifti.header.get_data_shape()[-1]):

            outputs = [volume[..., slice_index] for volume in volumes]

            yield (outputs, parameters.copy())

    return load_func


def numpy_clip(minimum, maximum):

    if not isinstance(minimum, numbers.Number):
        raise TypeError("Minimum must be a number! Received: %s" % type(minimum))

    if not isinstance(maximum, numbers.Number):
        raise TypeError("Maximum must be a number! Received: %s" % type(maximum))

    def clip_func(input_tuple):
        inputs, parameters = input_tuple

        np.clip(inputs[0], minimum, maximum, out=inputs[0])

        return (inputs, parameters)

    def wrap(input_tuple):
        inputs, parameters = input_tuple
        outputs = []

        for inp in inputs:
            outputs.append(clip_func((inp, parameters))[0])

        return (outputs, parameters)

    return wrap


# unsafe: doesn't check inputs for size and fit
def numpy_patches(core_size, overlap):

    if isinstance(core_size, numbers.Number):
        core_size = [core_size, core_size]
    if isinstance(core_size, collections.Sequence):
        core_size = np.asarray(core_size)
    if not isinstance(core_size, np.ndarray):
        raise TypeError("core_size must be a number, list, tuple or Numpy array! Received: %s" % type(core_size))

    if isinstance(overlap, numbers.Number):
        overlap = [overlap, overlap]
    if isinstance(overlap, collections.Sequence):
        overlap = np.asarray(overlap)
    if not isinstance(overlap, np.ndarray):
        raise TypeError("overlap must be a number, list, tuple or Numpy array! Received: %s" % type(overlap))

    def patch_func(input_tuple):

        inputs, parameters = input_tuple

        image = inputs[0]
        labels = inputs[1]

        shape = image.shape
        minimum = np.min(image)
        step_x = core_size[0] / 2.
        step_y = core_size[1] / 2.

        length = max(0, shape[0] - step_x - 1)
        xs = int(length // step_x + 1)

        length = max(0, shape[1] - step_y - 1)
        ys = int(length // step_y + 1)

        for x in xrange(xs):
            for y in xrange(ys):

                steps_x = int(round(x * step_x))
                steps_y = int(round(y * step_y))

                out_image = np.full([core_size[0] + 2 * overlap[0], core_size[1] + 2 * overlap[1]], minimum, dtype=image.dtype)
                out_labels = np.zeros(core_size, dtype=inputs[1].dtype)

                start_x = max(0, steps_x - overlap[0])
                end_x = min(image.shape[0], steps_x + core_size[0] + overlap[0])

                start_y = max(0, steps_y - overlap[1])
                end_y = min(image.shape[1], steps_y + core_size[1] + overlap[1])

                offset_x = max(0, -(steps_x - overlap[0]))
                last_x = offset_x + (end_x - start_x)

                offset_y = max(0, -(steps_y - overlap[1]))
                last_y = offset_y + (end_y - start_y)

                out_image[offset_x:last_x, offset_y:last_y] = image[start_x:end_x, start_y:end_y]

                start_x = steps_x
                end_x = min(image.shape[0], steps_x + core_size[0])

                start_y = steps_y
                end_y = min(image.shape[1], steps_y + core_size[1])

                last_x = end_x - start_x
                last_y = end_y - start_y

                out_labels[0:last_x, 0:last_y] = labels[start_x:end_x, start_y:end_y]

                yield ([out_image, out_labels], parameters.copy())

    return patch_func


def numpy_clahe():

    def equ_func(input_tuple):

        inputs, parameters = input_tuple

        image = inputs[0]

        minimum = np.min(image)
        maximum = np.max(image)

        image = (image - minimum) / (maximum - minimum)
        image = skimage.exposure.equalize_adapthist(image)
        inputs[0] = image * (maximum - minimum) + minimum

        return (inputs, parameters)

    def wrap(input_tuple):
        inputs, parameters = input_tuple
        outputs = []

        for inp in inputs:
            outputs.append(equ_func((inp, parameters))[0])

        return (outputs, parameters)

    return wrap


def numpy_translation2D(probability, factor=0.5, default_border=0.25, label_of_interest=1):

    def translation_func(input_tuple, rand_val, skip_transform):

        inputs, parameters = input_tuple

        if(not skip_transform):  # do not apply deformations always, just sometimes

            label = inputs[1]

            if label_of_interest not in label:
                xdist = default_border * label.shape[0] * factor
                ydist = default_border * label.shape[1] * factor

            else:
                itemindex = np.where(label == 1)

                xdist = min(np.min(itemindex[0]), label.shape[0] - np.max(itemindex[0])) * factor
                ydist = min(np.min(itemindex[1]), label.shape[1] - np.max(itemindex[1])) * factor

            ox = int(rand_val * xdist) if xdist > 0 else 0
            oy = int(rand_val * ydist) if ydist > 0 else 0

            def non(s):
                return s if s < 0 else None

            def mom(s):
                return max(0, s)

            for i in xrange(len(inputs)):
                shift_img = np.full_like(inputs[i], np.min(inputs[i]))
                shift_img[mom(ox):non(ox), mom(oy):non(oy)] = inputs[i][mom(-ox):non(-ox), mom(-oy):non(-oy)]
                inputs[i] = shift_img

            parameters["translation"] = (ox, oy)

        return (inputs, parameters)

    def wrap(input_tuple):
        inputs, parameters = input_tuple
        outputs = []

        rand_val = np.random.uniform(-1, 1)
        skip_transform = np.random.rand() > probability

        for inp in inputs:
            outputs.append(translation_func((inp, parameters), rand_val, skip_transform)[0])

        return (outputs, parameters)

    return wrap


def numpy_rotation2D(probability, upper_bound=20, min_val=None):

    def rotation_func(input_tuple, angle, skip, min_val):

        inputs, parameters = input_tuple

        if(not skip):  # do not apply deformations always, just sometimes

            if not min_val:
                min_val = np.min(inputs[0])

            inputs[0] = scipy.ndimage.interpolation.rotate(inputs[0], angle, reshape=False, order=1, cval=min_val, prefilter=False)  # order = 1 => biliniear interpolation
            inputs[1] = scipy.ndimage.interpolation.rotate(inputs[1], angle, reshape=False, order=0, cval=np.min(inputs[1]), prefilter=False)  # order = 0 => nearest neighbour

            parameters["rotation"] = angle

        return (inputs, parameters)

    def wrap(input_tuple):
        inputs, parameters = input_tuple
        outputs = []

        angle = np.random.randint(-upper_bound, upper_bound)
        angle = (360 + angle) % 360

        skip = np.random.rand() > probability

        for inp in inputs:
            outputs.append(rotation_func((inp, parameters), angle, skip, min_val)[0])

        return (outputs, parameters)

    return wrap

def numpy_pullup(min_val = 0):
    def renorm(input_tuple):
        inputs, parameters = input_tuple
        outputs = []

        for inp in inputs:
            min_v = np.min(inp[0])
            
            # shift smallest value to zero
            inp[0] = np.add(inp[0], -min_v+min_val)

            outputs.append(inp)

        return (outputs, parameters)

    return renorm

def numpy_mask_background(mask_label, mask_value):

    def mask_func(input_tuple):
        inputs, parameters = input_tuple

        inputs[0][inputs[1] == mask_label] = mask_value

        return (inputs, parameters)

    def wrap(input_tuple):
        inputs, parameters = input_tuple
        outputs = []

        for inp in inputs:
            outputs.append(mask_func((inp, parameters))[0])

        return (outputs, parameters)

    return wrap

def numpy_static_scaler(minval, maxval):
    interv = float(maxval-minval)

    def scale_func(input_tuple):
        # get image
        inputs, parameters = input_tuple
        image = inputs[0]

        # push to 0 as minval
        image -= minval

        # scale down to 0 - 2
        image /= (interv / 2)

        # print ("image", image.min(), image.max())

        return (inputs, parameters)

    def wrap(input_tuple):
        inputs, parameters = input_tuple
        outputs = []

        for inp in inputs:
            outputs.append(scale_func((inp, parameters))[0])

        return (outputs, parameters)

    return wrap

def numpy_robust_scaling(dictionary=None):

    if dictionary is not None:

        scaler = skpre.RobustScaler(copy=False)

        def scale_func(input_tuple):
            # get image and file name
            inputs, parameters = input_tuple
            file_name = parameters["file_names"][0]
            image = inputs[0]

            # flatten
            old_shape = image.shape
            image = image.reshape(-1, 1)

            # scale
            scaler_params = dictionary[file_name]
            scaler.center_ = scaler_params[0]
            scaler.scale_ = scaler_params[1]
            image = scaler.transform(image)

            # reshape
            inputs[0] = image.reshape(old_shape)

            return (inputs, parameters)

    else:
        def scale_func(input_tuple):
            # get image
            inputs, parameters = input_tuple
            image = inputs[0]

            # flatten
            old_shape = image.shape
            image = image.reshape(-1, 1)

            # scale
            image = skpre.robust_scale(image, copy=False)

            # reshape
            inputs[0] = image.reshape(old_shape)

            return (inputs, parameters)

    def wrap(input_tuple):
        inputs, parameters = input_tuple
        outputs = []

        for inp in inputs:
            outputs.append(scale_func((inp, parameters))[0])

        return (outputs, parameters)

    return wrap


def simpleITK_rescale_intensitiy(minimum=0, maximum=1):

    if not isinstance(minimum, numbers.Number):
        raise TypeError("Minimum must be a number! Received: %s" % type(minimum))

    if not isinstance(maximum, numbers.Number):
        raise TypeError("Maximum must be a number! Received: %s" % type(maximum))

    rescale = sitk.RescaleIntensityImageFilter()
    rescale.SetOutputMinimum(minimum)
    rescale.SetOutputMaximum(maximum)

    def rescale_func(input_tuple):
        inputs, parameters = input_tuple

        inputs[0] = rescale.Execute(inputs[0])

        return (inputs, parameters)

    def wrap(input_tuple):
        inputs, parameters = input_tuple
        outputs = []

        for inp in inputs:
            outputs.append(rescale_func((inp, parameters))[0])

        return (outputs, parameters)

    return wrap


def numpy_pad(pixels, mode='reflect', input_only=True):

    if input_only:

        def pad_func(input_tuple):
            inputs, parameters = input_tuple

            inputs[0] = np.pad(inputs[0], pixels, mode=mode)

            return (inputs, parameters)

    else:

        def pad_func(input_tuple):
            inputs, parameters = input_tuple

            for i in range(len(inputs)):
                inputs[i] = np.pad(inputs[i], pixels, mode=mode)

            return (inputs, parameters)

    def wrap(input_tuple):
        inputs, parameters = input_tuple
        outputs = []

        for inp in inputs:
            outputs.append(pad_func((inp, parameters))[0])

        return (outputs, parameters)

    return wrap


def fuse_labels():
    return fuse_labels_greater_than(1)


def fuse_labels_greater_than(label_number):

    if not isinstance(label_number, numbers.Number):
        raise TypeError("Label Number must be a number! Received: %s" % type(label_number))

    def label_func(input_tuple):
        inputs, parameters = input_tuple

        inputs[1] = (inputs[1] > (label_number - 0.5)).astype(float)

        return (inputs, parameters)

    def wrap(input_tuple):
        inputs, parameters = input_tuple
        outputs = []

        for inp in inputs:
            outputs.append(label_func((inp, parameters))[0])

        return (outputs, parameters)

    return wrap


def keep_label(label_number):

    if not isinstance(label_number, numbers.Number):
        raise TypeError("Label Number must be a number! Received: %s" % type(label_number))

    def label_func(input_tuple):
        inputs, parameters = input_tuple

        inputs[1] = (np.isclose(inputs[1], label_number)).astype(float)

        return (inputs, parameters)

    def wrap(input_tuple):
        inputs, parameters = input_tuple
        outputs = []

        for inp in inputs:
            outputs.append(label_func((inp, parameters))[0])

        return (outputs, parameters)

    return wrap


def numpy_zoom(desired_size):

    if not isinstance(desired_size, collections.Sequence) and not isinstance(desired_size, np.ndarray):
        TypeError("Desired Size must be a sequence or array! Received: %s" % type(desired_size))

    desired_size = np.asarray(desired_size, dtype=np.int)

    def zoom_func(input_tuple):

        inputs, parameters = input_tuple

        zooms = desired_size / np.array(inputs[0].shape, dtype=np.float)

        inputs[0] = scipy.ndimage.zoom(inputs[0], zooms, order=1)  # order = 1 => biliniear interpolation
        inputs[1] = scipy.ndimage.zoom(inputs[1], zooms, order=0)  # order = 0 => nearest neighbour

        parameters["spacing"] = tuple(np.array(parameters["spacing"]) * (parameters["size"] / desired_size.astype(np.float)))
        parameters["size"] = tuple(desired_size)

        return (inputs, parameters)

    def wrap(input_tuple):
        inputs, parameters = input_tuple
        outputs = []

        for inp in inputs:
            outputs.append(zoom_func((inp, parameters))[0])

        return (outputs, parameters)

    return wrap


def numpy_random_zoom2D(probability, desired_size, lower_bound=0.9, upper_bound=1.1):

    if not isinstance(desired_size, collections.Sequence) and not isinstance(desired_size, np.ndarray):
        TypeError("Desired Size must be a sequence or array! Received: %s" % type(desired_size))

    desired_size = np.asarray(desired_size, dtype=np.int)

    def zoom_func(input_tuple, rand_val, skip_transform):
        inputs, parameters = input_tuple

        if(skip_transform):  # do not apply deformations always, just sometimes

            factor = rand_val * (upper_bound - lower_bound) + lower_bound

            if factor > 1.:

                zooms = (desired_size * factor) / np.array(inputs[0].shape, dtype=np.float)
                scaled_size = np.rint(desired_size * factor).astype(np.int)

                x_start = (scaled_size[0] - desired_size[0]) / 2
                y_start = (scaled_size[1] - desired_size[1]) / 2
                x_end = x_start + desired_size[0]
                y_end = y_start + desired_size[1]

                output = np.zeros(desired_size, dtype=inputs[0].dtype)

                image = scipy.ndimage.zoom(inputs[0], zooms, order=1)  # order = 1 => biliniear interpolation
                output[:, :] = image[x_start:x_end, y_start:y_end]
                inputs[0] = output

                output = np.zeros(desired_size, dtype=inputs[1].dtype)

                image = scipy.ndimage.zoom(inputs[1], zooms, order=0)  # order = 0 => nearest neighbour
                output[:, :] = image[x_start:x_end, y_start:y_end]
                inputs[1] = output

                parameters["spacing"] = tuple(np.array(parameters["spacing"]) * (parameters["size"] / scaled_size.astype(np.float)))
                parameters["size"] = tuple(desired_size)

            elif factor < 1.:

                zooms = (desired_size * factor) / np.array(inputs[0].shape, dtype=np.float)
                scaled_size = np.rint(desired_size * factor).astype(np.int)

                x_start = (desired_size[0] - scaled_size[0]) / 2
                y_start = (desired_size[1] - scaled_size[1]) / 2
                x_end = x_start + scaled_size[0]
                y_end = y_start + scaled_size[1]

                output = np.full(desired_size, np.min(inputs[0]), dtype=inputs[0].dtype)

                image = scipy.ndimage.zoom(inputs[0], zooms, order=1)  # order = 1 => biliniear interpolation
                output[x_start:x_end, y_start:y_end] = image[:, :]
                inputs[0] = output

                output = np.zeros(desired_size, dtype=inputs[1].dtype)

                image = scipy.ndimage.zoom(inputs[1], zooms, order=0)  # order = 0 => nearest neighbour
                output[x_start:x_end, y_start:y_end] = image[:, :]
                inputs[1] = output

                parameters["spacing"] = tuple(np.array(parameters["spacing"]) * (parameters["size"] / scaled_size.astype(np.float)))
                parameters["size"] = tuple(desired_size)
        else:

            zooms = desired_size / np.array(inputs[0].shape, dtype=np.float)

            inputs[0] = scipy.ndimage.zoom(inputs[0], zooms, order=1)  # order = 1 => biliniear interpolation
            inputs[1] = scipy.ndimage.zoom(inputs[1], zooms, order=0)  # order = 0 => nearest neighbour

            parameters["spacing"] = tuple(np.array(parameters["spacing"]) * (parameters["size"] / desired_size.astype(np.float)))
            parameters["size"] = tuple(desired_size)

        return (inputs, parameters)

    def wrap(input_tuple):
        inputs, parameters = input_tuple
        outputs = []

        rand_val = np.random.rand()
        skip_transform = np.random.rand() > probability

        for inp in inputs:
            outputs.append(zoom_func((inp, parameters), rand_val, skip_transform)[0])

        return (outputs, parameters)

    return wrap


def simpleITK_zoom(desired_size, desired_spacing):

    if not isinstance(desired_size, collections.Sequence) and not isinstance(desired_size, np.ndarray):
        TypeError("Desired Size must be a sequence or array! Received: %s" % type(desired_size))

    if not isinstance(desired_spacing, collections.Sequence) and not isinstance(desired_spacing, np.ndarray):
        TypeError("Desired Spacing must be a sequence or array! Received: %s" % type(desired_spacing))

    desired_size = np.asarray(desired_size, dtype=float)
    desired_spacing = np.asarray(desired_spacing, dtype=float)

    def zoom_func(input_tuple):
        inputs, parameters = input_tuple

        # make sure spacing is set correct
        # it gets lost during conversations to numpy arrays

        spacing = np.asarray(parameters["spacing"], dtype=float)
        inputs[0].SetSpacing(spacing)
        inputs[1].SetSpacing(spacing)

        # compute the new size of the volume
        # - size is measured in voxels
        # - spacing is measured in millimetres per voxel

        old_size = np.asarray(inputs[0].GetSize(), dtype=float)
        physical_size = old_size * spacing
        new_size = physical_size / desired_spacing  # mm / (mm / voxel) = voxel
        new_size = np.max([new_size, desired_size], axis=0)
        new_size = np.round(new_size)

        minmax = sitk.MinimumMaximumImageFilter()
        minmax.Execute(inputs[0])
        minimum = minmax.GetMinimum()

        # scale the image volume using linear interpolation

        resampler = sitk.ResampleImageFilter()
        resampler.SetReferenceImage(inputs[0])
        resampler.SetDefaultPixelValue(minimum)
        resampler.SetOutputSpacing(list(desired_spacing))
        resampler.SetSize(list(new_size.astype(int)))
        resampler.SetInterpolator(sitk.sitkLinear)

        inputs[0] = resampler.Execute(inputs[0])

        # scale the label volume using nearest neighbor interpolation

        resampler.SetInterpolator(sitk.sitkNearestNeighbor)
        resampler.SetDefaultPixelValue(0)

        inputs[1] = resampler.Execute(inputs[1])

        # if the volume is bigger than the desired_size (this may happen
        # depending on the desired_spacing) extract a new volume of the
        # desired size centered in the old volume

        volume_origin = np.round((new_size - desired_size) / 2.0).astype(int)

        extractor = sitk.RegionOfInterestImageFilter()
        extractor.SetSize(desired_size.astype(int))  # cast to list?
        extractor.SetIndex(volume_origin)  # cast to list?

        inputs[0] = extractor.Execute(inputs[0])
        inputs[1] = extractor.Execute(inputs[1])

        parameters["size"] = list(desired_size.astype(int))
        parameters["spacing"] = list(desired_spacing)

        return (inputs, parameters)

    def wrap(input_tuple):
        inputs, parameters = input_tuple
        outputs = []

        for inp in inputs:
            outputs.append(zoom_func((inp, parameters))[0])

        return (outputs, parameters)

    return wrap


def numpy_transpose(definition=None):

    if definition is None:
        def transpose_func(input_tuple):
            inputs, parameters = input_tuple

            for i in range(len(inputs)):
                inputs[i] = np.transpose(inputs[i])

            parameters["size"] = tuple(reversed(parameters["size"]))
            parameters["spacing"] = tuple(reversed(parameters["spacing"]))

            return (inputs, parameters)
    else:

        if not isinstance(definition, collections.Sequence) and not isinstance(definition, np.ndarray):
            TypeError("Definition must be a sequence or array! Received: %s" % type(definition))

        def transpose_func(input_tuple):
            inputs, parameters = input_tuple

            for i in range(len(inputs)):
                inputs[i] = np.transpose(inputs[i], definition)

            # switch the size and spacing parameters for the different
            # dimensions according to the transpose definition
            size = np.asarray(parameters["size"])
            spacing = np.asarray(parameters["spacing"])

            parameters["size"] = tuple(size[definition])
            parameters["spacing"] = tuple(spacing[definition])

            return (inputs, parameters)

    def wrap(input_tuple):
        inputs, parameters = input_tuple
        outputs = []

        for inp in inputs:
            outputs.append(transpose_func((inp, parameters))[0])

        return (outputs, parameters)

    return wrap


def numpy_to_simpleITK():

    warn("Converting from NumPy to SimpleITK reverses all axis!")

    def convert_func(input_tuple):
        inputs, parameters = input_tuple

        parameters["size"] = tuple(reversed(parameters["size"]))
        parameters["spacing"] = tuple(reversed(parameters["spacing"]))

        for i in range(len(inputs)):
            spacing = tuple([np.float64(x) for x in parameters["spacing"]])
            inputs[i] = sitk.GetImageFromArray(inputs[i])
            inputs[i].SetSpacing(spacing)

        return (inputs, parameters)

    def wrap(input_tuple):
        inputs, parameters = input_tuple
        outputs = []

        for inp in inputs:
            outputs.append(convert_func((inp, parameters))[0])

        return (outputs, parameters)

    return wrap


def simpleITK_to_numpy():

    warn("Converting from SimpleITK to NumPy reverses all axis!")

    def convert_func(input_tuple):
        inputs, parameters = input_tuple

        for i in range(len(inputs)):
            inputs[i] = sitk.GetArrayFromImage(inputs[i])

        parameters["size"] = tuple(reversed(parameters["size"]))
        parameters["spacing"] = tuple(reversed(parameters["spacing"]))

        return (inputs, parameters)

    def wrap(input_tuple):
        inputs, parameters = input_tuple
        outputs = []

        for inp in inputs:
            outputs.append(convert_func((inp, parameters))[0])

        return (outputs, parameters)

    return wrap


def numpy_data_normalization(mean=None, standard_deviation=None):

    if mean is None and standard_deviation is None:
        def norm_func(input_tuple):
            inputs, parameters = input_tuple

            mean = np.mean(inputs[0])
            std = np.std(inputs[0])

            inputs[0] -= mean
            inputs[0] /= std

            return (inputs, parameters)

    elif mean is not None and standard_deviation is not None:

        if not isinstance(mean, numbers.Number):
            raise TypeError("Mean must be a number! Received: %s" % type(mean))

        if not isinstance(standard_deviation, numbers.Number):
            raise TypeError("Standard Deviation must be a number! Received: %s" % type(standard_deviation))

        def norm_func(input_tuple):
            inputs, parameters = input_tuple

            inputs[0] -= mean
            inputs[0] /= standard_deviation

            return (inputs, parameters)

    else:
        raise ValueError("Please provide a value each for mean and standard "
                         "deviation or non at all. If no value is provided "
                         "the mean and standard deviation of the current "
                         "volume will be used.")


def numpy_histogram_matching():

    def match_func(source_tuple, template_tuple):
        """
        Adjust the pixel values of a grayscale image such that its histogram
        matches that of a target image

        Arguments:
        -----------
            source: np.ndarray
                Image to transform; the histogram is computed over the flattened
                array
            template: np.ndarray
                Template image; can have different dimensions to source
        Returns:
        -----------
            matched: np.ndarray
                The transformed output image
        """

        source_inputs, source_parameters = source_tuple
        template_inputs, _ = template_tuple

        source = source_inputs[0]
        template = template_inputs[0]

        old_shape = source.shape
        source = source.ravel()
        template = template.ravel()

        # get the set of unique pixel values and their corresponding indices and
        # counts
        _, indices, s_counts = np.unique(source, return_inverse=True,
                                         return_counts=True)
        t_values, t_counts = np.unique(template, return_counts=True)

        # take the cumsum of the counts and normalize by the number of pixels to
        # get the empirical cumulative distribution functions for the source and
        # template images (maps pixel value --> quantile)
        s_quantiles = np.cumsum(s_counts).astype(np.float64)
        s_quantiles /= s_quantiles[-1]
        t_quantiles = np.cumsum(t_counts).astype(np.float64)
        t_quantiles /= t_quantiles[-1]

        # interpolate linearly to find the pixel values in the template image
        # that correspond most closely to the quantiles in the source image
        interpolated_s_values = np.interp(s_quantiles, t_quantiles, t_values)

        source_inputs[0] = interpolated_s_values[indices].reshape(old_shape)

        return (source_inputs, source_parameters)

    return match_func


def simpleITK_deform(probability, control_points, std_def):

    def deform_func(input_tuple, skip_transform, params, transform):

        inputs, parameters = input_tuple

        if(not skip_transform):  # do not apply deformations always, just sometimes

            minmax = sitk.MinimumMaximumImageFilter()
            minmax.Execute(inputs[0])
            minimum = minmax.GetMinimum()

            if inputs[0].GetDimension() == 3:
                params[0:(len(params) / 3)] = 0.0  # remove z deformations! The resolution in z is too bad

            transform.SetParameters(tuple(params))

            resampler = sitk.ResampleImageFilter()
            resampler.SetReferenceImage(inputs[0])
            resampler.SetDefaultPixelValue(minimum)
            resampler.SetTransform(transform)
            resampler.SetInterpolator(sitk.sitkLinear)

            inputs[0] = resampler.Execute(inputs[0])

            minmax.Execute(inputs[1])
            minimum = minmax.GetMinimum()

            resampler.SetDefaultPixelValue(minimum)
            resampler.SetInterpolator(sitk.sitkNearestNeighbor)

            inputs[1] = resampler.Execute(inputs[1])

        return (inputs, parameters)

    def wrap(input_tuple):
        inputs, parameters = input_tuple
        outputs = []

        skip_transform = np.random.rand() > probability

        idx = len(inputs) // 2
        mesh_size = [control_points] * inputs[idx][0].GetDimension()

        transform = sitk.BSplineTransformInitializer(inputs[idx][0], mesh_size)

        params = np.asarray(transform.GetParameters(), dtype=float)
        params = params + np.random.randn(params.size) * std_def

        for inp in inputs:
            outputs.append(deform_func((inp, parameters), skip_transform, params, transform)[0])

        return (outputs, parameters)

    return wrap


def numpy_dont_use(prototxt, snapshot_directory,
                   snapshot=0,
                   volume_size=[],
                   iterations=0,
                   batch_size=1,
                   base_learning_rate=0.001,
                   weight_decay=0.005,
                   step_size=20000,
                   momentum=0.99):

    # Write a temporary solver text file because pycaffe is stupid
    with open("solver.prototxt", 'w') as f:
        f.write("net: \"" + prototxt + "\" \n")
        f.write("base_lr: " + str(base_learning_rate) + " \n")
        f.write("momentum: " + momentum + " \n")
        f.write("weight_decay: " + weight_decay + " \n")
        f.write("lr_policy: \"step\" \n")
        f.write("stepsize: " + step_size + " \n")
        f.write("gamma: 0.1 \n")
        f.write("display: 1 \n")
        f.write("snapshot: 500 \n")
        f.write("snapshot_prefix: \"" + snapshot_directory + "\" \n")

    f.close()
    solver = caffe.SGDSolver("solver.prototxt")
    os.remove("solver.prototxt")

    if snapshot > 0:
        solver.restore(snapshot_directory + "_iter_" + str(snapshot) + ".solverstate")

    # plt.ion()

    dimensions = [batch_size, 1]
    dimensions = dimensions.extend(volume_size)

    images = np.zeros(dimensions, dtype=np.float32)
    labels = np.zeros(dimensions, dtype=np.float32)
    weights = np.zeros(dimensions, dtype=np.float32)

    def train_func(source):

        for i in range(batch_size):

            try:
                inputs, parameters = source.next()
            except StopIteration:
                raise StopIteration

            images[i, 0, :] = inputs[0]
            labels[i, 0, :] = inputs[1]

            for label in np.unique(inputs[1]):
                weights[i, 0, inputs[1] == label] = np.prod(inputs[1].shape) / np.sum(inputs[1] == label).astype(np.float32)

        solver.net.blobs['data'].data[...] = images
        solver.net.blobs['label'].data[...] = labels
        solver.net.blobs['labelWeight'].data[...] = weights

        solver.step(1)  # this does the training

        return solver.net.blobs['loss'].data

    return train_func
