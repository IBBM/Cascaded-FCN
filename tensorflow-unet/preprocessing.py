import os.path
import pickle

import nibabel
import numpy as np
import sklearn.preprocessing as prep

import processing.pipeline as pipe
import processing.transformations.segmentation as seg
import sys
import oversampler

import matplotlib.pyplot as plt

import scipy
np.set_printoptions(threshold=np.nan)

countg = 0
countg_reg = 0
countg_liv = 0
def custom_load_slices(label_of_interest=2, label_required=1, min_frequency=0.8, max_tries=10, slice_type='axial', n_neighboringslices=5, oversampler=None):

    counter = [float(0)]  # counts the number of slices with the label_of_interest
    # a list is necessary because python's scoping rules (are demented and)
    # only allow reading, but not modifying variables defined in the outter scope

    total = [0.03125]  # small float to avoid division by zero. it's 1/(2^5), I thought base 2 would be neat

    def get_slice_from_type(image_obj,slice=None,slice_type='axial'):

        if slice_type == 'axial':
            limit = image_obj.header.get_data_shape()[-1]
            slice_index = np.random.randint(limit) if slice==None else slice
            slc = image_obj.dataobj[..., slice_index]
            # print ("image_obj.header.get_data_shape()", image_obj.header.get_data_shape(), slice_index)
        elif slice_type == 'sagital':
            limit = image_obj.header.get_data_shape()[-2]
            slice_index = np.random.randint(limit) if slice==None else slice
            slc = image_obj.dataobj[:, slice_index, :]
        elif slice_type == 'coronal':
            limit = image_obj.header.get_data_shape()[0]
            slice_index = np.random.randint(limit) if slice==None else slice
            slc = image_obj.dataobj[slice_index, ...]
        else:
            print 'No slice type defined'

        return slc, slice_index

    def load_pair(image_volume, label_volume, idx):
        outputs = []
        # image slice first
        outputs.append(np.asarray(get_slice_from_type(image_volume,slice=idx,slice_type=slice_type)[0]).astype(np.float32))

        label_slice = np.asarray(get_slice_from_type(label_volume,slice=idx,slice_type=slice_type)[0]).astype(np.float32)
        # label_slice = clean_labels(label_slice)
        outputs.append(label_slice)

        return outputs

    def clean_labels(label):
        # print (label.shape)
        if label.shape[0] == 2:
            label = np.argmax(label, 0)
            label = scipy.misc.imresize(label, (512,512), interp='nearest')

        # print (label.shape)
        return label

    def load_func(input_tuple):
        global countg
        global countg_reg
        global countg_liv

        inputs, parameters = input_tuple
        middle_index = int(n_neighboringslices / 2)

        label_volume = nibabel.load(inputs[1])
        label_slice, slice_index = np.asarray(get_slice_from_type(label_volume, slice_type=slice_type))

        # label_slice = clean_labels(label_slice)

        lc = 0
        while True:
            lc += 1

            # found lesion in the current slice
            if label_of_interest in label_slice and (not oversampler or oversampler.slice_loadprob_deviation(inputs[1], slice_index) < 0.02):
                counter[0] += 1.
                # if 2 in label_slice:
                countg_reg += 1
                break

            # evaluates to true if either the max_lesion_tries are exceeded or the other cond is met
            if lc > max_tries:
                if label_required in label_slice:
                    countg += 1
                    break

            label_slice, slice_index = np.asarray(get_slice_from_type(label_volume, slice_type=slice_type))

        total[0] += 1

        image_volume = nibabel.load(inputs[0])

        output_arr = []
        limit = image_volume.header.get_data_shape()[-1]

        for idx in range(n_neighboringslices):
            new_index = slice_index - middle_index + idx

            if new_index < 0:
                new_index = 0
            elif new_index >= limit - 1:
                new_index = limit - 1

            output_arr.append(load_pair(image_volume, label_volume, new_index))

        # inform the sampler that a specific slice was loaded
        if oversampler:
            oversampler.loaded(inputs[1], slice_index, output_arr[middle_index][0])
        
        parameters["spacing"] = image_volume.header.get_zooms()[:2]  # the last value is the time between scans, no need to keep it
        parameters["original_spacing"] = parameters["spacing"]

        parameters["size"] = image_volume.header.get_data_shape()[:2]
        parameters["original_size"] = parameters["size"]

        parameters["slices_total"] = total[0]
        parameters["slices_label_of_interest"] = counter[0]
        

        return (output_arr, parameters)

    return load_func


def get_scaling_dict(name, data_dir):

    name = name + '.p'

    if os.path.isfile(name):

        print 'Found dictionary file "{0}". Loading dictionary...'.format(name)

        infile = open(name, 'rb')
        dictionary = pickle.load(infile)
        infile.close()

    else:

        print 'No dictionary file "{0}" found. Creating dictionary...'.format(name)

        dictionary = {}

        scale = prep.RobustScaler(copy=False)
        #LITS
        dct = pipe.Reader(seg.file_paths_ordered(data_dir, iterations=1, image_identifier='volume', label_identifier='segmentation'), name="Read File Names")
        #3dircad
        #dct = pipe.Reader(seg.file_paths_ordered(data_dir, iterations=1, image_identifier='image', label_identifier='label'), name="Read File Names")
        dct = dct.transform(seg.numpy_load_volume(), name="Load Volumes")
        dct = dct.transform(seg.numpy_clip(-100, 400), name="Clip Pixel Values")
        dct = dct.transform(seg.numpy_mask_background(0, -100.))

        for inputs, parameters in dct:

            file_name = parameters["file_names"][0]
            volume = inputs[0]
            volume = np.ma.masked_values(volume, -100.)
            volume = np.ma.masked_values(volume, 400., copy=False)
            volume = np.ma.compressed(volume)
            volume = volume.reshape(-1, 1)
            print 'Filename %s has %s labels' % (file_name,np.unique(inputs[1], return_counts=True))
            scale.fit(volume)
            dictionary[file_name] = (scale.center_, scale.scale_)


        outfile = open(name, 'wb')
        pickle.dump(dictionary, outfile)
        outfile.close()

    print 'Dictionary loaded.'

    return dictionary


def get_crop_dict(name, data_dir):

    name = name + '.p'

    if os.path.isfile(name):

        print 'Found dictionary file "{0}". Loading dictionary...'.format(name)

        infile = open(name, 'rb')
        dictionary = pickle.load(infile)
        infile.close()

    else:

        print 'No dictionary file "{0}" found. Creating dictionary...'.format(name)

        dictionary = {}

        dct = pipe.Reader(seg.file_paths_ordered(data_dir, iterations=1, image_identifier='volume', label_identifier='segmentation'), name="Read File Names")
        dct = dct.transform(seg.numpy_load_volume(), name="Load Volumes")

        for inputs, parameters in dct:

            file_name = parameters["file_names"][0]
            volume = inputs[1]

            x = np.any(volume, axis=(1, 2))
            y = np.any(volume, axis=(0, 2))
            z = np.any(volume, axis=(0, 1))

            xmin, xmax = np.where(x)[0][[0, -1]]
            ymin, ymax = np.where(y)[0][[0, -1]]
            zmin, zmax = np.where(z)[0][[0, -1]]

            dictionary[file_name] = (xmin, xmax, ymin, ymax, zmin, zmax)

        outfile = open(name, 'wb')
        pickle.dump(dictionary, outfile)
        outfile.close()

    print 'Dictionary loaded.'

    return dictionary


def crop(dictionary=None):

    def crop_func(input_tuple):

        # get image and file name
        inputs, parameters = input_tuple

        image = inputs[0]
        labels = inputs[1]

        parameters['original_labels'].append(inputs[1])
        parameters['original_images'].append(inputs[0])

        if len(inputs) > 2:
            parameters['original_labels_orig'].append(inputs[2])
        else:
            parameters['original_labels_orig'].append(inputs[1])

        shape = image.shape

        x = np.any(labels, axis=1)
        y = np.any(labels, axis=0)

        x_where = np.where(x)
        if len(x_where) == 0 or len(x_where[0]) == 0:
            return (inputs, parameters)
        xmin, xmax = x_where[0][[0, -1]]

        y_where = np.where(y)
        if len(y_where) == 0 or len(y_where[0]) == 0:
            return (inputs, parameters)
        ymin, ymax = y_where[0][[0, -1]]

        x_length = xmax - xmin
        y_length = ymax - ymin

        if x_length < 60:
            xmin -= (60 - x_length) / 2
            xmax += (60 - x_length + 1) / 2
            x_length = xmax - xmin

        if y_length < 60:
            ymin -= (60 - y_length) / 2
            ymax += (60 - y_length + 1) / 2
            y_length = ymax - ymin

        # find and extend the shorter side for square crop
        if x_length < y_length:

            # find new x_length. extend xmin and xmax for centered crop placement.
            xmin -= (y_length - x_length) / 2
            xmax += (y_length - x_length + 1) / 2
            x_length = y_length

        else:

            # find new y_length. extend ymin and ymax for centered crop placement.
            ymin -= (x_length - y_length) / 2
            ymax += (x_length - y_length + 1) / 2
            y_length = x_length

        # start and end on the new canvas. if the centered crop extends over
        # the boarders of the old canvas those parts will be filled with
        # the minimum value.
        xstart = 0
        xend = x_length

        ystart = 0
        yend = y_length

        if xmin < 0:
            xstart = abs(xmin)
            xmin = 0
        if xmax > shape[0]:
            xend = xend - (xmax - shape[0])
            xmax = shape[0]

        if ymin < 0:
            ystart = abs(ymin)
            ymin = 0
        if ymax > shape[1]:
            yend = yend - (ymax - shape[1])
            ymax = shape[1]

        fill_value = np.min(image[xmin:xmax, ymin:ymax])

        image_crop = np.full((x_length, y_length), fill_value, dtype=np.float)
        image_crop[xstart:xend, ystart:yend] = image[xmin:xmax, ymin:ymax]

        labels_crop = np.full((x_length, y_length), 0, dtype=np.float)
        labels_crop[xstart:xend, ystart:yend] = labels[xmin:xmax, ymin:ymax]

        # labels_crop2 = np.full((x_length, y_length), 0, dtype=np.float)
        # labels_crop2[xstart:xend, ystart:yend] = inputs[2][xmin:xmax, ymin:ymax]

        # crop
        inputs[0] = image_crop
        inputs[1] = labels_crop
        # inputs[2] = labels_crop2

        parameters['crop_indices'] = [xmin, xmax, ymin, ymax]

        return (inputs, parameters)

    def wrap(input_tuple):
        inputs, parameters = input_tuple
        outputs = []

        parameters['original_labels'] = []
        parameters['original_images'] = []
        parameters['original_labels_orig'] = []

        # print ("applyincg crop", len(inputs))
        for inp in inputs:
            outputs.append(crop_func((inp, parameters))[0])

        return (outputs, parameters)

    return wrap


def training(train_data_dir,slice_type='axial',n_neighboringslices=5,image_size=320, oversample=True, label_of_interest=2, label_required=1, max_tries=500, min_frequency=1.0, fuse_labels=False, apply_crop=True):

    dict_name = os.path.basename(os.path.dirname(train_data_dir))
    # scaling_dict = get_scaling_dict(dict_name + "_scaling", train_data_dir)

    sampler = None
    # Instantiate the oversampling class which is used in the custom slice loader to oversample outliers
    if oversample:
        sampler = oversampler.Oversampler(label_of_interest=2, label_required=label_required, train_data_dir=train_data_dir, debug=False)

    # crop_dict = get_crop_dict(dict_name + "_cropping", train_data_dir)

    # Load slice
    tr = pipe.Reader(seg.file_paths_random(train_data_dir, iterations=0, image_identifier='volume', label_identifier='segmentation'), name="Read File Names")
    tr = tr.transform(custom_load_slices(label_of_interest=label_of_interest, label_required=label_required, min_frequency=min_frequency, max_tries=max_tries,slice_type=slice_type, n_neighboringslices=n_neighboringslices, oversampler=sampler), name="Load Slices")

    # Random transformations
    tr = tr.transform(seg.numpy_rotation2D(1.0, upper_bound=90, min_val=-350.), name="Random Rotation")
    tr = tr.transform(seg.numpy_random_zoom2D(1.0, [image_size, image_size], lower_bound=0.8, upper_bound=1.2), name="Random Slice Scaling")
    tr = tr.transform(seg.numpy_translation2D(0.5, factor=0.25, default_border=0.25, label_of_interest=1), name="Random Translation")

    tr = _test_val_tail(tr, None, image_size=image_size, label_of_interest=label_of_interest, label_required=label_required,fuse_labels=fuse_labels, apply_crop=apply_crop)

    return tr


def validation(validation_data_dir,slice_type='axial',n_neighboringslices=5,image_size=320, label_of_interest=2, label_required=1, max_tries=100, fuse_labels=False, apply_crop=False):

    dict_name = os.path.basename(os.path.dirname(validation_data_dir))
    # scaling_dict = get_scaling_dict(dict_name + "_scaling", validation_data_dir)
    # crop_dict = get_crop_dict(dict_name + "_cropping", validation_data_dir)

    vld = pipe.Reader(seg.file_paths_random(validation_data_dir, iterations=0, image_identifier='volume', label_identifier='segmentation'), name="Read File Names")
    vld = vld.transform(custom_load_slices(label_of_interest=label_of_interest, label_required=label_required, min_frequency=1.0, max_tries=max_tries,slice_type=slice_type, n_neighboringslices=n_neighboringslices), name="Load Slices")
    vld = _test_val_tail(vld, None, image_size=image_size, label_of_interest=label_of_interest, label_required=label_required,fuse_labels=fuse_labels, apply_crop=apply_crop)

    return vld


def test(test_data_dir, label_of_interest=2, label_required=1,fuse_labels=False):

    dict_name = os.path.basename(os.path.dirname(test_data_dir))
    # scaling_dict = get_scaling_dict(dict_name + "_scaling", test_data_dir)
    # crop_dict = get_crop_dict(dict_name + "_cropping", test_data_dir)

    tst = pipe.Reader(seg.file_paths_ordered(test_data_dir, iterations=1, image_identifier='volume', label_identifier='segmentation'), name="Read File Names")
    tst = tst.run_on(1)
    tst = tst.multiply(seg.numpy_load_all_slices(), name="Load Slices")
    # tst = _test_val_tail(tst, scaling_dict, crop_dict)
    tst = _test_val_tail(tst, None, label_of_interest=label_of_interest, label_required=label_required,fuse_labels=fuse_labels)
    tst = tst.run_on(4)

    return tst

def custom_numpy_load_all_slices(n_neighboringslices=1):

    def load_func(input_tuple):

        inputs, parameters = input_tuple

        niftis = [nibabel.load(inpt) for inpt in inputs]

        volumes = [np.asarray(nifti.dataobj).astype(np.float32) for nifti in niftis]

        nifti = niftis[0]

        parameters["spacing"] = nifti.header.get_zooms()[:2]  # the last value is the time between scans, no need to keep it
        parameters["original_spacing"] = parameters["spacing"]

        parameters["size"] = nifti.header.get_data_shape()[:2]
        parameters["original_size"] = parameters["size"]

        parameters["header"] = nifti.header

        print ("loading all slices", len(volumes), len(niftis), nifti.header.get_data_shape()[:3], niftis[1].header.get_data_shape()[:3], niftis[2].header.get_data_shape()[:3])

        middle_index = n_neighboringslices // 2
        limit = nifti.header.get_data_shape()[-1]

        for slice_index in xrange(limit):
            output_arr = []

            for idx in range(n_neighboringslices):
                new_index = slice_index - middle_index + idx

                if new_index < 0:
                    new_index = 0
                elif new_index >= limit - 1:
                    new_index = limit - 1

                outp = [volume[..., new_index] for volume in volumes]
                output_arr.append(outp)

            # only use one label
            for out in output_arr:
                out[1] = output_arr[n_neighboringslices//2][1]

            yield (output_arr, parameters.copy())

    return load_func

def generate_volumes(test_data_dir, n_neighboringslices=1, new_postfix='prediction', image_size=320):
    dict_name = os.path.basename(os.path.dirname(test_data_dir))

    tst = pipe.Reader(seg.file_paths_ordered(test_data_dir, iterations=1, image_identifier='volume', label_identifier='segmentation', new_postfix=new_postfix), name="Read File Names")
    tst = tst.multiply(custom_numpy_load_all_slices(n_neighboringslices=n_neighboringslices), name="Load Slices")
    tst = _test_val_tail(tst, None, label_of_interest=1, label_required=0, fuse_labels=True, apply_crop=False, image_size=image_size)

    return tst

def generate_volumes_lesion(test_data_dir, n_neighboringslices=1, new_postfix='', prediction_postfix='prediction', image_size=320):
    dict_name = os.path.basename(os.path.dirname(test_data_dir))

    tst = pipe.Reader(seg.file_paths_ordered(test_data_dir, iterations=1, image_identifier='volume', label_identifier='segmentation', prediction_postfix=prediction_postfix, new_postfix=new_postfix), name="Read File Names")
    tst = tst.multiply(custom_numpy_load_all_slices(n_neighboringslices=n_neighboringslices), name="Load Slices")
    tst = _test_val_tail(tst, None, label_of_interest=2, label_required=1, fuse_labels=False, apply_crop=True, discard_labels=False, image_size=image_size)

    return tst

def _test_val_tail(node, scaling_dict, image_size=320, label_of_interest=2, label_required=1, fuse_labels=False, apply_crop=False, discard_labels=True):
    if apply_crop:
        node = node.transform(crop())

    node = node.transform(seg.numpy_clip(-300, 500), name="Clip Pixel Values")

    if fuse_labels:
        node = node.transform(seg.fuse_labels_greater_than(label_of_interest), name="Fuse Labels")
    else:
        node = node.transform(seg.numpy_mask_background(0, -350.))

    node = node.transform(seg.numpy_static_scaler(-350., 500.), name="Static Scaling")

    # Prepare as network input
    node = node.transform(seg.numpy_zoom([image_size, image_size]), name="Scale Slice")
    # node = node.transform(seg.numpy_transpose(), name="Transpose")

    if discard_labels:
        node = node.transform(seg.keep_label(label_of_interest), name="Filter Labels")

    return node
