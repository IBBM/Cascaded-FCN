import os.path
import argparse

import numpy as np
import tensorflow as tf

import preprocessing
import unetv2 as unet

import sys

import matplotlib.pyplot as plt

import medpy
import medpy.filter

import scipy

import utils.niifs as niifs
import utils.niiplot as niiplot
import utils.niismooth as niismooth

# Command Line Arguments
parser = argparse.ArgumentParser()

parser.add_argument('--model', '-m', help='the model file.')
parser.add_argument('--data_directory', '-d', help='the directory which contains the validation data set.')
parser.add_argument('--out_postfix', '-o', help='this postfix will be added to all files.', default="_prediction")
parser.add_argument('--liver_mode', '-l', help='the liver mode determines if the liver or the lesion segmentations are generated.', default=False)

args = parser.parse_args()

if args.liver_mode == "true" or args.liver_mode == "1" or args.liver_mode == "True":
    args.liver_mode = True

class Generator:

    def __init__(self, args, smooth=True, plot_mode=True, remove_small_objects=True):
        self.remove_small_objects = remove_small_objects
        self.args = args
        self.liver_mode = args.liver_mode

        self._prepare_filepaths()

        if self.liver_mode:
            self._set_liver_params()
        else:
            self._set_lesion_params()

        self.batch_size = 1

        self.size_orig = 512
        self.middle_index = self.n_neighboringslices // 2

        self._tf_setup()
        self._pipeline_setup()

        self.plot_mode = plot_mode
        debug_lvl = 3
        if plot_mode:
            self.plotter = niiplot.Overlay(debug_lvl, label_max=2, label_min=0)

        self.smooth = smooth
        if self.smooth:
            self.smoother = niismooth.Smoothutil()

        self.run()

    def _prepare_filepaths(self):
        # Training Parameters
        self.model = self.args.model
        if os.path.isfile(self.model + '.index'):
            print ("Model {} found".format(self.model))
        else:
            print ("Unable to find model {}. Still continuing in case the model format is not known.".format(self.model))

        self.data_dir = self.args.data_directory
        self.data_dir = os.path.join(self.data_dir, '')

    def _set_liver_params(self):
        print ("Generating liver segmentations!")
        self.n_neighboringslices = 1
        self.size_training = 400
        self.n_filters = 80

    def _set_lesion_params(self):
        print ("Generating lesion segmentations!")
        self.n_neighboringslices = 3
        # Todo, remove that override
        self.args.out_postfix = '_prediction_les'
        self.liver_postfix = '__prediction'
        self.size_training = 256
        self.n_filters = 80

    def _tf_setup(self):
        tf.reset_default_graph()

        # Inputs
        with tf.name_scope('inputs'):
            self.images_var = tf.placeholder(tf.float32, shape=(1, self.size_training, self.size_training, self.n_neighboringslices), name='images')
            self.is_training = tf.placeholder(tf.bool, shape=[], name='is_training')

        # Model
        print ("Setting up the architecture")

        logits_op = unet.inference(self.images_var, self.is_training, n_filters=self.n_filters)
        self.softmax_op = tf.nn.softmax(logits_op)

        self.image_input = np.zeros([self.batch_size, self.size_training, self.size_training, self.n_neighboringslices], dtype=np.float32)

        # Saver
        self.saver = tf.train.Saver(max_to_keep=100, keep_checkpoint_every_n_hours=1)
        self.sess = tf.Session()

        print ("Loading model {}".format(self.model))
        self.saver.restore(self.sess, self.model)

    def _pipeline_setup(self):
        if self.liver_mode:
            self.source = preprocessing.generate_volumes(self.data_dir, new_postfix=self.args.out_postfix, image_size=self.size_training)
        else:
            self.source = preprocessing.generate_volumes_lesion(self.data_dir, self.n_neighboringslices, prediction_postfix=self.liver_postfix, new_postfix=self.args.out_postfix, image_size=self.size_training)

    def _get_next_input(self):
        try:
            inputs, parameters = self.source.next()
        except StopIteration:
            # In case the pipeline returned a wrong amount of slices or something went wrong during processing or the volume ran out.
            if self.niivol:
                # Also saves the file.
                self.niivol.check_error()

            # Return value forces a break. Generation most likely unsucessful for some Volumes.
            return -1, -1

        return inputs, parameters

    def _handle_new_volume(self, parameters):
        # If the name changes, a new volume has to be constructed.
        if not self.niivol or parameters["file_names"][1] != self.niivol.name:
            if self.niivol:
                # Also savess the file
                self.niivol.check_error()

                # print ("parameters", parameters, self.niivol.name)

            parameters = parameters.copy()
            self.niivol = niifs.Volume(parameters["header"], name=parameters["file_names"][1], target_dir=self.data_dir, postfix=self.args.out_postfix, largest_connected_component=self.liver_mode)

    def _run_inference(self, inputs):
        for idx in range(self.n_neighboringslices):
            self.image_input[0, :, :, idx] = inputs[idx][0]

        feed_dict = {
            self.images_var: self.image_input,
            self.is_training: False
        }

        predictions = self.sess.run(self.softmax_op, feed_dict=feed_dict)
        return np.transpose(predictions, (3, 2, 1, 0))  # Change from batch, height, width, likelihoods to likelihoods, width, height, batch

    def _undo_cropping(self, parameters, predictions):
        resized_predictions = np.zeros((self.size_orig, self.size_orig), dtype=np.float32)
        # Retrieve the crop-indices from the pipeline
        crop_indices = parameters["crop_indices"]
        side_lengths = [abs(crop_indices[0]-crop_indices[1]), abs(crop_indices[2] - crop_indices[3])]

        # Undo the scaling to self.size_training pixels side length
        zooms = np.asarray(side_lengths, dtype=np.float32) / np.asarray([float(self.size_training), float(self.size_training)], dtype=np.float32)
        temp_slice = np.transpose(predictions[1, :, :, 0])

        # best interpolation so far
        temp_slice = scipy.ndimage.zoom(temp_slice, zooms, order=1)

        # Fit the prediction into the full sized slice
        resized_predictions[crop_indices[0]:crop_indices[1], crop_indices[2]:crop_indices[3]] = temp_slice
        # resized_predictions = np.transpose(resized_predictions)
        return resized_predictions

    def run(self):
        i = 0
        desired_size = np.asarray((self.size_orig,self.size_orig), dtype=np.int32)
        self.niivol = None

        while True:
            inputs, parameters = self._get_next_input()
            if inputs == -1:
                print ("End of pipeline reached. Breaking out of main loop.")
                break

            # The pipeline only provides slices and does not notify us if the volume changes.
            # Therefore we have to monitor the changes in the parameters.
            self._handle_new_volume(parameters)

            mid_label = inputs[self.middle_index][1]
            quantified = np.zeros((self.size_orig, self.size_orig), dtype=np.int8)
            # Only run predictions if we are generating the liver or if there is a liver segmentation available in the current slice.
            if self.liver_mode or 1 in mid_label:

                predictions = self._run_inference(inputs)

                if not self.liver_mode:
                    resized_predictions = self._undo_cropping(parameters, predictions)
                else:
                    resized_predictions = np.zeros((self.size_orig, self.size_orig), dtype=np.float32)
                    zooms = desired_size / np.array(predictions[1, :, :, 0].shape, dtype=np.float32)
                    resized_predictions = scipy.ndimage.zoom(predictions[1, :, :, 0], zooms, order=1)

                """
                Values found by empirical approach.
                """
                if self.smooth:
                    resized_convolved = self.smoother.convolve2d(resized_predictions)

                    quantified[resized_convolved > 0.51] = 1
                    quantified[resized_convolved <= 0.51] = 0

                    # If the network is sure that a pixel does or doesn't belong to the segmentation, override the smoothed values.
                    quantified[resized_predictions <= 0.44] = 0
                    quantified[resized_predictions >= 0.57] = 1
                else:
                    quantified[resized_predictions > 0.51] = 1
                    quantified[resized_predictions <= 0.51] = 0

                if self.remove_small_objects:
                    quantified = medpy.filter.binary.size_threshold(quantified, 16, 'lt')
            
            if not self.liver_mode:
                # Readd the segmentations with the original liver label, loaded from the volume and the quantified predictions.
                res_label = parameters['original_labels'][self.middle_index]
                res_label[res_label > 1] = 1
                quantified = quantified + res_label

            # Undo the transpose from the trainings pipeline
            if self.liver_mode:
                quantified = np.transpose(quantified)

            # Plot functions for debugging purposes.
            if self.plot_mode and self.liver_mode:
                self.plotter.segmentations_over_slice(scipy.misc.imresize(inputs[self.middle_index][0], (self.size_orig, self.size_orig), interp='nearest').astype(np.float32), [quantified, inputs[self.middle_index][2] ], segmentation_labels=['Quantified', 'Ground Truth'])  
            
            if self.plot_mode and not self.liver_mode:
                self.plotter.segmentations_over_slice(scipy.misc.imresize(parameters['original_images'][self.middle_index], (self.size_orig, self.size_orig), interp='nearest').astype(np.float32), [quantified, parameters['original_labels_orig'][self.middle_index] ], segmentation_labels=['Quantified', 'Ground Truth'])
                # self.plotter.segmentations_over_slice(scipy.misc.imresize(parameters['original_images'][self.middle_index], (self.size_orig, self.size_orig), interp='nearest').astype(np.float32), [quantified], segmentation_labels=['Quantified'])

            # Adds the slice to the volume which will be stored by the niivol.save call once the volume is finished.
            self.niivol.add_slice(quantified)

            i += 1
            if i % 500 == 0:
                print ("{} Slices processed".format(i))

Generator(args)
# After Test
print ("Done!")
