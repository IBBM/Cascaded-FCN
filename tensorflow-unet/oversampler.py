from __future__ import print_function

import numpy as np
import copy
import pickle

import processing.pipeline as pipe
import processing.transformations.segmentation as seg
import os.path

import sys

from pprint import pprint
import matplotlib.pyplot as plt

import math

def imhist(im):
    # calculates normalized histogram of an image
    m = len(im)

    h = [0.0] * 256
    for j in range(m):
        h[int(im[j])]+=1

    return np.array(h)/(m)

def cumsum(h):
    # finds cumulative sum of a numpy array, list
    return [sum(h[:i+1]) for i in range(len(h))]

def histeq(im):
    im = np.array(im)
    #calculate Histogram
    h = imhist(im)
    cdf = np.array(cumsum(h)) #cumulative distribution function
    sk = np.uint8(255 * cdf) #finding transfer function values
    s = len(im)
    Y = np.zeros_like(im)
    # applying transfered values for each pixels
    for j in range(s):
        Y[j] = sk[int(im[j])]

    # H = imhist(Y)
    #return transformed image, original and new istogram, 
    # and transform function
    return Y

class Oversampler(object):
    def __init__(self, label_of_interest=2, label_required=1, n_classes=80, sort_by='median', debug=True, train_data_dir='./', plot=False):
        self.total_slices = 0

        self.label_of_interest = label_of_interest
        self.label_required = label_required
        self.n_classes = n_classes
        self.cache = {}

        self.train_data_dir = train_data_dir

        self.sort_by = sort_by

        self.buckets = [{'slices': [], 'intensities': [], 'median_intensities': [], 'lesion_sizes': [], 'upper_intensity': 0, 'lower_intensity': 0, 'index': x} for x in range(n_classes)]
        self.max_intensity = 555
        self.min_intensity = -655

        self.initial_bucketsize = (self.max_intensity - self.min_intensity) // (self.n_classes - 1)
        for idx, bucket in enumerate(self.buckets[1:]):
            bucket['lower_intensity'] = self.initial_bucketsize * idx + self.min_intensity
            bucket['upper_intensity'] = self.initial_bucketsize * (idx + 1) + self.min_intensity

        # fix rounding
        self.buckets[-1]['upper_intensity'] = self.max_intensity

        self.statistics = {'cache_hits': 0, 'cache_misses': 0, 'ct': 0}
        self.debug = debug
        # only set this if you are running an ipython session or any session which has the display variable set for matplotlib
        self.plot = plot

        self.load_buckets = [{'expected_probability': 0, 'total_bucket_loads': 0, 'probability_modifier': 0, 'oversampled_probability': 0} for x in range(n_classes)]

        self.equalize_hist = True

        self.total_loaded_slices = 0

        self.build_oversampling_dict()
        self.calc_distribution()

    def calc_distribution(self):
        self.n_filled_buckets = 0

        for bucket in self.buckets[1:]:
            for slices in bucket['slices']:
                self.total_slices += 1

            if len(bucket['slices']) > 0:
                self.n_filled_buckets += 1

        intens_dif = float(self.max_intensity - self.min_intensity)
        total_oversampled_prob = 0
        for idx, bucket in enumerate(self.buckets):
            if idx == 0:
                continue

            bucket_len = float(len(bucket['slices']))

            if bucket_len == 0:
                continue

            self.load_buckets[idx]['expected_probability'] = bucket_len / self.total_slices

            bucket_center = (bucket['upper_intensity'] - bucket['lower_intensity']) // 2 + bucket['lower_intensity']

            if self.equalize_hist:
                probability_modifier = 1
            else:
                probability_modifier = (bucket_len / self.total_slices) + abs(bucket_center - self.global_intensity) ** 0.76 / intens_dif

            self.load_buckets[idx]['probability_modifier'] = probability_modifier
            self.load_buckets[idx]['oversampled_probability'] = float(1) / self.n_filled_buckets * probability_modifier # float(bucket_len) / self.total_slices * probability_modifier

            total_oversampled_prob += self.load_buckets[idx]['oversampled_probability']

        # norm the probability to 1
        for idx, bucket in enumerate(self.buckets):
            if idx == 0:
                continue

            self.load_buckets[idx]['oversampled_probability'] /= total_oversampled_prob

        self.dump_statistics()

    # gives feedback if the current slice is acceptable
    def slice_loadprob_deviation(self, file_path, slice_index):
        key = self.get_key(file_path, slice_index)

        if not key in self.cache:
            return 0

        bucket_idx = self.cache[key]['bucket']

        if not bucket_idx:
            return 0

        if self.total_loaded_slices == 0:
            return 0

        dev =  float(self.load_buckets[bucket_idx]['total_bucket_loads']) / self.total_loaded_slices - self.load_buckets[bucket_idx]['oversampled_probability']
        return dev

    def loaded(self, file_path, slice_index, label_map):
        key = self.get_key(file_path, slice_index)
        if not key in self.cache:
            print ("Warning: ", key, " not found in the cache dict, loaded without oversampled probability distribution.", file_path, slice_index)
            return

        bucket_idx = self.cache[key]['bucket']

        # Liver-slice will be loaded
        if not bucket_idx:
            # print ("Warning: ", key, " is not sorted into any buckets.", file_path, slice_index, self.cache[key])
            return

        self.load_buckets[bucket_idx]['total_bucket_loads'] += 1

        if self.label_of_interest in label_map:
            self.total_loaded_slices += 1

        self.dump_probability_buckets()

    def calculate_global_intensity(self):
        self.global_intensity = 0
        liver_ct = 0
        lesion_ct = 0
        med_arr = []
        for key in self.cache:
            # ignore all liver-only slices
            if self.cache[key]['bucket'] > 0:
                med_arr.append(self.cache[key][self.sort_by + '_intensity'])
                lesion_ct += 1
            liver_ct += 1

        if self.sort_by == 'average':
            self.global_intensity = np.average(np.array(med_arr))
        else:
            self.global_intensity = np.median(np.array(med_arr))

        print ("The global ", self.sort_by, " intensity for all liver-lesion slices is: ", self.global_intensity)
        print ("Liver slices found: ", liver_ct, " Lesion slices found: ", lesion_ct)
        
    def build_oversampling_dict(self):
        dict_name = os.path.basename(os.path.dirname(self.train_data_dir))
        name = dict_name + '_oversampling' + '.p'

        if os.path.isfile(name):

            print ('Found oversampling file "{0}". Loading dictionary...'.format(name))

            infile = open(name, 'rb')
            self.cache, self.buckets, self.global_intensity = pickle.load(infile)
            infile.close()

        else:
            print ('No oversampling file "{0}" found. Creating dictionary...'.format(name))

            #LITS
            dct = pipe.Reader(seg.file_paths_ordered(self.train_data_dir, iterations=1, image_identifier='volume', label_identifier='segmentation'), name="Read File Names")
            dct = dct.multiply(seg.numpy_gen_slices(), name="Load Slices")

            for outputs, parameters, inputs, slice_index in dct:
                if self.label_required in outputs[1]:
                    self.register_slice(inputs[1], slice_index, outputs)

            self.calculate_global_intensity()

            outfile = open(name, 'wb')
            pickle.dump([self.cache, self.buckets, self.global_intensity], outfile)
            outfile.close()

        print ('Oversampling loaded.')

    def register_slice(self, file_path, slice_index, input_pair):
        key = self.get_key(file_path, slice_index)

        if not key in self.cache:
            self.add_slice(key, input_pair)
            self.statistics['cache_misses'] += 1
        else:
            self.statistics['cache_hits'] += 1

        self.statistics['ct'] += 1
        if self.statistics['ct'] % 300 == 0:
            self.dump_statistics()

    def dump_bucket(self, bucket):
        idx = bucket['index']
        avg_intensity = 0
        avg_median_intensity = 0
        avg_size = 0

        if len(bucket['intensities']) > 0:
            for intens in bucket['intensities']:
                avg_intensity += intens
            avg_intensity /= len(bucket['intensities'])

            for intens in bucket['median_intensities']:
                avg_median_intensity += intens
            avg_median_intensity /= len(bucket['median_intensities'])

            for size in bucket['lesion_sizes']:
                avg_size += size
            avg_size /= len(bucket['lesion_sizes'])

        print ("Bucket_" + str(idx) + ": " + str(bucket['lower_intensity']) + " - " + str(bucket['upper_intensity']) + " count: " + str(len(bucket['slices'])) + " int: " + str(avg_intensity) + " med: " + str(avg_median_intensity) + " avg px size: " + str(avg_size))

    def dump_probability_buckets(self):
        if not self.debug:
            return

        print("All probability buckets: ")
        for idx, load_bucket in enumerate(self.load_buckets):
            print (
                    "Load_bucket_" + str(idx),
                    " exp_pro: ",           str(round(load_bucket['expected_probability'], 4)),
                    " os_prob: ",           str(round(load_bucket['oversampled_probability'], 4)),
                    " prob_mod: ",          str(round(load_bucket['probability_modifier'], 2)),
                    " total_loads: ",       str(round(load_bucket['total_bucket_loads'], 4)),
                    " rel_loads: " ,        str(round(float(load_bucket['total_bucket_loads'])/self.total_loaded_slices, 4)) if self.total_loaded_slices > 0 else None, 
                    sep='\t'
                )
        
        if self.plot:
            highest_val = 0
            vals = []
            probs = []
            for idx, bucket in enumerate(self.buckets):
                if idx > 0:
                    if len(bucket['median_intensities']) == 0:
                        continue

                    for var in bucket['median_intensities']:
                        vals.append(var)

                    oversampled_prob = self.load_buckets[idx]['oversampled_probability']
                    center = (bucket['upper_intensity'] - bucket['lower_intensity']) / 2 + bucket['lower_intensity']

                    print ("Bucket dump median intensity, lower, upper: ", bucket['median_intensities'][-1], center, bucket['lower_intensity'], bucket['upper_intensity'], int(self.total_slices*oversampled_prob), len(bucket['median_intensities']))
                    for prob in range(int(self.total_slices*oversampled_prob)):
                        probs.append(center)

            self.show_overlay_hist(np.array(probs), np.array(vals))


    def dump_buckets(self):
        print ("### Dumping oversample buckets ###")
        vals = []
        vals_avg = []
        for idx, bucket in enumerate(self.buckets):
            self.dump_bucket(bucket)

            # First bucket is not interesting
            if idx > 0:
                for var in bucket['median_intensities']:
                    vals.append(var)

                for var in bucket['intensities']:
                    vals_avg.append(var)

        if self.plot:
            print ("Median histogram:")
            self.show_histogram(vals)
            print ("Average histogram:")
            self.show_histogram(vals_avg)

            # self.show_histogram(histeq(vals))

    def show_overlay_hist(self, vals1, vals2):
        if len(vals1) == 0:
            return

        x = np.array(vals1)
        binwidth = float((self.max_intensity - self.min_intensity)) / self.n_classes
        hist, bins = np.histogram(x, bins=np.arange(min(vals1), max(vals1) + binwidth, binwidth))

        x2 = np.array(vals2)
        hist2, bins2 = np.histogram(x2, bins=bins)

        width = 0.7 * (bins[1] - bins[0])
        center = (bins[:-1] + bins[1:]) / 2

        # b1 = plt.bar(center, hist, align='center', width=width, alpha=0.5, label='Oversampled')
        # b2 = plt.bar(center, hist2, align='center', width=width, alpha=0.5, label='Line 2')
        # plt.legend(handles=[b1, b2])
        # plt.show()

        b1 = plt.bar(center, hist, align='center', width=width, alpha=0.2, label='Oversampled')
        b2 = plt.bar(center, hist2, align='center', width=width, alpha=0.5, label='Original')
        plt.legend(handles=[b1, b2])
        plt.show()

    def show_histogram(self, vals):
        x = np.array(vals)
        hist, bins = np.histogram(x, bins=50)
        width = 0.7 * (bins[1] - bins[0])
        center = (bins[:-1] + bins[1:]) / 2
        plt.bar(center, hist, align='center', width=width)
        plt.show()

    def dump_statistics(self):
        if not self.debug:
            return

        pprint(self.statistics)
        print ("Sorted into the buckets by: ", self.sort_by)
        self.dump_buckets()

        self.dump_probability_buckets()

    def get_key(self, file_path, slice_index):
        return str(file_path) + str(slice_index)

    def add_slice(self, key, input_pair):
        image = input_pair[0]
        label = input_pair[1]
        interesting = self.label_of_interest in label
        required = self.label_required in label

        average_intensity = 0
        median_intensity = 0
        lesion_size = 0
        if interesting:
            average_intensity, lesion_size = self.get_dif_avg_intensity(image, label)
            median_intensity, lesion_size = self.get_dif_median_intensity(image, label)

        bucket_index = self.get_bucket(interesting, average_intensity, median_intensity, lesion_size, key)

        self.cache[key] = {'interesting': interesting, 'required': required, 'average_intensity': average_intensity, 'median_intensity': median_intensity, 'bucket': bucket_index, 'size': lesion_size}

    def get_average_intensity(self, image, label, label_to_check=2):
        average_intensity = 0
        interesting_pixels = 0

        for x,y in zip(*np.where(label==label_to_check)):
            interesting_pixels += 1
            average_intensity += image[x][y]

        return average_intensity // interesting_pixels, 0

    def get_median_intensity(self, image, label, label_to_check=2):
        intensity_values = []

        intensity_values = image[label == label_to_check]
        return np.median(intensity_values), len(intensity_values)

    def get_dif_avg_intensity(self, image, label):
        liver_avg, liver_size = self.get_average_intensity(image, label, self.label_required)
        lesion_avg, lesion_size = self.get_average_intensity(image, label, self.label_of_interest)
        return (liver_avg - lesion_avg, lesion_size)

    def get_dif_median_intensity(self, image, label):
        liver_med, liver_size = self.get_median_intensity(image, label, self.label_required)
        lesion_med, lesion_size = self.get_median_intensity(image, label, self.label_of_interest)
        return (liver_med - lesion_med, lesion_size)

    def get_bucket(self, interesting, average_intensity, median_intensity, lesion_size, key):
        # first bucked is always used for slices without the label of interest (lesion) present in the slice
        if not interesting:
            self.add_to_bucket(0, key, average_intensity, median_intensity, 0)
            return 0

        if self.sort_by == 'average':
            sort_value = average_intensity
        elif self.sort_by == 'median':
            sort_value = median_intensity
        else:
            raise Exception("Sort by " + self.sort_by + " - mode is not implemented!")

        # disregard first bucket since it is reserved for boring slices <= and >= to cover intensity self.min_intensity and max (theoretically)
        lc = 1
        for bucket in self.buckets[1:]:
            if bucket['lower_intensity'] <= sort_value and bucket['upper_intensity'] > sort_value:
                self.add_to_bucket(lc, key, average_intensity, median_intensity, lesion_size)
                return lc

            lc += 1

    def add_to_bucket(self, bucket_idx, key, average_intensity, median_intensity, lesion_size):
        self.buckets[bucket_idx]['slices'].append(key)
        self.buckets[bucket_idx]['intensities'].append(average_intensity)
        self.buckets[bucket_idx]['median_intensities'].append(median_intensity)
        self.buckets[bucket_idx]['lesion_sizes'].append(lesion_size)



