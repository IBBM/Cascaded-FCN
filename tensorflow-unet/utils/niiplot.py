import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

import medpy
import medpy.metric
import medpy.metric.binary

def is_kernel():
    if 'IPython' not in sys.modules:
        # IPython hasn't been imported, definitely not
        return False
    from IPython import get_ipython
    # check for `kernel` attribute on the IPython instance
    return getattr(get_ipython(), 'kernel', None) is not None   
display_available = is_kernel()

def check_display():
    def pass_through(func):
        return func

    def skip_func(func):
        def pass_func(*args, **kwargs):
            pass

        return pass_func

    return pass_through if display_available else skip_func

def my_decorator(condition):
    def wrap(fn):
        def new_f(*args,**kwargs):
            # some code that uses the function fn (fn holds a reference to the function you have decorated)
            pass

        if condition:
            return new_f
        else:
            return fn

    return wrap

class Overlay:
    """
    Used to plot slices and segmentations from nii format to visually check overlaps and errors.
    All given slices must be a numpy array.
    """
    def __init__(self, debug_lvl=3, label_max=None, label_min=None, slice_size=512):
        # Debug lvl 0 = no debugging, 1 = some, 2 = much, 3 = all prints
        self.debug_lvl = debug_lvl

        # label_max/min is used for rescaling of segmentations, give either both or leave both at None
        self.label_max = label_max
        self.label_min = label_min

        self.slice_size = slice_size
        self.dpi = 50

        self.cmaps = ['winter','autumn','summer','spring','cool']

        if not display_available and debug_lvl > 0:
            print ("Warning, no Display was found in the os environment. There will be no plots for this session.")

    @check_display()
    def rescale_slices(self, nii_slices):
        """
        Necessary because segmentations only have low values which will not feature a good contrast
        when using sequential colormaps. This function performs a copy, not to alter values in place.
        """
        max_seq_val = 255

        max_value = self.label_max if self.label_max else 0
        min_value = self.label_min if self.label_min else 0

        # Get the max value in case it is not given.
        if not self.label_max:
            for slc in nii_slices:
                if slc.max() > max_value:
                    max_value = slc.max()
                elif slc.min() < min_value:
                    min_value = slc.min()

        res_slices = []
        for slc in nii_slices:
            # Type conversion is important for plotting. Only floats can receive the np.nan which cuts out the basic label.
            slc_cp = slc.astype(np.float32, copy=True)

            # Translate to zero
            slc_cp -= min_value

            # Scale so that the max value is at 255
            slc_cp *= max_seq_val // max_value

            res_slices.append(slc_cp)

        return res_slices

    @check_display()
    def get_cmap(self, i):
        i = i % len(self.cmaps)
        return self.cmaps[i]

    @check_display()
    def get_2dintersection(self, arrays_to_check, label_to_purge=0):
        """
        Returns an array the same size like the n inputs which contains 1 at all points where the arrays match exactly, 0 where they do not mach AND
        an array which contains 1 everywhere the intersection failed.

        The arrays_to_check parameter must not be a numpy array but a regular list of numpy arrays! Make sure the dimensions match!
        """

        if len(arrays_to_check) < 2:
            raise ValueError('niiplot.Overlay.get_2dintersection only received 1 array. Intersections cannot be calculated.')

        intersect = np.zeros(arrays_to_check[0].shape)
        not_intersect = np.zeros(arrays_to_check[0].shape)

        for i, np_arr in enumerate(arrays_to_check[1:]):
            intersect[(np_arr == arrays_to_check[i]) & (np_arr != label_to_purge)] += 1
            not_intersect[(np_arr != arrays_to_check[i])] = 1

        intersect[intersect == len(arrays_to_check)-1] = 1
        intersect[intersect != len(arrays_to_check)-1] = 0

        return intersect, not_intersect

    @check_display()
    def segmentations_over_slice(self, nii_slice, segmentations, segmentation_labels=[], label_to_purge=0, extra_label=None, ignore_minlabel=True, skip_empty=True, stats=True):
        seg_len = len(segmentations)

        plt.figure(figsize=(self.slice_size/self.dpi,self.slice_size/self.dpi), dpi=self.dpi)

        # Underlying nii picture slice
        plt.imshow(nii_slice, alpha=1, cmap="gray", zorder=0, aspect='auto')

        # Rescale for sequential colormaps. Otherwise the color difference is not big enough.
        segmentations_rescaled = self.rescale_slices(segmentations)

        # sorted list of all values in data, necessary for labeling correctly
        label_values = np.unique(np.asarray(segmentations).ravel())
        # a sorted list of all rescaled values in data
        values = np.unique(np.asarray(segmentations_rescaled).ravel())
        
        patches = []

        seg_found = False
        for i, seg in enumerate(segmentations_rescaled):
            # If there is no segmentation to plot, skip
            if seg.max() == self.label_min and skip_empty:
                continue

            seg_found = True

            # The basic label (no segmentation), will not be plotted.
            if self.label_min is not None and ignore_minlabel:
                seg[seg == self.label_min] = np.nan

            im = plt.imshow(seg, alpha=0.4, cmap=self.get_cmap(i), zorder=i, interpolation='none', aspect='auto')

            # get the colors of the values, according to the
            # colormap used by imshow
            colors = [ im.cmap(im.norm(value)) for value in values]
            
            seg_label = segmentation_labels[i] if len(segmentation_labels) else "Segmentation"
            # create a patch (proxy artist) for every color, only show it if the value is really available 
            patches.extend([ mpatches.Patch(color=colors[y], label="{it} {lb}, val {l}".format(l=label_values[y], it=i, lb=seg_label) ) for y in range(len(values)) if values[y] in seg])


        if stats and len(segmentations) > 1:
            intersection, exclusions = self.get_2dintersection(segmentations, label_to_purge=label_to_purge)        
            patches.append(mpatches.Patch(color=(0,0,0,0), label="Intersection Pixels: {isec}".format(isec=intersection.sum())))
            patches.append(mpatches.Patch(color=(0,0,0,0), label="Exclusion Pixels: {fails}".format(fails=exclusions.sum())))
            if label_to_purge is not None:
                patches.append(mpatches.Patch(color=(0,0,0,0), label="The label {purged} has been excluded from the intersection analysis.".format(purged=label_to_purge)))

            patches.append(mpatches.Patch(color=(0,0,0,0), label="Dice segmentation 0 vs 1: {dc}".format(dc=medpy.metric.binary.dc(segmentations[0], segmentations[1]))))



        # put those patched as legend-handles into the legend
        plt.legend(handles=patches, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

        if seg_found or not skip_empty:
            plt.show()

        # make sure we do not flood with figures
        plt.close()
