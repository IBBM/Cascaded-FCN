import os
import nibabel

import numpy as np

import medpy
import medpy.filter

class Volume:

    def __init__(self, header, name=None, target_dir=None, postfix='', debug_lvl=1, largest_connected_component=False):
        self.header = header

        dimensions = list(self.header.get_data_shape())
        self.out_volume = np.zeros(dimensions, dtype=np.int8)

        self.join_targetpath(postfix, name, target_dir)

        self.largest_connected_component = largest_connected_component

        self.debug_lvl = debug_lvl

        self.slice_counter = 0

        if self.debug_lvl > 0:
            print ("Creating new volume: ", name, dimensions)

    def check_error(self):
        if self.slice_counter != self.out_volume.shape[2]:
            raise RuntimeError("slice_counter: {}, volume.shape: {}, for last volume: {}".format(self.slice_counter, self.out_volume.shape, self.name))

        self.save()

    def join_targetpath(self, postfix, name, data_dir):
        name_parts = os.path.splitext(os.path.basename(name))
        save_name = os.path.join(data_dir, name_parts[0] + postfix + name_parts[1])
        
        self.name = name
        self.save_path = save_name

    def add_slice(self, slice):
        self.out_volume[:, :, self.slice_counter] = slice.copy()
        self.slice_counter += 1
    
    def save(self, save_path=None):
        if save_path is None:
            save_path = self.save_path

        if self.largest_connected_component:
            self.out_volume = medpy.filter.binary.largest_connected_component(self.out_volume)

        img = nibabel.Nifti1Image(self.out_volume, self.header.get_base_affine(), header=self.header)
        nibabel.save(img, save_path)

        if self.debug_lvl > 0:
            print ('Saved nii at %s' % self.save_path)