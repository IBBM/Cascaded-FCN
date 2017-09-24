import argparse
import os
import sys

import tensorflow as tf

from segmenter import Segmenter

# Command Line Arguments

parser = argparse.ArgumentParser()

parser.add_argument('--model', '-m', help='the model file', default=None)
parser.add_argument('--reset_counter', '-r', action='store_true')
parser.add_argument('--logdir', '-l', help='parent directory where logfiles, checkpoints, graphs are stored', default='/code/litsruns')
parser.add_argument('--runname', '-name', help='name of the run. if empty, number will be chosen', default=None)


args = parser.parse_args()

# Training Parameters

load_path = args.model
log_dir = args.logdir
run_name = args.runname

if not os.path.exists(log_dir):
    os.makedirs(log_dir)

if not run_name is None:
    data_path = log_dir + '/' + run_name

    if os.path.exists(data_path):
        raise Exception('Error: Data for the run ' + run_name + ' already exists. Please delete or chose another.')
else:
    run_name = 'Run_' + str(len([name for name in os.listdir(log_dir) if os.path.isdir(log_dir + "/" + name)]))
    data_path = log_dir + '/' + run_name

print ("Starting run: ", run_name)

snapshot_dir = data_path + '/snapshots/'
summaries_dir = data_path + '/summaries/'
snapshot_path = os.path.join(snapshot_dir, 'unet_model')

if not os.path.exists(snapshot_dir):
    os.makedirs(snapshot_dir)

if not os.path.exists(summaries_dir):
    os.makedirs(summaries_dir)

train_data_dir = "/code/LITS/TRB2/"
test_data_dir = "/code/LITS/TRB1/"

# Liver config
# segmentor_instance = Segmenter(validation_examples = 100, 
#                                 validation_interval = 800,
#                                 max_steps = 1000000000,
#                                 batch_size = 4,
#                                 n_neighboringslices = 1,
#                                 input_size = 400,
#                                 output_size = 400,
#                                 slice_type = 'axial',
#                                 oversample = False,
#                                 load_path = load_path,
#                                 reset_counter = args.reset_counter,
#                                 summaries_dir = summaries_dir,
#                                 snapshot_path = snapshot_path,
#                                 label_of_interest=1,
#                                 label_required=0,
#                                 magic_number=26.91, # 16.4
#                                 max_slice_tries_val = 0,
#                                 max_slice_tries_train = 2,
#                                 fuse_labels=True,
#                                 apply_crop=False)

# Lesion config
segmentor_instance = Segmenter(validation_examples = 200, 
                                validation_interval = 800,
                                max_steps = 1000000000,
                                batch_size = 8,
                                n_neighboringslices = 3,
                                input_size = 256,
                                output_size = 256,
                                slice_type = 'axial',
                                oversample = False,
                                load_path = load_path,
                                reset_counter = args.reset_counter,
                                summaries_dir = summaries_dir,
                                snapshot_path = snapshot_path,
                                label_of_interest=2,
                                label_required=1,
                                magic_number=8.5,
                                max_slice_tries_val = 0,
                                max_slice_tries_train = 0,
                                fuse_labels=False,
                                apply_crop=True)

# Instantiate the preprocessing
segmentor_instance.setup_preprocessing(train_data_dir, test_data_dir)

print ("Preprocessing setup done.")

# Fill the validation set with slices
segmentor_instance.setup_validation()

print ("Validation setup done.")

# Initiate the training
segmentor_instance.go()
