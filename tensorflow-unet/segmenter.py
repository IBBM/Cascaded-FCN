import tensorflow as tf
import numpy as np
import preprocessing
import unetv2 as unet
import sys
import time

# Choose Slice Type from axial, sagital and coronal
class Segmenter(object):
    def __init__(self, validation_examples = 300, 
                        validation_interval = 100,
                        max_steps = 1000000000,
                        batch_size = 4,
                        n_neighboringslices = 1,
                        input_size = 320,
                        output_size = 320,
                        oversample = False,
                        slice_type = 'axial',
                        load_path = None,
                        reset_counter = False,
                        summaries_dir = None,
                        snapshot_path = None,
                        label_of_interest = 1,
                        label_required = 1,
                        magic_number = 7.5,
                        max_slice_tries_val = 100,
                        max_slice_tries_train = 100,
                        fuse_labels=False,
                        apply_crop=True
                        ):
        
        self.validation_examples = validation_examples
        self.validation_interval = validation_interval
        self.max_steps = max_steps
        self.batch_size = batch_size
        self.n_neighboringslices = n_neighboringslices
        self.input_size = input_size
        self.output_size = output_size

        self.label_idx = n_neighboringslices // 2
        self.slice_type = slice_type

        self.load_path = load_path
        self.reset_counter = reset_counter

        self.oversample = oversample

        self.summaries_dir = summaries_dir
        self.snapshot_path = snapshot_path

        self.label_of_interest = label_of_interest
        self.label_required = label_required

        self.magic_number = magic_number
        self.max_slice_tries_val = max_slice_tries_val
        self.max_slice_tries_train = max_slice_tries_train
        self.fuse_labels = fuse_labels

        self.apply_crop = apply_crop

    def check_arguments(self):
        if self.validation_interval % self.batch_size != 0:
            raise RuntimeError("The validation interval ({0}) must be a multiple of the batch size ({1})!".format(
                       self.validation_interval, self.batch_size))

        if self.validation_examples % self.batch_size != 0:
            raise RuntimeError("The number of validation examples ({0}) must be a multiple of the batch size ({1})!".format(
                       self.validation_examples, self.batch_size))

    def setup_preprocessing(self, train_data_dir, test_data_dir):
        # Preprocessing
        print ("Setting up preprocessing")

        self.training_pipeline = preprocessing.training(
                train_data_dir,
                slice_type=self.slice_type,
                n_neighboringslices=self.n_neighboringslices,
                image_size=self.input_size, 
                oversample=self.oversample,
                label_of_interest=self.label_of_interest,
                label_required=self.label_required,
                max_tries=self.max_slice_tries_train,
                fuse_labels=self.fuse_labels,
                apply_crop=self.apply_crop
            )

        # In the validation pipeline, oversampling does not make any sense.
        self.validation_pipeline = preprocessing.validation(
                test_data_dir,
                slice_type=self.slice_type,
                n_neighboringslices=self.n_neighboringslices,
                image_size=self.input_size,
                label_of_interest=self.label_of_interest,
                label_required=self.label_required,
                max_tries=self.max_slice_tries_val,
                fuse_labels=self.fuse_labels,
                apply_crop=self.apply_crop
            )

    def setup_validation(self):
        self.validation_set = []
        source = self.validation_pipeline.run_on(1, num_datapoints=self.validation_examples)

        ct = 0
        for inputs, _ in source:
            ct += 1
            sys.stdout.write("\rValidation examples loaded %i/%i" % (ct, self.validation_examples))
            self.validation_set.append(inputs)

        sys.stdout.flush()
        source.close()

    def restore_model(self, saver, sess, init):
        if self.load_path is not None:

            print "Loading model {0}".format(self.load_path)
            saver.restore(sess, self.load_path)

            if self.reset_counter:

                print "Resetting global_step to 0"

                found = False
                for x in tf.all_variables():
                    if x.name == "train/global_step:0":
                        print "Found in tf.all_variables()"
                        sess.run(x.assign(0))
                        found = True
                        break
                if not found:
                    for x in tf.model_variables():
                        if x.name == "train/global_step:0":
                            print "Found in tf.model_variables()"
                            sess.run(x.assign(0))
                            found = True
                            break
                    if not found:
                        for x in tf.local_variables():
                            if x.name == "train/global_step:0":
                                print "Found in tf.local_variables()"
                                sess.run(x.assign(0))
                                found = True
                                break
                        if not found:
                            raise Exception("Variable global_step not found!")

        else:
            print "Initializing new model"
            sess.run(init)

    def go(self):
        tf.reset_default_graph()
        # Inputs

        with tf.name_scope('inputs'):

            images_var = tf.placeholder(tf.float32, shape=(self.batch_size, self.input_size, self.input_size, self.n_neighboringslices), name='images')
            labels_var = tf.placeholder(tf.int32, shape=(self.batch_size, self.output_size, self.output_size), name='labels')

            is_training = tf.placeholder(tf.bool, shape=[], name='is_training')

            weights = tf.constant([1, self.magic_number], dtype=tf.float32, name='weights')

            keep_prob_var = tf.placeholder(tf.float32, shape=[], name='keep_prob')

        # Model

        print "Setting up the model"

        logits_op = unet.inference(images_var, is_training)
        # loss_op, weight_map = unet.loss(logits_op, labels_var, uniform_weight_map, class_of_interest=1)
        # loss_op, weight_map = unet.loss(logits_op, labels_var, weights)
        loss_op, weight_map, dice_op = unet.loss_with_binary_dice(logits_op, labels_var, weights)

        train_op, global_step = unet.training(loss_op)
        # dice_op, precision_op, sensitivity_op = unet.measurements(logits_op, labels_var, label=1)
        num_prediction_op, num_ground_truth_op, num_intersection_op = unet.measurements(logits_op, labels_var, label=1)

        init = tf.initialize_all_variables()

        # Saver
        saver = tf.train.Saver(max_to_keep=100, keep_checkpoint_every_n_hours=2)

        # Start Session

        sess = tf.Session()

        self.restore_model(saver, sess, init)

        # Summaries

        with tf.name_scope('summaries'):

            with tf.name_scope('inputs'):

                size = tf.constant([self.output_size, self.output_size], dtype=tf.int32, name='size')
                offset = tf.zeros([1, 2], dtype=tf.float32, name='offset')

                image_crop = tf.image.extract_glimpse(images_var[0:1,:,:,self.label_idx:self.label_idx+1], size, offset, centered=True, name='crop')
                image_out = tf.reshape(image_crop, [self.output_size, self.output_size], name='reshape_image')

                # temp_crop = tf.image.extract_glimpse(images_var[:,:,:,idx:idx+1], size, offset, centered=True, name='crop')
                # slices_out = tf.reshape(images_var[:,:,:,:], [n_neighboringslices, output_size, output_size, 1], name='reshape_image')
                slices_out = tf.transpose(images_var[0:1,:,:,:], [3, 1, 2, 0])

                labels_float = tf.to_float(labels_var[0:1,:,:])
                labels_out = tf.reshape(labels_float, [self.output_size, self.output_size], name='reshape_labels')

                with tf.device('/cpu:0'):
                    image_summary_in = tf.placeholder(tf.float32, shape=(1, self.output_size * 2, self.output_size * 2, 1), name='image_summary_in')
                    image_summary = tf.summary.image('image', image_summary_in, max_outputs=5)

                    slices_summary_in = tf.placeholder(tf.float32, shape=(self.n_neighboringslices, self.output_size, self.output_size, 1), name='slices_summary_in')
                    slices_summary = tf.summary.image('slices', slices_summary_in, max_outputs=20)

            with tf.name_scope('outputs'):

                prediction = tf.to_float(tf.argmax(logits_op, 3, name='prediction_values'))
                prediction_out = tf.reshape(prediction[0:1,:,:], [self.output_size, self.output_size], name='reshape_prediction')

                weight_map_out = tf.reshape(weight_map[0:weight_map.get_shape().as_list()[0]//self.batch_size], [self.output_size, self.output_size], name='reshape_weight_map')

                with tf.device('/cpu:0'):
                    prediction_summary_in = tf.placeholder(tf.float32, shape=(1, self.output_size * 2, self.output_size * 2, 3), name='prediction_summary_in')
                    prediction_summary = tf.summary.image('prediction', prediction_summary_in, max_outputs=5)

                    weight_map_summary_in = tf.placeholder(tf.float32, shape=(1, self.output_size * 2, self.output_size * 2, 1), name='weight_map_summary_in')
                    weight_map_summary = tf.summary.image('weight map', weight_map_summary_in, max_outputs=5)

            with tf.device('/cpu:0'):
                image_summary_op = tf.summary.merge([image_summary, slices_summary, prediction_summary, weight_map_summary])

            with tf.name_scope('measurements'):

                mean_duration = tf.placeholder(tf.float64, shape=[], name='mean_duration')
                mean_loss = tf.placeholder(tf.float64, shape=[], name='mean_loss')
                mean_dice = tf.placeholder(tf.float64, shape=[], name='mean_dice')
                mean_precision = tf.placeholder(tf.float64, shape=[], name='mean_precision')
                mean_sensitivity = tf.placeholder(tf.float64, shape=[], name='mean_sensitivity')

                duration_summary = tf.summary.scalar('duration', mean_duration)
                loss_summary = tf.summary.scalar('loss', mean_loss)
                dice_summary = tf.summary.scalar('dice score', mean_dice)
                precision_summary = tf.summary.scalar('precision', mean_precision)
                sensitivity_summary = tf.summary.scalar('sensitivity', mean_sensitivity)

            measurements_summary_op = tf.summary.merge([duration_summary, loss_summary, dice_summary, precision_summary, sensitivity_summary], name="measurements")

        # Summary Writers

        train_writer = tf.summary.FileWriter(self.summaries_dir + '/train', sess.graph)
        test_writer = tf.summary.FileWriter(self.summaries_dir + '/val', sess.graph)

        image_input = np.zeros([self.batch_size, self.input_size, self.input_size, self.n_neighboringslices], dtype=np.float32)
        label_input = np.zeros([self.batch_size, self.output_size, self.output_size], dtype=np.float32)

        img_vis = np.zeros([1, self.output_size * 2, self.output_size * 2, 1], dtype=np.float32)

        slices_vis = np.zeros([self.n_neighboringslices, self.output_size, self.output_size, 1], dtype=np.float32)

        pred_vis = np.zeros([1, self.output_size * 2, self.output_size * 2, 3], dtype=np.float32)
        weight_vis = np.zeros([1, self.output_size * 2, self.output_size * 2, 1], dtype=np.float32)
        image_index = 0

        duration_list = []
        loss_list = []
        sum_prediction = np.float64(0.)
        sum_ground_truth = np.float64(0.)
        sum_intersection = np.float64(0.)

        print "Starting session"

        step = sess.run(global_step)


        # Helper Function

        def fill_batch(image_input, label_input, source):
            for i in range(self.batch_size):
                try:
                    inputs, _ = source.next()
                except StopIteration:
                    raise StopIteration

                for j in range(self.n_neighboringslices):
                    image_input[i, :, :, j] = inputs[j][0]
                
                label_input[i, :, :] = inputs[self.label_idx][1]


        print "Starting training"

        while step < self.max_steps:

            # Training Interval

            source = self.training_pipeline.run_on(1, num_datapoints=(self.validation_interval))

            val_intervalsteps = self.validation_interval / self.batch_size
            for _ in xrange(val_intervalsteps):

                fill_batch(image_input, label_input, source)
                        
                feed_dict = {
                    images_var: image_input,
                    labels_var: label_input,
                    keep_prob_var: 0.7,
                    is_training: True
                }

                start_time = time.time()

                # sys.stdout.write("\rTraining step %i. Next validation at %i steps." % (step, step // val_intervalsteps * (val_intervalsteps + 1) ))
                print ("Training step: ", step)

                if step % 100 == 0:  # this is on purpose to get visualizations on odd steps

                    # compute indices to place the image (output_sizexoutput_size) on the 2x2 canvas (776x776)
                    w_s = (image_index % 2) * self.output_size
                    w_e = w_s + self.output_size
                    h_s = (image_index / 2) * self.output_size
                    h_e = h_s + self.output_size

                    image_index = (image_index + 1) % 4

                    # execute one step and place the image output on the 2x2 canvas
                    _, step, loss, num_prediction, num_ground_truth, num_intersection, \
                        img_vis[0, h_s:h_e, w_s:w_e, 0], \
                        slices_vis, \
                        pred_vis[0, h_s:h_e, w_s:w_e, 1], \
                        pred_vis[0, h_s:h_e, w_s:w_e, 0], \
                        weight_vis[0, h_s:h_e, w_s:w_e, 0] = sess.run(
                            [train_op, global_step, loss_op, num_prediction_op, num_ground_truth_op, num_intersection_op,
                             image_out, slices_out, labels_out, prediction_out, weight_map_out],
                            feed_dict=feed_dict
                        )

                    # prepare to feed the canvases to the summary writer
                    feed_dict = {
                        image_summary_in: img_vis,
                        slices_summary_in: slices_vis,
                        prediction_summary_in: pred_vis,
                        weight_map_summary_in: weight_vis
                    }

                    # feed the canvases to the summary writer
                    summary = sess.run(image_summary_op, feed_dict=feed_dict)
                    train_writer.add_summary(summary, step)
                    train_writer.flush()

                else:

                    # normal training step
                    _, step, loss, dice, num_prediction, num_ground_truth, num_intersection = sess.run(
                        [train_op, global_step, loss_op, dice_op, num_prediction_op, num_ground_truth_op, num_intersection_op],
                        feed_dict=feed_dict
                    )
                    print ("dice: ", dice, loss, num_prediction, num_ground_truth, num_intersection)

                duration = time.time() - start_time

                duration_list.append(duration)
                loss_list.append(loss)
                sum_prediction += np.float64(num_prediction)
                sum_ground_truth += np.float64(num_ground_truth)
                sum_intersection += np.float64(num_intersection)

            # After Training before Validation

            source.close()

            temp_loss = np.mean(loss_list)
            print "Iteration: {0}, loss: {1}".format(step, temp_loss)

            feed_dict = {
                mean_duration: np.mean(duration_list),
                mean_loss: temp_loss,
                mean_dice: 2. * sum_intersection / (sum_ground_truth + sum_prediction),
                mean_precision: sum_intersection / sum_prediction,
                mean_sensitivity: sum_intersection / sum_ground_truth,
                is_training: False
            }

            summary = sess.run(measurements_summary_op, feed_dict=feed_dict)
            train_writer.add_summary(summary, step)
            train_writer.flush()

            img_vis[:] = 0
            slices_vis[:] = 0
            pred_vis[:] = 0
            weight_vis[:] = 0
            image_index = 0

            duration_list = []
            loss_list = []
            sum_prediction = np.float64(0.)
            sum_ground_truth = np.float64(0.)
            sum_intersection = np.float64(0.)

            saver.save(sess, self.snapshot_path, global_step=step)

            # Validation Interval

            for i in xrange(self.validation_examples // self.batch_size):

                for j in xrange(self.batch_size):

                    inputs = self.validation_set[i * self.batch_size + j]

                    for y in range(self.n_neighboringslices):
                        image_input[j, :, :, y] = inputs[y][0]

                    label_input[j, :, :] = inputs[self.label_idx][1]

                feed_dict = {
                    images_var: image_input,
                    labels_var: label_input,
                    keep_prob_var: 1.0,
                    is_training: False
                }

                start_time = time.time()

                if i % 100 == 0:  # this is on purpose to get visualizations on odd steps

                    # compute indices to place the image (output_sizexoutput_size) on the 2x2 canvas (776x776)
                    w_s = (image_index % 2) * self.output_size
                    w_e = w_s + self.output_size
                    h_s = (image_index / 2) * self.output_size
                    h_e = h_s + self.output_size

                    image_index = (image_index + 1) % 4

                    # execute one step and place the image output on the 2x2 canvas
                    loss, num_prediction, num_ground_truth, num_intersection, \
                        img_vis[0, h_s:h_e, w_s:w_e, 0], \
                        slices_vis, \
                        pred_vis[0, h_s:h_e, w_s:w_e, 1], \
                        pred_vis[0, h_s:h_e, w_s:w_e, 0], \
                        weight_vis[0, h_s:h_e, w_s:w_e, 0] = sess.run(
                            [loss_op, num_prediction_op, num_ground_truth_op, num_intersection_op,
                             image_out, slices_out, labels_out, prediction_out, weight_map_out],
                            feed_dict=feed_dict
                        )

                    # prepare to feed the canvases to the summary writer
                    feed_dict = {
                        image_summary_in: img_vis,
                        slices_summary_in: slices_vis,
                        prediction_summary_in: pred_vis,
                        weight_map_summary_in: weight_vis
                    }

                    # feed the canvases to the summary writer
                    summary = sess.run(image_summary_op, feed_dict=feed_dict)
                    test_writer.add_summary(summary, step)
                    test_writer.flush()

                else:

                    # normal execution
                    loss, num_prediction, num_ground_truth, num_intersection = sess.run(
                        [loss_op, num_prediction_op, num_ground_truth_op, num_intersection_op],
                        feed_dict=feed_dict
                    )

                duration = time.time() - start_time

                duration_list.append(duration)
                loss_list.append(loss)
                sum_prediction += np.float64(num_prediction)
                sum_ground_truth += np.float64(num_ground_truth)
                sum_intersection += np.float64(num_intersection)

            # After Validation

            temp_loss = np.mean(loss_list)
            print "Validation loss: {0}".format(temp_loss)

            feed_dict = {
                mean_duration: np.mean(duration_list),
                mean_loss: temp_loss,
                mean_dice: 2. * sum_intersection / (sum_ground_truth + sum_prediction),
                mean_precision: sum_intersection / sum_prediction,
                mean_sensitivity: sum_intersection / sum_ground_truth,
                is_training: False
            }

            summary = sess.run(measurements_summary_op, feed_dict=feed_dict)
            test_writer.add_summary(summary, step)
            test_writer.flush()

            img_vis[:] = 0
            pred_vis[:] = 0
            weight_vis[:] = 0
            image_index = 0

            duration_list = []
            loss_list = []
            sum_prediction = np.float64(0.)
            sum_ground_truth = np.float64(0.)
            sum_intersection = np.float64(0.)




