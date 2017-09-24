import numpy as np
import tensorflow as tf

import tensorflow.contrib.slim as slim


def static_weighted_softmax_cross_entropy_loss(logits, labels, weights, factor=0.5):
    logits = tf.reshape(logits, [-1, tf.shape(logits)[3]], name='flatten_logits')
    labels = tf.reshape(labels, [-1], name='flatten_labels')

    # get predictions from likelihoods
    prediction = tf.argmax(logits, 1, name='predictions')

    # get maps of class_of_interest pixels
    predictions_hit = tf.to_float(tf.equal(prediction, 1), name='predictions_weight_map')
    labels_hit = tf.to_float(tf.equal(labels, 1), name='labels_weight_map')

    predictions_weight_map = predictions_hit * ((factor * weights[1]) - weights[0]) + weights[0]
    labels_weight_map = labels_hit * (weights[1] - weights[0]) + weights[0]

    weight_map = tf.maximum(predictions_weight_map, labels_weight_map, name='combined_weight_map')
    weight_map = tf.stop_gradient(weight_map, name='stop_gradient')

    # compute cross entropy loss
    """
    - new tf version!
    Positional arguments are not allowed anymore. was (logits, labels, name=) instead of (logits=logits, labels=labels, name=)
    """
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels, name='cross_entropy_softmax')

    # apply weights to cross entropy loss
    """
    - new tf version!
    tf.mul -> tf.multiply
    """
    weighted_cross_entropy = tf.multiply(weight_map, cross_entropy, name='apply_weights')

    # get loss scalar
    loss = tf.reduce_mean(weighted_cross_entropy, name='loss')

    # print ("loss", loss)

    return loss, weight_map

"""
Pre-activation residual block as in resnet version 2
"""
@slim.add_arg_scope
def residual_block(inputs, n_filters, scope=None, sub_sample=False):
    with tf.variable_scope("bottleneck"):
        if sub_sample:
            first_stride = 2

            # Projection shortcut
            shortcut = slim.conv2d(inputs, n_filters, [1, 1], stride=2, padding="SAME", scope="projection")
        else:
            shortcut = inputs
            first_stride = 1
        # Compress
        l = slim.batch_norm(inputs, activation_fn=tf.nn.relu, scope="bn_0")
        l = slim.conv2d(l, n_filters // 4, [1, 1], stride=first_stride, padding="SAME", scope="conv_0")
        # Do transform
        l = slim.batch_norm(l, activation_fn=tf.nn.relu, scope="bn_1")
        l = slim.conv2d(l, n_filters // 4, [3, 3], stride=1, padding="SAME", scope="conv_1")
        # Decompress
        l = slim.batch_norm(l, activation_fn=tf.nn.relu, scope="bn_2")
        l = slim.conv2d(l, n_filters, [1, 1], stride=1, padding="SAME", scope="conv_2")             
        return shortcut + l

"""
Up-sampling residual block with pre-activations and deconvoluational projection
"""
@slim.add_arg_scope
def residual_block_up(inputs, n_filters):
    with tf.variable_scope("bottleneck"):
        l = slim.batch_norm(inputs, activation_fn=tf.nn.relu, scope="bn_0")
        # Compress
        l = slim.conv2d(l, n_filters // 4, [1, 1], stride=1, padding="SAME", scope="conv_0")
        l = slim.batch_norm(l, activation_fn=tf.nn.relu, scope="bn_1")
        # In this case, this dude samples up
        l = slim.conv2d_transpose(l, n_filters // 4, [3, 3], stride=2, padding="SAME", scope="conv_1")
        l = slim.batch_norm(l, activation_fn=tf.nn.relu, scope="bn_2")
        # Decompress
        l = slim.conv2d(l, n_filters, [1, 1], stride=1, padding="SAME", scope="conv_2")
        
        shortcut = slim.conv2d_transpose(inputs, n_filters, [1, 1], stride=2, padding="SAME", scope="projection")

        return shortcut + l


def inference(images, is_training, n_filters=80, block_config=[3, 3, 3, 3]):
    with slim.arg_scope([slim.conv2d, slim.conv2d_transpose],
                      weights_initializer=tf.truncated_normal_initializer(stddev=0.01),
                      weights_regularizer=slim.l2_regularizer(0.005)):
        with slim.arg_scope([slim.batch_norm],
                      is_training=is_training,
                      updates_collections=None, center=True, scale=True, decay=0.9):
            
            
            net = slim.conv2d(images, n_filters, [3, 3], padding="SAME", stride=1, activation_fn=tf.nn.relu)
            net = slim.batch_norm(net)
            
            # list that holds the anchors to forward features with
            forwards = []
            
            with tf.variable_scope("contract"):
                for idx, n_blocks in enumerate(block_config):
                    with tf.variable_scope("segment_%i" % idx):
                        for ydx in range(1, n_blocks):
                            with tf.variable_scope("block_%i" % ydx):
                                net = residual_block(net, n_filters)
                        # Save anchor for later
                        forwards.append(net)
                        # sub_sample
                        n_filters *= 2
                        with tf.variable_scope("block_%i" % (n_blocks)):
                            net = residual_block(net, n_filters, sub_sample=True)
                    
            with tf.variable_scope("expand"):
                for idx, n_blocks in enumerate(block_config):
                    with tf.variable_scope("segment_%i" % idx):
                        for ydx in range(1, n_blocks):
                            with tf.variable_scope("block_%i" % ydx):
                                net = residual_block(net, n_filters)
                        n_filters /= 2
                        with tf.variable_scope("block_%i" % (n_blocks)):
                            net = residual_block_up(net, n_filters)
                            
                        # Concat and pool relevant channels by using 1x1 conv followed by bn
                        with tf.variable_scope("merge_%i" % idx):
                            anchor = forwards[::-1][idx]
                            merged = tf.concat([net, anchor], 3)
                            # 1x1 conv to select relevant features and compress the channel dim
                            net = slim.conv2d(merged, n_filters, [1, 1], stride=1, padding="SAME")
                            # normalize lol
                            net = slim.batch_norm(net, activation_fn=tf.nn.relu)
                        
            # Do some final transform and map to logits
            with tf.variable_scope("final"):
                net = slim.conv2d(net, n_filters, [3, 3], padding="SAME", stride=1, activation_fn=tf.nn.relu)
                net = slim.batch_norm(net)
                
                # Get all them logits!!!111!!einself
                logits = slim.conv2d(net, 2, [1, 1], stride=1, padding="SAME", scope="logits")
                return logits

def loss_with_binary_dice(logits, labels, weights, axis=[1,2], smooth=1e-7):
    weighted_cross_entropy, weight_map = loss(logits, labels, weights)
        
    softmaxed = tf.nn.softmax(logits)[:,:,:,1]
   
    cond = tf.less(softmaxed, 0.5)
    output = tf.where(cond, tf.zeros(tf.shape(softmaxed)), tf.ones(tf.shape(softmaxed)))

    target = labels #tf.one_hot(labels, depth=2)
    
    # Make sure inferred shapes are equal during graph construction.
    # Do a static check as shapes won't change dynamically.
    assert output.get_shape().as_list() == target.get_shape().as_list()

    with tf.name_scope('dice'):
        output = tf.cast(output, tf.float32)
        target = tf.cast(target, tf.float32)
        inse = tf.reduce_sum(output * target, axis=axis)
        l = tf.reduce_sum(output, axis=axis)
        r = tf.reduce_sum(target, axis=axis)
        dice = (2. * inse + smooth) / (l + r + smooth)
        dice = tf.reduce_mean(dice)
    
    with tf.name_scope('final_cost'):
        final_cost = (weighted_cross_entropy + (1 - dice) + (1 / dice) ** 0.3 - 1) / 3

    return final_cost, weight_map, dice
    
            
def loss(logits, labels, weights):

    with tf.name_scope('loss'):
        loss, weight_map = static_weighted_softmax_cross_entropy_loss(logits, labels, weights)

    return loss, weight_map


def measurements(logits, ground_truth, label=1):
    """
    Takes 2 2-D arrays with class labels, and returns the dice score, precision, and sensitivity.
    Only the given label is considered.
    """

    with tf.name_scope('measurements'):

        label_const = tf.constant(label, dtype=tf.int32, shape=[], name='label_of_interest')

        prediction = tf.to_int32(tf.argmax(logits, 3, name='prediction'))

        prediction_label = tf.equal(prediction, label_const)
        ground_truth_label = tf.equal(ground_truth, label_const)

        sum_ground_truth = tf.reduce_sum(tf.to_float(ground_truth_label), name='sum_ground_truth')
        sum_prediction = tf.reduce_sum(tf.to_float(prediction_label), name='sum_prediction')

        with tf.name_scope('intersection'):
            intersection = tf.reduce_sum(tf.to_float(tf.logical_and(prediction_label, ground_truth_label)))

    return sum_prediction, sum_ground_truth, intersection


def training(loss_op):

    with tf.name_scope('train'):

        global_step = tf.Variable(0, name='global_step', trainable=False)

        optimizer = tf.train.AdamOptimizer(epsilon=0.1)
        train_op = optimizer.minimize(loss_op, global_step=global_step, name='minimize_loss')

    return train_op, global_step
