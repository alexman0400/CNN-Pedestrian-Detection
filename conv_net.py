from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf
from pedestrian_conv_net.project_utils import getDims
from pedestrian_conv_net.estimator_test import get_learning_rate

HEIGHT, WIDTH = getDims()   # discovered through helper java classes
CLASSES = 3
NUM_CHANNELS = 3  # 3 for RGB, 1 for grayscale
# learning_rate = 0.001
learning_rate = get_learning_rate()
num_outputs = 512


def create_conv_layer(inputs,
                      stride,
                      conv_filter_size,
                      num_filters,
                      relu):
    activation = None
    if relu:
        activation = tf.nn.relu
    layer = tf.layers.conv2d(inputs=inputs,
                             filters=num_filters,
                             strides=[stride, stride],
                             kernel_size=[conv_filter_size, conv_filter_size],
                             padding='same',
                             activation=activation)
    return layer

# First CNN Model
def cnn_model_fn_0(features, labels, mode):
    with tf.name_scope('input_layer'):
        input_layer = tf.reshape(features["x"], [-1, WIDTH, HEIGHT, 3])

    with tf.name_scope('conv1'):
        conv1 = create_conv_layer(inputs=input_layer,
                                  stride=2,
                                  conv_filter_size=11,
                                  num_filters=3,
                                  relu=True)

    with tf.name_scope('pool1'):
        pool1 = tf.contrib.layers.max_pool2d(inputs=conv1,
                                             kernel_size=3,
                                             stride=2,
                                             padding='SAME')

    with tf.name_scope('conv1'):
        conv2 = create_conv_layer(inputs=pool1,
                                  stride=1,
                                  conv_filter_size=5,
                                  num_filters=96,
                                  relu=True)

    with tf.name_scope('pool2'):
        pool2 = tf.contrib.layers.max_pool2d(inputs=conv2,
                                             kernel_size=3,
                                             stride=2,
                                             padding='SAME')

    with tf.name_scope('conv3'):
        conv3 = create_conv_layer(inputs=pool2,
                                  stride=1,
                                  conv_filter_size=3,
                                  num_filters=128,
                                  relu=True)

    with tf.name_scope('conv4'):
        conv4 = create_conv_layer(inputs=conv3,
                                  stride=1,
                                  conv_filter_size=3,
                                  num_filters=256,
                                  relu=True)

    with tf.name_scope('conv5'):
        conv5 = create_conv_layer(inputs=conv4,
                                  stride=1,
                                  conv_filter_size=3,
                                  num_filters=256,
                                  relu=True)

    with tf.name_scope('pool3'):
        pool3 = tf.contrib.layers.max_pool2d(inputs=conv5,
                                             kernel_size=3,
                                             stride=2,
                                             padding='SAME')

    with tf.name_scope('layer_flat'):
        layer_flat = tf.layers.flatten(inputs=pool3)

    with tf.name_scope('dense_1'):
        dense_1 = tf.layers.dense(inputs=layer_flat, units=4096, activation=tf.nn.relu)

    with tf.name_scope('dense_2'):
        dense_2 = tf.layers.dense(inputs=dense_1, units=1024, activation=tf.nn.relu)

    with tf.name_scope('drp'):
        drp = tf.layers.dropout(inputs=dense_2)

    with tf.name_scope('logits'):
        # logits = tf.contrib.layers.fully_connected(inputs=drp,
        #                                            num_outputs=CLASSES) # tf.layers.dense(inputs=drp, units=CLASSES)
        logits = tf.layers.dense(inputs=drp, units=CLASSES)

    # Calculate Predictions
    predictions = {
        # Generate predictions (for PREDICT and EVAL mode)
        "classes": tf.argmax(input=logits, axis=1),
        # Add `softmax_tensor` to the graph. It is used for PREDICT and by the `logging_hook`.
        "probabilities": tf.nn.softmax(logits=logits,
                                       name="softmax_tensor")
    }

    print("\n", mode)
    if mode == tf.estimator.ModeKeys.PREDICT:
        spec = tf.estimator.EstimatorSpec(mode=mode,
                                          predictions=predictions)
        return spec

    # Calculate Loss (for both TRAIN and EVAL modes)
    onehot_labels = tf.one_hot(indices=tf.cast(labels, tf.int32),
                               depth=3)
    print(onehot_labels)

    cross_entropy = tf.losses.softmax_cross_entropy(onehot_labels=onehot_labels,
                                                    logits=logits)  # softmax_cross_entropy(onehot_labels=onehot_labels, logits=logits)
    tf.summary.scalar('cross_entropy', cross_entropy)

    loss = tf.reduce_mean(cross_entropy)
    tf.summary.scalar('loss', loss)

    # Configure the Training Op (for TRAIN mode)
    if mode == tf.estimator.ModeKeys.TRAIN:
        # -------------------------- try different optimizers(GradientDescentOptimizer) -----------------------
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate, name="Adam_Optimizer")
        # optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate, name="SGD")
        train_op = optimizer.minimize(loss=loss, global_step=tf.train.get_global_step())
        spec = tf.estimator.EstimatorSpec(mode=mode,
                                          loss=loss,
                                          train_op=train_op)
        return spec

    # Add evaluation metrics (for EVAL mode)
    eval_metric_ops = {"accuracy": tf.metrics.accuracy(labels=labels,
                                                       predictions=predictions["classes"])}
    merged = tf.summary.merge_all()

    spec = tf.estimator.EstimatorSpec(mode=mode,
                                      loss=loss,
                                      eval_metric_ops=eval_metric_ops)
    return spec

# Second CNN Model
def cnn_model_fn_1(features, labels, mode):
    with tf.name_scope('input_layer'):
        input_layer = tf.reshape(features["x"], [-1, WIDTH, HEIGHT, 3])

    with tf.name_scope('conv1'):
        conv1 = create_conv_layer(inputs=input_layer,
                                  stride=2,
                                  conv_filter_size=5,
                                  num_filters=32,
                                  relu=False)

    with tf.name_scope('pool1'):
        pool1 = tf.contrib.layers.max_pool2d(inputs=conv1,
                                             kernel_size=3,
                                             stride=2,
                                             padding='SAME')

    with tf.name_scope('conv2'):
        conv2 = create_conv_layer(inputs=pool1,
                                  stride=1,
                                  conv_filter_size=1,
                                  num_filters=32,
                                  relu=False)

    with tf.name_scope('layer_flat'):
        layer_flat = tf.layers.flatten(inputs=conv2)

    with tf.name_scope('dense_1'):
        dense_1 = tf.layers.dense(inputs=layer_flat, units=512)

    with tf.name_scope('drp'):
        drp = tf.layers.dropout(inputs=dense_1,
                                training=mode == tf.estimator.ModeKeys.TRAIN)

    with tf.name_scope('dense_2'):
        dense_1 = tf.layers.dense(inputs=drp, units=512)

    with tf.name_scope('logits'):
        # logits = tf.contrib.layers.fully_connected(inputs=drp,
        #                                            num_outputs=CLASSES) # tf.layers.dense(inputs=drp, units=CLASSES)
        logits = tf.layers.dense(inputs=dense_1, units=CLASSES)

    # Calculate Predictions
    predictions = {
        # Generate predictions (for PREDICT and EVAL mode)
        "classes": tf.argmax(input=logits, axis=1),
        # Add `softmax_tensor` to the graph. It is used for PREDICT and by the `logging_hook`.
        "probabilities": tf.nn.softmax(logits=logits,
                                       name="softmax_tensor")
    }

    print("\n", mode)
    if mode == tf.estimator.ModeKeys.PREDICT:
        spec = tf.estimator.EstimatorSpec(mode=mode,
                                          predictions=predictions)
        return spec

    # Calculate Loss (for both TRAIN and EVAL modes)
    onehot_labels = tf.one_hot(indices=tf.cast(labels, tf.int32),
                               depth=3)
    print(onehot_labels)

    cross_entropy = tf.losses.softmax_cross_entropy(onehot_labels=onehot_labels,
                                                    logits=logits)  # softmax_cross_entropy(onehot_labels=onehot_labels, logits=logits)
    tf.summary.scalar('cross_entropy', cross_entropy)

    loss = tf.reduce_mean(cross_entropy)
    tf.summary.scalar('loss', loss)

    # Configure the Training Op (for TRAIN mode)
    if mode == tf.estimator.ModeKeys.TRAIN:
        # -------------------------- try different optimizers(GradientDescentOptimizer) -----------------------
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate, name="Adam_Optimizer")
        # optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate, name="SGD")
        train_op = optimizer.minimize(loss=loss, global_step=tf.train.get_global_step())
        spec = tf.estimator.EstimatorSpec(mode=mode,
                                          loss=loss,
                                          train_op=train_op)
        return spec

    # Add evaluation metrics (for EVAL mode)
    eval_metric_ops = {"accuracy": tf.metrics.accuracy(labels=labels,
                                                       predictions=predictions["classes"])}
    merged = tf.summary.merge_all()

    spec = tf.estimator.EstimatorSpec(mode=mode,
                                      loss=loss,
                                      eval_metric_ops=eval_metric_ops)
    return spec

# Third CNN Model
def cnn_model_fn_2(features, labels, mode):
    with tf.name_scope('input_layer'):
        input_layer = tf.reshape(features["x"], [-1, WIDTH, HEIGHT, 3])

    with tf.name_scope('conv1'):
        conv1 = create_conv_layer(inputs=input_layer,
                                  stride=2,
                                  conv_filter_size=11,
                                  num_filters=96,
                                  relu=False)


    with tf.name_scope('pool1'):
        pool1 = tf.contrib.layers.max_pool2d(inputs=conv1,
                                             kernel_size=3,
                                             stride=2,
                                             padding='SAME')

    with tf.name_scope('conv2'):
        conv2 = create_conv_layer(inputs=pool1,
                                  stride=1,
                                  conv_filter_size=5,
                                  num_filters=128,
                                  relu=False)

    with tf.name_scope('pool2'):
        pool2 = tf.contrib.layers.max_pool2d(inputs=conv2,
                                             kernel_size=3,
                                             stride=2,
                                             padding='SAME')

    with tf.name_scope('conv3'):
        conv3 = create_conv_layer(inputs=pool2,
                                  stride=1,
                                  conv_filter_size=3,
                                  num_filters=256,
                                  relu=False)

    with tf.name_scope('conv4'):
        conv4 = create_conv_layer(inputs=conv3,
                                  stride=1,
                                  conv_filter_size=3,
                                  num_filters=256,
                                  relu=False)

    with tf.name_scope('conv5'):
        conv5 = create_conv_layer(inputs=conv4,
                                  stride=1,
                                  conv_filter_size=3,
                                  num_filters=256,
                                  relu=False)

    with tf.name_scope('conv6'):
        conv6 = create_conv_layer(inputs=conv5,
                                  stride=1,
                                  conv_filter_size=3,
                                  num_filters=256,
                                  relu=False)

    with tf.name_scope('pool2'):
        pool3 = tf.contrib.layers.max_pool2d(inputs=conv6,
                                             kernel_size=3,
                                             stride=2,
                                             padding='SAME')

    with tf.name_scope('layer_flat'):
        layer_flat = tf.layers.flatten(inputs=pool3)

    with tf.name_scope('dense_1'):
        dense_1 = tf.layers.dense(inputs=layer_flat, units=4096)

    with tf.name_scope('dense_2'):
        dense_2 = tf.layers.dense(inputs=dense_1, units=1024)

    with tf.name_scope('logits'):
        logits = tf.layers.dense(inputs=dense_2, units=CLASSES)

    # Calculate Predictions
    predictions = {
        # Generate predictions (for PREDICT and EVAL mode)
        "classes": tf.argmax(input=logits, axis=1),
        # Add `softmax_tensor` to the graph. It is used for PREDICT and by the `logging_hook`.
        "probabilities": tf.nn.softmax(logits=logits,
                                       name="softmax_tensor")
    }

    print("\n", mode)
    if mode == tf.estimator.ModeKeys.PREDICT:
        spec = tf.estimator.EstimatorSpec(mode=mode,
                                          predictions=predictions)
        return spec

    # Calculate Loss (for both TRAIN and EVAL modes)
    onehot_labels = tf.one_hot(indices=tf.cast(labels, tf.int32),
                               depth=3)
    print(onehot_labels)

    cross_entropy = tf.losses.softmax_cross_entropy(onehot_labels=onehot_labels,
                                                    logits=logits)
    tf.summary.scalar('cross_entropy', cross_entropy)

    loss = tf.reduce_mean(cross_entropy)
    tf.summary.scalar('loss', loss)

    # Configure the Training Op (for TRAIN mode)
    if mode == tf.estimator.ModeKeys.TRAIN:
        # -------------------------- try different optimizers(GradientDescentOptimizer) -----------------------
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate, name="Adam_Optimizer")
        # optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate, name="SGD")
        train_op = optimizer.minimize(loss=loss, global_step=tf.train.get_global_step())
        spec = tf.estimator.EstimatorSpec(mode=mode,
                                          loss=loss,
                                          train_op=train_op)
        return spec

    # Add evaluation metrics (for EVAL mode)
    eval_metric_ops = {"accuracy": tf.metrics.accuracy(labels=labels,
                                                       predictions=predictions["classes"])}
    merged = tf.summary.merge_all()

    spec = tf.estimator.EstimatorSpec(mode=mode,
                                      loss=loss,
                                      eval_metric_ops=eval_metric_ops)
    return spec

# Fourth CNN Model
def cnn_model_fn_3(features, labels, mode):
    with tf.name_scope('input_layer'):
        input_layer = tf.reshape(features["x"], [-1, WIDTH, HEIGHT, 3])

    with tf.name_scope('conv1'):
        conv1 = create_conv_layer(inputs=input_layer,
                                  stride=1,
                                  conv_filter_size=7,
                                  num_filters=32,
                                  relu=False)

    with tf.name_scope('pool1'):
        pool1 = tf.contrib.layers.max_pool2d(inputs=conv1,
                                             kernel_size=3,
                                             stride=2,
                                             padding='SAME')

    with tf.name_scope('conv2'):
        conv2 = create_conv_layer(inputs=pool1,
                                  stride=1,
                                  conv_filter_size=5,
                                  num_filters=64,
                                  relu=False)

    with tf.name_scope('pool2'):
        pool2 = tf.contrib.layers.max_pool2d(inputs=conv2,
                                             kernel_size=3,
                                             stride=2,
                                             padding='SAME')

    with tf.name_scope('conv3'):
        conv3 = create_conv_layer(inputs=pool2,
                                  stride=1,
                                  conv_filter_size=3,
                                  num_filters=128,
                                  relu=False)

    with tf.name_scope('conv4'):
        conv4 = create_conv_layer(inputs=conv3,
                                  stride=1,
                                  conv_filter_size=3,
                                  num_filters=128,
                                  relu=False)

    with tf.name_scope('pool3'):
        pool3 = tf.contrib.layers.max_pool2d(inputs=conv4,
                                             kernel_size=3,
                                             stride=2,
                                             padding='SAME')

    with tf.name_scope('layer_flat'):
        layer_flat = tf.layers.flatten(inputs=pool3)

    with tf.name_scope('logits'):
        # logits = tf.contrib.layers.fully_connected(inputs=drp,
        #                                            num_outputs=CLASSES) # tf.layers.dense(inputs=drp, units=CLASSES)
        logits = tf.layers.dense(inputs=layer_flat, units=CLASSES)

        # Calculate Predictions
    predictions = {
        # Generate predictions (for PREDICT and EVAL mode)
        "classes": tf.argmax(input=logits, axis=1),
        # Add `softmax_tensor` to the graph. It is used for PREDICT and by the `logging_hook`.
        "probabilities": tf.nn.softmax(logits=logits,
                                       name="softmax_tensor")
    }

    print("\n", mode)
    if mode == tf.estimator.ModeKeys.PREDICT:
        spec = tf.estimator.EstimatorSpec(mode=mode,
                                          predictions=predictions)
        return spec

    # Calculate Loss (for both TRAIN and EVAL modes)
    onehot_labels = tf.one_hot(indices=tf.cast(labels, tf.int32),
                               depth=3)
    print(onehot_labels)

    cross_entropy = tf.losses.softmax_cross_entropy(onehot_labels=onehot_labels,
                                                    logits=logits)  # softmax_cross_entropy(onehot_labels=onehot_labels, logits=logits)
    tf.summary.scalar('cross_entropy', cross_entropy)

    loss = tf.reduce_mean(cross_entropy)
    tf.summary.scalar('loss', loss)

    # Configure the Training Op (for TRAIN mode)
    if mode == tf.estimator.ModeKeys.TRAIN:
        # -------------------------- try different optimizers(GradientDescentOptimizer) -----------------------
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate, name="Adam_Optimizer")
        # optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate, name="SGD")
        train_op = optimizer.minimize(loss=loss, global_step=tf.train.get_global_step())
        spec = tf.estimator.EstimatorSpec(mode=mode,
                                          loss=loss,
                                          train_op=train_op)
        return spec

    # Add evaluation metrics (for EVAL mode)
    eval_metric_ops = {"accuracy": tf.metrics.accuracy(labels=labels,
                                                       predictions=predictions["classes"])}
    merged = tf.summary.merge_all()

    spec = tf.estimator.EstimatorSpec(mode=mode,
                                      loss=loss,
                                      eval_metric_ops=eval_metric_ops)
    return spec

# Fifth CNN Model
def cnn_model_fn_5(features, labels, mode):
    with tf.name_scope('input_layer'):
        input_layer = tf.reshape(features["x"], [-1, WIDTH, HEIGHT, 3])

    with tf.name_scope('conv1'):
        conv1 = create_conv_layer(inputs=input_layer,
                                  stride=4,
                                  conv_filter_size=11,
                                  num_filters=1,
                                  relu=False)

    with tf.name_scope('pool1'):
        pool1 = tf.contrib.layers.max_pool2d(inputs=conv1,
                                             kernel_size=3,
                                             stride=2,
                                             padding='SAME')

    with tf.name_scope('conv2'):
        conv2 = create_conv_layer(inputs=pool1,
                                  stride=1,
                                  conv_filter_size=5,
                                  num_filters=1,
                                  relu=False)

    with tf.name_scope('pool2'):
        pool2 = tf.contrib.layers.max_pool2d(inputs=conv2,
                                             kernel_size=3,
                                             stride=2,
                                             padding='SAME')

    with tf.name_scope('conv3'):
        conv3 = create_conv_layer(inputs=pool2,
                                  stride=1,
                                  conv_filter_size=3,
                                  num_filters=1,
                                  relu=False)

    with tf.name_scope('conv4'):
        conv4 = create_conv_layer(inputs=conv3,
                                  stride=1,
                                  conv_filter_size=3,
                                  num_filters=1,
                                  relu=False)

    with tf.name_scope('conv5'):
        conv5 = create_conv_layer(inputs=conv4,
                                  stride=1,
                                  conv_filter_size=3,
                                  num_filters=1,
                                  relu=False)

    with tf.name_scope('pool3'):
        pool3 = tf.contrib.layers.max_pool2d(inputs=conv5,
                                             kernel_size=3,
                                             stride=2,
                                             padding='SAME')

    with tf.name_scope('conv6'):
        conv6 = create_conv_layer(inputs=pool3,
                                  stride=1,
                                  conv_filter_size=1,
                                  num_filters=4096,
                                  relu=False)

    with tf.name_scope('conv7'):
        conv7 = create_conv_layer(inputs=conv6,
                                  stride=1,
                                  conv_filter_size=1,
                                  num_filters=4096,
                                  relu=False)

    with tf.name_scope('conv8'):
        conv8 = create_conv_layer(inputs=conv7,
                                  stride=1,
                                  conv_filter_size=1,
                                  num_filters=3,
                                  relu=False)

    with tf.name_scope('layer_flat'):
        layer_flat = tf.layers.flatten(inputs=conv8)

    with tf.name_scope('logits'):
        # logits = tf.contrib.layers.fully_connected(inputs=drp,
        #                                            num_outputs=CLASSES) # tf.layers.dense(inputs=drp, units=CLASSES)
        logits = tf.layers.dense(inputs=layer_flat, units=CLASSES)

    # Calculate Predictions
    predictions = {
        # Generate predictions (for PREDICT and EVAL mode)
        "classes": tf.argmax(input=logits, axis=1),
        # Add `softmax_tensor` to the graph. It is used for PREDICT and by the `logging_hook`.
        "probabilities": tf.nn.softmax(logits=logits,
                                       name="softmax_tensor")
    }

    print("\n", mode)
    if mode == tf.estimator.ModeKeys.PREDICT:
        spec = tf.estimator.EstimatorSpec(mode=mode,
                                          predictions=predictions)
        return spec

    # Calculate Loss (for both TRAIN and EVAL modes)
    onehot_labels = tf.one_hot(indices=tf.cast(labels, tf.int32),
                               depth=3)
    print(onehot_labels)

    cross_entropy = tf.losses.softmax_cross_entropy(onehot_labels=onehot_labels,
                                                    logits=logits)  # softmax_cross_entropy(onehot_labels=onehot_labels, logits=logits)
    tf.summary.scalar('cross_entropy', cross_entropy)

    loss = tf.reduce_mean(cross_entropy)
    tf.summary.scalar('loss', loss)

    # Configure the Training Op (for TRAIN mode)
    if mode == tf.estimator.ModeKeys.TRAIN:
        # -------------------------- try different optimizers(GradientDescentOptimizer) -----------------------
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate, name="Adam_Optimizer")
        # optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate, name="SGD")
        train_op = optimizer.minimize(loss=loss, global_step=tf.train.get_global_step())
        spec = tf.estimator.EstimatorSpec(mode=mode,
                                          loss=loss,
                                          train_op=train_op)
        return spec

    # Add evaluation metrics (for EVAL mode)
    eval_metric_ops = {"accuracy": tf.metrics.accuracy(labels=labels,
                                                       predictions=predictions["classes"])}
    merged = tf.summary.merge_all()

    spec = tf.estimator.EstimatorSpec(mode=mode,
                                      loss=loss,
                                      eval_metric_ops=eval_metric_ops)
    return spec

# Sixth CNN Model
def cnn_model_fn_6(features, labels, mode):
        with tf.name_scope('input_layer'):
            input_layer = tf.reshape(features["x"], [-1, WIDTH, HEIGHT, 3])

        with tf.name_scope('conv1'):
            conv1 = create_conv_layer(inputs=input_layer,
                                      stride=2,
                                      conv_filter_size=5,
                                      num_filters=32,
                                      relu=False)

        with tf.name_scope('pool1'):
            pool1 = tf.contrib.layers.max_pool2d(inputs=conv1,
                                                 kernel_size=3,
                                                 stride=1,
                                                 padding='SAME')

        with tf.name_scope('conv2'):
            conv2 = create_conv_layer(inputs=pool1,
                                      stride=1,
                                      conv_filter_size=1,
                                      num_filters=32,
                                      relu=False)

        with tf.name_scope('layer_flat'):
            layer_flat = tf.layers.flatten(inputs=conv2)

        with tf.name_scope('dense_1'):
            dense_1 = tf.layers.dense(inputs=layer_flat, units=512)

        with tf.name_scope('drp'):
            drp = tf.layers.dropout(inputs=dense_1,
                                    training=mode == tf.estimator.ModeKeys.TRAIN)
        with tf.name_scope('logits'):
            logits = tf.layers.dense(inputs=drp, units=CLASSES)

        # Calculate Predictions
        predictions = {
            # Generate predictions (for PREDICT and EVAL mode)
            "classes": tf.argmax(input=logits, axis=1),
            # Add `softmax_tensor` to the graph. It is used for PREDICT and by the `logging_hook`.
            "probabilities": tf.nn.softmax(logits=logits,
                                           name="softmax_tensor")
        }

        print("\n", mode)
        if mode == tf.estimator.ModeKeys.PREDICT:
            spec = tf.estimator.EstimatorSpec(mode=mode,
                                              predictions=predictions)
            return spec

        # Calculate Loss (for both TRAIN and EVAL modes)
        onehot_labels = tf.one_hot(indices=tf.cast(labels, tf.int32),
                                   depth=3)
        print(onehot_labels)

        cross_entropy = tf.losses.softmax_cross_entropy(onehot_labels=onehot_labels,
                                                        logits=logits)    #softmax_cross_entropy(onehot_labels=onehot_labels, logits=logits)
        tf.summary.scalar('cross_entropy', cross_entropy)

        loss = tf.reduce_mean(cross_entropy)
        tf.summary.scalar('loss', loss)

        # Configure the Training Op (for TRAIN mode)
        if mode == tf.estimator.ModeKeys.TRAIN:
            # -------------------------- try different optimizers(GradientDescentOptimizer) -----------------------
            optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate, name="Adam_Optimizer")
            # optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate, name="SGD")
            train_op = optimizer.minimize(loss=loss, global_step=tf.train.get_global_step())
            spec = tf.estimator.EstimatorSpec(mode=mode,
                                              loss=loss,
                                              train_op=train_op)
            return spec

        # Add evaluation metrics (for EVAL mode)
        eval_metric_ops = {"accuracy": tf.metrics.accuracy(labels=labels,
                                                           predictions=predictions["classes"])}
        merged = tf.summary.merge_all()

        spec = tf.estimator.EstimatorSpec(mode=mode,
                                          loss=loss,
                                          eval_metric_ops=eval_metric_ops)
        return spec

# Seventh CNN Model
def cnn_model_fn_7(features, labels, mode):
    # with tf.device('/device:CPU:0'):
        with tf.name_scope('input_layer'):
            input_layer = tf.reshape(features["x"], [-1, WIDTH, HEIGHT, 3])

        with tf.name_scope('conv_1'):
            conv1 = create_conv_layer(inputs=input_layer,
                                      stride=2,
                                      conv_filter_size=5,
                                      num_filters=64,
                                      relu=False)

        with tf.name_scope('pool_1'):
            pool1 = tf.contrib.layers.max_pool2d(inputs=conv1,
                                                 kernel_size=2,
                                                 stride=1,
                                                 padding='SAME')

        with tf.name_scope('conv_2'):
            conv2 = create_conv_layer(inputs=pool1,
                                      stride=1,
                                      conv_filter_size=3,
                                      num_filters=128,
                                      relu=False)

        with tf.name_scope('pool_2'):
            pool2 = tf.contrib.layers.max_pool2d(inputs=conv2,
                                                 kernel_size=2,
                                                 stride=1,
                                                 padding='SAME')

        with tf.name_scope('conv_3'):
            conv3 = create_conv_layer(inputs=pool2,
                                      stride=1,
                                      conv_filter_size=3,
                                      num_filters=128,
                                      relu=False)

        with tf.name_scope('conv_4'):
            conv4 = create_conv_layer(inputs=conv3,
                                      stride=1,
                                      conv_filter_size=3,
                                      num_filters=128,
                                      relu=False)

        with tf.name_scope('conv_5'):
            conv5 = create_conv_layer(inputs=conv4,
                                      stride=1,
                                      conv_filter_size=3,
                                      num_filters=256,
                                      relu=False)

        with tf.name_scope('pool_3'):
            pool3 = tf.contrib.layers.max_pool2d(inputs=conv5,
                                                 kernel_size=3,
                                                 stride=1,
                                                 padding='SAME')

        with tf.name_scope('layer_flat'):
            layer_flat = tf.layers.flatten(inputs=pool3)

        with tf.name_scope('dense_1'):
            dense_1 = tf.layers.dense(inputs=layer_flat, units=4096)

        with tf.name_scope('drp_1'):
            drp_1 = tf.layers.dropout(inputs=dense_1, training=mode == tf.estimator.ModeKeys.TRAIN)

        with tf.name_scope('dense_2'):
            dense_2 = tf.layers.dense(inputs=drp_1, units=4096)

        with tf.name_scope('drp_2'):
            drp_2 = tf.layers.dropout(inputs=dense_2, training=mode == tf.estimator.ModeKeys.TRAIN)

        with tf.name_scope('logits'):
            # logits = tf.contrib.layers.fully_connected(inputs=drp_2,
            #                                            num_outputs=CLASSES) # tf.layers.dense(inputs=drp, units=CLASSES)
            logits = tf.layers.dense(inputs=drp_2, units=CLASSES)

        # Calculate Predictions
        predictions = {
            # Generate predictions (for PREDICT and EVAL mode)
            "classes": tf.argmax(input=logits, axis=1),
            # Add `softmax_tensor` to the graph. It is used for PREDICT and by the `logging_hook`.
            "probabilities": tf.nn.softmax(logits=logits,
                                           name="softmax_tensor")
        }

        print("\n", mode)
        if mode == tf.estimator.ModeKeys.PREDICT:
            spec = tf.estimator.EstimatorSpec(mode=mode,
                                              predictions=predictions)
            return spec

        # Calculate Loss (for both TRAIN and EVAL modes)
        onehot_labels = tf.one_hot(indices=tf.cast(labels, tf.int32),
                                   depth=3)
        print(onehot_labels)

        cross_entropy = tf.losses.softmax_cross_entropy(onehot_labels=onehot_labels,
                                                        logits=logits)  # softmax_cross_entropy(onehot_labels=onehot_labels, logits=logits)
        tf.summary.scalar('cross_entropy', cross_entropy)

        loss = tf.reduce_mean(cross_entropy)
        tf.summary.scalar('loss', loss)

        # Configure the Training Op (for TRAIN mode)
        if mode == tf.estimator.ModeKeys.TRAIN:
            # -------------------------- try different optimizers(GradientDescentOptimizer) -----------------------
            optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate, name="Adam_Optimizer")
            # optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate, name="SGD")
            train_op = optimizer.minimize(loss=loss, global_step=tf.train.get_global_step())
            spec = tf.estimator.EstimatorSpec(mode=mode,
                                              loss=loss,
                                              train_op=train_op)
            return spec

        # Add evaluation metrics (for EVAL mode)
        eval_metric_ops = {"accuracy": tf.metrics.accuracy(labels=labels,
                                                           predictions=predictions["classes"])}
        merged = tf.summary.merge_all()

        spec = tf.estimator.EstimatorSpec(mode=mode,
                                          loss=loss,
                                          eval_metric_ops=eval_metric_ops)
        return spec
