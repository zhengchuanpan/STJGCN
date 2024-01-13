import tensorflow as tf

def conv2d(x, output_dims, kernel_size, stride = [1, 1], padding = 'SAME',
           use_bias = True, activation = tf.nn.relu, is_training = None):
    input_dims = x.get_shape()[-1].value
    kernel_shape = kernel_size + [input_dims, output_dims]
    kernel = tf.Variable(
        tf.glorot_uniform_initializer()(shape = kernel_shape),
        dtype = tf.float32, trainable = True)
    x = tf.nn.conv2d(x, kernel, [1] + stride + [1], padding = padding)
    if use_bias:
        bias = tf.Variable(
            tf.zeros_initializer()(shape = [output_dims]),
            dtype = tf.float32, trainable = True)
        x = tf.nn.bias_add(x, bias)
    if activation is not None:
        x = batch_norm(x, is_training = is_training)
        x = activation(x)
    return x

def batch_norm(x, is_training):
    input_dims = x.get_shape()[-1].value
    moment_dims = list(range(len(x.get_shape()) - 1))
    beta = tf.Variable(
        tf.zeros_initializer()(shape = [input_dims]),
        dtype = tf.float32, trainable = True)
    gamma = tf.Variable(
        tf.ones_initializer()(shape = [input_dims]),
        dtype = tf.float32, trainable = True)
    batch_mean, batch_var = tf.nn.moments(x, moment_dims)
    ema = tf.train.ExponentialMovingAverage(decay = 0.9)
    # Operator that maintains moving averages of variables.
    ema_apply_op = tf.cond(
        is_training,
        lambda: ema.apply([batch_mean, batch_var]), lambda: tf.no_op())
    # Update moving average and return current batch's avg and var.
    def mean_var_with_update():
        with tf.control_dependencies([ema_apply_op]):
            return tf.identity(batch_mean), tf.identity(batch_var)
    # ema.average returns the Variable holding the average of var.
    mean, var = tf.cond(
        is_training, mean_var_with_update,
        lambda: (ema.average(batch_mean), ema.average(batch_var)))
    x = tf.nn.batch_normalization(x, mean, var, beta, gamma, 1e-3)
    return x
