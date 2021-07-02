
import tensorflow as tf


def conv2d(input_feature, f_size, f_num, s, normal=False, activation='relu', name='conv2d'):
    with tf.variable_scope(name):
        weights = init_weights('weights', [f_size, f_size, input_feature.shape[3], f_num]) 
        biases = init_biases('biases', [f_num])

        conv = tf.add(tf.nn.conv2d(input_feature, weights, strides=[1, s, s, 1], 
                                   padding='SAME', name=name), biases)
        if normal:
            conv = tf.layers.batch_normalization(conv, name='batch_normal')
        if activation == 'relu':
            output = tf.nn.relu(conv)
        elif activation == 'tanh':
            output = tf.nn.tanh(conv)
        # elif activation == 'leakyrelu':
        #     output = tf.nn.maximum(0.2*conv, conv)
        else:
            output = conv
        return output

def full_conv(input_layer, in_num, out_num, name='fc'):
    with tf.variable_scope(name):
        weights = init_weights('weights', [in_num, out_num])
        biases = init_biases('biases', [out_num])
        fc = tf.add(tf.matmul(input_layer, weights, name=name), biases)
        output = tf.nn.relu(fc)
        return output


def init_weights(name, shape, mean=0.0, stddev=0.01):
    weights = tf.get_variable(name, shape,
                          initializer=tf.truncated_normal_initializer(mean=mean, 
                                                                   stddev=stddev,
                                                                   dtype=tf.float32))
    # weights = tf.get_variable(name, shape, initializer=tf.contrib.layers.xavier_initializer())
            
    return weights

def init_biases(name, shape, constant=0.0):
    return tf.get_variable(name, shape,
              initializer=tf.constant_initializer(constant))
