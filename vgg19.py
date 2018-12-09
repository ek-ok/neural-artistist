import numpy as np
import tensorflow as tf


def build(rgb, pool_method, pool_stride):
    """
    input image is rgb image [batch, height, width, 3]
    Load params of VGG19 trained on imagenet; the params are downloaded from
    https://github.com/machrisaa/tensorflow-vgg as numpy compressed (npz) file
    """
    params = np.load('vgg19.npy', encoding='latin1').item()

    def _pool(input_, name, method, stride):
        if method == 'avg':
            return tf.nn.avg_pool(input_,
                                  ksize=[1, stride, stride, 1],
                                  strides=[1, stride, stride, 1],
                                  padding='SAME',
                                  name=name)
        elif method == 'max':
            return tf.nn.max_pool(input_,
                                  ksize=[1, stride, stride, 1],
                                  strides=[1, stride, stride, 1],
                                  padding='SAME',
                                  name=name)
        else:
            raise ValueError('Invalid pool_method')

    def _conv_layer(input_, name):
        kernel = tf.constant(params[name][0], name='kernel')
        bias = tf.constant(params[name][1], name='bias')

        conv_output = tf.nn.conv2d(input_, kernel,
                                   strides=[1, 1, 1, 1], padding='SAME')
        relu_output = tf.nn.relu(conv_output + bias)
        return relu_output

    graph = {}
    graph['conv1_1'] = _conv_layer(rgb, 'conv1_1')
    graph['conv1_2'] = _conv_layer(graph['conv1_1'], 'conv1_2')
    graph['pool1'] = _pool(graph['conv1_2'], 'pool1', pool_method, pool_stride)

    graph['conv2_1'] = _conv_layer(graph['pool1'], 'conv2_1')
    graph['conv2_2'] = _conv_layer(graph['conv2_1'], 'conv2_2')
    graph['pool2'] = _pool(graph['conv2_2'], 'pool2', pool_method, pool_stride)

    graph['conv3_1'] = _conv_layer(graph['pool2'], 'conv3_1')
    graph['conv3_2'] = _conv_layer(graph['conv3_1'], 'conv3_2')
    graph['conv3_3'] = _conv_layer(graph['conv3_2'], 'conv3_3')
    graph['conv3_4'] = _conv_layer(graph['conv3_3'], 'conv3_4')
    graph['pool3'] = _pool(graph['conv3_4'], 'pool3', pool_method, pool_stride)

    graph['conv4_1'] = _conv_layer(graph['pool3'], 'conv4_1')
    graph['conv4_2'] = _conv_layer(graph['conv4_1'], 'conv4_2')
    graph['conv4_3'] = _conv_layer(graph['conv4_2'], 'conv4_3')
    graph['conv4_4'] = _conv_layer(graph['conv4_3'], 'conv4_4')
    graph['pool4'] = _pool(graph['conv4_4'], 'pool4', pool_method, pool_stride)

    graph['conv5_1'] = _conv_layer(graph['pool4'], 'conv5_1')
    graph['conv5_2'] = _conv_layer(graph['conv5_1'], 'conv5_2')
    graph['conv5_3'] = _conv_layer(graph['conv5_2'], 'conv5_3')
    graph['conv5_4'] = _conv_layer(graph['conv5_3'], 'conv5_4')
    graph['pool5'] = _pool(graph['conv5_4'], 'pool5', pool_method, pool_stride)

    return graph
