import tensorflow as tf
import numpy as np


def build(vgg19_npy_path, rgb):
    """
    input image is rgb image [batch, height, width, 3]
    Load params of VGG19 trained on imagenet; the params are downloaded from
    https://github.com/machrisaa/tensorflow-vgg as numpy compressed (npz) file
    """
    # params = np.load(vgg19_npy_path, encoding='latin1').item()

    # Load params from Keras
    from tensorflow.python.keras.applications.vgg19 import VGG19
    names = iter(['conv1_1', 'conv1_2',
                  'conv2_1', 'conv2_2',
                  'conv3_1', 'conv3_2', 'conv3_3', 'conv3_4',
                  'conv4_1', 'conv4_2', 'conv4_3', 'conv4_4',
                  'conv5_1', 'conv5_2', 'conv5_3', 'conv5_4'])
    keras = VGG19(include_top=False)
    params = {next(names): layer.get_weights() for layer in keras.layers[1:] if 'conv' in layer.name}

    def _avg_pool(input, name):
        return tf.nn.avg_pool(input, ksize=[1, 2, 2, 1],
                              strides=[1, 2, 2, 1], padding='SAME', name=name)

    def _conv_layer(input, name):
        kernel = tf.constant(params[name][0], name='kernel')
        bias = tf.constant(params[name][1], name='bias')

        conv_output = tf.nn.conv2d(input, kernel, [1, 1, 1, 1], padding='SAME')
        relu_output = tf.nn.relu(conv_output + bias)
        return relu_output

    graph = {}
    graph['conv1_1'] = _conv_layer(rgb, 'conv1_1')
    graph['conv1_2'] = _conv_layer(graph['conv1_1'], 'conv1_2')
    graph['pool1'] = _avg_pool(graph['conv1_2'], 'pool1')

    graph['conv2_1'] = _conv_layer(graph['pool1'], 'conv2_1')
    graph['conv2_2'] = _conv_layer(graph['conv2_1'], 'conv2_2')
    graph['pool2'] = _avg_pool(graph['conv2_2'], 'pool2')

    graph['conv3_1'] = _conv_layer(graph['pool2'], 'conv3_1')
    graph['conv3_2'] = _conv_layer(graph['conv3_1'], 'conv3_2')
    graph['conv3_3'] = _conv_layer(graph['conv3_2'], 'conv3_3')
    graph['conv3_4'] = _conv_layer(graph['conv3_3'], 'conv3_4')
    graph['pool3'] = _avg_pool(graph['conv3_4'], 'pool3')

    graph['conv4_1'] = _conv_layer(graph['pool3'], 'conv4_1')
    graph['conv4_2'] = _conv_layer(graph['conv4_1'], 'conv4_2')
    graph['conv4_3'] = _conv_layer(graph['conv4_2'], 'conv4_3')
    graph['conv4_4'] = _conv_layer(graph['conv4_3'], 'conv4_4')
    graph['pool4'] = _avg_pool(graph['conv4_4'], 'pool4')

    graph['conv5_1'] = _conv_layer(graph['pool4'], 'conv5_1')
    graph['conv5_2'] = _conv_layer(graph['conv5_1'], 'conv5_2')
    graph['conv5_3'] = _conv_layer(graph['conv5_2'], 'conv5_3')
    graph['conv5_4'] = _conv_layer(graph['conv5_3'], 'conv5_4')
    graph['pool5'] = _avg_pool(graph['conv5_4'], 'pool5')

    return graph
