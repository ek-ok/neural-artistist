import tensorflow as tf
import time
import numpy as np

class VGG19Truncate:
    def __init__(self, vgg19_params_path=None):
        # Load params of VGG19 trained on imagenet; the params are downloaded from
        # https://github.com/machrisaa/tensorflow-vgg as numpy compressed (npz) file
        self.params_dict = np.load(vgg19_params_path, encoding='latin1').item()
        self.vgg_mean = [103.939, 116.779, 123.68]
        print("VGG19 pre-trained params loaded")

    def build(self, input_image):

        """
        input image is rgb image [batch, height, width, 3]
        """

        start_time = time.time()
        print("start building model")

        input_image = input_image * 255.0
        print("input_image.shape", input_image.shape)

        # Convert RGB to BGR
        red, green, blue = tf.split(axis=3, num_or_size_splits=3, value=input_image)
        bgr = tf.concat(axis=3, values=[blue - self.vgg_mean[0],
                                        green - self.vgg_mean[1],
                                        red - self.vgg_mean[2]])

        self.conv1_1 = self.conv_layer(bgr, "conv1_1")
        self.conv1_2 = self.conv_layer(self.conv1_1, "conv1_2")
        self.pool1 = self.avg_pool(self.conv1_2, "pool1")

        self.conv2_1 = self.conv_layer(self.pool1, "conv2_1")
        self.conv2_2 = self.conv_layer(self.conv2_1, "conv2_2")
        self.pool2 = self.avg_pool(self.conv2_2, "pool2")

        self.conv3_1 = self.conv_layer(self.pool2, "conv3_1")
        self.conv3_2 = self.conv_layer(self.conv3_1, "conv3_2")
        self.conv3_3 = self.conv_layer(self.conv3_2, "conv3_3")
        self.conv3_4 = self.conv_layer(self.conv3_3, "conv3_4")
        self.pool3 = self.avg_pool(self.conv3_4, "pool3")

        self.conv4_1 = self.conv_layer(self.pool3, "conv4_1")
        self.conv4_2 = self.conv_layer(self.conv4_1, "conv4_2")
        self.conv4_3 = self.conv_layer(self.conv4_2, "conv4_3")
        self.conv4_4 = self.conv_layer(self.conv4_3, "conv4_4")
        self.pool4 = self.avg_pool(self.conv4_4, "pool4")

        self.conv5_1 = self.conv_layer(self.pool4, "conv5_1")
        self.conv5_2 = self.conv_layer(self.conv5_1, "conv5_2")
        self.conv5_3 = self.conv_layer(self.conv5_2, "conv5_3")
        self.conv5_4 = self.conv_layer(self.conv5_3, "conv5_4")
        self.pool5 = self.avg_pool(self.conv5_4, "pool5")

        # This clears away the memory used for data_dict
        self.data_dict = None
        print("model finished building; time taken = {}".format(time.time() - start_time))

    def avg_pool(self, input, name):
        with tf.variable_scope(name):
            return tf.nn.avg_pool(input, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=name)

    def conv_layer(self, input, name):
        with tf.variable_scope(name):
            kernel = self.get_conv_kernel(name)

            conv_output = tf.nn.conv2d(input, kernel, [1, 1, 1, 1], padding='SAME')

            conv_bias = self.get_bias(name)
            conv_output_w_bias = tf.nn.bias_add(conv_output, conv_bias)

            output = tf.nn.relu(conv_output_w_bias)
            return output

    def get_conv_kernel(self, name):
        return tf.constant(self.params_dict[name][0], name="kernel")

    def get_bias(self, name):
        return tf.constant(self.params_dict[name][1], name="bias")