import os
import time

import numpy as np
import pandas as pd
import tensorflow as tf
from PIL import Image

import vgg19


VGG_MEAN = [123.68, 116.779, 103.939]


def read_image(filename, new_width):
    """
    Load an image and convert from rgb to bgr.
    Resize image if new_width is set
    """
    rgb = Image.open(os.path.join('inputs', filename))

    # Resize image if new_width is set
    if new_width:
        original_size = rgb.size
        new_height = int(new_width * 0.75)  # hardcode the size to be 4x6
        rgb = rgb.resize((new_width, new_height), Image.ANTIALIAS)

        msg = '{} was resized from {} to {}'
        print(msg.format(filename, original_size, rgb.size))

    rgb = np.float32(rgb) - np.array(VGG_MEAN).reshape((1, 1, 3))

    bgr = rgb[:, :, ::-1]  # rgb to bgr
    bgr = bgr.reshape((1,) + bgr.shape)
    return bgr


def write_image(dirname, filename, bgr):
    """Convert an image from bgr to rgb and save to a file"""
    bgr = bgr[0]

    rgb = bgr[:, :, ::-1]  # bgr to rgb
    rgb += np.array(VGG_MEAN).reshape((1, 1, 3))
    rgb = np.clip(rgb, 0, 255).astype('uint8')

    image = Image.fromarray(rgb)
    path = os.path.join('outputs', dirname, filename)
    image.save(path)
    return image


def generate_noisey_image(content, noise_ratio):
    """Add some noise to the content image"""
    noise = np.random.uniform(low=-20, high=20, size=content.shape)
    image = noise*noise_ratio + content*(1 - noise_ratio)
    return image


def calculate_content_loss(vgg_image, vgg_content, layer_num):
    """define loss function for content"""
    layer_list = ['conv1_2', 'conv2_2', 'conv3_2', 'conv4_2', 'conv5_2']

    layer = layer_list[layer_num - 1]
    image_layer = vgg_image[layer]
    content_layer = vgg_content[layer]

    n = tf.shape(content_layer)[3]
    m = tf.shape(content_layer)[1] * tf.shape(content_layer)[2]
    n, m = map(lambda x: tf.cast(x, tf.float32), [n, m])

    loss = tf.nn.l2_loss(image_layer - content_layer) / (2 * (n**2 * m**2))
    return loss


def calculate_style_loss(vgg_image, vgg_style, ws, num_layers):
    """define loss function for style"""
    def _gram_matrix(conv):
        n = tf.shape(conv)[3]
        m = tf.shape(conv)[1] * tf.shape(conv)[2]

        conv = tf.reshape(conv, (m, n))
        gram = tf.matmul(tf.transpose(conv), conv)
        gram, n, m = map(lambda x: tf.cast(x, tf.float32), [gram, n, m])
        return gram, n, m

    layers = ['conv1_1', 'conv2_1', 'conv3_1', 'conv4_1', 'conv5_1']
    total_loss = 0

    for layer_num in range(num_layers):
        gram_style, n, m = _gram_matrix(vgg_style[layers[layer_num]])
        gram_image, _, _ = _gram_matrix(vgg_image[layers[layer_num]])

        loss = tf.nn.l2_loss(gram_style - gram_image) / (2 * (n**2 * m**2))
        total_loss += loss * ws[layer_num]

    return total_loss


def apply(content_file, style_file, learning_rate, iters, alpha, beta,
          noise_ratio, optimizer='adam', new_width=None,
          pool_method='avg', pool_stride=2,
          style_loss_layers_w=(0.2, 0.2, 0.2, 0.2, 0.2), style_num_layers=5,
          content_layer_num=4):
    """Apply neural style transfer to content image"""

    def _save_results(filename, i, total_loss, content_loss, style_loss):
        results_file = 'outputs/results.csv'
        vals = [filename, i, total_loss, content_loss, style_loss,
                learning_rate, iters, alpha, beta, noise_ratio, optimizer,
                new_width, pool_method, pool_stride, style_loss_layers_w[-1],
                style_num_layers, content_layer_num]

        cols = ['filename', 'i', 'total_loss', 'content_loss', 'style_loss',
                'learning_rate', 'iters', 'alpha', 'beta',
                'noise_ratio', 'optimizer', 'new_width', 'pool_method',
                'pool_stride', 'last_style_loss_layer_w', 'style_num_layers',
                'content_layer_num']

        results = pd.DataFrame([vals], columns=cols)
        header = not os.path.isfile(results_file)
        results.to_csv(results_file, mode='a', header=header, index=False)

    # Create output_file as output_content_style.jpg
    output_file = '{content}_{style}_w{w}_{i}_lr{lr}_a{a}_b{b}_nr{nr}_{opt}_{pm}_ps{ps}_sw{sw}_sn{sn}_cn{cn}_tm_rt{rt}.jpg' # noqa E501
    without_ext = lambda f: os.path.splitext(f)[0]  # noqa E731
    output_file = output_file.format(
            content=without_ext(content_file),
            style=without_ext(style_file),
            w=new_width, i='{i}', lr=learning_rate, a=alpha, b=beta,
            nr=noise_ratio, opt=optimizer, pm=pool_method, ps=pool_stride,
            sw=style_loss_layers_w[-1],
            sn=style_num_layers, cn=content_layer_num, rt='{rt}')

    # Load images and construct a noisy image
    content_image = read_image(content_file, new_width)
    style_image = read_image(style_file, new_width)
    image = generate_noisey_image(content_image, noise_ratio=noise_ratio)

    shape = content_image.shape[1:-1]
    vgg = vgg19.build(shape, pool_method, pool_stride)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    # Calculate content loss
    sess.run([vgg['input'].assign(content_image)])
    content_loss = calculate_content_loss(sess.run(vgg), vgg,
                                          content_layer_num)

    # Calculate style loss
    sess.run([vgg['input'].assign(style_image)])
    style_loss = calculate_style_loss(sess.run(vgg), vgg,
                                      style_loss_layers_w, style_num_layers)

    # Calculate total loss
    total_loss = alpha*content_loss + beta*style_loss

    # Feed data to vgg and optimize
    t_start = time.time()
    msg = 'Iteration:{}, total loss:{}, content loss:{}, style loss:{}'

    if optimizer == "adam":
        step = tf.train.AdamOptimizer(learning_rate).minimize(total_loss)
        sess.run(tf.global_variables_initializer())
        sess.run(vgg['input'].assign(image))

        for i in range(iters):
            sess.run(step)
            if i % 100 == 0:
                tf_outputs = sess.run([vgg['input'], total_loss,
                                       content_loss, style_loss])
                output_image = tf_outputs[0]
                output_file = output_file.format(i=i, rt='')
                write_image('interim', output_file, tf_outputs[0])
                _save_results(output_file, i, *tf_outputs[1:])

                # Edit just to print
                tf_outputs[0] = i
                print(msg.format(*tf_outputs))

    elif optimizer == 'lbfgs':
        train_step = tf.contrib.opt.ScipyOptimizerInterface(
                total_loss, method='L-BFGS-B', options={'maxiter': iters})
        sess.run(tf.global_variables_initializer())
        sess.run(vgg['input'].assign(image))
        train_step.minimize(sess)
        tf_outputs = sess.run([vgg['input'], total_loss,
                              content_loss, style_loss])
        output_image = tf_outputs[0]

    # All done, save the final image and results
    t_end = time.time()
    run_time = (t_end - t_start) / iters
    output_file = output_file.format(i=iters, rt=run_time)
    _save_results(output_file, iters, *tf_outputs[1:])
    image = write_image('final', output_file, output_image)

    return image
