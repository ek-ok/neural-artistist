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


def write_image_csv(dirname, image, total_loss, content_loss, style_loss,
                    content_file, style_file, params, i, run_time):
    # Create an image file name
    without_ext = lambda f: os.path.splitext(f)[0]  # noqa E731
    content = without_ext(content_file)
    style = without_ext(style_file)
    sw = params['style_loss_layers_w'][-1]  # noqa F841
    image_filename = (f"{content}_{style}_"
                      f"w{params['new_width']}_"
                      f"i{i}_"
                      f"lr{params['learning_rate']}_"
                      f"a{params['alpha']}_"
                      f"b{params['beta']}_"
                      f"nr{params['noise_ratio']}_"
                      f"{params['optimizer']}_"
                      f"{params['pool_method']}_"
                      f"ps{params['pool_stride']}_"
                      f"sw{sw}_"
                      f"sn{params['style_num_layers']}_"
                      f"cn{params['content_layer_num']}_"
                      f"rt{run_time:.1f}.jpg")
    print(image_filename)

    # Save results to a csv
    csv_filename = 'outputs/results.csv'

    params['run_time'] = run_time
    params['total_loss'] = total_loss
    params['content_loss'] = content_loss
    params['style_loss'] = style_loss
    params['i'] = i
    params['last_style_loss_layers_w'] = sw

    cols = ['filename', 'i', 'run_time', 'total_loss', 'content_loss',
            'style_loss', 'learning_rate', 'iters', 'alpha', 'beta',
            'noise_ratio', 'optimizer', 'new_width', 'pool_method',
            'pool_stride', 'last_style_loss_layer_w', 'style_num_layers',
            'content_layer_num']

    header = not os.path.isfile(csv_filename)
    results = pd.DataFrame([params], columns=cols)
    results.to_csv(csv_filename, mode='a', header=header, index=False)

    # Write image
    img = write_image(dirname, image_filename, image)
    return img


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


def apply(content_file, style_file, **kwargs):
    """Apply neural style transfer to content image"""

    # Load images and construct a noisy image
    content_image = read_image(content_file, kwargs['new_width'])
    style_image = read_image(style_file, kwargs['new_width'])
    image = generate_noisey_image(content_image,
                                  noise_ratio=kwargs['noise_ratio'])

    shape = content_image.shape[1:-1]
    vgg = vgg19.build(shape, kwargs['pool_method'], kwargs['pool_stride'])

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    # Calculate content loss
    sess.run([vgg['input'].assign(content_image)])
    content_loss = calculate_content_loss(sess.run(vgg), vgg,
                                          kwargs['content_layer_num'])

    # Calculate style loss
    sess.run([vgg['input'].assign(style_image)])
    style_loss = calculate_style_loss(sess.run(vgg), vgg,
                                      kwargs['style_loss_layers_w'],
                                      kwargs['style_num_layers'])

    # Calculate total loss
    total_loss = kwargs['alpha']*content_loss + kwargs['beta']*style_loss

    # Feed data to vgg and optimize
    t_start = time.time()

    if kwargs['optimizer'] == "adam":
        step = tf.train.AdamOptimizer(kwargs['learning_rate']) \
                       .minimize(total_loss)
        sess.run(tf.global_variables_initializer())
        sess.run(vgg['input'].assign(image))

        for i in range(kwargs['iters']):
            sess.run(step)
            if i % 100 == 0:
                tf_outputs = sess.run([vgg['input'], total_loss,
                                       content_loss, style_loss])

                # Save interim image and outputs
                write_image_csv('interim', *tf_outputs, content_file, 
                                style_file, kwargs, i=i, run_time=0)

    elif kwargs['optimizer'] == 'lbfgs':
        train_step = tf.contrib.opt.ScipyOptimizerInterface(
                total_loss, method='L-BFGS-B',
                options={'maxiter': kwargs['iters']})
        sess.run(tf.global_variables_initializer())
        sess.run(vgg['input'].assign(image))
        train_step.minimize(sess)
        tf_outputs = sess.run([vgg['input'], total_loss,
                              content_loss, style_loss])

    # All done, save the final image and results
    t_end = time.time()
    run_time = (t_end - t_start) / kwargs['iters']
    image = write_image_csv('final', *tf_outputs, content_file, style_file,
                            kwargs, i=kwargs['iters'], run_time=run_time)

    return image
