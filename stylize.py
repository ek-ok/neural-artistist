import os

import numpy as np
import tensorflow as tf
from PIL import Image
import vgg19

VGG_MEAN = [123.68, 116.779, 103.939]

def load_image(filename, small_img):
    if small_img:
        rgb = Image.open(os.path.join('images_resized', filename))
    else:
        rgb = Image.open(os.path.join('images', filename))

    rgb = np.float32(rgb) - np.array(VGG_MEAN).reshape((1, 1, 3))

    bgr = rgb[:, :, ::-1]  # rgb to bgr
    bgr = bgr.reshape((1,) + bgr.shape)
    return bgr


def generate_noisy_image(content, noisy_img_content, noise_ratio):

    noise = np.random.uniform(low=-20, high=20, size=content.shape)

    if noisy_img_content:
        # add noise to content image as input
        image = noise*noise_ratio + content*(1 - noise_ratio)
    else:
        image = noise
    # else:
    #     # generate a white noise image as input
    #     image_shape = content.shape
    #     image_num_pixels = np.array(image_shape)
    #     image_num_pixels = image_num_pixels[0] * image_num_pixels[1] * image_num_pixels[2] * image_num_pixels[3]
    #     # ranf returns values  in [0, 1) drawing from a uniform distribution
    #     image = np.random.ranf(image_num_pixels) * 255
    #     image = image.reshape(image_shape)

    return image


def save_image(filename, bgr, output_clip_hard):
    bgr = bgr[0]
    rgb = bgr[:, :, ::-1]
    rgb += np.array(VGG_MEAN).reshape((1, 1, 3))

    if output_clip_hard:
        rgb = np.clip(rgb, 0, 255).astype('uint8')
    else:
        r = rgb[:, :, 0]
        g = rgb[:, :, 1]
        b = rgb[:, :, 2]

        r = (r - np.min(r)) / (np.max(r) - np.min(r))
        g = (g - np.min(g)) / (np.max(g) - np.min(g))
        b = (b - np.min(b)) / (np.max(b) - np.min(b))

        rgb = np.stack((r, g, b), axis=-1)
        rgb *= 255
        rgb = rgb.astype('uint8')

    image = Image.fromarray(rgb)
    image.save(os.path.join('output', filename))


def calculate_content_loss(vgg_image, vgg_content, layer_num=4, add_coef=False):
    """define loss function for content"""
    # layer = 'conv4_2'
    layer_list = ['conv1_2', 'conv2_2', 'conv3_2', 'conv4_2', 'conv5_2']

    loss = tf.nn.l2_loss(vgg_image[layer_list[layer_num - 1]] - vgg_content[layer_list[layer_num - 1]])

    n = tf.shape(vgg_content[layer_list[layer_num - 1]])[3]
    m = tf.shape(vgg_content[layer_list[layer_num - 1]])[1] * tf.shape(vgg_content[layer_list[layer_num - 1]])[2]

    print("type(loss), type(n), type(m)", type(loss), type(n), type(m))
    print("type(loss), type(n), type(m)", loss.dtype.as_numpy_dtype, n.dtype.as_numpy_dtype, m.dtype.as_numpy_dtype)

    if add_coef:
        coef = 1 / (2 * (n ** 2 * m ** 2))
        loss *= coef

    return loss


def calculate_style_loss(vgg_image, vgg_style, ws, num_layers=5, add_coef=False):
    """define loss function for style"""
    def _gram_matrix(conv):
        n = tf.shape(conv)[3]
        m = tf.shape(conv)[1] * tf.shape(conv)[2]

        conv = tf.reshape(conv, (m, n))
        gram = tf.matmul(tf.transpose(conv), conv)
        return gram, n, m

    layers = ['conv1_1', 'conv2_1', 'conv3_1', 'conv4_1', 'conv5_1']
    total_loss = 0

    for layer_num in range(num_layers):
        gram_style, n, m = _gram_matrix(vgg_style[layers[layer_num]])
        gram_image, _, _ = _gram_matrix(vgg_image[layers[layer_num]])

        loss = tf.nn.l2_loss(gram_style - gram_image)

        if add_coef:
            coef = 1 / (2 * (n ** 2 * m ** 2))
            loss *= coef

        total_loss += loss * ws[layer_num]

    return total_loss


# def apply(content_file, style_file, learning_rate, iterations, alpha, beta, noise_ratio):
def apply(content_file, style_file, config):

    vgg_npy_file = config['vgg_params_path']
    noise_ratio =  config['noise_ratio']
    alpha = config['alpha']
    beta = config['beta']
    iters = config['iters']
    learning_rate = config['learning_rate']
    noisy_img_content = config['noisy_img_content']
    style_loss_layers_w = config['style_loss_layers_w']
    num_style_layers = config['num_style_layers']
    content_layer_num = config['content_layer_num']
    small_content_img = config['small_size_content_img']
    small_style_img = config['small_size_style_img']
    optimizer = config['optimizer']
    loss_type = config['loss']
    add_coef = config['add_coef']
    output_clip_hard = config['output_clip_hard']
    pooling_method = config['pooling_method']
    pooling_stride = config['pooling_stride']

    # Load images and construct a noisy image
    content_image = load_image(content_file, small_content_img)
    style_image = load_image(style_file, small_style_img)
    image = generate_noisy_image(content_image, noisy_img_content, noise_ratio=noise_ratio)

    tf_content_image = tf.constant(content_image, dtype=tf.float32)
    tf_style_image = tf.constant(style_image, dtype=tf.float32)
    tf_image = tf.Variable(image, dtype=tf.float32)

    vgg_content = vgg19.build(vgg_npy_file, tf_content_image, pooling_method, pooling_stride)
    vgg_style = vgg19.build(vgg_npy_file, tf_style_image, pooling_method, pooling_stride)
    vgg_image = vgg19.build(vgg_npy_file, tf_image, pooling_method, pooling_stride)

    # get loss
    content_loss = calculate_content_loss(vgg_image, vgg_content, content_layer_num, add_coef)
    style_loss = calculate_style_loss(vgg_image, vgg_style, style_loss_layers_w, num_style_layers, add_coef)

    if loss_type == "all":
        total_loss = alpha * content_loss + beta * style_loss
    elif loss_type == "content":
        total_loss = alpha * content_loss
    else:
        total_loss = beta * style_loss

    if optimizer == "adam":
        step = tf.train.AdamOptimizer(learning_rate).minimize(total_loss)
    else:
        step = tf.train.AdamOptimizer(learning_rate).minimize(total_loss)

    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init)

        for i in range(iters):
            _, cur_total_loss, cur_content_loss, cur_style_loss = sess.run(
                    [step, total_loss, content_loss, style_loss])

            if i == 0:
                total_losses = [cur_total_loss]
                content_losses = [cur_content_loss]
                style_losses = [cur_style_loss]
            else:
                total_losses.append(cur_total_loss)
                content_losses.append(cur_content_loss)
                style_losses.append(cur_style_loss)

            if i % 100 == 0:
                print("Iteration: {}, total loss: {}, content loss: {}, style loss: {}".format(
                    i, cur_total_loss, cur_content_loss, cur_style_loss))

                output_image = tf_image.eval()
                save_image('output_{}.jpeg'.format(i), output_image, output_clip_hard)

        output_image = tf_image.eval()
        save_image('output_{}.jpeg'.format(i), output_image, output_clip_hard)
        np.save('output/total_losses.npy', np.array(total_losses))
        np.save('output/content_losses.npy', np.array(content_losses))
        np.save('output/style_losses.npy', np.array(style_losses))
        return output_image, total_losses, content_losses, style_losses