import numpy as np
import tensorflow as tf
from PIL import Image
import vgg19_truncate

# define loss function for content
def get_content_loss(vgg_image, vgg_content, layer=4):

    content_op_list = [vgg_content.conv1_2, vgg_content.conv2_2, vgg_content.conv3_2,
                       vgg_content.conv4_2, vgg_content.conv5_2]

    img_op_list = [vgg_image.conv1_2, vgg_image.conv2_2, vgg_image.conv3_2,
                   vgg_image.conv4_2, vgg_image.conv5_2]

    content_loss = tf.nn.l2_loss(content_op_list[layer-1] - img_op_list[layer-1])

    return content_loss


# define loss function for style
def get_style_loss(vgg_image, vgg_style, layers=5):

    style_op_list = [vgg_style.conv1_2, vgg_style.conv2_2, vgg_style.conv3_2, vgg_style.conv4_2, vgg_style.conv5_2]
    img_op_list = [vgg_image.conv1_2, vgg_image.conv2_2, vgg_image.conv3_2, vgg_image.conv4_2, vgg_image.conv5_2]

    total_style_loss = 0

    for layer_op_num in range(layers):

        # since there is only one style image, there is only one output; get the first output
        cur_style_op = style_op_list[layer_op_num][0]

        #flatten the w and h dim of output of current layer; retain channel as a separate dim
        cur_style_op = tf.reshape(cur_style_op, shape=(-1, style_op_list[layer_op_num][3]))

        # do the same for output for desired image
        cur_img_op = img_op_list[layer_op_num][0]
        cur_img_op = tf.reshape(cur_img_op, shape=(-1, img_op_list[layer_op_num][3]))

        # get style loss from current layer
        cur_style_loss = tf.nn.l2_loss(tf.matmul(tf.transpose(cur_style_op), cur_style_op) -
                                    tf.matmul(tf.transpose(cur_img_op), cur_img_op))

        cur_style_loss /= tf.to_float(tf.multiply(tf.square(tf.shape(style_op_list[layer_op_num])[1]),
                                  tf.square(tf.shape(style_op_list[layer_op_num])[3]))) * 2

        total_style_loss += cur_style_loss


    # scale the loss by the number of layers
    total_style_loss /= layers

    return total_style_loss


# define total loss function
def get_total_loss(alpha, beta, content_loss, style_loss):

    total_loss = alpha * content_loss + beta * style_loss

    return total_loss


# define gradient descent function
def train_step(loss, learning_rate=5e-3, optimizer="adam"):

    with tf.name_scope('train_step'):
        if optimizer == "adam":
            print("in step")
            step = tf.train.AdamOptimizer(learning_rate).minimize(loss)
        else:
            step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)

    return step

def normalize_img(img_array):
    # normalize image to range of [0, 1]
    img_array = (img_array - np.min(img_array)) / (np.max(img_array) - np.min(img_array))
    img_array = img_array.reshape((1,) + img_array.shape)

    return img_array


def get_image(image_path, content_image, style_image, vgg_params_path, iterations=10, alpha_beta_ratio=0.001):

    content_image1 = np.array(Image.open(image_path + content_image), dtype='float32')
    style_image1  = np.array(Image.open(image_path + style_image), dtype='float32')

    # normalize image to range of [0, 1]
    content_image1 = normalize_img(content_image1)
    style_image1 = normalize_img(style_image1)

    # construct noisy image
    image_shape = content_image1.shape
    image_num_pixels = np.array(image_shape)
    image_num_pixels = image_num_pixels[0] * image_num_pixels[1] * image_num_pixels[2] * image_num_pixels[3]
    # ranf returns values  in [0, 1) drawing from a uniform distribution
    image = np.random.ranf(image_num_pixels)
    image = image.reshape(image_shape)

    # alpha + beta = 1
    beta = 1 / (1 + alpha_beta_ratio)
    alpha = alpha_beta_ratio * beta
    print("beta = {}, alpha = {}".format(beta, alpha))

    tf_image = tf.Variable(initial_value=image, dtype=tf.float32)
    tf_content_image = tf.constant(value=content_image1, dtype=tf.float32)
    tf_style_image = tf.constant(value=style_image1, dtype=tf.float32)

    # create 3 truncated VGG models - one for image, one for content, one for style
    vgg_image = vgg19_truncate.vgg19_truncate(vgg_params_path)
    vgg_content = vgg19_truncate.vgg19_truncate(vgg_params_path)
    vgg_style = vgg19_truncate.vgg19_truncate(vgg_params_path)

    vgg_image.build(tf_image)
    vgg_content.build(tf_content_image)
    vgg_style.build(tf_style_image)


    # get loss
    content_loss = get_content_loss(vgg_image, vgg_content)
    style_loss = get_style_loss(vgg_image, vgg_style)
    total_loss = get_total_loss(alpha, beta, content_loss, style_loss)

    # perform gradient descent on image
    step = train_step(total_loss)

    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init)

        for i in range(iterations):
            print(i)
            _, cur_total_loss, cur_content_loss, cur_style_loss = sess.run([step,
                                                                            total_loss,
                                                                            content_loss,
                                                                            style_loss])

            print("Total loss = {}, content loss = {}, style loss = {}:".format(cur_total_loss,
                                                                                cur_content_loss,
                                                                                cur_style_loss))

        final_image = tf_image.eval()

        print("final_image.shape", final_image.shape)
        b = final_image[:, :, :, 0]
        g = final_image[:, :, :, 1]
        r = final_image[:, :, :, 2]

        final_image = np.stack((r, g, b), axis=-1)

    np.save("./image.npy", final_image)
    return final_image