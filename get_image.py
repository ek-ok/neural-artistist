import numpy as np
import tensorflow as tf
from PIL import Image
import vgg19_truncate

# define loss function for content
def get_content_loss(vgg_image, vgg_content, layer="conv4_2"):

    if layer == "conv4_2":
        content_loss = tf.nn.l2_loss(vgg_content.conv4_2 - vgg_image.conv4_2)

    # print("content_loss=", content_loss)

    return content_loss

# define loss function for style
def get_style_loss(vgg_image, vgg_style, layer="conv5_2"):

    if layer == "conv5_2":

        # since there is only one style image, there is only one output; get the first output
        conv5_2_style_op = vgg_style.conv5_2[0]
        #flatten the w and h dim of output of conv5_2; retain channel as a separate dim
        conv5_2_style_op = tf.reshape(conv5_2_style_op, shape=(-1, vgg_style.conv5_2.shape[3]))
        # do the same for output for desired image
        conv5_2_img_op = vgg_image.conv5_2[0]
        conv5_2_img_op = tf.reshape(conv5_2_img_op, shape=(-1, vgg_image.conv5_2.shape[3]))

        #print("conv5_2_style_op.shape=", conv5_2_style_op.shape)

        # get style loss from conv5_2
        style_loss5 = tf.nn.l2_loss(tf.matmul(tf.transpose(conv5_2_style_op), conv5_2_style_op) -
                                    tf.matmul(tf.transpose(conv5_2_img_op), conv5_2_img_op))

        #print("style_loss.dtype", style_loss.dtype)

        # get style loss from conv5_2
        style_loss5 /= tf.to_float(tf.multiply(tf.square(tf.shape(vgg_style.conv5_2)[1]),
                                  tf.square(tf.shape(vgg_style.conv5_2)[3]))) * 2


        # do the same for conv4_2 like we do for conv5_2
        conv4_2_style_op = vgg_style.conv4_2[0]
        conv4_2_style_op = tf.reshape(conv4_2_style_op, shape=(-1, vgg_style.conv4_2.shape[3]))
        conv4_2_img_op = vgg_image.conv4_2[0]
        conv4_2_img_op = tf.reshape(conv4_2_img_op, shape=(-1, vgg_image.conv4_2.shape[3]))

        style_loss4 = tf.nn.l2_loss(tf.matmul(tf.transpose(conv4_2_style_op), conv4_2_style_op) -
                                    tf.matmul(tf.transpose(conv4_2_img_op), conv4_2_img_op))

        style_loss4 /= tf.to_float(tf.multiply(tf.square(tf.shape(vgg_style.conv4_2)[1]),
                                  tf.square(tf.shape(vgg_style.conv4_2)[3]))) * 2

        conv3_2_style_op = vgg_style.conv3_2[0]
        conv3_2_style_op = tf.reshape(conv3_2_style_op, shape=(-1, vgg_style.conv3_2.shape[3]))
        conv3_2_img_op = vgg_image.conv3_2[0]
        conv3_2_img_op = tf.reshape(conv3_2_img_op, shape=(-1, vgg_image.conv3_2.shape[3]))

        style_loss3 = tf.nn.l2_loss(tf.matmul(tf.transpose(conv3_2_style_op), conv3_2_style_op) -
                                    tf.matmul(tf.transpose(conv3_2_img_op), conv3_2_img_op))

        style_loss3 /= tf.to_float(tf.multiply(tf.square(tf.shape(vgg_style.conv3_2)[1]),
                                  tf.square(tf.shape(vgg_style.conv3_2)[3]))) * 2

        conv2_2_style_op = vgg_style.conv2_2[0]
        conv2_2_style_op = tf.reshape(conv2_2_style_op, shape=(-1, vgg_style.conv2_2.shape[3]))
        conv2_2_img_op = vgg_image.conv2_2[0]
        conv2_2_img_op = tf.reshape(conv2_2_img_op, shape=(-1, vgg_image.conv2_2.shape[3]))

        style_loss2 = tf.nn.l2_loss(tf.matmul(tf.transpose(conv2_2_style_op), conv2_2_style_op) -
                                    tf.matmul(tf.transpose(conv2_2_img_op), conv2_2_img_op))

        style_loss2 /= tf.to_float(tf.multiply(tf.square(tf.shape(vgg_style.conv2_2)[1]),
                                  tf.square(tf.shape(vgg_style.conv2_2)[3]))) * 2

        conv1_2_style_op = vgg_style.conv1_2[0]
        conv1_2_style_op = tf.reshape(conv1_2_style_op, shape=(-1, vgg_style.conv1_2.shape[3]))
        conv1_2_img_op = vgg_image.conv1_2[0]
        conv1_2_img_op = tf.reshape(conv1_2_img_op, shape=(-1, vgg_image.conv1_2.shape[3]))

        style_loss1 = tf.nn.l2_loss(tf.matmul(tf.transpose(conv1_2_style_op), conv1_2_style_op) -
                                    tf.matmul(tf.transpose(conv1_2_img_op), conv1_2_img_op))

        style_loss1 /= tf.to_float(tf.multiply(tf.square(tf.shape(vgg_style.conv1_2)[1]),
                                  tf.square(tf.shape(vgg_style.conv1_2)[3]))) *  2


        style_loss = style_loss5 + style_loss4 + style_loss3 + style_loss2 + style_loss1

        style_loss *= 0.2

    # print("style_loss=", style_loss)

    return style_loss


# define total loss function
def get_total_loss(alpha, beta, vgg_image, vgg_content, vgg_style):

    total_loss = alpha * get_content_loss(vgg_image, vgg_content) + beta * get_style_loss(vgg_image, vgg_style)

    print("total_loss=", total_loss)

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


def get_image(image_path, content_image, style_image, vgg_params_path, iterations=10, alpha_beta_ratio=0.001):

    content_image = np.array(Image.open(image_path + content_image), dtype='float32')
    style_image  = np.array(Image.open(image_path + style_image), dtype='float32')

    content_image1 = content_image.reshape((1,) + content_image.shape)
    style_image1 = style_image.reshape((1,) + style_image.shape)

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

    vgg_image = vgg19_truncate.vgg19_truncate(vgg_params_path)
    vgg_content = vgg19_truncate.vgg19_truncate(vgg_params_path)
    vgg_style = vgg19_truncate.vgg19_truncate(vgg_params_path)

    vgg_image.build(tf_image)
    vgg_content.build(tf_content_image)
    vgg_style.build(tf_style_image)

    content_loss = get_content_loss(vgg_image, vgg_content)
    style_loss = get_style_loss(vgg_image, vgg_style)
    total_loss = get_total_loss(alpha, beta, vgg_image, vgg_content, vgg_style)

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

    np.save("./image.npy", final_image)
    return final_image