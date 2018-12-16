# neural-artistist

`neural-artistist` is an implementation of [Image Style Transfer Using Convolutional Neural Networks](https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Gatys_Image_Style_Transfer_CVPR_2016_paper.pdf)

# How to run

You will need to download  [our data on Google Drive](https://drive.google.com/open?id=1rzTQRcquj63vCXT5xkue8sVcXjhf22t5) and place it to the root directory of `neural-artistist`. It contains the pre-trained VGG19 model, `vgg19.npy`, input images are stored in `inputs`, and output images and summary statistics will be exported to `outputs`

Then apply style to a content image by

    import stylize

    content = 'tubingen.jpg'
    style = 'shipwreck.jpg'

    stylize.apply(content, style, learning_rate=1.0,iters=100,
                  alpha=1, beta=5, noise_ratio=0.1, new_width=300, pool_method='avg',
                  pool_stride=2, style_loss_layers_w=(0.2, 0.2, 0.2, 0.2, 0.2),
                  style_num_layers=5, content_layer_num=4, optimizer='lbfgs')

# Our results

You can see our final results in [stylize_image.ipynb](stylize_image.ipynb).
