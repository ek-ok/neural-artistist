# Neural Artistist

This project reproduced the results of [A Neural Algorithm of Artistic Style](https://arxiv.org/pdf/1508.06576.pdf) by Leon A. Gatys, Alexander S. Ecker and Matthias Bethge in 2015.
Also conducted a series of experimentations to obtain deeper insights into deep learning mechanisms for style and content transfer in images.

The above paper was followed by [Image Style Transfer Using Convolutional Neural Networks](https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Gatys_Image_Style_Transfer_CVPR_2016_paper.pdf) in 2016 IEEE Conference on Computer Vision and Pattern Recognition, by the same authors.

# Results

Output | Input Content | Input Style
:-----:|:--------:|:--------:
<img src="outputs/hawaii_composition7_w600_i2000_lr1.0_alpha1_beta2_nr0.1_lbfgs_avg_ps2_sllw0.2_snl5_cln4_time2.6.jpg" alt="drawing" width="300"/> | <img src="inputs/hawaii.jpg" alt="drawing" width="300"/> | <img src="inputs/composition7.jpg" alt="drawing" width="300"/>
<img src="outputs/hawaii2_wave_w600_i200_lr1.0_alpha1_beta4_nr0.1_lbfgs_avg_ps2_sllw0.2_snl5_cln4_time2.2.jpg" alt="drawing" width="300"/> | <img src="inputs/hawaii2.jpg" alt="drawing" width="300"/> | <img src="inputs/wave.jpg" alt="drawing" width="300"/>
<img src="outputs/final/csjl_starry_night_w300_i70_lr1.0_alpha1_beta10_nr0.6_lbfgs_avg_ps2_sllw0.2_snl5_cln4_time0.2.jpg" alt="drawing" width="300"/> | <img src="inputs/csjl.jpeg" alt="drawing" width="300"/> | <img src="inputs/starry_night.jpg" alt="drawing" width="300"/>

Other results replicating the paper

Output | Input Content | Input Style
:-----:|:--------:|:--------:
<img src="outputs/final/tubingen_shipwreck_w300_i100_lr1.0_alpha1_beta5_nr0.1_lbfgs_avg_ps2_sllw0.2_snl5_cln4_time0.2.jpg" alt="drawing" width="300"/> | <img src="inputs/tubingen.jpg" alt="drawing" width="300"/> | <img src="inputs/shipwreck.jpg" alt="drawing" width="300"/>
<img src="outputs/final/tubingen_starry_night_w300_i100_lr1.0_alpha100_beta1_nr0.1_lbfgs_avg_ps2_sllw0.2_snl5_cln4_time1.1.jpg" alt="drawing" width="300"/> | <img src="inputs/tubingen.jpg" alt="drawing" width="300"/> | <img src="inputs/starry_night.jpg" alt="drawing" width="300"/>
<img src="outputs/final/tubingen_scream_w300_i40_lr2.0_alpha1_beta1000_nr0.1_lbfgs_avg_ps2_sllw0.2_snl5_cln4_time0.2.jpg" alt="drawing" width="300"/> | <img src="inputs/tubingen.jpg" alt="drawing" width="300"/> | <img src="inputs/scream.jpg" alt="drawing" height="230"/>
<img src="outputs/final/tubingen_seated_nude_w300_i200_lr1.0_alpha1_beta1000_nr0.7_lbfgs_avg_ps2_sllw0.2_snl5_cln4_time0.2.jpg" alt="drawing" width="300"/> | <img src="inputs/tubingen.jpg" alt="drawing" width="300"/> | <img src="inputs/seated_nude.jpg" alt="drawing" height="230"/>
<img src="outputs/final/tubingen_composition7_w300_i300_lr1.0_alpha1_beta1000_nr0.6_lbfgs_avg_ps2_sllw0.2_snl5_cln4_time0.2.jpg" alt="drawing" width="300"/> | <img src="inputs/tubingen.jpg" alt="drawing" width="300"/> | <img src="inputs/composition7.jpg" alt="drawing" width="300"/>


# Data

- `inputs`: It contains the style and content images used to reproduce all the results of the original pager. It is also used for our additional experiments.

- `vgg19.npy`: You would need to download the VGG19 pre-trained weights, [vgg19.npy](https://drive.google.com/open?id=1dvv0XiR1nmJVO06EdqLcesJoNYqIcPx7) from Google Drive and place `vgg19.npy` in the root directory of this project.

# Code

- `vgg19.py`: creates vgg19 network

- `stylize.py`: performs style transfer; saves the output

- `stylize_image.ipynb`: notebook to present the result


# How to run

As mentioned above, please make sure to download the pre-trained VGG19 model, [vgg19.npy](https://drive.google.com/open?id=1dvv0XiR1nmJVO06EdqLcesJoNYqIcPx7) on Google Drive and place it in the root directory of this project.

Then apply style to a content image by running the following in the Jupyter Notebook

    import stylize

    content = 'tubingen.jpg'
    style = 'shipwreck.jpg'

    stylize.apply(content, style, learning_rate=1.0,iters=100,
                  alpha=1, beta=5, noise_ratio=0.1, new_width=300, pool_method='avg',
                  pool_stride=2, style_loss_layers_w=(0.2, 0.2, 0.2, 0.2, 0.2),
                  style_num_layers=5, content_layer_num=4, optimizer='lbfgs')

Output images will be exported to [outputs](outputs)
