# neural-artistist

`neural-artistist` is an implementation of [Image Style Transfer Using Convolutional Neural Networks](https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Gatys_Image_Style_Transfer_CVPR_2016_paper.pdf)

# Results

Our team standing in front of Columbia University with the Starry Night by Vincent van Gogh

Result | Our Team | Starry Night
:-----:|:--------:|:--------:
<img src="inputs/output_csjl_starry_night.jpg" alt="drawing" width="300"/> | <img src="inputs/csjl.jpeg" alt="drawing" width="300"/> | <img src="inputs/starry_night.jpg" alt="drawing" width="300"/>

Other results replicating the paper

Result | Tubingen | Shipwreck
:-----:|:--------:|:--------:
<img src="inputs/output_tubingen_shipwreck.jpg" alt="drawing" width="300"/> | <img src="inputs/tubingen.jpg" alt="drawing" width="300"/> | <img src="inputs/shipwreck.jpg" alt="drawing" width="300"/>

Result | Tubingen | Starry Night
:-----:|:--------:|:--------:
<img src="inputs/output_tubingen_starry_night.jpg" alt="drawing" width="300"/> | <img src="inputs/tubingen.jpg" alt="drawing" width="300"/> | <img src="inputs/starry_night.jpg" alt="drawing" width="300"/>

Result | Tubingen | Scream
:-----:|:--------:|:--------:
<img src="inputs/output_tubingen_scream.jpg" alt="drawing" width="300"/> | <img src="inputs/tubingen.jpg" alt="drawing" width="300"/> | <img src="inputs/scream.jpg" alt="drawing" height="230"/>

Result | Tubingen | Seated Nude
:-----:|:--------:|:--------:
<img src="inputs/output_tubingen_seated_nude.jpg" alt="drawing" width="300"/> | <img src="inputs/tubingen.jpg" alt="drawing" width="300"/> | <img src="inputs/seated_nude.jpg" alt="drawing" height="230"/>

Result | Tubingen | Somposition
:-----:|:--------:|:--------:
<img src="inputs/output_tubingen_somposition.jpg" alt="drawing" width="300"/> | <img src="inputs/tubingen.jpg" alt="drawing" width="300"/> | <img src="inputs/somposition.jpg" alt="drawing" width="300"/>

# How to run
You will need to download the pre-trained VGG19 model from [here](https://github.com/machrisaa/tensorflow-vgg) and place it to the root directory of `neural-artistist`.

Then apply slyle to a content image by

    import stylize

    content = 'tubingen.jpg'
    style = 'starry_night.jpg'

    stylize.apply(content, style, learning_rate=2.0, iterations=1001,
                  alpha=1, beta=1000, noise_ratio=0.6, new_width=300)
