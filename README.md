# neural-artistist

`neural-artistist` is an implementation of [Image Style Transfer Using Convolutional Neural Networks](https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Gatys_Image_Style_Transfer_CVPR_2016_paper.pdf)

# Results

#### Us standing in front of Columbia University with the Starry Night style

Result | Our Team | Starry Night
:-----:|:--------:|:--------:
<img src="images/output_csjl_starry_night.jpg" alt="drawing" width="300"/> | <img src="images/csjl.jpeg" alt="drawing" width="300"/> | <img src="images/starry_night.jpg" alt="drawing" width="300"/>

#### Other results replicated the paper

Result | Tubingen | Shipwreck
:-----:|:--------:|:--------:
<img src="images/output_tubingen_shipwreck.jpg" alt="drawing" width="300"/> | <img src="images/tubingen.jpg" alt="drawing" width="300"/> | <img src="images/shipwreck.jpg" alt="drawing" width="300"/>

Result | Tubingen | Starry Night
:-----:|:--------:|:--------:
<img src="images/output_tubingen_starry_night.jpg" alt="drawing" width="300"/> | <img src="images/tubingen.jpg" alt="drawing" width="300"/> | <img src="images/starry_night.jpg" alt="drawing" width="300"/>

Result | Tubingen | Scream
:-----:|:--------:|:--------:
<img src="images/output_tubingen_scream.jpg" alt="drawing" width="300"/> | <img src="images/tubingen.jpg" alt="drawing" width="300"/> | <img src="images/scream.jpg" alt="drawing" height="200"/>

Result | Tubingen | Seated Nude
:-----:|:--------:|:--------:
<img src="images/output_tubingen_seated_nude.jpg" alt="drawing" width="300"/> | <img src="images/tubingen.jpg" alt="drawing" width="300"/> | <img src="images/seated_nude.jpg" alt="drawing" height="200"/>

Result | Tubingen | Somposition
:-----:|:--------:|:--------:
<img src="images/output_tubingen_somposition.jpg" alt="drawing" width="300"/> | <img src="images/tubingen.jpg" alt="drawing" width="300"/> | <img src="images/somposition.jpg" alt="drawing" width="300"/>

# How to run
You will need to download the pre-trained VGG19 model from [here](https://github.com/machrisaa/tensorflow-vgg) and place it to the root directory of `neural-artistist`.

Then apply slyle to a content image by

    import stylize

    content = 'tubingen.jpg'
    style = 'starry_night.jpg'

    stylize.apply(content, style, learning_rate=2.0, iterations=1001,
                  alpha=1, beta=1000, noise_ratio=0.6, new_width=300)
