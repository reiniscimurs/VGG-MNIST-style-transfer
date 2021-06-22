# VGG-MNIST-style-transfer

VGG based calligraphy style transfer for hand written numbers from MNIST dataset. The goal of this project is to embed a certain caligraphy style to hand drawn numbers. First, we create a composite image of 10x10 random numbers from the MNIST dataset. Then, create a number of caligraphy style images from which the style features are extracted from. The features are applied to the original hand drawn image to change the numbers so that they would possess the selected style but still represent the original contents as much as possible.

Implementation taken from the paper:

Image style transfer using convolutional neural networks
https://openaccess.thecvf.com/content_cvpr_2016/html/Gatys_Image_Style_Transfer_CVPR_2016_paper.html
