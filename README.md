# VGG-MNIST-style-transfer

VGG based calligraphy style transfer for hand written numbers from MNIST dataset. The goal of this project is to embed a certain caligraphy style to hand drawn numbers and test if m,ultiple style image use improves the transfer. First, we create a composite image of 10x10 random numbers from the MNIST dataset. Then, create a number of caligraphy style images from which the style features are extracted from. We augment each individual style number to create a better feature represantation and matching to MNIST numbers. The features are applied to the original hand drawn image to change the numbers so that they would possess the selected style but still represent the original contents as much as possible.

Original paper:  
Image Style Transfer Using Convolutional Neural Networks, Gatys L.A. et al, 2016  
https://openaccess.thecvf.com/content_cvpr_2016/html/Gatys_Image_Style_Transfer_CVPR_2016_paper.html


**Main dependencies:**   
* [PyTorch](https://pytorch.org/)
* [Torchvision](https://pytorch.org/vision/stable/index.html)
* [idx2numpy](https://pypi.org/project/idx2numpy/)
* [OpenCV](https://pypi.org/project/opencv-python/)
* [Augmentor](https://github.com/mdbloice/Augmentor)



Before running the code, extract the train-images-idx3-ubyte.zip (MNIST dataset) file in the /mnist folder.



# Results  
**Multiple Styles**  
3 styles used for style feature extraction  
| ![mnist.jpg](https://github.com/reiniscimurs/VGG-MNIST-style-transfer/blob/main/mnist.jpg) | ![out.jpg](https://github.com/reiniscimurs/VGG-MNIST-style-transfer/blob/main/out.jpg) | ![out.gif](https://github.com/reiniscimurs/VGG-MNIST-style-transfer/blob/main/output.gif) | ![style.jpg](https://github.com/reiniscimurs/VGG-MNIST-style-transfer/blob/main/style.jpg)|
|:--:|:--:| :--:|:--:|
| MNIST image |Output |Output GIF |Style example |


**Style 2**  
1 style image used for feature extraction:  
| ![mnist.jpg](https://github.com/reiniscimurs/VGG-MNIST-style-transfer/blob/main/results/res%20num2-1/mnist.jpg) | ![out.jpg](https://github.com/reiniscimurs/VGG-MNIST-style-transfer/blob/main/results/res%20num2-1/out.jpg) | ![style.jpg](https://github.com/reiniscimurs/VGG-MNIST-style-transfer/blob/main/results/res%20num2-1/style.jpg) |
|:--:|:--:| :--:|
| MNIST image |Output |Style example  |

5 style images used for feature extraction:  
| ![mnist.jpg](https://github.com/reiniscimurs/VGG-MNIST-style-transfer/blob/main/results/res%20num2-5/mnist.jpg) | ![out.jpg](https://github.com/reiniscimurs/VGG-MNIST-style-transfer/blob/main/results/res%20num2-5/out.jpg) | ![style.jpg](https://github.com/reiniscimurs/VGG-MNIST-style-transfer/blob/main/results/res%20num2-5/style.jpg) |
|:--:|:--:| :--:|
| MNIST image |Output |Style example  |

100 style image used for feature extraction:  
| ![mnist.jpg](https://github.com/reiniscimurs/VGG-MNIST-style-transfer/blob/main/results/res%20num2-100/mnist.jpg) | ![out.jpg](https://github.com/reiniscimurs/VGG-MNIST-style-transfer/blob/main/results/res%20num2-100/out.jpg) | ![style.jpg](https://github.com/reiniscimurs/VGG-MNIST-style-transfer/blob/main/results/res%20num2-100/style.jpg) |
|:--:|:--:| :--:|
| MNIST image |Output |Style example  |



