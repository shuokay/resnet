# Deep Residual Net
Example code for [Deep Residual Learning for Image Recognition](http://arxiv.org/abs/1512.03385)  
* Run this script by ```python resnet-small.py``` for 100 epochs get a train accuracy around 89.47% and validation accuracy around 85.95%  
* Then change the learning rate to 0.01, running this training from 100th epoch for 50 iterations, and get a train accuracy around 98.72% and test accuracy around 89.77%  

## Differences to the Paper
* 1*1 convolution operators are used for increasing dimensions.
* This is a small residual net consists of 52 layers(can change to 20, 32, 44 layers by changing ```n``` in ```ResidualSymbol``` to 3, 5, 7)
* Using mxnet default data augmentation options include center crop (instead of random crop) and random mirror, no paddings on raw image data and the input image size is 28\*28(instead of 32\*32).