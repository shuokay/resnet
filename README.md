# resnet
This code is outdated and the example have moved to https://github.com/shuokay/mxnet/blob/master/example/image-classification/symbol_resnet-small.py.
Running that example by  
```python train_cifar10.py --network=resnet-small --lr=0.01 --lr-factor=0.1 --lr-factor-epoch=20 --num-epochs=30```  
I get a final train accuracy about 91.75% and test accuracy about 84.30%.

Example code for [Deep Residual Learning for Image Recognition](http://arxiv.org/abs/1512.03385)

I have only tested this code on [timyimagenet data](http://cs231n.stanford.edu/tiny-imagenet-200.zip)

