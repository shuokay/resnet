#!/usr/bin/env python 
'''
Author: Yushu Gao
Email: shuokay@gmail.com
MSRA Paper: http://arxiv.org/pdf/1512.03385v1.pdf
'''

import mxnet as mx
import logging
def ConvFactory(data, num_filter, kernel, stride=(1, 1), pad=(0, 0), act_type = 'relu'):
    conv = mx.symbol.Convolution(data = data, num_filter = num_filter, kernel = kernel, stride = stride, pad = pad)
    bn = mx.symbol.BatchNorm(data=conv)
    act = mx.symbol.Activation(data=bn, act_type=act_type)
    return act

def ResidualFactory(data, num_filter, diff_dim=False, stage1=False):
    if diff_dim:
        conv1 = ConvFactory(          data=data,  num_filter=num_filter[0], kernel=(1,1), stride=(2,2), pad=(0,0))
        conv2 = ConvFactory(          data=conv1, num_filter=num_filter[1], kernel=(3,3), stride=(1,1), pad=(1,1))
        conv3 = mx.symbol.Convolution(data=conv2, num_filter=num_filter[2], kernel=(1,1), stride=(1,1), pad=(0,0))
        _data = mx.symbol.Convolution(data=data,  num_filter=num_filter[2], kernel=(1,1), stride=(2,2), pad=(0,0))
        data  = _data+conv3
        bn    = mx.symbol.BatchNorm(data=data)
        act   = mx.symbol.Activation(data=bn, act_type='relu')
        return act
    else:
        _data=data
        conv1 = ConvFactory(data=data,  num_filter=num_filter[0], kernel=(1,1), stride=(1,1), pad=(0,0))
        conv2 = ConvFactory(data=conv1, num_filter=num_filter[1], kernel=(3,3), stride=(1,1), pad=(1,1))
        conv3 = ConvFactory(data=conv2, num_filter=num_filter[2], kernel=(1,1), stride=(1,1), pad=(0,0))
        data  = _data+conv3
        bn    = mx.symbol.BatchNorm(data=data)
        act   = mx.symbol.Activation(data=bn, act_type='relu')
        return act
def ResidualSymbol(data):
    "stage 1"
    for i in xrange(3):
        if i == 0:
            _data = mx.symbol.Convolution(data=data,  num_filter=256,  kernel=(1,1), stride=(1,1), pad=(0,0))
            conv1 = ConvFactory(          data=data,  num_filter=64,   kernel=(1,1), stride=(1,1), pad=(0,0))
            conv2 = ConvFactory(          data=conv1, num_filter=64,   kernel=(3,3), stride=(1,1), pad=(1,1))
            conv3 = mx.symbol.Convolution(data=conv2, num_filter=256,  kernel=(1,1), stride=(1,1), pad=(0,0))
            data  = _data+conv3
            bn    = mx.symbol.BatchNorm(data=data)
            act   = mx.symbol.Activation(data=bn, act_type='relu')
            data  = act
        else:
            data = ResidualFactory(data, (64, 64, 256))
    "stage 2"
    for i in xrange(8):
        if i == 0:
            data = ResidualFactory(data, (128, 128, 512), True)
        else:
            data = ResidualFactory(data, (128, 128, 512))
    "stage 3"
    for i in xrange(36):
        if i == 0:
            data = ResidualFactory(data, (256, 256, 1024), True)
        else:
            data = ResidualFactory(data, (256, 256, 1024))
    "stage 4"
    for i in xrange(3):
        if i==0:
            data = ResidualFactory(data, (512, 512, 2048), True)
        else:
            data = ResidualFactory(data, (512, 512, 2048))
    return data


def get_dataiter(batch_size = 8):
    train_dataiter = mx.io.ImageRecordIter(
        path_imgrec        = "./train.rec",
        rand_crop          = True,
        rand_mirror        = True,
        shuffle            = True,
        data_shape         = (3,224,224),
        batch_size         = batch_size,
        preprocess_threads = 4,
        prefetch_buffer    = 4,
        )
    test_dataiter = mx.io.ImageRecordIter(
        path_imgrec        = "./val.rec",
        rand_crop          = False,
        rand_mirror        = False,
        data_shape         = (3,224,224),
        batch_size         = batch_size,
        preprocess_threads = 4,
        prefetch_buffer    = 4,
        )
    return train_dataiter, test_dataiter

if __name__=='__main__':
    logging.basicConfig(level=logging.DEBUG)
    "before residual net"
    data    = ConvFactory(data=mx.symbol.Variable(name='data'), num_filter=64, kernel=(7,7), stride=(2,2), pad=(3,3))
    pool    = mx.symbol.Pooling(data=data, kernel=(3,3), stride=(2,2), pad=(0,0), pool_type='max')

    "get residual net"
    res     = ResidualSymbol(pool)

    "global pooling + classify"
    pool    = mx.symbol.Pooling(data=res, kernel=(7,7), pool_type='avg')
    flatten = mx.symbol.Flatten(data=pool, name='flatten')
    "set num_hidden=1000 when test on ImageNet competition dataset"
    fc      = mx.symbol.FullyConnected(data=flatten, num_hidden=200, name='fc1')
    softmax = mx.symbol.SoftmaxOutput(data=fc, name='softmax')

    "uncomment the following two line to visualize resnet"
    # g=mx.visualization.plot_network(softmax)
    # g.view()
    model = mx.model.FeedForward(ctx=mx.gpu(0), symbol=softmax, num_epoch=10, learning_rate=0.1, momentum=0.9, wd=0.0001,initializer=mx.init.Uniform(0.07))
    train_dataiter, test_dataiter = get_dataiter()
    model.fit(X=train_dataiter, eval_data=test_dataiter, batch_end_callback=mx.callback.Speedometer(8))

