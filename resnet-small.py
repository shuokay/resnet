#!/usr/bin/env python 
'''
MSRA Paper: http://arxiv.org/pdf/1512.03385v1.pdf
'''

import mxnet as mx
import logging
def ConvFactory(data, num_filter, kernel, stride=(1, 1), pad=(0, 0), act_type = 'relu',last=False):
    conv = mx.symbol.Convolution(data = data, num_filter = num_filter, kernel = kernel, stride = stride, pad = pad)
    if last:
        return conv
    else:
        bn = mx.symbol.BatchNorm(data=conv)
        act = mx.symbol.Activation(data=bn, act_type=act_type)
        return act

def ResidualFactory(data, num_filter, diff_dim=False):
    if diff_dim:
        conv1 = ConvFactory(          data=data,  num_filter=num_filter[0], kernel=(3,3), stride=(2,2), pad=(1,1), last=False)
        conv2 = ConvFactory(          data=conv1, num_filter=num_filter[1], kernel=(3,3), stride=(1,1), pad=(1,1), last=True)
        _data = mx.symbol.Convolution(data=data,  num_filter=num_filter[1], kernel=(3,3), stride=(2,2), pad=(1,1))
        data  = _data+conv2
        bn    = mx.symbol.BatchNorm(data=data)
        act   = mx.symbol.Activation(data=bn, act_type='relu')
        return act
    else:
        _data=data
        conv1 = ConvFactory(data=data,  num_filter=num_filter[0], kernel=(3,3), stride=(1,1), pad=(1,1), last=False)
        conv2 = ConvFactory(data=conv1, num_filter=num_filter[1], kernel=(3,3), stride=(1,1), pad=(1,1), last=True)
        data  = _data+conv2
        bn    = mx.symbol.BatchNorm(data=data)
        act   = mx.symbol.Activation(data=bn, act_type='relu')
        return act
def ResidualSymbol(data, n=9):
    "stage 1"
    for i in xrange(n):
        data = ResidualFactory(data, (16, 16))
    "stage 2"
    for i in xrange(n):
        if i == 0:
            data = ResidualFactory(data, (32, 32), True)
        else:
            data = ResidualFactory(data, (32, 32))
    "stage 3"
    for i in xrange(n):
        if i == 0:
            data = ResidualFactory(data, (64, 64), True)
        else:
            data = ResidualFactory(data, (64, 64))
    return data


def get_dataiter(batch_size=128):
    data_shape=(3,28,28)
    train_dataiter = mx.io.ImageRecordIter(
        path_imgrec        = "./data/cifar10/train.rec",
        mean_img           = "./data/cifar10/mean.bin",
        rand_crop          = True,
        rand_mirror        = True,
        shuffle            = True,
        data_shape         = data_shape,
        batch_size         = batch_size,
        preprocess_threads = 2,
        prefetch_buffer    = 2,
        )
    test_dataiter = mx.io.ImageRecordIter(
        path_imgrec        = "./data/cifar10/test.rec",
        mean_img           = "./data/cifar10/mean.bin",
        rand_crop          = False,
        rand_mirror        = False,
        data_shape         = data_shape,
        batch_size         = batch_size,
        preprocess_threads = 2,
        prefetch_buffer    = 2,
        )
    return train_dataiter, test_dataiter

if __name__=='__main__':
    logging.basicConfig(level=logging.DEBUG)
    data    = ConvFactory(data=mx.symbol.Variable(name='data'), num_filter=16, kernel=(3,3), stride=(1,1), pad=(1,1)) # before residual net
    res     = ResidualSymbol(data) # get residual net
    pool    = mx.symbol.Pooling(data=res, kernel=(7,7), pool_type='avg') # global pooling + classify
    flatten = mx.symbol.Flatten(data=pool, name='flatten')
    fc      = mx.symbol.FullyConnected(data=flatten, num_hidden=10, name='fc1') # set num_hidden=1000 when test on ImageNet competition dataset
    softmax = mx.symbol.SoftmaxOutput(data=fc, name='softmax')

    # uncomment the following two line to visualize resnet
    g=mx.visualization.plot_network(softmax)
    g.render(filename='resnet', cleanup=True)
    batch_size = 128
    train_dataiter, test_dataiter = get_dataiter(batch_size=batch_size)
    finetune=False
    if finetune==False:
        model = mx.model.FeedForward(ctx=mx.gpu(0), symbol=softmax, num_epoch=70, learning_rate=0.1, momentum=0.9, wd=0.0001, \
                                 initializer=mx.init.Xavier(rnd_type='gaussian', factor_type="in", magnitude=2),
                                 # initializer=mx.init.Xavier(),
                                 # initializer=mx.init.Normal(),
                                lr_scheduler=mx.lr_scheduler.FactorScheduler(step =100000000000, factor = 0.95)
                                 )
        moniter=mx.monitor.Monitor(1)
        batch_end_callback=[mx.callback.Speedometer(batch_size=batch_size, frequent=100),
                            # mx.callback.ProgressBar(50000.0/batch_size)
        ]
        model.fit(X=train_dataiter, eval_data=test_dataiter, monitor=None ,batch_end_callback=batch_end_callback, epoch_end_callback=mx.callback.do_checkpoint("./models/resnet"))
    else:
        loaded = mx.model.FeedForward.load('models/resnet', 100)
        continue_model = mx.model.FeedForward(ctx=mx.gpu(0), symbol = loaded.symbol, arg_params = loaded.arg_params, aux_params = loaded.aux_params, num_epoch=10000, learning_rate=0.01, momentum=0.9, wd=0.0001)
        continue_model.fit(X=train_dataiter, eval_data=test_dataiter, batch_end_callback=mx.callback.Speedometer(batch_size),epoch_end_callback=mx.callback.do_checkpoint("./models/resnet"))
