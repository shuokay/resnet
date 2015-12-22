import mxnet as mx

def ConvFactory(data, num_filter, kernel, stride=(1, 1), pad=(0, 0), act_type = 'relu'):
    conv = mx.symbol.Convolution(data = data, num_filter = num_filter, kernel = kernel, stride = stride, pad = pad)
    bn = mx.symbol.BatchNorm(data=conv)
    act = mx.symbol.Activation(data=bn, act_type=act_type)
    return act

def ResidualFactory(data, num_filter, diff_dim=False):
    if diff_dim:
        conv1=ConvFactory(          data=data, num_filter=num_filter[0], kernel=(1,1), stride=(2,2), pad=(0,0))
        conv2=ConvFactory(          data=conv1,num_filter=num_filter[1], kernel=(3,3), stride=(1,1), pad=(1,1))
        conv3=mx.symbol.Convolution(data=conv2,num_filter=num_filter[2], kernel=(1,1), stride=(1,1), pad=(0,0))
        _data=mx.symbol.Convolution(data=data, num_filter=num_filter[2], kernel=(1,1), stride=(2,2), pad=(0,0))
        data=_data+conv3
        bn=mx.symbol.BatchNorm(data=data)
        act=mx.symbol.Activation(data=bn, act_type='relu')
        return act
    else:
        _data=data
        conv1=ConvFactory(data=data, num_filter=num_filter[0], kernel=(1,1), stride=(1,1), pad=(0,0))
        conv2=ConvFactory(data=conv1,num_filter=num_filter[1], kernel=(3,3), stride=(1,1), pad=(1,1))
        conv3=ConvFactory(data=conv2,num_filter=num_filter[2], kernel=(1,1), stride=(1,1), pad=(0,0))
        data=_data+conv3
        bn=mx.symbol.BatchNorm(data=data)
        act=mx.symbol.Activation(data=bn, act_type='relu')
        return act
def ResidualSymbol(data):
    for i in xrange(3):
        if i == 0:
            data = ResidualFactory(data, (64, 64, 256), True)
        else:
            data = ResidualFactory(data, (64, 64, 256))
    for i in xrange(8):
        if i == 0:
            data = ResidualFactory(data, (128, 128, 512), True)
        else:
            data = ResidualFactory(data, (128, 128, 512))
    for i in xrange(36):
        if i == 0:
            data = ResidualFactory(data, (256, 256, 1024), True)
        else:
            data = ResidualFactory(data, (256, 256, 1024))
    for i in xrange(3):
        if i==0:
            data = ResidualFactory(data, (512, 512, 2048), True)
        else:
            data = ResidualFactory(data, (512, 512, 2048))
    return data

if __name__=='__main__':
    data=ConvFactory(data=mx.symbol.Variable(name='data'), num_filter=64, kernel=(7,7), stride=(2,2), pad=(3,3))
    pool=mx.symbol.Pooling(data=data, kernel=(3,3), stride=(2,2), pad=(0,0), pool_type='max')
    res=ResidualSymbol(pool)
    pool=mx.symbol.Pooling(data=res, kernel=(7,7), pool_type='avg')
    flatten=mx.symbol.Flatten(data=pool, name='flatten')
    fc=mx.symbol.FullyConnected(data=flatten, num_hidden=200, name='fc1')
    softmax=mx.symbol.SoftmaxOutput(data=fc, name='softmax')
    g=mx.visualization.plot_network(softmax)
    # g.view()
    batch_size = 5
    num_epoch = 1
    model = mx.model.FeedForward(ctx=mx.gpu(0), symbol=softmax, num_epoch=num_epoch, learning_rate=0.1, momentum=0.9, wd=0.0001,initializer=mx.init.Uniform(0.07))
    train_dataiter = mx.io.ImageRecordIter(
        path_imgrec="./train.rec",
        # rand_crop=True,
        # rand_mirror=True,
        shuffle = True,
        data_shape=(3,224,224),
        batch_size=batch_size,
        preprocess_threads=4,
        prefetch_buffer=4,
        )
    test_dataiter = mx.io.ImageRecordIter(
        path_imgrec="./val.rec",
        rand_crop=True,
        # rand_mirror=True,
        data_shape=(3,224,224),
        batch_size=batch_size,
        preprocess_threads=4,
        prefetch_buffer=4,
        )
    model.fit(X=train_dataiter, eval_data=test_dataiter, batch_end_callback=mx.callback.Speedometer(batch_size))
