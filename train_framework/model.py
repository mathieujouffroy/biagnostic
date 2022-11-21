import tensorflow.keras.layers as tfl

def ConvolutionBlock(x, n_filts, params, name):
    x = tfl.Conv3D(n_filts, **params, name=name+"_conv0")(x)
    if params.batch_normalization:
        x = tfl.BatchNormalization(name=name+"_bn0")(x)
    x = tfl.Activation("relu", name=name+"_relu0")(x)
    x = tfl.Conv3D(n_filts, **params, name=name+"_conv1")(x)
    if params.batch_normalization:
        x = tfl.BatchNormalization(name=name+"_bn1")(x)
    x = tfl.Activation("relu", name=name)(x)
    return x


def UpConvBlock(x, n_filts, pool_size, upsampling, name):
    if upsampling:
        x = tfl.UpSampling3D(size=pool_size, interpolation="bilinear", name=name)(x)
    else:
        x = tfl.Deconvolution3D(filters=n_filts,
                                kernel_size=(2, 2, 2),
                                strides=(2, 2, 2),
                                padding='same', name=name)(x)
    return x


def unet_model_3d(input_shape=(4, 160, 160, 32),
                  pool_size=(2, 2, 2), n_labels=3,
                  upsampling=False, depth=4, n_filts=32,
                  activation_name="sigmoid"):

    inputs = tfl.Input(input_shape)
    levels = list()

    params = dict(kernel_size=(3, 3, 3), strides=(1, 1, 1),
                  padding="same", kernel_initializer="he_uniform",
                  batch_normalization=False)

    # add levels with max pooling
    layer = inputs
    for layer_depth in range(depth):
        encode_block = ConvolutionBlock(layer, n_filts, params, f"encode_{layer_depth}")
        if layer_depth < depth - 1:
            layer = tfl.MaxPooling3D(name=f"pool_{layer_depth}", pool_size=pool_size)(encode_block)
        else:
            layer = encode_block
        levels.append([encode_block])

    print(encode_block._keras_shape[1]) # == 8
    print(levels[layer_depth])
    print(levels[layer_depth - 1])
    #up_conv = UpConvBlock(encode_block, pool_size=pool_size, n_filts=encode_block._keras_shape[1],  upsampling=upsampling)
    up_conv = UpConvBlock(encode_block, pool_size, n_filts*8,  upsampling, f"Up_{layer_depth}")
    concat = tfl.concatenate([up_conv, levels[layer_depth - 1]], axis=1, name=f"concat_{layer_depth-1}")
    decode = ConvolutionBlock(concat, f"decode_{layer_depth-2}", n_filts*8, params)

    # add levels with up-convolution or up-sampling
    n = 4
    for layer_depth in range(depth - 3, -1, -1):
        print(layer_depth)
        print(levels[layer_depth]._keras_shape[1])
        print(n)
        up_conv = UpConvBlock(decode, pool_size, n_filts*n,  upsampling, f"Up_{layer_depth}")
        concat = tfl.concatenate([up_conv, levels[layer_depth]], axis=1, name=f"concat_{layer_depth}")
        if layer_depth < 0:
            decode = ConvolutionBlock(concat, f"decode_{layer_depth-1}", n_filts*n, params)
        else:
            convout = ConvolutionBlock(concat, f"convOut", n_filts, params)
        n /= 2


    outputs = tfl.Conv3D(n_labels, (1, 1, 1), activation=activation_name, name="PredictionMask")(convout)
    model = tfl.Model(inputs=inputs, outputs=outputs)

    return model
