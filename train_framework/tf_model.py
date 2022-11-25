import tensorflow.keras.layers as tfl
import tensorflow as tf

def ConvolutionBlock(x, n_filts, params, batch_normalization, name):
    x = tfl.Conv3D(n_filts, **params, name=name+"_conv0")(x)
    if batch_normalization:
        x = tfl.BatchNormalization(name=name+"_bn0")(x)
    x = tfl.Activation("relu", name=name+"_relu0")(x)
    x = tfl.Conv3D(n_filts, **params, name=name+"_conv1")(x)
    if batch_normalization:
        x = tfl.BatchNormalization(name=name+"_bn1")(x)
    x = tfl.Activation("relu", name=name)(x)
    return x


def UpConvBlock(x, n_filts, pool_size, upsampling, name):
    if upsampling:
        x = tfl.UpSampling3D(size=pool_size, name=name)(x)
    else:
        x = tfl.Conv3DTranspose(filters=n_filts,
                                kernel_size=(2, 2, 2),
                                strides=(2, 2, 2),
                                padding='same', name=name)(x)
    return x


def unet_model_3d(input_shape=(4, 128, 128, 32),
                  pool_size=(2, 2, 2), n_labels=3,
                  upsampling=True, depth=5, n_filts=16,
                  activation_name="sigmoid", batch_norm=False):

    inputs = tfl.Input(input_shape)
    levels = list()

    params = dict(kernel_size=(3, 3, 3), strides=(1, 1, 1),
                  padding="same", kernel_initializer="he_uniform")

    # add levels with max pooling
    layer = inputs
    for layer_depth in range(depth):
        n_f = n_filts* (2 ** layer_depth)
        encode_block = ConvolutionBlock(layer, n_f, params, batch_norm, f"encode_{layer_depth}")
        if layer_depth < depth - 1:
            layer = tfl.MaxPooling3D(name=f"pool_{layer_depth}", pool_size=pool_size)(encode_block)
        else:
            layer = encode_block
        levels.append(encode_block)


    up_conv = UpConvBlock(encode_block, n_filts*8, pool_size, upsampling, f"Up_{layer_depth}")
    concat = tfl.concatenate([up_conv, levels[layer_depth-1]], axis=1, name=f"concat_{layer_depth-1}")
    decode = ConvolutionBlock(concat, n_filts*8, params, batch_norm, f"decode_{layer_depth-2}",)


    # add levels with up-convolution or up-sampling
    n = 4
    for layer_depth in range(depth - 3, -1, -1):
        up_conv = UpConvBlock(decode, n_filts*n,  pool_size, upsampling, f"Up_{layer_depth}")
        concat = tfl.concatenate([up_conv, levels[layer_depth]], axis=1, name=f"concat_{layer_depth}")
        if layer_depth > 0:
            decode = ConvolutionBlock(concat, n_filts*n, params, batch_norm, f"decode_{layer_depth-1}")
        else:
            convout = ConvolutionBlock(concat, n_filts, params, batch_norm, f"convOut")
        
        n /= 2

    outputs = tfl.Conv3D(n_labels, (1, 1, 1), activation=activation_name, name="PredictionMask")(convout)
    model =  tf.keras.Model(inputs=inputs, outputs=outputs)
    return model
