import tensorflow as tf
import tensorflow.keras.layers as tfl


# ADD Attention Unet 
# Create classes for 3d unet and 3d attention unet


def ConvolutionBlock(x, n_filts, params, batch_normalization, name):

    for i in range(2):
        x = tfl.Conv3D(n_filts, **params, name=f"{name}_conv{i}")(x)
        if batch_normalization:
            x = tfl.BatchNormalization(name=f"{name}_bn{i}")(x)
        x = tfl.Activation("relu", name=f"{name}_relu{i}")(x)
    
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


def unet_model_3d(m_name, input_shape=(160, 160, 64, 4),
                  pool_size=(2, 2, 2), n_labels=3, depth=5,
                  n_filts=16, activation_name="sigmoid",
                  upsampling=True, batch_norm=False, dropout=False):

    inputs = tfl.Input(input_shape)
    levels = list()

    params = dict(kernel_size=(3, 3, 3), strides=(1, 1, 1),
                  padding="same", kernel_initializer="he_uniform")

    layer = inputs

    # add levels with max pooling
    for layer_depth in range(depth):
        # filters -> 16, 32, 64, 128, 256 
        n_f = n_filts * (2 ** layer_depth)
        encode_block = ConvolutionBlock(layer, n_f, params, batch_norm, f"encode_{layer_depth}")
        
        if layer_depth < depth - 1:
            layer = tfl.MaxPooling3D(name=f"pool_{layer_depth}", pool_size=pool_size)(encode_block)        
        else:
            layer = encode_block

        if dropout and layer_depth == 0:
            layer = tfl.SpatialDropout3D(0.2, data_format='channels_first')(layer)
        
        levels.append(encode_block)

    up_conv = UpConvBlock(encode_block, n_filts*8, pool_size, upsampling, f"Up_{layer_depth}")
    concat = tfl.concatenate([up_conv, levels[layer_depth-1]], axis=-1, name=f"concat_{layer_depth-1}")
    decode = ConvolutionBlock(concat, n_filts*8, params, batch_norm, f"decode_{layer_depth-2}",)

    n = 4
    # iterate from layer[-2] to layer[0] to add up-convolution or up-sampling
    for layer_depth in range(depth - 3, -1, -1):
        up_conv = UpConvBlock(decode, n_filts*n,  pool_size, upsampling, f"Up_{layer_depth}")
        concat = tfl.concatenate([up_conv, levels[layer_depth]], axis=-1, name=f"concat_{layer_depth}")
        if layer_depth > 0:
            name = f"decode_{layer_depth-1}"
        else:
            name = "convOut"
        decode = ConvolutionBlock(concat, n_filts*n, params, batch_norm, name)
        n /= 2

    outputs = tfl.Conv3D(n_labels, (1, 1, 1), activation=activation_name, name="PredictionMask")(decode)
    model =  tf.keras.Model(inputs=inputs, outputs=outputs, name=m_name)
    return model
