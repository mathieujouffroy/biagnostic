import tensorflow as tf
import tensorflow.keras.layers as tfl

class Unet3D:

    def __init__(
        self,
        m_name,
        vol_shape=(160, 160, 64, 4),
        n_labels=3,
        depth=5,
        pool_size=(2, 2, 2),
        n_filts = 16,
        activation_name = 'sigmoid',
        upsampling=True,
        batch_norm=True,
        dropout=False
    ):
        self.m_name = m_name
        self.vol_shape = vol_shape
        self.n_labels = n_labels
        self.depth = depth
        self.pool_size = pool_size
        self.n_filts = n_filts
        self.activation_name = activation_name
        self.upsampling = upsampling
        self.batch_norm = batch_norm
        self.dropout = dropout
        self.params = dict(kernel_size=(3, 3, 3), strides=(1, 1, 1), padding="same", kernel_initializer="he_uniform")
        self.model = self.build()


    def conv_block(self, x, n_filts, name):

        for i in range(2):
            x = tfl.Conv3D(n_filts, **self.params, name=f"{name}_conv{i}")(x)
            if self.batch_norm:
                x = tfl.BatchNormalization(name=f"{name}_bn{i}")(x)
            x = tfl.Activation("relu", name=f"{name}_relu{i}")(x)

        return x


    def up_conv_block(self, x, n_filts, name):

        if self.upsampling:
            x = tfl.UpSampling3D(size=self.pool_size, name=name)(x)
        else:
            x = tfl.Conv3DTranspose(filters=n_filts,
                                    kernel_size=(2, 2, 2),
                                    strides=(2, 2, 2),
                                    padding='same', name=name)(x)
        return x


    def build(self):
        inputs = tfl.Input(self.vol_shape)
        levels = list()

        layer = inputs
        # add levels with max pooling
        for layer_depth in range(self.depth):
            # filters -> 16, 32, 64, 128, 256
            n_f = self.n_filts * (2 ** layer_depth)
            encode_block = self.conv_block(layer, n_f, f"encode_{layer_depth}")

            if layer_depth < self.depth - 1:
                layer = tfl.MaxPooling3D(name=f"pool_{layer_depth}", pool_size=self.pool_size)(encode_block)
            else:
                layer = encode_block

            if self.dropout and layer_depth == 0:
                layer = tfl.SpatialDropout3D(0.2, data_format='channels_first')(layer)

            levels.append(encode_block)

        up_conv = self.up_conv_block(encode_block, self.n_filts*8, f"Up_{layer_depth}")
        concat = tfl.concatenate([up_conv, levels[layer_depth-1]], axis=-1, name=f"concat_{layer_depth-1}")
        decode = self.conv_block(concat, self.n_filts*8, f"decode_{layer_depth-2}",)

        n = 4
        # iterate from layer[-2] to layer[0] to add up-convolution or up-sampling
        for layer_depth in range(self.depth - 3, -1, -1):
            up_conv = self.up_conv_block(decode, self.n_filts*n, f"Up_{layer_depth}")
            concat = tfl.concatenate([up_conv, levels[layer_depth]], axis=-1, name=f"concat_{layer_depth}")
            if layer_depth > 0:
                name = f"decode_{layer_depth-1}"
            else:
                name = "convOut"
            decode = self.conv_block(concat, self.n_filts*n, name)
            n /= 2

        outputs = tfl.Conv3D(self.n_labels, (1, 1, 1), activation=self.activation_name, name="PredictionMask")(decode)
        model =  tf.keras.Model(inputs=inputs, outputs=outputs, name=self.m_name)

        return model




class AttentionUnet3D(tf.keras.Model):
    def __init__(
        self,
        m_name,
        vol_shape=(160, 160, 64, 4),
        n_labels=3,
        depth=5,
        pool_size=(2, 2, 2),
        n_filts = 16,
        # softmax
        activation_name = 'sigmoid',
        upsampling=True,
        batch_norm=False,
        dropout=False
    ):
        self.m_name = m_name
        self.vol_shape = vol_shape
        self.n_labels = n_labels
        self.depth = depth
        self.pool_size = pool_size
        self.n_filts = n_filts
        self.activation_name = activation_name
        self.upsampling = upsampling
        self.batch_norm = batch_norm
        self.dropout = dropout
        self.params = dict(kernel_size=(3, 3, 3), strides=(1, 1, 1), padding="same", kernel_initializer="he_uniform")


    def conv_block(self, x, n_filts, name):
        for i in range(2):
            x = tfl.Conv3D(n_filts, **self.params, name=f"{name}_conv{i}")(x)
            if self.batch_norm:
                x = tfl.BatchNormalization(name=f"{name}_bn{i}")(x)
            x = tfl.Activation("relu", name=f"{name}_relu{i}")(x)
        return x


    def deconv_block(self, x, n_filts, name):
        x = tfl.UpSampling3D(size=self.pool_size, name=name)(x)
        x = tfl.Conv3D(n_filts, **self.params, name=f"{name}_conv")(x)
        if self.batch_norm:
            x = tfl.BatchNormalization(name=f"{name}_bn")(x)
        x = tfl.Activation("relu", name=f"{name}_relu")(x)
        return x


    def attention_block(self, F_g, F_l, n_filts):

        g = tfl.Conv3D(n_filts, 1, padding="valid")(F_g)
        g = tfl.BatchNormalization()(g)
        x = tfl.Conv3D(n_filts, 1, padding="valid")(F_l)
        x = tfl.BatchNormalization()(x)

        psi = tfl.Add()([g, x])
        psi = tfl.Activation("relu")(psi)
        psi = tfl.Conv3D(1, 1, padding="valid")(psi)
        psi = tfl.BatchNormalization()(psi)
        psi = tfl.Activation("sigmoid")(psi)
        return tfl.Multiply()([F_l, psi])


    def build(self):
        inputs = tfl.Input(self.vol_shape)
        levels = list()

        layer = inputs
        # add levels with max pooling
        for layer_depth in range(self.depth):
            # filters -> 16, 32, 64, 128, 256
            n_f = self.n_filts * (2 ** layer_depth)
            encode_block = self.conv_block(layer, n_f, f"encode_{layer_depth}")
            print(f"encode_{layer_depth} :  filters= {n_f}")

            if layer_depth < self.depth - 1:
                layer = tfl.MaxPooling3D(name=f"pool_{layer_depth}", pool_size=self.pool_size)(encode_block)
            else:
                layer = encode_block

            if self.dropout and layer_depth == 0:
                layer = tfl.SpatialDropout3D(0.2, data_format='channels_first',name=f"dropout_{layer_depth}")(layer)

            levels.append(encode_block)


        up_conv = self.deconv_block(encode_block, self.n_filts*8, f"Up_{layer_depth-1}")
        att = self.attention_block(up_conv, levels[layer_depth-1], self.n_filts*8)
        concat = tfl.concatenate([up_conv, att], axis=-1, name=f"concat_{layer_depth-1}")
        conv = self.conv_block(concat, self.n_filts*8, f"conv_up_{layer_depth-1}_encode")
        print(f"Up_{layer_depth-1}:             filters={self.n_filts*8}")
        print(f"concat_{layer_depth}")
        print(f"conv_up_{layer_depth}_encode")


        n = 4
        # iterate from layer[-2] to layer[0] to add up-convolution or up-sampling
        for layer_depth in range(self.depth - 3, -1, -1):
            up_conv = self.deconv_block(conv, self.n_filts*n, f"Up_{layer_depth}")
            att = self.attention_block(up_conv, levels[layer_depth], self.n_filts*n)
            concat = tfl.concatenate([up_conv, att], axis=-1, name=f"concat_{layer_depth}")
            if layer_depth > 0:
                name = f"conv_up_{layer_depth}_encode"
            else:
                name = "convOut"
            conv = self.conv_block(concat, self.n_filts*n, name)
            n /= 2
            print(f"Up_{layer_depth}:          filters={self.n_filts*n}")
            print(f"concat_{layer_depth}")
            print(f"{name}")


        outputs = tfl.Conv3D(self.n_labels, (1, 1, 1), activation=self.activation_name, name="PredictionMask")(conv)
        model =  tf.keras.Model(inputs=inputs, outputs=outputs, name=self.m_name)

        return model