import tensorflow as tf
import tensorflow.keras.layers as tfl


# ADD Attention Unet 
# Create classes for 3d unet and 3d attention unet

class Unet3D:#(tf.keras.Model):


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
        
        #return outputs
        return model




class AttentionUnet3D(tf.keras.Model):
    def __init__(
        self,
        m_name,
        vol_shape,
        n_labels,
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

    def conv_block(self, x, name):

        for i in range(2):
            x = tfl.Conv3D(self.n_filts, **self.params, name=f"{name}_conv{i}")(x)
            if self.batch_norm:
                x = tfl.BatchNormalization(name=f"{name}_bn{i}")(x)
            x = tfl.Activation("relu", name=f"{name}_relu{i}")(x)

        return x


    def deconv_block(self, x, name):
        x = tfl.UpSampling3D(size=self.pool_size, name=name)(x)
        x = tfl.Conv3D(self.n_filts, **self.params)(x)
        if self.batch_norm:
            x = tfl.BatchNormalization(name=name)(x)
        x = tfl.Activation("relu", name=name)(x)
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


    def call(self):
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
        
        up_conv = self.deconv_block(encode_block, self.n_filts*8, self.pool_size, self.upsampling, f"Up_{layer_depth}")
        conv6 = self.attention_block(up_conv, levels[layer_depth-1], self.n_filts*8)
        concat = tfl.concatenate([up_conv, conv6], axis=-1, name=f"concat_0")
        
        n = 4
        # iterate from layer[-2] to layer[0] to add up-convolution or up-sampling
        for layer_depth in range(self.depth - 3, -1, -1):
            up_conv = self.up_conv_block(decode, self.n_filts*n,  self.pool_size, self.upsampling, f"Up_{layer_depth}")
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


#up6 = deconv3d(conv5, 512)
#conv6 = attention_block(up6, conv4, 512)
#up6 = Concatenate()([up6, conv6])
#conv6 = conv3d(up6, 512)
#up7 = deconv3d(conv6, 256)
#conv7 = attention_block(up7, conv3, 256)
#up7 = Concatenate()([up7, conv7])
#conv7 = conv3d(up7, 256)
#up8 = deconv3d(conv7, 128)
#conv8 = attention_block(up8, conv2, 128)
#up8 = Concatenate()([up8, conv8])
#conv8 = conv3d(up8, 128)
#up9 = deconv3d(conv8, 64)
#conv9 = attention_block(up9, conv1, 64)
#up9 = Concatenate()([up9, conv9])
#conv9 = conv3d(up9, 64)
#output = Conv3D(4, 1, activation="softmax")(conv9)
#return Model(inputs=inputs, outputs=output, name="Attention_Unet")        

#https://github.com/mahsaama/BrainTumorSegmentation/blob/main/models/att_unet_model.py
#https://github.com/arkanivasarkar/Retinal-Vessel-Segmentation-using-variants-of-UNET/blob/main/model.py 
#https://github.com/lixiaolei1982/Keras-Implementation-of-U-Net-R2U-Net-Attention-U-Net-Attention-R2U-Net.-/blob/master/network.py 