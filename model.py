import tensorflow as tf
from tensorflow import keras
from keras import layers

class ResNetBlock(layers.Layer):

    def __init__(self, channels, only_conv = False, name="ResNetBlock", trainable=True, **kwargs):
        super(ResNetBlock, self).__init__(name=name, trainable=trainable, **kwargs)
        self.channels = channels
        self.only_conv = only_conv

        self.conv_1 = layers.Conv2D(self.channels, 1)
        self.conv_2 = layers.Conv2D(2*self.channels, 3, padding="same")
        self.conv_3 = layers.Conv2D(self.channels, 1)
        self.conv_1_skip = layers.Conv2D(self.channels, 1)


    def call(self, inputs):

        x = self.conv_1(inputs)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("gelu")(x)

        x = self.conv_2(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("gelu")(x)
        
        x = self.conv_3(x)

        x_skip = self.conv_1_skip(inputs)

        x = layers.Add()([x, x_skip])
        x = layers.BatchNormalization()(x)

        return layers.Activation("gelu")(x)


class ResNet(tf.keras.Model):

    def __init__(self):
        super(ResNet, self).__init__()
        
        self.block_1 = ResNetBlock(channels = 128, name="ResB1")  
        self.block_2 = ResNetBlock(channels = 256, name="ResB2")  
        self.block_3 = ResNetBlock(channels = 512, name="ResB3")
        # self.block_4 = ResNetBlock(channels = 4673, only_conv = True) # batch x 2 x 2 x 4673 --> NUMBER OF POSSIBLE MOVES + VALUE OF STATE  
        self.conv_1 = layers.Conv2D(74, 1)
        
        self.pooling = layers.GlobalMaxPooling3D()

        
    def call(self, inputs):

        x = self.block_1(inputs)
        x = self.block_2(x)
        x = self.block_3(x)
        # x = self.block_4(x)
        x = self.conv_1(x)

        action_values = layers.Lambda(lambda x: x[:,:,:,:-1], name="action_values")(x)            # remove last
        # action_values = layers.Reshape((8,8,-1))(action_values)
        
        state_value = layers.Lambda(lambda x: x[:,:,:,-1:], name="state_values_init")(x)              # keep only last
        state_value = self.pooling(state_value)
        state_value = layers.Lambda(lambda x: tf.clip_by_value(x, -1, 1), name="clip_layer")(state_value)        # clip to [-1, 1]
        state_value = layers.Flatten()(state_value)
        
        return action_values, state_value

    def summary(self):
        x = layers.Input(shape=[8,8,119])
        model = tf.keras.Model(inputs=[x], outputs=self.call(x))
        return model.summary()
