import tensorflow as tf
from tensorflow import keras
from keras import layers
from utils import Config
import numpy as np

conf = Config()

class LogitsMaskToSoftmax(keras.layers.Layer):
    def __init__(self, **kwargs):
        super(LogitsMaskToSoftmax, self).__init__(**kwargs)

    def call(self, *args):
        logits, mask = zip(*args)
        # masked_logits = tf.boolean_mask(logits, mask)
        masked_probs = tf.boolean_mask(logits, mask)
        # masked_probs = tf.nn.softmax(masked_logits)
        probs = tf.scatter_nd(tf.where(mask), masked_probs, tf.shape(mask, out_type=tf.int64))
        probs = tf.squeeze(probs, axis=0)
        return probs

    def get_config(self):
        return {}

    @classmethod
    def from_config(cls, config):
        return cls(**config)
        
# class TurnTensor(keras.layers.Layer):
#     def __init__(self, **kwargs):
#         super(TurnTensor, self).__init__(**kwargs)
#         self.mask = tf.stack(
#                         tf.fill((8,8,112),  False),
#                         tf.fill((8,8,1),    True),
#                         tf.fill((8,8,6),    False)
#         )

#     def call(self, planes):
#         batch_size = tf.shape(planes)[0]
#         mask = tf.repeat(self.mask, repeats=batch_size, axis=0)
#         turn_plane = tf.boolean_mask(planes, mask)
#         turn_value = tf.math.reduce_mean(turn_plane, axis=[1,2])
#         return probs

def create_model():
    
    l2_reg = 1e-4
    channels_convolution = 256
    channels_res_block = 256
    channels_policy = 128
    num_res_blocks = 8

    ###### USING FUNCTIONAL MODEL because the other one was giving an error ########
    input_legal_moves = layers.Input(shape=(8*8*73), name="legal_moves")                            # array with 1 for legal moves, 0 otherwise

    input_planes = layers.Input(shape=(8, 8, 119), name="planes")
    turn = layers.Lambda(lambda x: tf.expand_dims(x[..., -7], axis=-1))(input_planes) # 1 for white, -1 for black
    turn = layers.GlobalAveragePooling2D()(turn)
    turn = layers.Dense(8*8*73, kernel_initializer=tf.keras.initializers.Constant(value=1), use_bias=False, trainable=False)(turn)

    x = layers.Conv2D(channels_convolution, 3, padding="same", kernel_regularizer=tf.keras.regularizers.L2(l2_reg))(input_planes)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("gelu")(x)

    for i in range(num_res_blocks):
        x_skip = x
        x = layers.Conv2D(channels_res_block, 3, padding="same", kernel_regularizer=tf.keras.regularizers.L2(l2_reg))(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("gelu")(x)

        x = layers.Conv2D(channels_res_block, 3, padding="same", kernel_regularizer=tf.keras.regularizers.L2(l2_reg))(x)
        x = layers.BatchNormalization()(x)
        
        x = layers.Add()([x, x_skip])
        
        x = layers.Activation("gelu", name="activation_resblock_{}".format(i))(x)

    policy = layers.Conv2D(channels_policy, 1, padding="same", kernel_regularizer=tf.keras.regularizers.L2(l2_reg))(x)
    policy = layers.BatchNormalization()(policy)
    policy = layers.Activation("gelu")(policy)
    policy = layers.Conv2D(73, 1, kernel_regularizer=tf.keras.regularizers.L2(l2_reg))(policy)
    policy = layers.BatchNormalization()(policy)

    policy = layers.Flatten()(policy)
    # for "black" moves, we want the logits to be POSITIVE as well! otherwise the log from the crossentropy will kill them
    policy = layers.Multiply()([policy, turn]) # black turn --> swap signs
    policy = LogitsMaskToSoftmax(name="policy")([policy, input_legal_moves])

    value = layers.Conv2D(1, 1, kernel_regularizer=tf.keras.regularizers.L2(l2_reg))(x)                          # only one plane for the state value
    value = layers.BatchNormalization()(value)
    value = layers.Activation("gelu")(value)

    value = layers.Flatten()(value)
    value = layers.Dense(256, kernel_regularizer=tf.keras.regularizers.L2(l2_reg))(value)
    value = layers.Activation("gelu")(value)

    value = layers.Dense(1, kernel_regularizer=tf.keras.regularizers.L2(l2_reg))(value)
    value = layers.Activation("tanh", name="value")(value)

    model = tf.keras.Model(inputs=[input_planes, input_legal_moves], outputs=[policy, value])

    model.compile(
        optimizer=conf.OPTIMIZER,
        loss={"policy":conf.LOSS_FN_POLICY, "value":conf.LOSS_FN_VALUE},
        metrics={"policy":conf.METRIC_FN_POLICY}
    )

    return model


# class ResNetBlock(layers.Layer):

#     def __init__(self, channels, only_conv = False, name="ResNetBlock", trainable=True, **kwargs):
#         super(ResNetBlock, self).__init__(name=name, trainable=trainable, **kwargs)
#         self.channels = channels
#         self.only_conv = only_conv

#         self.conv_1 = layers.Conv2D(self.channels, 1)
#         self.conv_2 = layers.Conv2D(2*self.channels, 3, padding="same")
#         self.conv_3 = layers.Conv2D(self.channels, 1)
#         self.conv_1_skip = layers.Conv2D(self.channels, 1)


#     def call(self, inputs):

#         x = self.conv_1(inputs)
#         x = layers.BatchNormalization()(x)
#         x = layers.Activation("gelu")(x)

#         x = self.conv_2(x)
#         x = layers.BatchNormalization()(x)
#         x = layers.Activation("gelu")(x)
        
#         x = self.conv_3(x)

#         x_skip = self.conv_1_skip(inputs)

#         x = layers.Add()([x, x_skip])
#         x = layers.BatchNormalization()(x)

#         return layers.Activation("gelu")(x)


# class ResNet(tf.keras.Model):

#     def __init__(self):
#         super(ResNet, self).__init__()
        
#         self.block_1 = ResNetBlock(channels = 128, name="ResB1")  
#         self.block_2 = ResNetBlock(channels = 256, name="ResB2")  
#         self.block_3 = ResNetBlock(channels = 512, name="ResB3")
#         # self.block_4 = ResNetBlock(channels = 4673, only_conv = True) # batch x 2 x 2 x 4673 --> NUMBER OF POSSIBLE MOVES + VALUE OF STATE  
#         self.conv_1 = layers.Conv2D(74, 1)
        
#         self.pooling = layers.GlobalMaxPooling2D()

        
#     def call(self, inputs):

#         x = self.block_1(inputs)
#         x = self.block_2(x)
#         x = self.block_3(x)
#         # x = self.block_4(x)
#         x = self.conv_1(x)

#         action_values = layers.Lambda(lambda x: x[:,:,:,:-1], name="action_values")(x)            # remove last
#         # action_values = layers.Reshape((8,8,-1))(action_values)
        
#         state_value = layers.Lambda(lambda x: x[:,:,:,-1:], name="state_values_init")(x)              # keep only last
#         state_value = self.pooling(state_value)
#         state_value = layers.Lambda(lambda x: tf.clip_by_value(x, -1, 1), name="clip_layer")(state_value)        # clip to [-1, 1]
#         state_value = layers.Flatten()(state_value)
        
#         return action_values, state_value

#     def summary(self):
#         x = layers.Input(shape=[8,8,119])
#         model = tf.keras.Model(inputs=[x], outputs=self.call(x))
#         return model.summary()
