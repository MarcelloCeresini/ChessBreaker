import tensorflow as tf
from tensorflow import keras
from keras import layers


class ResNet(tf.keras.Model):

    def __init__(self, name="ResNet", **kwargs):
        super(ResNet, self).__init__(name=name, **kwargs)

        self.block_1 = layers.Conv2D(256, 3)  # batch x 6 x 6 x channels
        self.block_2 = layers.Conv2D(512, 3)  # batch x 4 x 4 x channels
        self.block_3 = layers.Conv2D(1024, 3) # batch x 2 x 2 x channels
        self.block_4 = layers.Conv2D(4673, 1) # batch x 2 x 2 x 4673 --> NUMBER OF POSSIBLE MOVES + VALUE OF STATE  
        self.pooling = layers.GlobalAveragePooling2D()
        
    def call(self, inputs):
        x = self.block_1(inputs)
        x = self.block_2(x)
        x = self.block_3(x)
        x = self.block_4(x)
        x = self.pooling(x)

        x = layers.Reshape((-1,1))(x)
        
        action_values = layers.Cropping1D((0,1))(x)             # remove last
        action_values = layers.Reshape((8,8,-1))(action_values)
        
        state_value = layers.Cropping1D((4672,0))(x)            # keep only last
        state_value = layers.Reshape((-1,))(state_value)

        return action_values, state_value

    def model(self):
        x = layers.Input(shape=(8,8,119))
        return tf.keras.Model(inputs=[x], outputs=self.call(x))
