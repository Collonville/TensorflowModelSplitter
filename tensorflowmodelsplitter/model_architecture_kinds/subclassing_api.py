import tensorflow as tf
from tensorflow.keras import layers


class SubclassingModel(tf.keras.Model):
    def __init__(self, *args, **kwargs) -> None:
        super(SubclassingModel, self).__init__(*args, **kwargs)

        self.dense_1 = layers.Dense(5, name="dense_1")
        self.dense_2 = layers.Dense(5, name="dense_2")

        self.merged_layer = layers.Concatenate(name="concat_layer")
        self.out_1 = layers.Dense(3)
        self.out_2 = layers.Dense(1, name="output")

    def call(self, inputs, training=False):
        dense_1 = self.dense_1(inputs[0])
        dense_2 = self.dense_2(inputs[1])

        merged_layer = self.merged_layer([dense_1, dense_2])
        out_1 = self.out_1(merged_layer)
        out_2 = self.out_2(out_1)

        return out_2
