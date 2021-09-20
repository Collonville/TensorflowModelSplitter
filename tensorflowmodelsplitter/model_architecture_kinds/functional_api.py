from tensorflow.keras import layers, losses, metrics, optimizers
from tensorflow.keras.models import Model


class FunctionalModel:
    def __init__(self) -> None:
        pass

    def _model(self) -> Model:
        input_1 = layers.Input(shape=(3,), name="input_1")
        input_2 = layers.Input(shape=(5,), name="input_2")

        dense_1 = layers.Dense(5, name="dense_1")(input_1)
        dense_2 = layers.Dense(5, name="dense_2")(input_2)

        merged_layer = layers.Concatenate(name="concat_layer")([dense_1, dense_2])
        out = layers.Dense(3)(merged_layer)
        out = layers.Dense(1, name="output")(out)

        return Model(inputs=[input_1, input_2], outputs=out)

    def get_model(self) -> Model:
        model = self._model()
        model.compile(
            optimizer=optimizers.Adam(), loss=losses.mse, metrics=[metrics.mse]
        )

        return model
