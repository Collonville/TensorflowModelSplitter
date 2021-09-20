import numpy as np
from tensorflow.keras.models import Model

from tensorflowmodelsplitter.model_architecture_kinds.functional_api import (
    FunctionalModel,
)


def extract_layer(model: Model) -> Model:
    input_layer = model.get_layer(name="input_1").input
    dense_output = model.get_layer(name="dense_1").output

    extract_model = Model(inputs=[input_layer], outputs=dense_output)
    extract_model.summary()

    return extract_model


def extract() -> np.ndarray:
    model = FunctionalModel().get_model()

    extract_model = extract_layer(model=model)
    p = extract_model.predict([[0, 1, 2]])

    return p
