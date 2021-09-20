import numpy as np
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model

from tensorflowmodelsplitter.model_architecture_kinds.subclassing_api import (
    SubclassingModel,
)


# Same as Functional API.
def extract_layer(model: Model) -> Model:
    input_layer = model.get_layer(name="input_1").input
    dense_output = model.get_layer(name="dense_1").output

    extract_model = Model(inputs=[input_layer], outputs=dense_output)
    extract_model.summary()

    return extract_model


def extract() -> np.ndarray:
    model = SubclassingModel()

    # Sub classing API cannot get layer information before build or predict.
    # input_1 = model.get_layer("dense_1")
    # print(input_1)

    # Run predict can get summary but not still get layer.
    # p = model.predict([[[0, 1, 2]], [[0, 1, 2]]])
    # print(p)
    # model.summary()

    # Adding Input layers to Model class and use call method can extract layers.
    # Do not run predict() before adding Input layer.
    input_layes = [Input(shape=(3,), name="input_1"), Input(shape=(5,), name="input_2")]
    model_with_input_layer = Model(inputs=input_layes, outputs=model.call(input_layes))

    extracted_model = extract_layer(model=model_with_input_layer)
    p = extracted_model.predict([[0, 1, 2]])

    return p
