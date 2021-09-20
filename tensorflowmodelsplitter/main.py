import tensorflow as tf

from tensorflowmodelsplitter.extractor.functional_api import (
    extract as functional_api_extractor,
)
from tensorflowmodelsplitter.extractor.subclassing_api import (
    extract as subclassing_api_extractor,
)
from tensorflowmodelsplitter.extractor.subclassing_api_with_original_layer import (
    extract as subclassing_api_with_original_layer_extractor,
)
from tensorflowmodelsplitter.utils.seeds import set_seed


def main():
    predict_result = functional_api_extractor()
    print(predict_result)

    # Run clear_session() to avoid loading duplicated models in GPU or CPU.
    tf.keras.backend.clear_session()
    set_seed()

    predict_result = subclassing_api_extractor()
    print(predict_result)

    tf.keras.backend.clear_session()
    set_seed()

    predict_result = subclassing_api_with_original_layer_extractor()
    print(predict_result)
