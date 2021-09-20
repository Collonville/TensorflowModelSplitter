import random

import numpy as np
import tensorflow as tf


def set_seed(seed: int = 0) -> None:
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
