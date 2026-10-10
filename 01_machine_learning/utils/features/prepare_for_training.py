"""Prepares the dataset for training"""

import numpy as np
from .normalize import normalize
from .generate_sinusoids import generate_sinusoids
from .generate_polynomials import generate_polynomials


def prepare_for_training(data, polynomial_degree=0, sinusoid_degree=0, normalize_data=True, features_mean=None, features_deviation=None):

    num_examples = data.shape[0]
    data_processed = np.copy(data)

    # 1. 先做特征变换（这里不再单独归一化）
    if sinusoid_degree > 0:
        sinusoids = generate_sinusoids(data_processed, sinusoid_degree)
        data_processed = np.concatenate((data_processed, sinusoids), axis=1)

    if polynomial_degree > 0:
        polynomials = generate_polynomials(data_processed, polynomial_degree)
        data_processed = np.concatenate((data_processed, polynomials), axis=1)

    # 2. 再统一归一化：训练时算标尺，预测时沿用训练存的标尺
    if normalize_data:
        if features_mean is not None and features_deviation is not None:
            data_processed = (data_processed - features_mean) / (features_deviation + 0.0000001)
        else:
            data_processed, features_mean, features_deviation = normalize(data_processed)

    # 3. 最后加一列 1（它不参与归一化）
    data_processed = np.hstack((np.ones((num_examples, 1)), data_processed))

    return data_processed, features_mean, features_deviation
