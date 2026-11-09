import numpy as np


def downsample(data: np.ndarray, rate: int):
    assert data.ndim in (1, 2)

    if data.ndim == 1:
        data_downsampled = np.zeros(data.shape[0] // rate, dtype=data.dtype)
    elif data.ndim == 2:
        data_downsampled = np.zeros((data.shape[0] // rate, data.shape[1]), dtype=data.dtype)

    for i in range(len(data_downsampled)):
        data_downsampled[i] = np.median(data[rate*i:rate*(i+1)], axis=0)

    return data_downsampled


def downsample_binary_labels(labels: np.ndarray, rate: int):
    """按时间块聚合二值标签；块内任一点异常即视为异常。

    连续特征可以使用中位数降采样，但异常标签不能使用中位数，否则短异常会被
    静默抹去。这里与滑动窗口标签的 ``any`` 语义保持一致。
    """
    labels = np.asarray(labels)
    if labels.ndim not in (1, 2):
        raise ValueError("binary labels must be one- or two-dimensional")
    rate = int(rate)
    if rate < 1:
        raise ValueError("downsample rate must be positive")
    block_count = labels.shape[0] // rate
    trimmed = labels[: block_count * rate]
    if labels.ndim == 1:
        return trimmed.reshape(block_count, rate).max(axis=1).astype(labels.dtype)
    return trimmed.reshape(block_count, rate, labels.shape[1]).max(axis=1).astype(
        labels.dtype
    )
