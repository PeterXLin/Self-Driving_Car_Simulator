import numpy as np

# TODO: sigmoid 輸出只有0-1
# TODO: relu 無法出現負數


class Dataset:
    def __init__(self, path, all_train: float = False):
        raw_data = np.loadtxt(path, dtype=np.float, delimiter=' ')
        training_data_size = ((raw_data.shape[0] * 2) // 3) + 1
        if all_train:
            training_data_size = raw_data.shape[0]
        self.feature = raw_data.shape[1] - 1
        # transfer class number to 0~n-1
        np.random.shuffle(raw_data)
        self.training_dataset = raw_data[:training_data_size]
        self.validation_dataset = raw_data[training_data_size:]
        self.training_data_size = training_data_size
        self.validation_data_size = raw_data.shape[0] - training_data_size
