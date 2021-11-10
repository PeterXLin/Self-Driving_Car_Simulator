import numpy as np

# TODO: sigmoid 輸出只有0-1
# TODO: relu 無法出現負數


class Dataset:
    def __init__(self, path, all_train: float = False, activation_function="ReLU"):
        raw_data = np.loadtxt(path, dtype=np.float, delimiter=' ')
        self.feature = raw_data.shape[1] - 1

        if activation_function == 'ReLU':
            for i in range(raw_data.shape[0]):
                raw_data[i][-1] = raw_data[i][-1] + 40
        elif activation_function == 'sigmoid':
            for i in range(raw_data.shape[0]):
                raw_data[i][-1] = (raw_data[i][-1] + 40) / 80

        np.random.shuffle(raw_data)
        # split raw_data into training_dataset and testing_dataset
        training_data_size = ((raw_data.shape[0] * 2) // 3) + 1
        if all_train:
            training_data_size = raw_data.shape[0]
        self.training_dataset = raw_data[:training_data_size]
        self.validation_dataset = raw_data[training_data_size:]
        self.training_data_size = training_data_size
        self.validation_data_size = raw_data.shape[0] - training_data_size
