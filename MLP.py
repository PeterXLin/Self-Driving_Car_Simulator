# this script of code implement MLP
import numpy as np


class MLP:
    """MLP model"""
    def __init__(self):
        """define the structure of model"""
        # modify here to change model structure(layer, neuron amount)
        neuron_amount = [(4, 10), (10, 1)]
        # [w11, w12],
        # [w21, w22]
        self.weight = set_weight(neuron_amount)
        # [-1, y1, y2......]
        self.output_tmp = set_output_buffer(neuron_amount)
        # [delta1, delta2,......]
        self.delta = set_delta(neuron_amount)
        self.label_buffer = 0
        self.activation_function = ReLU
        self.learning_rate = 0.5

    def forward(self, input_layer_input):
        """forward phase"""
        features, self.label_buffer = split_feature_and_label(input_layer_input)
        self.output_tmp[0][1:] = features.copy()
        for i in range(len(self.weight)):
            self.output_tmp[i+1][1:] = self.activation_function(self.output_tmp[i].dot(self.weight[i].transpose()))

    def backward(self):
        """calculate each weight's delta"""
        # calculate output layer's delta
        self.delta[-1] = self.derivative_of_activation_function(-1) * (self.label_buffer - self.output_tmp[-1][1:])

        # calculate hidden layer's delta
        for i in range(-2, -len(self.weight), -1):
            self.delta[i] = self.derivative_of_activation_function(i) * \
                            self.delta[i+1].dot(np.delete(self.weight[i+1], 0, 1))

    def derivative_of_activation_function(self, layer_index) -> np.ndarray:
        """calculate different activation's derivative"""
        if self.activation_function == ReLU:
            return self.output_tmp[layer_index][0][1:] > 0
        elif self.activation_function == sigmoid:
            return self.output_tmp[layer_index][0][1:] * \
                   (np.ones(self.output_tmp[layer_index].shape) - self.output_tmp[layer_index])[0][1:]

    def update(self):
        """update weight use gradient descent(related to activation function)"""
        for i in range(len(self.weight)):
            self.weight[i] += self.learning_rate * self.delta[i].transpose() * self.output_tmp[i]

    def evaluation(self):
        """get sigma e(n)^2 in output layer"""
        # TODO: finish
        pass

    def save_model(self):
        """store model"""
        # TODO: finish
        pass


def set_weight(each_layer_neuron: list) -> list:
    """create an ndarray list to store each layer's weight"""
    rng = np.random.default_rng(0)
    neuron_list = list()
    for input_d, output_d in each_layer_neuron:
        this_layer = rng.random((output_d, input_d))
        neuron_list.append(this_layer)
    return neuron_list


def set_output_buffer(each_layer_neuron: list) -> list:
    """create an ndarray list to temporarily store each layer's output and bias"""
    output_buffer = list()
    feature = np.zeros((1, each_layer_neuron[0][0] + 1))
    feature[0][0] = -1
    output_buffer.append(np.concatenate(feature))

    for input_d, output_d in each_layer_neuron:
        layer_output = np.zeros((1, output_d + 1))
        layer_output[0][0] = -1
        output_buffer.append(layer_output)
    return output_buffer


def set_delta(each_layer_neuron):
    """create an ndarray list to temporarily store each layer's delta"""
    delta = list()
    for input_d, output_d in each_layer_neuron:
        this_layer_delta = np.zeors((1, output_d))
        delta.append(this_layer_delta)
    return delta


def ReLU(v_n: np.ndarray) -> np.ndarray:
    """if v(n) >= 0 return v(n), else return 0"""
    v_n[v_n < 0] = 0
    return v_n


def sigmoid(v_n: np.ndarray) -> np.ndarray:
    """sigmoid function"""
    return 1/(1+np.exp(-v_n))


def split_feature_and_label(features_and_label):
    """[1, 2, 3,] -> [1, 2] , [3]"""
    # TODO: check input shape
    return features_and_label[:-1], features_and_label[-1:].reshape(1, 1)


if __name__ == '__main__':
    neuron = [(4, 3), (0, 0)]
    tmp = set_weight(neuron)
    array = np.array([1, 2, 3])
    print(tmp)
