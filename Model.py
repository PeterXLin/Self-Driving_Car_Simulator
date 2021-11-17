# this script of code implement MLP
import numpy as np


class MLP:
    """MLP model"""
    def __init__(self, neuron_amount: list, activation_function='sigmoid', lr: float = 0.05):
        """define the structure of model"""
        # modify here to change model structure(layer, neuron amount)
        self.neuron_amount = neuron_amount
        # [w10, w11, w12],
        # [w20, w21, w22]
        self.weight = set_weight(self.neuron_amount)
        # [-1, y1, y2......] -> include bias
        self.output_tmp = set_output_buffer(self.neuron_amount)
        # [delta1, delta2,......]
        self.delta = set_delta(self.neuron_amount)
        self.label_buffer = 0
        if activation_function == 'ReLU':
            self.activation_function = ReLU
        elif activation_function == 'sigmoid':
            self.activation_function = sigmoid
        self.learning_rate = lr

    def forward(self, input_layer_input):
        """forward phase"""
        features, self.label_buffer = split_feature_and_label(input_layer_input)
        # print(self.label_buffer)
        # keep bias -1
        self.output_tmp[0][0][1:] = features
        # print(self.output_tmp[0][0][1:])
        for i in range(len(self.weight)):
            # print(self.weight[i].transpose())
            # print(self.output_tmp[i].dot(self.weight[i].transpose()))
            self.output_tmp[i+1][0][1:] = self.activation_function(self.output_tmp[i].dot(self.weight[i].transpose()))

    def backward(self):
        """calculate each weight's delta"""
        # calculate output layer's delta, [1:] -> no bias in output layer
        self.delta[-1] = (self.label_buffer - self.output_tmp[-1][0][1:]) * self.derivative_of_activation_function(-1)

        # calculate hidden layer's delta
        for i in range(-2, -len(self.weight) - 1, -1):
            self.delta[i] = self.derivative_of_activation_function(i) * \
                            self.delta[i+1].dot(np.delete(self.weight[i+1], 0, 1))

    def derivative_of_activation_function(self, layer_index) -> np.ndarray:
        """calculate different activation's derivative"""
        if self.activation_function == ReLU:
            return self.output_tmp[layer_index][0][1:] > 0
        elif self.activation_function == sigmoid:
            # y * (1 - y)
            return self.output_tmp[layer_index][0][1:] * \
                   (np.ones(self.output_tmp[layer_index].shape) - self.output_tmp[layer_index])[0][1:]

    def update(self):
        """update weight use gradient descent(related to activation function)"""
        for i in range(len(self.weight)):
            self.weight[i] += self.learning_rate * self.delta[i].transpose().dot(self.output_tmp[i])

    def predict(self, features):
        """get output"""
        self.output_tmp[0][0][1:] = features.copy()
        # print(self.output_tmp[0])
        for i in range(len(self.weight)):
            self.output_tmp[i + 1][0][1:] = self.activation_function(self.output_tmp[i].dot(self.weight[i].transpose()))
        # return output layer([[1]])
        if self.activation_function == ReLU:
            return self.output_tmp[-1][0][1] - 40
        elif self.activation_function == sigmoid:
            return (self.output_tmp[-1][0][1] * 80) - 40

    def save_model(self, filename):
        """store model"""
        with open('./data/' + filename, 'w') as fd:
            # record activation function
            if self.activation_function == ReLU:
                fd.write('ReLU' + ' ')
            elif self.activation_function == sigmoid:
                fd.write('sigmoid' + ' ')
            fd.write('\n')

            # record model size
            for neuron_size in self.neuron_amount:
                fd.write(str(neuron_size[0]) + ' ' + str(neuron_size[1]) + ' ')
            fd.write('\n')

            # record weight
            for neuron_weight in self.weight:
                for row in neuron_weight:
                    for column in row:
                        fd.write(str(column))
                        fd.write(' ')
                    fd.write('\n')


def set_weight(each_layer_neuron: list) -> list:
    """create an ndarray list to store each layer's weight"""
    rng = np.random.default_rng(0)
    neuron_list = list()
    for input_d, output_d in each_layer_neuron:
        # input_d + 1 -> include bias' weight
        this_layer = rng.random((output_d, input_d + 1))
        neuron_list.append(this_layer)
    return neuron_list


def set_output_buffer(each_layer_neuron: list) -> list:
    """create an ndarray list to temporarily store each layer's output and bias, even last has bias block"""
    output_buffer = list()
    # store input features
    feature = np.zeros((1, each_layer_neuron[0][0] + 1))
    # set bias to zero
    feature[0][0] = -1
    output_buffer.append(feature)

    for input_d, output_d in each_layer_neuron:
        # store each layer's output
        layer_output = np.zeros((1, output_d + 1))
        layer_output[0][0] = -1
        output_buffer.append(layer_output)
    return output_buffer


def set_delta(each_layer_neuron):
    """create an ndarray list to temporarily store each layer's delta"""
    delta = list()
    for input_d, output_d in each_layer_neuron:
        # no delta for bias
        this_layer_delta = np.zeros((1, output_d))
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
    """[1, 2, 3] -> [1, 2] , [3]"""
    return features_and_label[:-1].reshape(1, features_and_label.shape[0] - 1), features_and_label[-1:].reshape(1, 1)


def load_model(filename):
    with open('./model/' + filename, 'r') as fp:
        model_activation_function = fp.readline().split(' ')[0]
        model_layer_raw = fp.readline().split(' ')
        model_layer = list()
        # -1 to avoid '\n'
        for i in range(0, len(model_layer_raw) - 1, 2):
            model_layer.append((int(model_layer_raw[i]), int(model_layer_raw[i+1])))

        my_model = MLP(model_layer, model_activation_function)
        # set weight
        for i in range(len(model_layer)):
            for j in range(model_layer[i][1]):
                tmp = fp.readline().split(' ')
                # len(tmp) include '\n\, so len - 1
                for k in range(len(tmp) - 1):
                    my_model.weight[i][j][k] = tmp[k]
    return my_model
