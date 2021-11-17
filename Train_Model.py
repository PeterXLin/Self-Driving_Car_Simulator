import Model
import Dataset
import numpy as np


def train_model(config: dict, dataset: Dataset, model: 'MLP'):
    min_error = 10000
    check_count = 0
    for n in range(config['epoch']):
        np.random.shuffle(dataset.training_dataset)
        for training_data in dataset.training_dataset:
            # print(training_data)
            model.forward(training_data)
            model.backward()
            model.update()
            # test if model is better than before
            # if check_count == config['check_packet_frequency']:
            #     tmp_error = 0
            #     for validation_data in dataset.validation_dataset:
            #         if model.predict(validation_data) != validation_data[-1]:
            #             tmp_error = tmp_error + 1
            #     if tmp_error < min_error:
            #         min_error = tmp_error
            #         model.best_neuron_list = model.neuron_list.copy()
            #         if dataset.validation_data_size > 0 and \
            #                 tmp_error/dataset.validation_data_size <= config['error_rate']:
            #             return model
            #     check_count = 0
            # else:
            #     check_count = check_count + 1
    return model


if __name__ == '__main__':
    my_config = {
        'epoch': 500,
        'check_packet_frequency': 100
    }
    my_dataset = Dataset.Dataset('./data/drive_by_rule.txt', True, 'sigmoid')

    my_model = Model.MLP([(3, 10), (10, 5), (5, 1)], 'sigmoid', 0.05)
    # my_model = Model.load_model('sigmoid_model_2.txt')
    my_model = train_model(my_config, my_dataset, my_model)

    print(my_model.weight)
    my_model.save_model('sigmoid_model_1_data_from_rule.txt')
