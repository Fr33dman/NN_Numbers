import numpy as np
import random

def New_data(size):
    random.seed()
    input_neuron = np.array([])
    weight = np.array([])
    for i in range(size):
        input_neuron = np.append(input_neuron,[random.randrange(0, 30)/10*((-1)**random.choice([1, 2]))])
        weight = np.append(weight,[random.randrange(0, 100)/100*((-1)**random.choice([1, 2]))])
    input_neuron = np.reshape(input_neuron, (3,1))
    weight = np.reshape(weight, (1, 3))
    true = round(random.random(), 1)*((-1)**random.choice([1, 2]))
    return weight, input_neuron, true

def Dot(weight, input_neuron):
    answer = np.dot(weight, input_neuron)[0]
    return answer

def float_to_int(arr):
    new_arr = [[0 for j in range(arr.shape[1])] for i in range(arr.shape[0])]
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            new_arr[i][j] = int(arr[i][j])
    return np.array(new_arr)

"""---------------------------------------------------------------------------------------------------------------"""
"""ERRORS"""

class Neural_network_error(Exception):
    pass


class Middle_neurons_error(Neural_network_error):
    def __init__(self, error):
        self.error1 = 'Не совпадает колличество заданных промежуточных слоев сети!'
        self.error2 = 'Неверно задано колличество нейронов в промежуточном слое!'
        self.error = error

    def __str__(self):
        return self.error

class Neural_layer_error(Neural_network_error):
    def __init__(self, error, operation = None):
        self.error1 = 'Не совпадает размер нейронного слоя!'
        if operation:
            self.error = error + '\tНомер записи давший ошибку: ' + str(operation)
        else:
            self.error = error

    def __str__(self):
        return self.error

"""---------------------------------------------------------------------------------------------------------------"""
"""CLASSES"""
"""----------------"""
"""WEIGHT"""

class Weight():
    def __init__(self, input_size, output_size, mode=0, arr=[]):
        self.input_size = input_size
        self.output_size = output_size
        if mode == 0:
            self.matrix = np.array([[random.random() for i in range(self.output_size)] for j in range(self.input_size)])
        elif mode == 1:
            self.matrix = arr


    def getsize(self):
        return [self.input_size, self.output_size]

    def getweight(self):
        return self.matrix

    def learn(self, weight_correction):
        self.matrix -= weight_correction

"""----------------"""
"""NEURAL_LAYER"""

class Neural_layer():
    def __init__(self, size):
        self.size = size
        self.layer = np.zeros((1, size))

    def take(self, vector):
        """Takes and writes down array-vector"""
        #if len(vector) != self.size:
        #    raise Neural_layer_error('Не совпадает размер нейронного слоя!')
        for i in range(self.size):
            self.layer = vector

    def vector(self):
        """Gets array-vector of this neuron layer"""
        return self.layer

    def getsize(self):
        """Gets size of this neuron layer"""
        return self.size

"""----------------"""
"""NEURAL_NETWORK"""

class Neural_network():
    def __init__(self, input_size, output_size, weights=None, quantity_mid_neurons = 0, mid_neuron_size = (), alpha = 1):
        self.middle_neurons = []
        self.input_size = input_size
        self.output_size = output_size
        self.quantity_mid_neurons = quantity_mid_neurons
        self.network_quality = self.quantity_mid_neurons + 2
        self.alpha = alpha
        if self.quantity_mid_neurons != 0:
            self.mid_neuron_size = mid_neuron_size
            if self.quantity_mid_neurons != len(self.mid_neuron_size):
                raise Middle_neurons_error('Не совпадает колличество заданных промежуточных слоев сети!')
            for i in self.mid_neuron_size:
                if (type(i) is not int) or (i < 0):
                    raise Middle_neurons_error('Неверно задано колличество нейронов в промежуточном слое!')
            for i in range(self.quantity_mid_neurons):
                self.middle_neurons.append(Neural_layer(mid_neuron_size[i]))
        self.input_neurons = Neural_layer(self.input_size)
        self.output_neurons = Neural_layer(self.output_size)
        self.network = [self.input_neurons, *self.middle_neurons, self.output_neurons]
        if weights:
            self.weights = [Weight(784, 10, mode=1, arr=np.load(f'{weights}\\weight0.npy', allow_pickle=True))]
        else:
            self.weights = [Weight(self.network[i].getsize(), self.network[i+1].getsize()) for i in range(self.network_quality - 1)]
        self.error = 0
        self.prediction = 0

    def take_information(self, vector, operation = None):
        try:
            self.network[0].take(vector)
        except Neural_layer_error:
            if operation:
                print(operation)

    def dot(self, layer, weight):
        try:
            ans = np.dot(layer, weight)
        except TypeError:
            print('Ошибка в ДОТ',layer, '\n\n', weight, '\n\n', ans)
        else:
            return ans

    def save_weights(self):
        for i in range(len(self.weights)):
            np.save('weights\\weight' + str(i), self.weights[i].getweight())

    def Learning(self, input_information, right_answer, operation = None, what_to_print = 0):
        if len(right_answer) != self.output_size:
            raise Neural_layer_error('Не совпадает размер нейронного слоя!')
        self.take_information(input_information, operation=operation)

        for i in range(self.network_quality - 1):
            self.network[i+1].take(self.dot(self.network[i].vector(), self.weights[i].getweight()))
        self.prediction = np.array(self.network[-1].vector())
        self.error = np.array(self.prediction - right_answer)

        for i in range(self.network_quality - 1):
            #weight_correction = np.array(self.dot(np.array(self.network[-(i+2)].vector()).reshape((self.network[-(i+2)].getsize(), 1)), self.error.reshape((1, self.network[-(i+1)].getsize()))))
            weight_correction = np.array(self.dot(self.error.reshape((self.network[-(i+1)].getsize(), 1)), np.array(self.network[-(i+2)].vector()).reshape(1, (self.network[-(i+2)].getsize())))).transpose()
            self.weights[-(i+1)].learn(self.alpha*weight_correction)

        if what_to_print == 0:
            pass
        elif what_to_print == 1:
            print('Prediction: ', self.prediction)
        return None

    def Work(self, input_information):
        self.take_information(input_information)
        for i in range(self.network_quality - 1):
            self.network[i + 1].take(self.dot(self.network[i].vector(), self.weights[i].getweight()))
        self.prediction = np.array(self.network[-1].vector())
        return self.prediction