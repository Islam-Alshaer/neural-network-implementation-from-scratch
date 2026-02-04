import numpy as np

def test_network_initialization():
    layer_sizes = [3, 2, 1]
    activation_types = ['relu', 'sigmoid', 'relu']
    network = Network(layer_sizes, activation_types, 3)

    #check how many layers are there
    expected_layers = len(layer_sizes)
    actual_layers = len(network.layers)
    if actual_layers != expected_layers:
        print(f'Test failed: expected {expected_layers} layers, got {actual_layers}')
    else:
        print('initialization test passed')

    #check weights of the first neuron in each layer
    # for layer in network.layers:
    #     print(layer.weights)


def test_feed_forward():
    layer_sizes = [2, 2]
    activation_types = ['relu', 'relu']
    network = Network(layer_sizes, activation_types, 2)
    input_data = [-1.0, 2.0]

    #change the weights to known values for testing
    network.layers[0].weights = np.array([[2.0, -1.0], [3.0, 1.0]])
    network.layers[1].weights = np.array([[4.0, 0.5], [5.0, 1.0]])
    network.feed_forward(input_data)

    #check output of the network
    output = network.layers[-1].outs
    expected_output = [31.0, 5.0]
    if (output != expected_output).any():
        print(f'Test feed forward 1 failed: expected output {expected_output}, got {output}')
    else:
        print('test feed forward 1 passed')


    layer_sizes = [2, 2]
    activation_types = ['poly', 'poly']
    network = Network(layer_sizes, activation_types, 2)
    input_data = [1.0, 1.0]
    network.layers[0].weights = np.array([[1.0, 2.0], [1.0, 1.0]])
    network.layers[1].weights = np.array([[2.0, 1.0], [1.0, 0.0]])
    network.feed_forward(input_data)
    if network.layers[-1].outs[0] != 289.0 or network.layers[-1].outs[1] != 16.0:
        print('Test feed forward 2 failed')
    else:
        print("Test feed forward 2 passed")

def test_back_propagation():
    layer_sizes = [2, 2]
    activation_types = ['poly', 'poly']
    network = Network(layer_sizes, activation_types, 2)
    input_data = [1.0, 1.0]
    network.layers[0].weights = np.array([[1.0, 2.0], [1.0, 1.0]])
    network.layers[1].weights = np.array([[2.0, 1.0], [1.0, 0.0]])
    network.feed_forward(input_data)
    target_output = np.array([290.0, 14.0])
    learning_rate = 0.5
    network.backpropagate(target_output, learning_rate, input_data)

    expected_weights_layer_0 = np.array([[105.0, 106.0], [103.0 ,103.0]])
    expected_weights_layer_1 = np.array([[70.0, 154.0], [-31.0, -72.0]])

    if (network.layers[0].weights != expected_weights_layer_0).any() or (network.layers[1].weights != expected_weights_layer_1).any():
        print("Test backpropagation 1 failed")
    else:
        print("Test backpropagation 1 passed")

    #test 2
    layer_sizes = [2, 2]
    activation_types = ['sigmoid', 'sigmoid']
    network = Network(layer_sizes, activation_types, 2)
    input_data = [0.05, 0.10]
    network.layers[0].weights = np.array([[0.15, 0.20], [0.25, 0.30]])
    network.layers[1].weights = np.array([[0.40, 0.45], [0.50, 0.55]])
    network.feed_forward(input_data)
    target_output = np.array([0.01, 0.99])
    learning_rate = 0.5
    network.backpropagate(target_output, learning_rate, input_data)

    expected_weights_layer_0 = np.array([[0.14988347, 0.19976693], [0.24984766, 0.29969532]])
    expected_weights_layer_1 = np.array([[0.36366666, 0.41353263], [0.52176661, 0.57184691]])
    if (network.layers[0].weights.round(8) != expected_weights_layer_0.round(8)).any() or (network.layers[1].weights.round(8) != expected_weights_layer_1.round(8)).any():
        print("Test backpropagation 2 failed")
    else:
        print("Test backpropagation 2 passed")

class Network:
    def __init__(self, layer_sizes, activation_types, n_inputs):
        self.layers = []
        for i in range(len(layer_sizes)):
            if i == 0:
                layer = Layer(layer_sizes[i], n_inputs, activation_types[i]) #the output of the layer before the first hidden layer is just the input data itself
            else:
                layer = Layer(layer_sizes[i], layer_sizes[i - 1], activation_types[i])

            self.layers.append(layer)
            #but for real, if you stop cereal industry, does that make you a serial killer? (for whatever reason I thought of that joke here)

    def feed_forward(self, input_data):
        #in the first hidden layer the input is just the input data
        for i in range(len(self.layers)):
            layer = self.layers[i]
            if i == 0:
                layer.nets = input_data @ layer.weights
            else:
                previous_layer = self.layers[i - 1]
                #calculate the output of the current layer
                layer.nets = previous_layer.outs @ layer.weights
            layer.activate_neurons()

    def backpropagate(self, target_output, learning_rate, input_data):
        #step 1: calculate partial E / partial net for every neuron in every layer
        #the base case is the output layer
        E = self.layers[-1].outs - target_output #1D array of partial E / partial out for output layer
        net_gradients = [E * self.layers[-1].activation_derivative()]  #2D list of partial E / partial net for every layer

        #for hidden layers going backwards :
        for i in range(len(self.layers) - 2, -1, -1): #the reason we go to -1 is because we want to include layer 0 which is input layer
            current_layer = self.layers[i]
            next_layer = self.layers[i + 1]
            partial_E_over_partial_net_next = net_gradients[-1]
            partial_net_next_over_partial_out_current = next_layer.weights
            partial_out_current_over_partial_net_current = current_layer.activation_derivative()

            partial_E_partial_net_current = (partial_E_over_partial_net_next @ partial_net_next_over_partial_out_current.T) * partial_out_current_over_partial_net_current
            net_gradients.append(partial_E_partial_net_current)

        net_gradients.reverse()  #reverse to match layer order

        #step 2: calculate partial net / partial w for every weight in every layer
        weights_gradients = [] # 3D array representing gradients for each layer
        for i in range(len(self.layers) - 1, -1, -1):
            previous_layer = self.layers[i - 1]
            partial_E_over_partial_net_next = np.array(net_gradients[i])
            if i == 0:
                output_of_previous_layer = np.array(input_data)
            else:
                output_of_previous_layer = np.array(previous_layer.outs)
            gradients = np.outer(partial_E_over_partial_net_next, output_of_previous_layer) # shape is m x n because 1xm.T @ 1xn = mxn
            weights_gradients.append(gradients) #every layer has mxn weights

        #step 3: update weights
        for i in range(len(self.layers)):
            current_layer = self.layers[i]
            current_layer.weights -= learning_rate * weights_gradients[i - 1]

class Layer:
    def __init__(self, m_neurons, n_entering_weights, activation_type):
        #we should have a matrix of weights n x m such that the previous layer has n neurons and the current layer has m neurons
        self.weights = []
        #for n times:
        for i in range(n_entering_weights):
            #create an array of random weights of size m and append it to the weights matrix
            self.weights.append(np.random.randn(1, m_neurons))
            # for testing purposes, we can set the weights to be all ones # self.weights.append(np.ones((1, m_neurons)))
        self.weights = np.vstack(self.weights)  #shape will be (n_entering_weights, m_neurons)
        self.nets = None
        self.outs = None
        self.activation_type = activation_type

    def activate_neurons(self):
        if self.nets is None:
            raise ValueError("Net inputs are not set.")

        if self.activation_type == 'poly':
            self.outs = self.nets ** 2
        elif self.activation_type == 'relu':
            self.outs = np.maximum(0, self.nets)
        elif self.activation_type == 'sigmoid':
            self.outs = 1 / (1 + np.exp(-self.nets))
        elif self.activation_type == 'softmax':
            exp_nets = np.exp(self.nets - np.max(self.nets))  # for numerical stability
            self.outs = exp_nets / np.sum(exp_nets)

    def activation_derivative(self):
        if self.activation_type == 'poly':
            return 2 * self.nets
        elif self.activation_type == 'relu':
            return np.where(self.nets > 0, 1, 0)
        elif self.activation_type == 'sigmoid':
            sigmoid_out = 1 / (1 + np.exp(-self.nets))
            return sigmoid_out * (1 - sigmoid_out)
        elif self.activation_type == 'softmax':
            softmax_out = self.outs
            return softmax_out * (1 - softmax_out)
        else:
            raise ValueError("Unknown activation type.")

if __name__ == "__main__":
    test_network_initialization()
    test_feed_forward()
    test_back_propagation()












