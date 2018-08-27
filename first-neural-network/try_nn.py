import numpy as np

from my_answers import NeuralNetwork


def MSE(y, Y):
    return np.mean((y-Y)**2)


inputs = np.array([[0.5, -0.2, 0.1]])
targets = np.array([[0.4]])
test_w_i_h = np.array([[0.1, -0.2],
                       [0.4, 0.5],
                       [-0.3, 0.2]])
test_w_h_o = np.array([[0.3],
                       [-0.1]])


network = NeuralNetwork(3, 2, 1, 0.5)
# Test that the activation function is a sigmoid
print(np.all(network.activation_function(0.5) == 1/(1+np.exp(-0.5))))


# Test that weights are updated correctly on training
network = NeuralNetwork(3, 2, 1, 0.5)
network.weights_input_to_hidden = test_w_i_h.copy()
network.weights_hidden_to_output = test_w_h_o.copy()

network.train(inputs, targets)
print(network.weights_hidden_to_output)
print(np.allclose(network.weights_hidden_to_output,
                  np.array([[0.37275328],
                            [-0.03172939]])))
print(np.allclose(network.weights_input_to_hidden,
                  np.array([[0.10562014, -0.20185996],
                            [0.39775194, 0.50074398],
                            [-0.29887597, 0.19962801]])))

# Test correctness of run method
network = NeuralNetwork(3, 2, 1, 0.5)
network.weights_input_to_hidden = test_w_i_h.copy()
network.weights_hidden_to_output = test_w_h_o.copy()

print(np.allclose(network.run(inputs), 0.09998924))
