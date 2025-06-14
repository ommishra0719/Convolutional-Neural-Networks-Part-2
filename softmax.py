import numpy as np

class Softmax:
    #A standard fully connected layer with softmax activation.
    #We’ll use a softmax layer with 10 nodes, one representing each digit, as the final layer in our CNN.
    def __init__(self, input_len, nodes):
        #We divide by input_len to reduce the variance of our initial values
        self.last_input = None
        self.weights = np.random.randn(input_len, nodes) / input_len
        self.biases = np.zeros(nodes)

    def forward(self, input):
        #Performs a forward pass of the softmax layer using the given input.
        #Returns a 1d array containing the respective probability values.
        #input can b any array with any dimensions.
        self.last_input_shape = input.shape

        input = input.flatten()
        self.last_input = input

        input_len, nodes = self.weights.shape

        totals = np.dot(input, self.weights) + self.biases
        self.last_totals = totals

        exp = np.exp(totals)
        return exp / np.sum(exp, axis = 0)

    def backdrop(self, d_L_d_out, learn_rate):
        # We know only 1 element of d_L_d_out will be nonzero
        for i, gradient in enumerate(d_L_d_out):
            if gradient == 0:
                continue

            # e^totals
            t_exp = np.exp(self.last_totals)

            # Sum of all e^totals
            S = np.sum(t_exp)

            # Gradients of out[i] against totals
            # if t is not t[i]
            d_out_d_t = -t_exp[i] * t_exp / (S ** 2)
            # if t is t[i]
            d_out_d_t[i] = t_exp[i] * (S - t_exp[i]) / (S ** 2)

            # Gradients of totals against weights/biases/input
            # t = w*input + b
            d_t_d_w = self.last_input
            d_t_d_b = 1
            d_t_d_inputs = self.weights
            # Gradients of loss against totals
            d_L_d_t = gradient * d_out_d_t
            # Gradients of loss agaisnt weights/biases/input
            d_L_d_w = d_t_d_w[np.newaxis].T @ d_L_d_t[np.newaxis]
            d_L_d_b = d_L_d_t * d_t_d_b
            d_L_d_inputs = d_t_d_inputs @ d_L_d_t

            # Update weights / biases
            self.weights -= learn_rate * d_L_d_w
            self.biases -= learn_rate * d_L_d_b
            return d_L_d_inputs.reshape(self.last_input_shape)