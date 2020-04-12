import numpy as np
import torch


def relu(z):
    if np.isscalar(z):
        result = np.max((z, 0))
    else:
        zero_aux = np.zeros(z.shape)
        meta_z = np.stack((z, zero_aux), axis=-1)
        result = np.max(meta_z, axis=-1)
    return result


def logsumexp(Z):  # NxC
    Zmax = np.max(Z, axis=1)[:, None]
    lse = Zmax + np.log(np.sum(np.exp(Z - Zmax), axis=1))[:, None]
    return lse  # N


def softmax(u):  # N x C
    u_exp = np.exp(u - np.max(u, 1)[:, None])
    return u_exp / np.sum(u_exp, axis=-1)[:, None]


def sigmoid(z):
    result = 1.0 / (1.0 + np.exp(-z))
    return result


def sigmoid_derivative(dA, z):
    sig = sigmoid(z)
    return dA * sig * (1 - sig)


def relu_derivative(dA, Z):
    dZ = np.array(dA, copy = True)
    dZ[Z <= 0] = 0;
    return dZ;


class MLP:

    def __init__(self, size_per_layer, activation_func):
        self.layer_size = size_per_layer
        self.n_layers = len(size_per_layer)
        self.activation_func = activation_func

        self.weights = self.initialize_weights()

    def train(self, X, Y, batch_size=10, epochs=1000, lr=.01):
        """
        params_values = init_layers(nn_architecture, 2)
        cost_history = []
        accuracy_history = []

        for i in range(epochs):
            Y_hat, cashe = full_forward_propagation(X, params_values, nn_architecture)
            cost = get_cost_value(Y_hat, Y)
            cost_history.append(cost)
            accuracy = get_accuracy_value(Y_hat, Y)
            accuracy_history.append(accuracy)

            grads_values = full_backward_propagation(Y_hat, Y, cashe, params_values, nn_architecture)
            params_values = update(params_values, grads_values, nn_architecture, learning_rate)

        return params_values, cost_history, accuracy_history
        """

        for epoch in range(epochs):
            # for our code, we do the fullforward and cost inside the backpropogation right away.
            gradients = self.backward(X, Y, learning_rate=lr, batch_size=batch_size)

            # Todo: need to update those weights after doing backpropagation but use a better function.
            self.weights = np.subtract(self.weights, gradients)

    def predict(self, X):
        Z, A = self.feedforward(X)
        return A[-1]

    def initialize_weights(self):
        weights = []
        for i in range(0, self.n_layers - 1):
            size_layer = self.layer_size[i]
            size_next_layer = self.layer_size[i + 1]
            if self.activation_func == 'sigmoid':
                theta_tmp = ((np.random.rand(size_next_layer, size_layer) * 2.0) - 1)
            elif self.activation_func == 'relu':
                theta_tmp = (np.random.rand(size_next_layer, size_layer))
            weights.append(theta_tmp)
        return np.asarray(weights)

    def batch(self, X, Y, batch_size=0.3):
        n = X.shape[0] * batch_size
        randoms = np.random.randint(X.shape[0], size=int(n))
        return X[randoms], Y[randoms]

    '''Just the first couple lines that were in the gradient function
        All we need here is Z (output of out first layer dot weights corresponding to that layer)
        and Yh (output) which is the Z dot W (weights of last layer)'''
    def feedforward(self, X):
        """
        :param X: Input dataset
        :return: all the activations as an ndarray, Yhat (prediction)
        """
        activations = []
        inputs = X
        for layer in range(self.n_layers - 1):
            if self.activation_func == 'relu':
                if layer == self.n_layers - 1:
                    A, Z = self.single_forward_propagation(inputs, self.weights[layer], softmax)
                else:
                    A, Z = self.single_forward_propagation(inputs, self.weights[layer], relu)
            elif self.activation_func == 'sigmoid':
                if layer == self.n_layers - 1:
                    A, Z = self.single_forward_propagation(inputs, self.weights[layer], softmax)
                else:
                    A, Z = self.single_forward_propagation(inputs, self.weights[layer], sigmoid)
            activations.append(Z)
        return np.asarray(activations), A

    def single_forward_propagation(self, inputs, weights, activation):
        Z = np.dot(weights, inputs)
        return activation(Z), Z


    '''Everything is in a while loop as was in GD, difference here is that this now takes as input W and V (could calculate it here too idk got confused)
    Then from the Z and Yh obtained from forward, we calculate everything else that was previously in gradients to obtain dW and dV
    , then finally subtract those values from W and V respectively
    
    Questions: why feed forward at every iteration?? Its doing that with regular GD as well. When GD calls gradient() at every iteration gradient is running feedforward
    WHYYYYYYYYYY AHHHHHHHHH, '''
    def backpropogation(self, X, Y, learning_rate=.1, eps=1e-9, max_iters=100000, batch_size=0.3):
        # batch the data radomly
        X_batch, Y_batch = self.batch(X, Y, batch_size=batch_size)

        # cost should be computed in the backprop stage
        Z, Yhat = self.feedforward(X_batch)
        dZ = [] * Z.shape
        dW = [] * Z.shape
        dV = [] * Z.shape

        dA = - (np.divide(Y, Yhat) - np.divide(1 - Y, 1 - Yhat))
        N, D = X.shape
        t = 0
        #while np.linalg.norm(dW) > eps and t < max_iters:
        for layer in np.arrange(self.n_layers - 2, 0, -1):

            dY = self.SSD(Yhat, Y_batch)  # cost

            if self.activation_func == 'sigmoid':
                dZ[layer] = sigmoid_derivative(dA, Z[layer]) # D x M
            elif self.activation_func == 'relu':
                dZ[layer] = relu_derivative(dA, Z[layer]) # D x M

            # Something happens after
            dW[layer] = np.dot(dZ[layer], Yhat[layer+1].T) / N  # M x K
            dV[layer] = np.sum(dZ[layer], axis=1, keepdims=True) / N
            dA = np.dot(self.weights[layer].T, dZ[layer])

        return dW, dV


    def SSD(self, Yh, Y):
        return np.sum((Yh - Y) ** 2)


    # def gradients(self,
    #               X,  # N x D
    #               Y,  # N x K
    #               W,  # M x K
    #               V,  # D x M
    #               ):
    #     # forwards
    #     Z = self.logistic(np.dot(X, V))  # N x M
    #     N, D = X.shape
    #     Yh = softmax(np.dot(Z, W))  # N x K
    #
    #     # cost function, lets use a better cost function!
    #     dY = Yh - Y  # N x K
    #
    #     # Something happen after
    #     dW = np.dot(Z.T, dY) / N  # M x K
    #     dZ = np.dot(dY, W.T)  # N x M
    #
    #     # calculate derivative
    #     dV = np.dot(X.T, dZ * Z * (1 - Z)) / N  # D x M
    #     return dW, dV

    # def feedforward(self, x):
    #     input_layer = x
    #     a = self.n_layers * [None]
    #     z = self.n_layers * [None]
    #     output_layer = 0.0
    #
    #     for layer in range(self.n_layers - 1):
    #         a[layer] = input_layer
    #         z[layer + 1] = np.dot(input_layer, self.weights[layer].tranpose())
    #
    #         if self.activation_func == 'sigmoid':
    #             # if last layer, apply softmax
    #             if layer + 1 == self.n_layers - 1:
    #                 output_layer = softmax(z[layer + 1])
    #             else:
    #                 output_layer = sigmoid(z[layer + 1])
    #         elif self.activation_func == 'relu':
    #             # if last layer, apply softmax
    #             if layer + 1 == self.n_layers - 1:
    #                 output_layer = softmax(z[layer + 1])
    #             else:
    #                 output_layer = relu(z[layer + 1])
    #
    #         input_layer = output_layer
    #
    #     a[self.n_layers - 1] = output_layer
    #     return a, z

    # def backpropagation(self, x, y):
    #     n_examples = x.shape[0]
    #     a, z = self.feedforward(y)
    #
    #     # backpropagation, calculate weights based on the softmax function (last layer)
    #     updates = self.n_layers * [None]
    #     updates[-1] = a[-1] - y
    #
    #     # update weights from the second last layer to the second layer
    #     for layer in np.arrange(self.n_layers - 2, 0, -1):
    #         if self.activation_func == 'sigmoid':
    #             updates[layer] = (np.dot(self.weights[layer].T, updates[layer + 1].T).T) * sigmoid_derivative(z[layer])
    #         elif self.activation_func == 'relu':
    #             updates[layer] = (np.dot(self.weights[layer].T, updates[layer + 1].T).T) * relu_derivative(z[layer])
    #
    #     # compute gradient
    #     gradients = (self.n_layer - 1) * [None]
    #     for layer in range(self.n_layers - 1):
    #         temp_grads = np.dot(updates[layer + 1].T, a[layer]) / n_examples
    #         temp_grads = temp_grads + (self.lambda_r / n_examples) * self.weights[layer][:, 1:]
    #
    #         gradients[layer] = temp_grads
    #
    #     return gradients

    # def GD(self, X, Y, M, learning_rate=.1, eps=1e-9, max_iters=100000):
    #     N, D = X.shape
    #     N, K = Y.shape
    #
    #     # make it ndarray right away!
    #     W = np.random.randn(M, K) * .01
    #     V = np.random.randn(D, M) * .01
    #     dW = np.inf * np.ones_like(W)
    #     t = 0
    #     while np.linalg.norm(dW) > eps and t < max_iters:
    #         # batch the data radomly first
    #         X_train, Y_train = self.batch(X, Y)
    #
    #         # do forward here, break up gradients function
    #         dW, dV = self.gradients(X, Y, W, V)
    #
    #         # last part of back propogation
    #         W = W - learning_rate * dW
    #         V = V - learning_rate * dV
    #         t += 1
    #     return W, V
