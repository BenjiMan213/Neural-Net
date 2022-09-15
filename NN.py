import numpy as np
from numba import jit, cuda
from scipy import signal

def relu(x): #activation function
    return (x > 0) * x
def d_relu(x): #gradient of the activation function
    return (x > 0) * 1
def mse(y, y_hat): #cost function
    return (y - y_hat) ** 2
def d_mse(y, y_hat): #gradient of the cost function
    return -2 * (y - y_hat)
def binary_cross_entropy(y, y_hat):
    return -np.mean(y * np.log(y_hat) + (1 - y) * np.log(1 - y_hat))
def d_binary_cross_entropy(y, y_hat):
    return ((1 - y) / (1 - y_hat) - y / y_hat) / np.size(y)
def sigmoid(x):
    return 1 / (1 + np.exp(-x))
def d_sigmoid(x):
    s = sigmoid(x)
    return s * (1 - s)
def softmax(x):
    exps = np.exp(x - np.max(x))
    return x / np.sum(exps)
def d_softmax(x):
    j = np.diag(x)
    for i in range(len(j)):
        for t in range(len(j)):
            j[i][t] = x[i] * (1 - x[i]) if i == t else -x[i] * x[t]
    return j @ x

class Dense(): #Linear Layer
    def __init__(self, n_neurons):
        self.type = 'dense'
        self.is_activation = False
        self.output_shape = n_neurons
        self.input_shape = None
        self.biases = np.random.uniform(-1, 1, n_neurons)
        self.input = None
            
    def initialize_weights(self):
        self.weights = np.random.uniform(-1, 1, (self.output_shape, self.input_shape))
        self.m_dw, self.m_db = np.zeros_like(self.weights), np.zeros(self.output_shape)
        self.v_dw, self.v_db = np.zeros_like(self.weights), np.zeros(self.output_shape)

    def forward(self, x):
        self.input = np.array(x)
        return np.dot(x, self.weights.transpose()) + self.biases

    def backward(self, derivative, lr, beta=0.99, eps=1e-8, descent=1.0): #RMSProp
        dw = np.array([derivative[i] * self.input for i in range(self.output_shape)])
        
        self.v_dw = beta * self.v_dw + (1 - beta) * dw**2
        self.v_db = beta * self.v_db + (1 - beta) * derivative ** 2
        
        self.biases -= lr / np.sqrt(self.v_db + eps) * derivative * descent
        self.weights -= lr / np.sqrt(self.v_dw + eps) * dw * descent
        
        return np.dot(derivative, self.weights)

class Activation(): #Layer with non linear activation
    def __init__(self, activation, d_activation):
        self.type = 'activation'
        self.is_flatten = False
        self.output_size = None
        self.act, self.d_act = activation, d_activation
        self.input_shape = None
        self.input = None

    def forward(self, x):
        self.input = np.array(x)
        return self.act(x)

    def backward(self, derivative, lr, descent=1.0):
        return self.d_act(self.input) * derivative

class Model(): #Model class
    def __init__(self, input_shape, lr=0.01, loss=mse, d_loss=d_mse):
        self.input_shape = input_shape
        self.lr = lr
        self.loss = loss
        self.d_loss = d_loss
        self.layers = []
        self.errors = []
        self.val_loss = []
        self.acc = []

        
    def add(self, layer):
        if len(self.layers) == 0:
            inp_shape = self.input_shape
        else:
            inp_shape = self.layers[-1].output_shape
        layer.input_shape = inp_shape
        if layer.type == 'dense':
            layer.initialize_weights()
        elif layer.type == 'activation' or layer.type == 'dropout':
            layer.output_shape = inp_shape
        elif layer.type == 'flatten':
            layer.output_shape = np.product(layer.input_shape)
        elif layer.type == 'pooling':
            layer.set_shape()
        elif layer.type == 'convolutional':
            layer.set_shape()
        self.layers.append(layer)
        
    def predict(self, x):
        inp = x
        for i in self.layers:
            inp = i.forward(inp)
        return inp
    
    def copy_params(self, target_model, tau=0.001):
        for i in range(len(target_model.layers)):
            if target_model.layers[i].type == 'convolutional':
                self.layers[i].kernels = target_model.layers[i].kernels * tau + (1 - tau) * self.layers[i].kernels
                self.layers.biases = target_model.layers[i].biases * tau + (1 - tau) * self.layers[i].biases
            elif target_model.layers[i].type == 'dense':
                self.layers[i].weights = target_model.layers[i].weights * tau + (1 - tau) * self.layers[i].weights
                self.layers[i].biases = target_model.layers[i].biases * tau + (1 - tau) * self.layers[i].biases

                
    def error(self, inps, outs, accuracy=False):
        total_error = 0
        correct = 0
        for i in range(len(inps)):
            y_hat = self.predict(inps[i])
            if accuracy:
                correct += 1 if np.argmax(y_hat) == np.argmax(outs[i]) else 0
            total_error += np.mean(self.loss(outs[i], y_hat))
        if accuracy:
            self.acc.append(100 * correct / len(inps))
        return total_error / len(inps)

    def rl_train(self, inps, advantages, indices, num):
        n_layers = len(self.layers)
        one_hots = np.eye(np.max(indices)+1)
        for i in range(num):
            total_error = 0
            for t in range(len(inps)):
                x = inps[t]
                y = one_hots[indices[t]]
                probs = self.predict(x)

                err = advantages[t] * self.d_loss(y, probs) #/ probs
                total_error += np.mean(self.loss(y, probs))
                #total_error += np.mean(-np.log(probs) * advantages[t])
                for e in range(n_layers):
                    err = self.layers[n_layers-e-1].backward(err, self.lr, descent=1.0)
            self.errors.append(total_error / len(inps))
            print('training loss:', self.errors[-1], ' --- ', 'update:',i)
                
    
    def train(self, inps, outs, num, val_data=None, accuracy=False):
        n_layers = len(self.layers)
        for i in range(num):
            total_error = 0
            for t in range(len(inps)):
                x = inps[t]                   
                y = outs[t]
                y_hat = self.predict(x)
                    
                err = self.d_loss(y, y_hat)
                total_error += self.loss(y, y_hat)
                for e in range(n_layers):
                    err = self.layers[n_layers - e - 1].backward(err, self.lr)

            self.errors.append(total_error/len(inps))        
            if not val_data == None:
                self.val_loss.append(self.error(val_data[0], val_data[1], accuracy=accuracy))
                if accuracy:
                    print('accuracy:', self.acc[-1], '% --- ', end='')
                print('validation loss:', self.val_loss[-1], ' --- ', end='')
            print('training loss:', self.errors[-1], ' --- ', 'update:',i)
        
class Convolutional():
    def __init__(self, n_kernels, kernel_size):
        self.type = 'convolutional'
        self.n_kernels = n_kernels
        self.kernel_size = kernel_size
        self.input_shape = None
        self.output_shape = None

    def set_shape(self):
        self.kernels = np.random.uniform(-1, 1, (self.n_kernels, self.input_shape[0], self.kernel_size, self.kernel_size))
        self.biases = np.random.uniform(-1, 1, (self.n_kernels, self.input_shape[1] - self.kernel_size + 1,
                                                self.input_shape[2] - self.kernel_size + 1))
        self.output_shape = self.biases.shape
        self.v_dk = np.zeros_like(self.kernels)
        self.v_db = np.zeros_like(self.biases)

    def forward(self, x):
        self.input = x
        out = np.copy(self.biases)
        for i in range(self.n_kernels):
            for t in range(self.input_shape[0]):
                out[i] += signal.correlate2d(self.input[t], self.kernels[i][t], 'valid')
        return out

    def backward(self, grad, lr, beta=0.98, eps=1e-8, descent=1.0): #RMSProp
        input_grad = np.zeros(self.input_shape)
        kernels_grad = np.zeros(self.kernels.shape)
        for i in range(self.n_kernels):
            for t in range(self.input_shape[0]):
                input_grad[t] = signal.convolve2d(grad[t], self.kernels[i][t], 'full')
                kernels_grad += signal.correlate2d(self.input[t], grad[i], 'valid')
                
        self.v_dk = beta * self.v_dk + (1 - beta) * kernels_grad ** 2
        self.v_db = beta * self.v_db + (1 - beta) * grad ** 2
        
        self.kernels -= lr * kernels_grad / np.sqrt(self.v_dk + eps) * descent
        self.biases -= lr * grad / np.sqrt(self.v_db + eps) * descent

        return input_grad

class Flatten():
    def __init__(self):
        self.type = 'flatten'
        self.is_activation = True
        self.is_flatten = True
        self.input_shape = None
        self.output_shape = None

    def forward(self, x):
        return x.reshape(self.output_shape)

    def backward(self, grad, lr, descent=1.0):
        return grad.reshape(self.input_shape)

class Dropout():
    def __init__(self, proportion):
        self.type = 'dropout'
        self.input_shape, self.output_shape = None, None
        self.p = proportion
        self.dropped = []

    def forward(self, x):
        for i in range(len(x)):
            if np.random.rand() < self.p:
                x[i] *= 0
                self.dropped.append(i)
        return x
    
    def backward(self, grad, lr, descent=1.0):
        for i in self.dropped:
            grad[i] *= 0
        return grad

class Pooling():
    def __init__(self, n_filters, stride):
        self.type = 'pooling'
        self.input_shape, self.output_shape = None, None
        self.f, self.s = n_filters, stride

    def forward(self, x):
        out = np.zeros(self.output_shape)
        for i in range(self.output_shape[0]):
            for t in range(0, self.input_shape[1] - self.f + 1, self.s):
                h = t // self.s
                for e in range(0, self.input_shape[2] - self.f + 1, self.s):
                    w = e // self.s
                    arg = x[i, t:t+self.f, e:e+self.f].argmax()
                    wh, ww = arg // self.f, arg % self.f
                    self.out[i][h][w] = x[i, t+wh, e+ww]
                    self.indices[i][h][w] = [t + wh, w + ww]
        return self.out

    def backward(self, grad, lr, descent=1.0):
        res = np.zeros(self.input_shape)
        for i in range(self.output_shape[0]):
            for t in range(self.output_shape[1]):
                for e in range(self.output_shape[2]):
                    res[i][self.indices[i][t][e][0]][self.indices[i][t][e][1]] = grad[i][t][e]
        return res

    def set_shape(self):
        self.output_shape = (self.input_shape[0], (self.input_shape[1] - self.f)//self.s + 1,
                             (self.input_shape[2] - self.f)//self.s + 1)
        self.indices = np.zeros(self.output_shape + (2,), dtype=int)
        self.out = np.zeros(self.output_shape)




    
