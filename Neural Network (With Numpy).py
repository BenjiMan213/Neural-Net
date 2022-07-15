#dependencies
from tensorflow.keras.datasets import boston_housing
import matplotlib.pyplot as plt
import numpy as np
import time

#data collection and preparation
(trainx, trainy), (testx, testy) = boston_housing.load_data()
class Normalizer():
    def __init__(self, data):
        self.min, self.max = np.min(data), np.max(data)
        self.range = self.max - self.min
    def refit(self, data):
        self.min, self.max = np.min(data), np.max(data)
        self.range = self.max - self.min
    def scale(self, x):
        return (x - self.min) / self.range
    def undo(self, x):
        return x * self.range + self.min

x_scalers = [Normalizer(trainx.T[i]) for i in range(len(trainx[0]))]
y_scaler = Normalizer(trainy)

y_train = y_scaler.scale(trainy)
y_test = y_scaler.scale(testy)
x_train = np.zeros_like(trainx)
x_test = np.zeros_like(testx)
for i in range(len(trainx)):
    for t in range(len(trainx[i])):
        x_train[i][t] = x_scalers[t].scale(trainx[i][t])
for i in range(len(testx)):
    for t in range(len(testx[i])):
        x_test[i][t] = x_scalers[t].scale(testx[i][t])

def relu(x): #activation function
    return [max(0,i) for i in x]
def d_relu(x): #gradient of the activation function
    return [0 if i < 0 else 1 for i in x]
def mse(y, y_hat): #cost function
    return (y - y_hat) ** 2
def d_mse(y, y_hat): #gradient of the cost function
    return -2 * (y - y_hat)

class DenseLayer(): #Linear Layer
    def __init__(self, n_neurons):
        self.activation = False
        self.output_size = n_neurons
        self.input_size = None
        self.biases = np.random.uniform(-1, 1, n_neurons)
        self.input = None
            
    def initialize_weights(self):
        self.weights = np.random.uniform(-1, 1, (self.output_size, self.input_size))
        self.m_dw, self.m_db = np.zeros_like(self.weights), np.zeros(self.output_size)
        self.v_dw, self.v_db = np.zeros_like(self.weights), np.zeros(self.output_size)
        
    def forward(self, x):
        self.input = np.array(x)
        return np.dot(x, self.weights.transpose()) + self.biases
    
    def backward(self, derivative, lr, beta=0.99, eps=1e-8): #RMSProp
        dw = np.array([derivative[i] * self.input for i in range(self.output_size)])
        
        self.v_dw = beta * self.v_dw + (1 - beta) * dw**2
        self.v_db = beta * self.v_db + (1 - beta) * derivative ** 2
        
        self.biases -= lr / np.sqrt(self.v_db + eps) * derivative
        self.weights -= lr / np.sqrt(self.v_dw + eps) * dw
        
        return np.dot(derivative, self.weights)

class ActivationLayer(): #Layer with non linear activation
    def __init__(self, activation, d_activation):
        self.activation = True
        self.output_size = None
        self.act, self.d_act = activation, d_activation
        self.input_size = None
        self.input = None
    
    def forward(self, x):
        self.input = np.array(x)
        return self.act(x)
    
    def backward(self, derivative, lr):
        return self.d_act(self.input) * derivative

class Model(): #Model class
    def __init__(self, input_size, lr=0.01, loss=mse, d_loss=d_mse):
        self.input_size = input_size
        self.lr = lr
        self.loss = loss
        self.d_loss = d_loss
        self.layers = []
        self.errors = []
        self.val_loss = []
        
    def add_layer(self, layer):
        if len(self.layers) == 0:
            inp_size = self.input_size
        else:
            inp_size = self.layers[-1].output_size
        layer.input_size = inp_size
        if layer.output_size == None:
            layer.output_size = self.layers[-1].output_size
        else:
            layer.initialize_weights()
        self.layers.append(layer)
        
    def predict(self, x):
        inp = x
        for i in self.layers:
            inp = i.forward(inp)
        return inp
    
    def copy_params(self, target_model):
        for i in range(len(target_model.layers)):
            if not target_model.layers[i].activation:
                self.layers[i].weights = target_model.layers[i].weights.copy()
                self.layers[i].biases = target_model.layers[i].biases.copy()
    def error(self, inps, outs):
        #total_error = np.zeros_like(outs[0])
        total_error = 0 if len(outs.shape) == 1 else np.zeros_like(outs[0])
        for i in range(len(inps)):
            y_hat = self.predict(inps[i])
            total_error += self.loss(outs[i], y_hat)
        return total_error / len(inps)
    
    def train(self, inputs, outputs, num, val_data=None):
        inps = inputs
        outs = outputs
        for i in range(num):
            for t in range(len(inps)):
                x = inps[t]
                y = outs[t]
                y_hat = self.predict(x)
                
                err = self.d_loss(y, y_hat)
                n_layers = len(self.layers)
                for e in range(n_layers):
                    err = self.layers[n_layers - e - 1].backward(err, self.lr)
            self.errors.append(self.error(inps, outs))
            if not val_data == None:
                self.val_loss.append(self.error(val_data[0], val_data[1]))
            print('update -- ', i)

#model instantiation
model = Model(13, lr=0.0005)
model.add_layer(DenseLayer(7))
model.add_layer(ActivationLayer(relu, d_relu))
model.add_layer(DenseLayer(7))
model.add_layer(ActivationLayer(relu, d_relu))
model.add_layer(DenseLayer(1))

#model training
start_time = time.time()
model.train(x_train, y_train, 200, val_data=(x_test, y_test))
print('time = ', time.time() - start_time, 's')

#plotting predictions and targets
y_hat = np.array([model.predict(x_test[i]) for i in range(len(x_test))])
true_y_hat = y_scaler.undo(y_hat)
plt.figure(figsize=(20,10))
plt.plot(true_y_hat)
#plt.plot(y_hat)
plt.plot(testy)

#plotting validation and training loss
plt.figure(figsize=(20,10))
plt.plot(np.log(model.errors))
print('current training loss: ', model.errors[-1])
plt.plot(np.log(model.val_loss))
print('current validation loss: ', model.val_loss[-1])
plt.axhline(0, color='black')

#plotting predictions over targets
plt.figure(figsize=(20,10))
plt.scatter(y_test, y_hat)
plt.plot([0.1 * i for i in range(0,11)],[0.1 * i for i in range(0,11)])
