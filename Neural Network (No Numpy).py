import random
import matplotlib.pyplot as plt
import time
from tensorflow.keras.datasets import boston_housing
import numpy as np #Numpy used exclusively for preparing the training dataset
#Dependencies

(trainx, trainy), (testx, testy) = boston_housing.load_data() #Data collection
class Normalizer(): #MinMax Scaler
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

#Scaling the data
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
    return max(0, x)
def d_relu(x): #gradient of the activation function
    return 0 if x < 0 else 1
def mse(y, y_hat): #loss function
    return (y - y_hat) ** 2
def d_mse(y, y_hat): #gradient of the loss function
    return -2 * (y - y_hat)

class DenseLayer(): #Layer with linear connections
    def __init__(self, n_neurons):
        self.activation = False
        self.output_size = n_neurons
        self.input_size = None
        self.biases = [random.uniform(-1,1) for i in range(self.output_size)]
        self.weights = None
        self.input = None
        
    def initialize_weights(self):
        self.weights = [[random.uniform(-1,1) for t in range(self.input_size)] for i in range(self.output_size)]
        self.m_dw = [[0 for t in range(self.input_size)] for i in range(self.output_size)]
        self.m_db = [0 for i in range(self.output_size)]
        self.v_dw = [[0 for t in range(self.input_size)] for i in range(self.output_size)]
        self.v_db = [0 for i in range(self.output_size)]
    
    def forward(self, x):
        self.input = x
        return [sum([x[t] * self.weights[i][t] for t in range(self.input_size)]) + self.biases[i] for i in range(self.output_size)]
    
    def backward(self, derivative, lr, beta=0.99, eps=1e-8): #Parameter optimization using RMSProp
        dw = [[derivative[i] * self.input[t] for t in range(self.input_size)]for i in range(self.output_size)]
        for i in range(self.output_size):
            self.v_db[i] = beta * self.v_db[i] + (1-beta) * derivative[i] ** 2
            self.biases[i] -= lr / (self.v_db[i] + eps) ** 0.5 * derivative[i]
            for t in range(self.input_size):
                self.v_dw[i][t] = beta * self.v_dw[i][t] + (1 - beta) * dw[i][t] ** 2
                self.weights[i][t] -= lr / (self.v_dw[i][t] + eps) ** 0.5 * dw[i][t]
        return [sum([self.weights[t][i] * derivative[t] for t in range(self.output_size)]) for i in range(self.input_size)]

class ActivationLayer(): #Layer for non linear activations
    def __init__(self, activation_function, d_activation_function):
        self.activation = True
        self.act = activation_function
        self.d_act = d_activation_function
        self.output_size = None
        self.input_size = None
        self.input = None
    
    def forward(self, x):
        self.input = x
        return [self.act(x[i]) for i in range(self.input_size)]
    
    def backward(self, derivative, lr): #No parameters to optimize
        return [self.d_act(self.input[i]) * derivative[i] for i in range(self.input_size)]        
    
class Model(): #Model class
    def __init__(self,input_size, lr=0.01, loss=mse, d_loss=d_mse):
        self.input_size = input_size
        self.loss, self.d_loss = loss, d_loss
        self.lr = lr
        self.layers = []
        self.errors = []
        self.val_loss = []
        
    def predict(self, x):
        inp = x.copy()
        for i in self.layers:
            inp = i.forward(inp)
        return inp
    
    def copy_params(self, target_model):
        for i in range(len(target_model.layers)):
            if not target_model.layers[i].activation:
                self.layers[i].weights = target_model.layers[i].weights.copy()
                self.layers[i].biases = target_model.layers[i].biases.copy()
    
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
            
    def error(self, inps, outs):
        
        if not type(outs[0]) == list:
            total_errors = 0
        else:
            total_errors = [0 for i in outs[0]]
        for i in range(len(inps)):
            y_hat = self.predict(inps[i])
            if self.layers[-1].output_size <= 1:
                total_errors += self.loss(outs[i], y_hat)
            else:
                for u in range(self.layers[-1].output_size):
                    total_errors[u] += self.loss(outs[i][u], y_hat[u])
        return total_errors / len(inps) if not type(total_errors) == list else [total_errors[e] / len(inps) for e in range(len(total_errors))]
        
    def train(self, inputs, outputs, num, val_data=None):
        inps = inputs
        outs = outputs
        for i in range(num):
            
            for t in range(len(inps)):
                x = inps[t]
                y = outs[t]
                y_hat = self.predict(x)
                
                error = self.d_loss(y, y_hat) if self.layers[-1].output_size <= 1 else [self.d_loss(y[q], y_hat[q]) for q in range(self.layers[-1].output_size)]
                for e in range(len(self.layers)):
                    error = self.layers[len(self.layers) - e - 1].backward(error, self.lr)
            self.errors.append(self.error(inps, outs))
            if not val_data == None:
                self.val_loss.append(self.error(val_data[0], val_data[1]))
            print('update -- ', i + 1)


#Model Instantiation
model = Model(13, lr=0.0005)
model.add_layer(DenseLayer(7))
model.add_layer(ActivationLayer(relu, d_relu))
model.add_layer(DenseLayer(7))
model.add_layer(ActivationLayer(relu, d_relu))
model.add_layer(DenseLayer(1))

start_time = time.time()
model.train(x_train, y_train, 200, val_data=(x_test, y_test))
print('time = ', time.time() - start_time, 's')

plt.figure(figsize=(20,10))
plt.title('Validation Predictions vs Targets')
y_hat = [model.predict(i) for i in x_test]
plt.plot(y_test)
plt.plot(y_hat)

plt.figure(figsize=(20,10))
plt.title('y over y_hat')
plt.scatter(y_test, y_hat)
plt.plot([0.1 * i for i in range(0,11)],[0.1 * i for i in range(0,11)])

plt.figure(figsize=(20,10))
plt.title('training vs validation loss')
plt.plot(model.errors)
print('current training loss: ', model.errors[-1])
plt.plot(model.val_loss)
print('current validation loss: ', model.val_loss[-1])
plt.axhline(0, color='black')
