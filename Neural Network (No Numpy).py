import random
import matplotlib.pyplot as plt
import time
#Dependencies

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
    def __init__(self,input_size, output_size, lr=0.01, loss=mse, d_loss=d_mse):
        self.input_size, self.output_size = input_size, output_size
        self.loss, self.d_loss = mse, d_mse
        self.lr = lr
        self.layers = []
        self.errors = []
        
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
            
    
    def train(self, inputs, outputs, num):
        inps = inputs
        outs = outputs
        for i in range(num):
            total_errors = [0 for i in outs[0]]
            for t in range(len(inps)):
                x = inps[t]
                y = outs[t]
                y_hat = self.predict(x)
                
                for u in range(len(y)):
                    total_errors[u] += self.loss(y[u], y_hat[u])
                
                error = [self.d_loss(y[q], y_hat[q]) for q in range(self.layers[-1].output_size)]
                for e in range(len(self.layers)):
                    error = self.layers[len(self.layers) - e - 1].backward(error, self.lr)
            self.errors.append([total_errors[i] / len(inps) for i in range(len(outs[0]))])
            print('update -- ', i + 1)

def func(x): #target function
    return [x[0]**2, x[0]**4 - x[0]**3 - x[0]**2 + x[0]]
def relu(x): #activation function
    return max(0, x)
def d_relu(x): #gradient of the activation function
    return 0 if x < 0 else 1
def mse(y, y_hat): #loss function
    return (y - y_hat) ** 2
def d_mse(y, y_hat): #gradient of the loss function
    return -2 * (y - y_hat)

x = [[i * 0.01] for i in range(-100, 100)]
y = [func(i) for i in x]

#Model Instantiation
model = Model(1, 2, lr=0.00015)
model.add_layer(DenseLayer(15))
model.add_layer(ActivationLayer(relu, d_relu))
model.add_layer(DenseLayer(15))
model.add_layer(ActivationLayer(relu, d_relu))
model.add_layer(DenseLayer(15))
model.add_layer(ActivationLayer(relu, d_relu))
model.add_layer(DenseLayer(15))
model.add_layer(ActivationLayer(relu, d_relu))
model.add_layer(DenseLayer(15))
model.add_layer(ActivationLayer(relu, d_relu))
model.add_layer(DenseLayer(2))

start_time = time.time()
model.train(x, y, 100)
print('time = ', time.time() - start_time, 's')

plt.figure(figsize=(20,10))
y_hat = [model.predict(i) for i in x]
plt.plot(x, y)
plt.plot(x, y_hat)

plt.figure(figsize=(20,10))
plt.plot(model.errors)
plt.axhline(0, color='black')
print('model error = ', model.errors[-1])

