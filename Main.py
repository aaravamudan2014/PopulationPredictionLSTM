from __future__ import division
from numba import cuda
import numpy
import math
import os
import signal
import pandas

import numpy as np
import matplotlib.pyplot as plt
from IPython import display
plt.style.use('seaborn-white')

os.environ['NUMBAPRO_NVVM']=r'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v9.1\nvvm\bin\nvvm64_32_0.dll'
os.environ['NUMBAPRO_LIBDEVICE']=r'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v9.1\nvvm\libdevice'

#reading in the data
X_size = 51 #number of states for net migration
number_of_years = 17 # number of migration years we are considering for
data_size = X_size * number_of_years
global sheet_no
sheet_no = 0


def fetchStateData():
    global sheet_no
    global data
    df = pandas.read_excel('migrationData.xlsx', sheet_name=sheet_no)
    #print(pandas.ExcelFile('migrationData.xlsx').sheet_names[sheet_no])
    data = df.as_matrix()
    data = numpy.transpose(data)
    sheet_no += 1

H_size = 51  #hidden layer size
T_steps = 10   #number of time steps: number of consecutive years for consideration
learning_rate = 5e-2  #learning rate for gradient descent , stochastic with adagrad
weight_sd = 0.5 # weight standard deviation for random initial vals
z_size = H_size + X_size
#defining all the activation functions needed
def sigmoid(x):
    return 1 / (1 + numpy.exp(-x))


def dsigmoid(y):
    return y * (1 - y)


def tanh(x):
    return numpy.tanh(x)


def dtanh(y):
    return 1 - y * y


class DelayedKeyboardInterrupt(object):
    def __enter__(self):
        self.signal_received = False
        self.old_handler = signal.signal(signal.SIGINT, self.handler)

    def handler(self, sig, frame):
        self.signal_received = (sig, frame)
        print('SIGINT received. Delaying KeyboardInterrupt.')

    def __exit__(self, type, value, traceback):
        signal.signal(signal.SIGINT, self.old_handler)
        if self.signal_received:
            self.old_handler(*self.signal_received)
class Param:
    def __init__(self, name, value):
        self.name = name
        self.v = value #parameter value
        self.d = numpy.zeros_like(value) #derivative
        self.m = numpy.zeros_like(value) #momentum for AdaGrad needed for stochastic gradient descent
class Parameters:
    def __init__(self):
        self.W_f = Param('W_f',
                         numpy.random.randn(H_size, z_size) * weight_sd + 0.5)
        self.b_f = Param('b_f',
                         numpy.zeros((H_size, 1)))

        self.W_i = Param('W_i',
                         numpy.random.randn(H_size, z_size) * weight_sd + 0.5)
        self.b_i = Param('b_i',
                         numpy.zeros((H_size, 1)))

        self.W_C = Param('W_C',
                         numpy.random.randn(H_size, z_size) * weight_sd)
        self.b_C = Param('b_C',
                         numpy.zeros((H_size, 1)))

        self.W_o = Param('W_o',
                         numpy.random.randn(H_size, z_size) * weight_sd + 0.5)
        self.b_o = Param('b_o',
                         numpy.zeros((H_size, 1)))

        # For final layer to predict the next year migration
        self.W_v = Param('W_v',
                         numpy.random.randn(X_size, H_size) * weight_sd)
        self.b_v = Param('b_v',
                         numpy.zeros((X_size, 1)))

    def all(self):
        return [self.W_f, self.W_i, self.W_C, self.W_o, self.W_v,
                self.b_f, self.b_i, self.b_C, self.b_o, self.b_v]

# this represents all the parameters of the LSTM model
parameters = Parameters()

def forward(x, h_prev, C_prev, p=parameters):
    x = numpy.reshape(x,(X_size,1))
    assert x.shape == (X_size,1 )
    assert h_prev.shape == (H_size, 1)
    assert C_prev.shape == (H_size, 1)

    z = numpy.row_stack((h_prev, x))
    f = sigmoid(numpy.dot(p.W_f.v, z) + p.b_f.v)
    i = sigmoid(numpy.dot(p.W_i.v, z) + p.b_i.v)
    C_bar = tanh(numpy.dot(p.W_C.v, z) + p.b_C.v)

    C = f * C_prev + i * C_bar
    o = sigmoid(numpy.dot(p.W_o.v, z) + p.b_o.v)
    h = o * tanh(C)

    v = numpy.dot(p.W_v.v, h) + p.b_v.v
    #print(v)
    m = max(v)
    #print(m)
    #wait = input("PRESS ENTER TO CONTINUE.")
    y = numpy.exp(v-m) / numpy.sum(numpy.exp(v-m))  # softmax


    return z, f, i, C_bar, C, o, h, v, y


def backward(target, dh_next, dC_next, C_prev,
             z, f, i, C_bar, C, o, h, v, y,
             p=parameters):
    assert z.shape == (X_size + H_size, 1)
    assert v.shape == (X_size, 1)
    assert y.shape == (X_size, 1)

    for param in [dh_next, dC_next, C_prev, f, i, C_bar, C, o, h]:
        assert param.shape == (H_size, 1)

    dv = numpy.copy(y).ravel()
    dv = dv-target
    dv = dv.reshape(X_size, 1)
    p.W_v.d += numpy.dot( dv,h.T)
    p.b_v.d += dv

    dh = numpy.dot(p.W_v.v.T, dv)
    dh += dh_next
    do = dh * tanh(C)
    do = dsigmoid(o) * do
    p.W_o.d += numpy.dot(do, z.T)
    p.b_o.d += do

    dC = numpy.copy(dC_next)
    dC += dh * o * dtanh(tanh(C))
    dC_bar = dC * i
    dC_bar = dtanh(C_bar) * dC_bar
    p.W_C.d += numpy.dot(dC_bar, z.T)
    p.b_C.d += dC_bar

    di = dC * C_bar
    di = dsigmoid(i) * di
    p.W_i.d += numpy.dot(di, z.T)
    p.b_i.d += di

    df = dC * C_prev
    df = dsigmoid(f) * df
    p.W_f.d += numpy.dot(df, z.T)
    p.b_f.d += df

    dz = (numpy.dot(p.W_f.v.T, df)
          + numpy.dot(p.W_i.v.T, di)
          + numpy.dot(p.W_C.v.T, dC_bar)
          + numpy.dot(p.W_o.v.T, do))
    dh_prev = dz[:H_size, :]
    dC_prev = f * dC

    return dh_prev, dC_prev
def cross_entropy(predictions, targets, epsilon=1e-12):
    """
    Computes cross entropy between targets (encoded as one-hot vectors)
    and predictions.
    Input: predictions (N, k) ndarray
           targets (N, k) ndarray
    Returns: scalar
    """
    predictions = np.clip(predictions, epsilon, 1. - epsilon)
    N = predictions.shape[0]
    ce = -np.sum(np.sum(targets*np.log(predictions+1e-9)))/N
    return ce
# to clear gradients during backprop
def clear_gradients(params = parameters):
    for p in params.all():
        p.d.fill(0)

#to cull exploding gradients
def clip_gradients(params = parameters):
    for p in params.all():
        numpy.clip(p.d, -1, 1, out=p.d)


def forward_backward(inputs, targets, h_prev, C_prev):
    global paramters

    # To store the values for each time step
    x_s, z_s, f_s, i_s, = {}, {}, {}, {}
    C_bar_s, C_s, o_s, h_s = {}, {}, {}, {}
    v_s, y_s = {}, {}

    # Values at t - 1
    h_s[-1] = numpy.copy(h_prev)
    C_s[-1] = numpy.copy(C_prev)

    loss = 0
    # Loop through time steps
    assert len(inputs) == T_steps
    for t in range(len(inputs)):
        x_s[t] = numpy.zeros((X_size, 1))
        x_s[t] = inputs[t]  # Input character

        (z_s[t], f_s[t], i_s[t],
         C_bar_s[t], C_s[t], o_s[t], h_s[t],
         v_s[t], y_s[t]) = \
            forward(x_s[t], h_s[t - 1], C_s[t - 1])  # Forward pass


        loss += -cross_entropy(y_s[t],targets[t])  # Loss for at t
        #loss += np.square(y_s[t]-targets[t]).mean()
        #loss += (1.0/number_of_years) * (numpy.dot(numpy.log(y_s), targets.T) + numpy.dot(numpy.log(1-y_s), (1-targets).T))
    clear_gradients()

    loss = abs(loss)
    #print("loss is\n")
    #print(loss)
    dh_next = numpy.zeros_like(h_s[0])  # dh from the next character
    dC_next = numpy.zeros_like(C_s[0])  # dh from the next character
    for t in reversed(range(len(inputs))):
        # Backward pass
        dh_next, dC_next = \
            backward(target=targets[t], dh_next=dh_next,
                     dC_next=dC_next, C_prev=C_s[t - 1],
                     z=z_s[t], f=f_s[t], i=i_s[t], C_bar=C_bar_s[t],
                     C=C_s[t], o=o_s[t], h=h_s[t], v=v_s[t],
                     y=y_s[t])

    clip_gradients()

    return loss, h_s[len(inputs) - 1], C_s[len(inputs) - 1]


def sample(h_prev, C_prev, first_char_idx, sentence_length):
    x = numpy.zeros((X_size, 1))
    x = first_char_idx

    h = h_prev
    C = C_prev

    indexes = []

    _, _, _, _, C, _, h, _, p = forward(x, h, C)
    return p


def update_status(inputs, h_prev, C_prev):
    # initialized later
    global plot_iter, plot_loss
    global smooth_loss

    # Get predictions for 200 letters with current model

    sample_idx = sample(h_prev, C_prev, inputs[0], number_of_years)
    txt = ' '.join(str(idx) for idx in sample_idx)

    # Clear and plot

    plt.plot(plot_iter, plot_loss)
    display.clear_output(wait=True)
    plt.show()

    # Print prediction and loss
    print("----\n %s \n----" % (txt,))
    print("iter %d, loss %f" % (iteration, numpy.mean(smooth_loss)))
def update_paramters(params = parameters):
    #stochastic gradient descent
    #for p in params.all():
    #    p.v -= learning_rate*p.d
    for p in params.all():
        p.m += p.d * p.d # Calculate sum of gradients
        #print(learning_rate * dparam)
        p.v += -(learning_rate * p.d / numpy.sqrt(p.m + 1e-8))

# CUDA kernel
@cuda.jit
def my_kernel(io_array):
    pos = cuda.grid(1)
    if pos < io_array.size:
        for i in range(0,25000000):
            io_array[pos] += 1 # do the computation


# Host code
#data = numpy.ones(256)
#threadsperblock = 256
#blockspergrid = math.ceil(data.shape[0] / threadsperblock)
#my_kernel[blockspergrid, threadsperblock](data)

#print(data)


smooth_loss = -numpy.log(1.0 / X_size) * T_steps

iteration, pointer = 0, 0

# For the graph
plot_iter = numpy.zeros((0))
plot_loss = numpy.zeros((0))

fetchStateData();
epoch = 0
while True:
    try:
        with DelayedKeyboardInterrupt():
            # Reset
            if pointer + T_steps >= len(data) or iteration == 0:
                g_h_prev = numpy.zeros((H_size, 1))
                g_C_prev = numpy.zeros((H_size, 1))
                print("Starting with new set of samples with sheet" + str(sheet_no))
                if iteration != 0:
                    fetchStateData()
                if sheet_no > 50:
                    sheet_no = 0
                    epoch += 1
                    print('starting new epoch ', epoch)
                    if epoch ==200:
                        break

                pointer = 0


            inputs = ([ch for ch in data[pointer: pointer + T_steps]])
            targets = ([ch for ch in data[pointer + 1: pointer + T_steps + 1]])


            inputs = (inputs - numpy.amin(inputs))/(numpy.amax(inputs) -numpy.amin(inputs) )
            targets = (targets - numpy.amin(targets)) / (numpy.amax(targets) - numpy.amin(targets))

            #inputs = inputs / numpy.amax(inputs)
            #targets = targets / numpy.amax(targets)




            loss, g_h_prev, g_C_prev = \
                forward_backward(inputs, targets, g_h_prev, g_C_prev)
            smooth_loss = smooth_loss * 0.999 + loss * 0.001

            # Print every 5 steps
            if iteration % 5 == 0:
                update_status(inputs, g_h_prev, g_C_prev)

            update_paramters()

            plot_iter = numpy.append(plot_iter, [iteration])
            plot_loss = numpy.append(plot_loss, [loss])

            pointer += T_steps
            iteration += 1
    except KeyboardInterrupt:
        update_status(inputs, g_h_prev, g_C_prev)
        break

