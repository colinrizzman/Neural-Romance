# github.com/colinrizzman
# pip install numpy tensorflow
import sys
import os
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import Callback
from time import time_ns
from os.path import isfile
from os.path import isdir
from os import mkdir
from pathlib import Path
from datetime import datetime

# disable warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['CUDA_VISIBLE_DEVICES'] = '-1' # disable gpu, assuming nvidia

# print everything / no truncations
np.set_printoptions(threshold=sys.maxsize)

# hyperparameters
project = "neural_romance"
optimiser = 'adam'
activator = 'relu'
inputsize = 27
outputsize = 1
layers = 6
layer_units = 32 # 32, 96
batches = 512 # 24, 512
epoches = 33333 # 12500, 33333
topo = 1
earlystop = 0 # 0 = off, anything above is the patience value

# load options
argc = len(sys.argv)
if argc >= 2:
    layers = int(sys.argv[1])
    print("layers:", layers)
if argc >= 3:
    layer_units = int(sys.argv[2])
    print("layer_units:", layer_units)
if argc >= 4:
    batches = int(sys.argv[3])
    print("batches:", batches)
if argc >= 5:
    epoches = int(sys.argv[4])
    print("epoches:", epoches)
if argc >= 6:
    activator = sys.argv[5]
    print("activator:", activator)
if argc >= 7:
    optimiser = sys.argv[6]
    print("optimiser:", optimiser)
if argc >= 8:
    topo = sys.argv[7]
    print("topo:", topo)

# make sure save dir exists
if not isdir('models'): mkdir('models')
model_name = 'models/' + activator + '_' + optimiser + '_' + str(layers) + '_' + str(layer_units) + '_' + str(batches) + '_' + str(epoches) + '_' + str(topo)

##########################################
#   LOAD
##########################################
print("\n--Loading Dataset")
st = time_ns()

dataset_size = sum(1 for _ in Path('training_data.txt').open())
print("Dataset Size:", "{:,}".format(dataset_size))

if isfile("train_x.npy"):
    train_x = np.load("train_x.npy")
    train_y = np.load("train_y.npy")
else:
    data = np.loadtxt('training_data.txt')
    train_x = data[:, :27] # first 27 columns
    train_y = data[:, 27]  # 28th column
    np.save("train_x.npy", train_x)
    np.save("train_y.npy", train_y)

# print(train_x.shape)
# print(train_x)
# print(train_y.shape)
# print(train_y)
# sys.exit()

timetaken = (time_ns()-st)/1e+9
print("Time Taken:", "{:.2f}".format(timetaken), "seconds")

##########################################
#   TRAIN
##########################################
print("\n--Training Model")

# construct neural network
model = Sequential()
model.add(Input(shape=(inputsize,)))
if topo == 0:
    for x in range(layers): model.add(Dense(layer_units, activation='relu'))
elif topo == 1:
    for x in range(layers-1): model.add(Dense(layer_units, activation='relu'))
    model.add(Dense(int(layer_units/2), activation='relu'))
elif topo == 2:
    dunits = layer_units
    for x in range(layers):
        model.add(Dense(int(dunits), activation='relu'))
        dunits=dunits/2
#model.add(Dense(outputsize, activation='sigmoid'))
model.add(Dense(outputsize))

# output summary
model.summary()

if optimiser == 'adam':
    optim = keras.optimizers.Adam(learning_rate=0.001)
elif optimiser == 'sgd':
    optim = keras.optimizers.SGD(learning_rate=0.01, momentum=0.0, nesterov=False)
elif optimiser == 'sgd_decay':
    #decay = keras.optimizers.schedules.ExponentialDecay(initial_learning_rate=0.3, decay_steps=epoches*dataset_size, decay_rate=0.1)
    decay = keras.optimizers.schedules.ExponentialDecay(initial_learning_rate=0.1, decay_steps=epoches*dataset_size, decay_rate=0.01)
    optim = keras.optimizers.SGD(learning_rate=decay, momentum=0.0, nesterov=False)
elif optimiser == 'momentum':
    optim = keras.optimizers.SGD(learning_rate=0.01, momentum=0.9, nesterov=False)
elif optimiser == 'nesterov':
    optim = keras.optimizers.SGD(learning_rate=0.01, momentum=0.9, nesterov=True)
elif optimiser == 'nadam':
    optim = keras.optimizers.Nadam(learning_rate=0.001)
elif optimiser == 'adagrad':
    optim = keras.optimizers.Adagrad(learning_rate=0.001)
elif optimiser == 'rmsprop':
    optim = keras.optimizers.RMSprop(learning_rate=0.001)
elif optimiser == 'adadelta':
    optim = keras.optimizers.Adadelta(learning_rate=0.001)
elif optimiser == 'adamax':
    optim = keras.optimizers.Adamax(learning_rate=0.001)
elif optimiser == 'ftrl':
    optim = keras.optimizers.Ftrl(learning_rate=0.001)

class PrintFullLoss(Callback):
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        numeric = {k: v for k, v in logs.items() if isinstance(v, (int, float))}
        parts = [f"{k}: {v:.10f}" for k, v in numeric.items()]
        print(f" - " + " - ".join(parts))
print_loss = PrintFullLoss()

early_stop = EarlyStopping(
    monitor='loss',
    patience=earlystop,
    min_delta=1e-7,
    verbose=1,
    mode='min'
)

model.compile(optimizer=optim, loss='mean_squared_error')
if earlystop == 0:  history = model.fit(train_x, train_y, epochs=epoches, batch_size=batches, callbacks=[print_loss])
else:               history = model.fit(train_x, train_y, epochs=epoches, batch_size=batches, callbacks=[early_stop, print_loss])
model_name = model_name + "_" + "L[{:.6f}]".format(history.history['loss'][-1])
timetaken = (time_ns()-st)/1e+9
print("\nTime Taken:", "{:.2f}".format(timetaken), "seconds")

##########################################
#   EXPORT
##########################################
print("\n--Exporting Model")
st = time_ns()

# save weights for C array
print("\nExporting weights...")
li = 0
f = open(model_name + "_layers.h", "w")
#f.write("#ifndef " + project + "_layers\n#define " + project + "_layers\n\n")
f.write("#ifndef " + project + "_layers\n#define " + project + "_layers\n\n// loss: " + "{:.8f}".format(history.history['loss'][-1]) + "\n\n")
if f:
    for layer in model.layers:
        total_layer_weights = layer.get_weights()[0].transpose().flatten().shape[0]
        total_layer_units = layer.units
        layer_weights_per_unit = total_layer_weights / total_layer_units
        print("+ Layer:", li)
        print("Total layer weights:", total_layer_weights)
        print("Total layer units:", total_layer_units)
        print("Weights per unit:", int(layer_weights_per_unit))

        f.write("const float " + project + "_layer" + str(li) + "[] = {")
        isfirst = 0
        wc = 0
        bc = 0
        if layer.get_weights() != []:
            for weight in layer.get_weights()[0].transpose().flatten():
                wc += 1
                if isfirst == 0:
                    f.write(str(weight))
                    isfirst = 1
                else:
                    f.write("," + str(weight))
                if wc == layer_weights_per_unit:
                    f.write(", /* bias */ " + str(layer.get_weights()[1].transpose().flatten()[bc]))
                    wc = 0
                    bc += 1
        f.write("};\n\n")
        li += 1
f.write("#endif\n")
f.close()

# save keras model
model.save(model_name + '.keras')

timetaken = (time_ns()-st)/1e+9
print("\nTime Taken:", "{:.2f}".format(timetaken), "seconds\n")
print(datetime.now().strftime("%Y-%m-%d %H:%M:%S") + ": " + model_name + "\n")
