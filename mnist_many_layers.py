from keras.datasets import mnist
from keras import models
from keras import layers
from keras.utils import to_categorical
import numpy as np
import matplotlib.pyplot as plt
import LBL_utils as utils
# Method for growing network
def grow_network(old_network,new_layer_size=32,output_size=10):
    new_network=models.Sequential()
    nbr_of_layers = len(old_network.layers)
    for i in range(nbr_of_layers-1):
        old_network.layers[i].trainable=False
        new_network.add(old_network.layers[i])
    new_network.add(layers.Dense(new_layer_size,activation='relu'))
    new_network.add(layers.Dense(output_size,activation='softmax'))
    #new_network.summary()
    return new_network

train_images,train_labels,test_images,test_labels=utils.get_mnist_data_flattened()

print('Successfully converted labels. Initializing the training')

network_a = models.Sequential()
network_a.add(layers.Dense(512,activation='relu',input_shape=(28 * 28,)))
network_a.add(layers.Dense(10,activation='softmax'))
network_a.compile(optimizer='rmsprop',loss='categorical_crossentropy',metrics=['accuracy'])

nbr_of_layers=50 #nbr of internal layers

network_a.summary()
network_a.fit(train_images, train_labels, epochs=5, batch_size=128)

for i in range(nbr_of_layers):
    if(i%2==0):
        network_b=grow_network(network_a,32,10)
        network_b.compile(optimizer='rmsprop',loss='categorical_crossentropy',metrics=['accuracy'])
        network_b.summary()
        network_b.fit(train_images, train_labels, epochs=5, batch_size=128)
        del(network_a)
    else:
        network_a=grow_network(network_b)
        network_a.compile(optimizer='rmsprop',loss='categorical_crossentropy',metrics=['accuracy'])
        network_a.summary()
        network_a.fit(train_images, train_labels, epochs=5, batch_size=128)
        del(network_b)
        print('b')
if(i%2==1):
    network = network_a
else:
    network = network_b

print('Creating and fitting replica of second network and using traditional training methods')
network_traditional=models.Sequential()
network_traditional.add(layers.Dense(512,activation='relu',input_shape=(28 * 28,)))
for n in range(nbr_of_layers):
    network_traditional.add(layers.Dense(32,activation='relu'))#not supposed to be here
network_traditional.add(layers.Dense(10,activation='softmax'))
network_traditional.summary()
network_traditional.compile(optimizer='rmsprop',loss='categorical_crossentropy',metrics=['accuracy'])
network_traditional.fit(train_images, train_labels, epochs=5, batch_size=128)

# Open a file
fo = open("runs_many_layers.txt", "a")
fo.write('nbr_of_layers: ')
fo.write(str(nbr_of_layers))
# Evaluate network1 perfomance
test_loss, test_acc = network.evaluate(test_images,test_labels)
print('Grown model test_acc = ',test_acc)
#fo.write('short network test_acc = ')

fo.write(str(test_acc))
fo.write('\n')
# Evaluate network2 perfomance

test_loss, test_acc = network_traditional.evaluate(test_images,test_labels)
print('traditional network test_acc = ',test_acc)
#fo.write('traditional network test_acc = ')
fo.write(str(test_acc))
fo.write('\n')

# Close opend file
fo.close()
