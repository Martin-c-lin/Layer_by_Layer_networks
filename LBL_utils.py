from keras import layers,models
from keras import optimizers
from keras.preprocessing.image import ImageDataGenerator

import os,sys
# Switch compiler?
def get_normal_network_CD(conv_layers,dense_layers,input_shape=(150,150,3)):
    #Returns a compiled small network with conv layers and dense_layers
    from keras.models import Sequential
    model = Sequential()

    if(len(conv_layers)>0):
        ### Add convolutional layers and pooling

        ### Add input layer
        model.add(layers.Conv2D(conv_layers[0],(3,3),activation='relu',input_shape=input_shape))
        model.add(layers.MaxPooling2D(2,2))
        ### Add hidden convólutional layers
        for conv_size in conv_layers[1:]:
            model.add(layers.Conv2D(conv_size,(3,3),activation='relu'))
            model.add(layers.MaxPooling2D(2,2))
        ### Prepare for dense by adding a flatten
        model.add(layers.Flatten())

    else:
        print("No convolutional layers in model")
        print("Error, no layers in model")
        return 0
    for dense_size in dense_layers:
        model.add(layers.Dense(dense_size,activation='relu'))
    model.add(layers.Dense(1,activation='sigmoid'))
    return model
def train_comparison_networks_CD(conv_layers,dense_layers,train_generator,validation_generator,nbr_epochs=80,network_name="CD_normal",path="",input_shape=(150,150,3)):
    """
    Function for generting comparison networks and training them
    """
    for i in range(len(conv_layers)):
        nbr_layers_added = i+1
        model = get_normal_network_CD(conv_layers[0:nbr_layers_added],dense_layers=[],input_shape=input_shape)
        model.compile(loss='binary_crossentropy',
              optimizer=optimizers.RMSprop(lr=1e-4),
              metrics=['acc'])
        model.summary()
        history = model.fit_generator(
                  train_generator,
                  steps_per_epoch=100,
                  epochs=nbr_epochs,
                  validation_data=validation_generator,
                  validation_steps=50)
        model_name = network_name+"_conv_"+str(nbr_epochs)+"e_"+str(nbr_layers_added)+"L"
        file_name = model_name+"results.txt"#change so that this
        print_results_to_file(model,history,file_name,model_name,path=path)
        model_name = path+model_name+".h5"
        model.save(model_name)
    ### Then add the dense layers
    for i in range(len(dense_layers)):
        nbr_layers_added = i+1
        model = get_normal_network_CD(conv_layers,dense_layers[0:nbr_layers_added],input_shape=input_shape)
        model.compile(loss='binary_crossentropy',
              optimizer=optimizers.RMSprop(lr=1e-4),
              metrics=['acc'])
        model.summary()
        history = model.fit_generator(
                  train_generator,
                  steps_per_epoch=100,
                  epochs=nbr_epochs,
                  validation_data=validation_generator,
                  validation_steps=50)
        model_name = network_name+"_dense_"+str(nbr_epochs)+"e_"+str(nbr_layers_added)+"L"
        file_name = model_name+"results.txt"#change so that this
        print_results_to_file(model,history,file_name,model_name,path=path)
        model_name = path+model_name+".h5"
        model.save(model_name)

    return "Success"
def create_growing_conv_network(input_shape = (28,28,1),conv_base_dim=32,number_of_outputs=1):
    """Creates and compiles a deep learning network with only input and output layers.
    Starts from a convolutional layer(input), a max-pooling layer,flatten and dense(output)
    Model suitable for adding intermediate layers between theese.

    Inputs:
    input_shape: size of the images to be analyzed [3-ple of positive integers, x-pixels by y-pixels by color channels]
    conv_layers_dimensions: number of convolutions in each convolutional layer [tuple of positive integers]
    Output:
    network: deep learning network conv base
    """
    network = models.Sequential()
    conv_layer_name = 'conv_1'
    conv_layer = layers.Conv2D(
            conv_base_dim,
            (3, 3),
            activation='relu',
            input_shape=input_shape,
            name=conv_layer_name)
    network.add(conv_layer)

    pooling_layer_name = 'pooling_2'
    pooling_layer = layers.MaxPooling2D(2, 2, name=pooling_layer_name)
    network.add(pooling_layer)
    # FLATTENING
    flatten_layer_name = 'flatten'
    flatten_layer = layers.Flatten(name=flatten_layer_name)
    network.add(flatten_layer)

    # OUTPUT LAYER
    output_layer = layers.Dense(number_of_outputs,activation='softmax',name='output')
    network.add(output_layer)
    network.compile(loss='binary_crossentropy',
              optimizer=optimizers.RMSprop(lr=1e-4),
              metrics=['acc'])

    return network
def create_growing_dense_network(input_size=64,layer_size=64,output_size=10):
    network = models.Sequential()
    network.add(layers.Dense(layer_size,activation='relu',input_shape=input_size))
    network.add(layers.Dense(output_size,activation='softmax'))
    network.compile(loss='binary_crossentropy',
              optimizer=optimizers.RMSprop(lr=1e-4),
              metrics=['acc'])
    return network
def add_conv_layer(old_network,new_layer_output_size=32,number_of_outputs=1):
    from keras import models, layers
    # Function for adding a new conv layer to a network and freezing previously existing weights
    # Input: Network to be grown
    # Output: New network with one untrained layer (excluding output layer)
    new_network=models.Sequential()
    nbr_of_layers = len(old_network.layers)
    for i in range(nbr_of_layers-2): # assumes there is a flatten layer before the last dense one
        old_network.layers[i].trainable=False
        new_network.add(old_network.layers[i])
    conv_layer_name = 'conv_' + str(nbr_of_layers)
    conv_layer = layers.Conv2D(new_layer_output_size,
                                (3, 3),
                                activation='relu',
                                name=conv_layer_name)
    new_network.add(conv_layer)

    # POOLIING LAYER
    pooling_layer_name = 'pooling_' + str(nbr_of_layers+1)
    pooling_layer = layers.MaxPooling2D(2, 2, name=pooling_layer_name)
    new_network.add(pooling_layer)

    # FLATTENING
    flatten_layer_name = 'flatten'
    flatten_layer = layers.Flatten(name=flatten_layer_name)
    new_network.add(flatten_layer)

    # OUTPUT LAYER
    output_layer = layers.Dense(number_of_outputs,activation='softmax',name='output') # or sigmoid?
    new_network.add(output_layer)
    new_network.compile(loss='binary_crossentropy',
              optimizer=optimizers.RMSprop(lr=1e-4),
              metrics=['acc'])
    return new_network
def get_mnist_data_flattened(type=0):
    """
    Returns mnist
    """
    import numpy as np
    from keras.datasets import mnist
    from keras.utils import to_categorical
    # Load data
    (train_images, train_labels), (test_images,test_labels) = mnist.load_data()
    s1=np.shape(train_images)
    s2=np.shape(test_images)

    # Reformat input & labels
    train_images = train_images.reshape((s1[0], s1[1]*s1[2] ))
    train_images = train_images.astype('float32')/255

    test_images=test_images.reshape((s2[0], s2[1]*s2[2] ))
    test_images = test_images.astype('float32')/255

    print('Successfully reformated the images')

    train_labels = to_categorical(train_labels)
    test_labels = to_categorical(test_labels)
    return train_images,train_labels,test_images,test_labels
def dense_mnist_LBL(nbr_of_layers=50,epochs=5,results_path="",save_models=False,models_path=""):
    """
    Function for generating
    """
    from keras import models,layers
    import numpy as np
    train_images,train_labels,test_images,test_labels = get_mnist_data_flattened()

    network_a = models.Sequential()
    network_a.add(layers.Dense(512,activation='relu',input_shape=(28 * 28,)))
    network_a.add(layers.Dense(10,activation='softmax'))
    network_a.compile(optimizer='rmsprop',loss='categorical_crossentropy',metrics=['accuracy'])

     #nbr of internal layers
    results = []
    network_a.summary()
    network_a.fit(train_images, train_labels, epochs=epochs, batch_size=128)

    for i in range(nbr_of_layers):
        if(i%2==0):
            network_b=grow_network_mnist(network_a,32,10)
            network_b.compile(optimizer='rmsprop',loss='categorical_crossentropy',metrics=['accuracy'])
            network_b.summary()
            network_b.fit(train_images, train_labels, epochs=epochs, batch_size=128)
            if(save_models):
                model_name = "MNIST_LBL_layer"+str(i+2)+"trainedfor"+str(epochs)+"e.h5"
                network_b.save(models_path+model_name)
            results.append(network_b.evaluate(test_images,test_labels))
            del(network_a)
        else:
            network_a=grow_network_mnist(network_b)
            network_a.compile(optimizer='rmsprop',loss='categorical_crossentropy',metrics=['accuracy'])
            network_a.summary()
            network_a.fit(train_images, train_labels, epochs=epochs, batch_size=128)
            if(save_models):
                model_name = "MNIST_LBL_layer"+str(i+2)+"trainedfor"+str(epochs)+"e.h5"
                network_a.save(models_path+model_name)
            results.append(network_a.evaluate(test_images,test_labels))
            del(network_b)
            print('b')
    if(i%2==1):
        network = network_a
    else:
        network = network_b
    result_name = 'MNIST_LBL_res_'+str(epochs)+'epochs_'+str(nbr_of_layers)+'layers'
    np.save(results_path+result_name,results)
    return network
def dense_mnist_normal(nbr_of_layers=50,epochs=5,results_path="",save_models=False,models_path=""):
    """
    Function which trains a set of networks the normal way as comparsion
    """
    from keras import models,layers
    import numpy as np
    train_images,train_labels,test_images,test_labels = get_mnist_data_flattened()

    results = []
    for m in range(nbr_of_layers):
        network_traditional=models.Sequential()
        network_traditional.add(layers.Dense(512,activation='relu',input_shape=(28 * 28,)))
        for n in range(m):
            network_traditional.add(layers.Dense(32,activation='relu'))#not supposed to be here
        network_traditional.add(layers.Dense(10,activation='softmax'))
        network_traditional.summary()
        network_traditional.compile(optimizer='rmsprop',loss='categorical_crossentropy',metrics=['accuracy'])
        network_traditional.fit(train_images, train_labels, epochs=epochs, batch_size=128)
        if(save_models):
            model_name = "MNIST_Normal_layer"+str(m+2)+"trainedfor"+str(epochs)+"e.h5"
            network_traditional.save(models_path+model_name)
    result_name = 'MNIST_normal_res_'+str(epochs)+'epochs_'+str(nbr_of_layers)+'layers'
    np.save(results_path+result_name,results)
    return network_traditional
def grow_network_mnist(old_network,new_layer_size=32,output_size=10):
    new_network=models.Sequential()
    nbr_of_layers = len(old_network.layers)
    for i in range(nbr_of_layers-1):
        old_network.layers[i].trainable=False
        new_network.add(old_network.layers[i])
    new_network.add(layers.Dense(new_layer_size,activation='relu'))
    new_network.add(layers.Dense(output_size,activation='softmax'))
    return new_network

def get_partial_output(network,training_data,offset=2):
    """
    Function for getting all the outputs from early convolutional layers and putting
    it into an array.
    Potentially saving training time.
    Input:
        network - network to be analyzed (must be Sequential at this stage)
        training data - data to be converted to array/file
        offset - How many layers at the end of network one needs to skip
    Output:
        array with the result of all the training data when run through the
        network(excluding the end of it).
    """
    # Transfer network into a new model
    output_network = models.Sequential()
    assert(offset>0)
    for i in range(len(network.layers)-offset):
        print(network.layers[i].name)
        output_network.add(network.layers[i])
    output_network.compile(loss='binary_crossentropy',
              optimizer=optimizers.RMSprop(lr=1e-4),
              metrics=['acc'])

    output = output_network.predict(training_data)
    del(output_network)
    return output
# def grow_conv_network(conv_layers, dense_layers,train_generator,validation_generator,output_size = 1,nbr_epochs=20):
#     """
#         function for creating, training and growing a conv network.
#     Input :
#         training data, size of and number of convolutional and dense layers
#     Output :
#         A conv net trained using the growing technique on the train images
#     """
#     final_network = models.Sequential() # final grown network
#     nbr_conv_layers = len(conv_layers)
#     nbr_dense_layers = len(dense_layers)
#     offset = 2
#
#
#     ### Create and train conv network base
#
#     for i in range(nbr_conv_layers):
#
#         # Create and train "single" conv layer network
#         # network_a and network_b temporary networks used during training
#         if(i==0):
#             network_a = create_growing_conv_network(input_shape=(150, 150, 3),conv_base_dim=conv_layers[0],number_of_outputs=output_size)
#             # network_b = add_conv_layer(network_a,new_layer_output_size=conv_layers[1],number_of_outputs=output_size)
#             # del(network_a)
#             # network_a=network_b
#             # for l in range(len(network_a.layers)):
#             #     network_a.layers[l].trainable=True
#             # del(network_b)
#             network_a.compile(loss='binary_crossentropy',
#                   optimizer=optimizers.RMSprop(lr=1e-4),
#                   metrics=['acc'])
#         else:
#             network_b = add_conv_layer(network_a,new_layer_output_size=conv_layers[i],number_of_outputs=output_size)
#             del(network_a)
#             network_a=network_b
#             del(network_b)
#         network_a.summary()
#         history = network_a.fit_generator(
#                   train_generator,
#                   steps_per_epoch=100,
#                   epochs=nbr_epochs,
#                   validation_data=validation_generator,
#                   validation_steps=50)
#
#         ### Add conv & pooling layers to final model
#         # speed up if there is no reflow of data thorugh model
#         print(network_a.layers[-4].name)
#         final_network.add(network_a.layers[-4])
#         conv_name = 'conv_layer_'+str(i)
#         final_network.layers[-1].name=conv_name# names may not be the same
#
#         final_network.add(network_a.layers[-3])
#         pooling_name = 'pooling_layer_'+str(i)
#         final_network.layers[-1].name=pooling_name
#
#         # Transfer also flatten layer if we are at the last iteration
#         if(i==nbr_conv_layers-1):
#             final_network.add(network_a.layers[2])
#             offset = 1# want to get the flatten output
#         print("final_network:")
#         final_network.summary()
#     offset=1
#
#     ### Add and train the dense top
#     for i in range(nbr_dense_layers):
#         print(output_size)
#         network_a = create_growing_dense_network(
#                             input_size=input_shape,
#                             layer_size=dense_layers[i],
#                             output_size=output_size)
#         network_a.summary()
#         network_a.fit(prev_output, train_labels, epochs=nbr_epochs, batch_size=64)
#         ### Add dense layer to final model
#         final_network.add(network_a.layers[0])
#         dense_name = 'dense_layer_'+str(i)
#         final_network.layers[-1].name=dense_name# names may not be the same
#         if(i==nbr_dense_layers-1):
#             final_network.add(network_a.layers[1])
#         prev_output = get_partial_output(network_a,prev_output,offset=offset)
#         input_shape = prev_output.shape[1:] #shape may change
#
#     final_network.summary()
#     final_network.compile(loss='binary_crossentropy',
#               optimizer=optimizers.RMSprop(lr=1e-4),
#               metrics=['acc'])
#     return final_network
def LBL_network_classification(
        conv_layers,
        dense_layers,
        train_generator,
        validation_generator,
        input_shape=(150,150,3),
        nbr_outputs=1,
        nbr_epochs=20,
        save_results=False,
        network_name="CD",
        result_dir=""):
    from keras import Input
    input_tensor = Input(input_shape)
    final_layers_list=[]

    #Create first layer model
    conv_layer = layers.Conv2D(conv_layers[0],(3,3),activation='relu')(input_tensor)
    pooling = layers.MaxPooling2D((2,2))(conv_layer)
    output = layers.Flatten()(pooling)
    output = layers.Dense(nbr_outputs,activation="sigmoid")(output)
    model = models.Model(input_tensor,output)
    final_layers_list.append(conv_layer)
    final_layers_list.append(pooling)
    model.compile(loss='binary_crossentropy',
          optimizer=optimizers.RMSprop(lr=1e-4),
          metrics=['acc'])
    model.summary()
    history = model.fit_generator(
              train_generator,
              steps_per_epoch=100,
              epochs=nbr_epochs,
              validation_data=validation_generator,
              validation_steps=50)

    final_layers_list[-1].trainable=False
    ### add subsequent layers

    idx=1 # Counts the layer indices
    if(save_results):
        model_name = network_name+"_conv_"+str(nbr_epochs)+"e_"+str(idx)+"L"
        file_name = model_name+"results.txt"#change so that this
        print_results_to_file(model,history,file_name,model_name,path=result_dir)
        model_name = result_dir+model_name+".h5"
        model.save(model_name)

    for conv_size in conv_layers[1:]:
        idx += 1

        ### Create new layer along with pooling etc

        conv_layer = layers.Conv2D(conv_size,(3,3),activation='relu')(final_layers_list[-1])
        pooling = layers.MaxPooling2D((2,2))(conv_layer)
        flatten = layers.Flatten()(pooling)
        output = layers.Dense(nbr_outputs,activation="sigmoid")(flatten)
        model = models.Model(input_tensor,output)

        final_layers_list.append(conv_layer)
        final_layers_list.append(pooling)

        ### Set layers in model untrainable
        for i in range(len(model.layers)-4):
            model.layers[i].trainable=False

        ### compile and fit model
        model.compile(loss='binary_crossentropy',
              optimizer=optimizers.RMSprop(lr=1e-4),
              metrics=['acc'])
        model.summary()
        history = model.fit_generator(
                  train_generator,
                  steps_per_epoch=100,
                  epochs=nbr_epochs,
                  validation_data=validation_generator,
                  validation_steps=50)

        # Save reuslts to a file
        if(save_results):
            model_name = network_name+"_conv_"+str(nbr_epochs)+"e_"+str(idx)+"L"
            file_name = model_name+"results.txt"#change so that this
            print_results_to_file(model,history,file_name,model_name,path=result_dir)
            model_name = result_dir+model_name+".h5"
            model.save(model_name)

    ## Add the flatten layer to bridge from conv to dense layers
    final_layers_list.append(flatten)

    ### Add the dense layers
    for dense_size in dense_layers:
        idx += 1
        dense_layer = layers.Dense(dense_size,activation='relu')(final_layers_list[-1])
        output = layers.Dense(nbr_outputs,activation="sigmoid")(dense_layer)
        model = models.Model(input_tensor,output)
        final_layers_list.append(dense_layer)
        model = models.Model(input_tensor,output)
        for i in range(len(model.layers)-2):
            model.layers[i].trainable=False
        model.summary()
        model.compile(loss='binary_crossentropy',
              optimizer=optimizers.RMSprop(lr=1e-4),
              metrics=['acc'])
        model.summary()
        history = model.fit_generator(
                  train_generator,
                  steps_per_epoch=100,
                  epochs=nbr_epochs,
                  validation_data=validation_generator,
                  validation_steps=50)
        if(save_results):
            model_name = network_name+"_dense_"+str(nbr_epochs)+"e_"+str(idx)+"L"
            file_name = model_name+"results.txt"#change so that this
            print_results_to_file(model,history,file_name,model_name,path=result_dir)
            model_name = result_dir+model_name+".h5"

            model.save(model_name)
    return model,final_layers_list

def print_results_to_file(network,history,file_name,model_name,path=""):
    """
    Prints history and model setup

    Inputs:
        Network, network history and network/model name
    Ouputs:
        file with model summary and the data from the history
    """
    import numpy as np
    file_trad = open(path+file_name, "w")

    # Print to file
    orig_std_out = sys.stdout
    sys.stdout = file_trad
    print(network.summary())
    sys.stdout = orig_std_out
    file_trad.write(model_name+'acc = ')
    file_trad.write(str(history.history['acc']))
    file_trad.write(';\n')
    file_trad.write(model_name+'val_acc = ')
    file_trad.write(str(history.history['val_acc']))
    file_trad.write(';\n')
    file_trad.write(model_name+'loss = ')
    file_trad.write(str(history.history['loss']))
    file_trad.write(';\n')
    file_trad.write(model_name+'val_loss = ')
    file_trad.write(str(history.history['val_loss']))
    file_trad.write(';\n')
    file_trad.close()

    # Save to numpy array
    file_length = len(history.history['acc'])
    np_results = np.zeros((4,file_length))
    np_results[0,:] = history.history['acc']
    np_results[1,:] = history.history['val_acc']
    np_results[2,:] = history.history['loss']
    np_results[3,:] = history.history['val_loss']
    np_filname = path+model_name+"np_res"
    np.save(np_filname,np_results)

def get_small_CD_data_generators(augument_data=False):
    """
    Function which returns a train and a test data generator for the small CD dataset.
    Input:
        augument_data - if images are to be agumented
    Output:
        train_generator, validation generator - generator for fetching training
        and validation images.
    """
    base_dir = 'C:/Users/Simulator/Desktop/Martin Selin/layer_by_layer/small_cats_dogs'
    train_dir = os.path.join(base_dir, 'train')
    validation_dir = os.path.join(base_dir, 'validation')
    test_dir = os.path.join(base_dir, 'test')
    train_cats_dir = os.path.join(train_dir, 'cats')
    train_dogs_dir = os.path.join(train_dir, 'dogs')
    validation_cats_dir = os.path.join(validation_dir, 'cats')
    validation_dogs_dir = os.path.join(validation_dir, 'dogs')
    test_cats_dir = os.path.join(test_dir, 'cats')
    test_dogs_dir = os.path.join(test_dir, 'dogs')
    if(augument_data):
        train_datagen = ImageDataGenerator(rescale=1./255,
            rotation_range=40,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True)
    else:
        train_datagen = ImageDataGenerator(rescale=1./255)
    test_datagen = ImageDataGenerator(rescale=1./255)

    train_generator = train_datagen.flow_from_directory(
            # This is the target directory
            train_dir,
            # All images will be resized to 150x150
            target_size=(150, 150),
            batch_size=20,
            # Since we use binary_crossentropy loss, we need binary labels
            class_mode='binary')

    validation_generator = test_datagen.flow_from_directory(
            validation_dir,
            target_size=(150, 150),
            batch_size=20,
            class_mode='binary')
    return train_generator,validation_generator
