"""
This script uses transfer learning to classify pokemon images by generation.
"""

# data tools
import os
import glob
import argparse
import numpy as np
import pandas as pd

# tf tools
import tensorflow as tf
from tensorflow.keras.preprocessing.image import (load_img,
                                                  ImageDataGenerator)
from tensorflow.keras.applications.vgg16 import (preprocess_input,
                                                 decode_predictions,
                                                 VGG16)
from tensorflow.keras.layers import (Flatten, 
                                     Dense,
                                     BatchNormalization)
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.optimizers import Adam             

#scikit-learn
from sklearn.preprocessing import LabelBinarizer, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.pipeline import Pipeline

# scikeras wrapper
from scikeras.wrappers import KerasClassifier

# for plotting
import matplotlib.pyplot as plt

#################################################################

# function that specifies the required arguments
def parse_args():
    # Initialise argparse
    ap = argparse.ArgumentParser()
    
    # command line parameters
    ap.add_argument("-d_a", "--data_augmentation", required = True, help = "specify data augmentation with 1 or 0", type = int)
    ap.add_argument("-ke_re", "--regularizer", required = False, help = "the kernel regularizer: l1, l2 or l1_l2")
    ap.add_argument("-epochs", "--epoch_num", required = False, help = "number of epochs", type = int)
    ap.add_argument("-batch_size", "--batch_size", required = False, help = "the batch size", type = int)
        
    args = vars(ap.parse_args())
    return args


# function to plot the loss and accuracy curves of the model
def plot_history(H, epochs):
    plt.style.use("seaborn-colorblind")

    plt.figure(figsize=(12,6))
    plt.subplot(1,2,1)
    plt.plot(np.arange(0, epochs), H.history["loss"], label="train_loss")
    plt.plot(np.arange(0, epochs), H.history["val_loss"], label="val_loss", linestyle=":")
    plt.title("Loss curve")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.tight_layout()
    plt.legend()

    plt.subplot(1,2,2)
    plt.plot(np.arange(0, epochs), H.history["accuracy"], label="train_acc")
    plt.plot(np.arange(0, epochs), H.history["val_accuracy"], label="val_acc", linestyle=":")
    plt.title("Accuracy curve")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.tight_layout()
    plt.legend()
    
    return 

# function to load the pokemon data
def load_data():
    # specify path of the folder containing the comparison images
    file_list = glob.glob(os.path.join('data', 'images','images', '*g'))
    file_list.sort()
    
    # fix the X values
    X = []
    for image in file_list:
        image = load_img(image)
        X.append(np.array(image)) # convert the image pixels to a numpy array

    # fix the y values
    file_path = os.path.join('data', 'pokemon_all.csv')
    df = pd.read_csv(file_path)

    y = df['Generation'].values
    y = np.array(y)
    
    return X, y


# function to split the data and binarize the labels
def split_data(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    test_size = 0.2, 
                                                    random_state = 1)
    
    X_train = np.array(X_train)
    X_test = np.array(X_test)
    
    # integers to one-hot vectors
    lb = LabelBinarizer()
    y_train = lb.fit_transform(y_train)
    y_test = lb.fit_transform(y_test)
 
    return X_train, X_test, y_train, y_test


# function to load the vgg_16 model
def load_vgg16():
    tf.keras.backend.clear_session()

    # load model without classifier layers
    model = VGG16(include_top=False, 
                  pooling='avg',
                  input_shape=(120, 120, 3)) # input shape of the images

    # mark loaded layers as not trainable
    for layer in model.layers:
        layer.trainable = False
        
    return model


# function to specify the new classifier layer 
def classifier_layer(model):
    
    args = parse_args()
    ke_re = args['regularizer']
    
    # specify kernel regularizer
    if ke_re == None:
        ke_re = 'l1'
    else:
        pass
    
    flat1 = Flatten()(model.layers[-1].output)
    bn = BatchNormalization()(flat1)
    class1 = Dense(256, 
                   activation='relu', kernel_regularizer = ke_re)(bn) 
    class2 = Dense(128, 
                   activation='relu')(class1)
    output = Dense(7, 
                   activation='softmax')(class2)
    
    # define new model
    model = Model(inputs = model.inputs,
                  outputs = output)
    
    # compile model
    model.compile(optimizer="adam",
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])    
    
    return model


# function to fit the model to the data
def fit_model(model, X_train, X_test, y_train, y_test):
    
    # parse arguments
    args = parse_args()
    data_aug = args['data_augmentation']
    
    epoch_num = args['epoch_num']
    batchsize = args['batch_size']
    ke_re = args['regularizer']
    
    # specify standard values of parameters
    if epoch_num == None:
        epoch_num = 10
    else:
        pass
    
    if batchsize == None:
        batchsize = 32
    else:
        pass
    
    if ke_re == None:
        ke_re = 'l1'
    else:
        pass
    
    # fit model with or without data augmentation 
    if data_aug == 1:
        da = 'DA' # variable for output names
        
        # create augmentations
        datagen = ImageDataGenerator(horizontal_flip=True, 
                                     rotation_range=30,
                                     brightness_range=[0.2,1.5],
                                     zoom_range=[0.5, 1.0])

        datagen.fit(X_train)
        
        # fits the model on batches with real-time data augmentation:
        H = model.fit(datagen.flow(X_train, y_train,
                                   batch_size = batchsize),
                                   validation_data = (X_test, y_test),
                                   epochs = epoch_num,
                                   verbose = 2)
        
    elif data_aug == 0: 
        da = ''
        
        # train the model
        H = model.fit(X_train, y_train, 
                      validation_data = (X_test, y_test), 
                      batch_size = batchsize, 
                      epochs = epoch_num,
                      verbose = 2)
    else:
        print('specify whether to use data augmentation: 1 = true, 0 = false')
        return
    
    predictions = model.predict(X_test, batch_size=32)
    
    # plot and save model history
    plot_history(H, epoch_num)
    fig = plt.gcf()
    fig.savefig(f'output/plot_{ke_re}_{da}_{epoch_num}_{batchsize}.png')
    
    
    # initialize label names for pokemon dataset
    labelNames = ['gen 1', 'gen 2', 'gen 3', 'gen 4', 'gen 5', 'gen 6', 'gen 7']
    
    report = classification_report(y_test.argmax(axis=1),
                                predictions.argmax(axis=1),
                                target_names=labelNames)
    
    cm = pd.DataFrame(confusion_matrix(y_test.argmax(axis=1),
                              predictions.argmax(axis=1)), 
                              index=labelNames, columns=labelNames)
    
    print(report) # display the classification report 
    
    # save the classification report along with the confusion matrix
    with open(f"output/report_{ke_re}_{da}_{epoch_num}_{batchsize}.txt", "w") as f:
        print(report, file=f)
        print(cm, file = f)
    
    return 


def main():
    
    # load data
    X, y = load_data()
    X_train, X_test, y_train, y_test = split_data(X, y)        
               
    #load model
    model = load_vgg16()
    model = classifier_layer(model)
               
    # fit process
    fit_model(model, X_train, X_test, y_train, y_test)
    
    print('success')         
    return


if __name__ == '__main__':
    main()