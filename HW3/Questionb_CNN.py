import numpy as np
from PIL import Image
import os
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from keras.utils import to_categorical

plt.style.use('ggplot')


from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
import matplotlib.pyplot as plt
from keras.utils import plot_model
from keras.layers.normalization import BatchNormalization
from Questionb import *


def basicCNNbuild(batch_size=32, epoch=30, input_shape=(64, 64, 3), num_classes=15):
    model = Sequential()


    model.add(Conv2D(32, kernel_size=(5, 5), strides=(1, 1),
                    activation='relu',
                    input_shape=input_shape))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Conv2D(64, (5, 5), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(1000, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))

    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    return model


def train_model(model, train_generator, validation_generator, batch_size=10):
    history = model.fit_generator(
        train_generator,
        steps_per_epoch=3000//batch_size,
        epochs=10,
        validation_data=validation_generator,
        validation_steps=40 // batch_size)
    return history


def plot_loss(history):
    # summarize history for accuracy
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()


def show_model(model):
    plot_model(model, to_file='images/cnn_model.png')

def augment(X):
    datagen = ImageDataGenerator(
        featurewise_center=True,
        featurewise_std_normalization=True,
        horizontal_flip=True)
    datagen.fit(X)
    return datagen

def main():
    model = basicCNNbuild(input_shape=(243,320,1))
    print(model.summary())
    show_model(model)
    X_origin, X_flatten, label = read_faces()
    print(X_origin.shape)
    le = encode(label)
    y = le.transform(label)
    encoded_y = to_categorical(y)
    X = np.expand_dims(X_origin,axis=3)
    datagen = augment(X)
    history = model.fit(X,encoded_y,epochs=10,validation_split=0.1)
    #plot_loss(history)
    #model.fit_generator(datagen.flow(X, encoded_y, batch_size=32),
    #                   steps_per_epoch=len(X) / 32, epochs=100)

if __name__ == "__main__":
    main()
