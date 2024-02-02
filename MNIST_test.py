import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import numpy as np

def load_dataset():
    keras.datasets.mnist.load_data(path="mnist.npz")

    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    # reshape dataset to have a single channel
    trainX = x_train.reshape((x_train.shape[0], 28, 28, 1))
    testX = x_test.reshape((x_test.shape[0], 28, 28, 1))

    # one hot encode target values
    trainY =  keras.utils.to_categorical(y_train)
    testY =  keras.utils.to_categorical(y_test)

    print('dataset downloaded')
    return trainX, trainY, testX, testY

def normalization(train,test):
    # convert from integers to floats
    train_norm = train.astype('float32')
    test_norm = test.astype('float32')

    # normalize to range 0-1
    train_norm = train_norm / 255.0
    test_norm = test_norm / 255.0

    return train_norm, test_norm

def create_model():
    #define model
    model = keras.models.Sequential()
    model.add(keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', input_shape=(28, 28, 1)))
    model.add(keras.layers.MaxPooling2D((2, 2)))
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(100, activation='relu', kernel_initializer='he_uniform'))
    model.add(keras.layers.Dense(10, activation='softmax'))

    #compile model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model

def train_model(model,x_train, y_train,x_test, y_test):
    scores, histories = list(), list()
    n_epochs = 3
    print('Learning statrted...')
    history = model.fit(x_train, y_train, epochs=n_epochs, batch_size=32, validation_data=(x_test, y_test), verbose=2)
    print('Learning ended...')
    # Evaluate the model
    score = model.evaluate(x_test, y_test, verbose=0)
    print(score)
    scores.append(score)
    histories.append(history)
    return scores, histories

# plot diagnostic learning curves
def summarize_diagnostics(histories):
    for i in range(len(histories)):
        # plot loss
        plt.subplot(2, 1, 1)
        plt.title('Cross Entropy Loss')
        plt.plot(histories[i].history['loss'], color='blue', label='train')
        plt.plot(histories[i].history['val_loss'], color='orange', label='test')
        # plot accuracy
        plt.subplot(2, 1, 2)
        plt.title('Classification Accuracy')
        plt.plot(histories[i].history['accuracy'], color='blue', label='train')
        plt.plot(histories[i].history['val_accuracy'], color='orange', label='test')
        plt.legend()
        plt.show()

# summarize model performance
def summarize_performance(scores):
    # print summary
    print('Accuracy: mean=%.3f std=%.3f, n=%d' % (np.mean(scores)*100, np.std(scores)*100, len(scores)))
    # box and whisker plots of results
    plt.boxplot(scores)
    plt.show()

def main():
    train_x, train_y, test_x, test_y = load_dataset()

    #normalization
    train_x,train_y = normalization(train_x,train_y)

    model = create_model()

    scores, histories = train_model(model,train_x,train_y,test_x,test_y)

    # learning curves
    summarize_diagnostics(histories)
    # summarize estimated performance
    summarize_performance(scores)

main()