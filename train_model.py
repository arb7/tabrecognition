# (c) 2020 Ari Ball-Burack

# Convolutional Neural Network architecture inspired by: https://towardsdatascience.com/image-classification-in-10-minutes-with-mnist-dataset-54c35b77a38d
# import cv2 Uncomment to generate example images
import glob
import argparse
import numpy as np
import tensorflow as tf


def run(lines):
    if lines:
        fname = 'saved_model/with_lines'
    else:
        fname = 'saved_model/without_lines'
    model_glob = glob.glob(fname)
    if model_glob:
        print('{} already exists -- overwrite? [y/N]'.format(fname))
        if not input() == 'y':
            return

    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

    # Reshaping the array to 4-dims so that it can work with the Keras API
    x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
    x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
    input_shape = (28, 28, 1)
    # Making sure that the values are float so that we can get decimal points after division
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')

    if lines:
        print('Adding lines...')
        for pic in x_train:
            # Uncomment the below to generate example images
            """rand = np.random.randint(100000)
            if rand < 99980:
                continue"""
            middle = int(np.random.normal(loc=pic.shape[1]/2, scale=2)+0.5)
            width = int(np.random.normal(loc=1.6, scale=.3)+0.5)
            middle, width = max(middle, 0), max(width, 0)
            while True:
                if middle-width < 0:
                    middle += 1
                    width -= 1
                    width = max(width, 0)
                elif middle+width > pic.shape[0]:
                    width = pic.shape[0]-middle
                    middle -= 1
                    width -= 1
                    width = max(width, 0)
                else:
                    break
            pic[middle-width:middle+width] = np.multiply(np.ones((2*width, pic.shape[1], 1)), 255)
            # Uncomment the below to generate example images
            # cv2.imwrite('img/classifier_eg/{}.png'.format(rand), pic)
        print('Done')
        # return For when generating example images

    # Normalizing the RGB codes by dividing it to the max RGB value.
    x_train /= 255
    x_test /= 255

    # Importing the required Keras modules containing model and layers
    from keras.models import Sequential
    from keras.layers import Dense, Conv2D, Dropout, Flatten, MaxPooling2D
    # Creating a Sequential Model and adding the layers
    model = Sequential()
    model.add(Conv2D(28, kernel_size=(3, 3), input_shape=input_shape))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())  # Flattening the 2D arrays for fully connected layers
    model.add(Dense(128, activation=tf.nn.relu))
    model.add(Dropout(0.2))
    model.add(Dense(10, activation=tf.nn.softmax))

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    model.fit(x=x_train, y=y_train, epochs=10)
    print('Saving model to {}'.format(fname))
    model.save(fname)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-l', '--lines', action='store_true',
                        help="Train model with artificial stave lines")
    args = parser.parse_args()
    run(args.lines)
