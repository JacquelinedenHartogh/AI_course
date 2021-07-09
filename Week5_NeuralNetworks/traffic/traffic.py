import cv2
import numpy as np
import os
import sys
import glob
import tensorflow as tf
from matplotlib import pyplot

from sklearn.model_selection import train_test_split

EPOCHS = 10
IMG_WIDTH = 30
IMG_HEIGHT = 30
NUM_CATEGORIES = 43
TEST_SIZE = 0.3


def main():

    # Check command-line arguments
    if len(sys.argv) not in [2, 3]:
        sys.exit("Usage: python traffic.py data_directory [model.h5]")

    # Get image arrays and labels for all image files
    images, labels = load_data(sys.argv[1])

    # Split data into training and testing sets
    labels = tf.keras.utils.to_categorical(labels)
    x_train, x_test, y_train, y_test = train_test_split(
        np.array(images), np.array(labels), test_size=TEST_SIZE
    )

    # Get a compiled neural network
    model = get_model()

    # Fit model on training data
    history = model.fit(x_train, y_train, epochs=EPOCHS)

    # Evaluate neural network performance
    train_acc = model.evaluate(x_train,  y_train, verbose=2)
    test_acc = model.evaluate(x_test,  y_test, verbose=2)    
    
    # Save model to file
    if len(sys.argv) == 3:
        filename = sys.argv[2]
        model.save(filename)
        print(f"Model saved to {filename}.")
        
    # plot loss during training
    pyplot.subplot(211)
    pyplot.title('Loss')
    pyplot.plot(history.history['loss'], label='train')
    pyplot.legend()
    # plot accuracy during training
    pyplot.subplot(212)
    pyplot.title('Accuracy')
    pyplot.plot(history.history['accuracy'], label='train')
    pyplot.tight_layout()
    pyplot.legend()
    pyplot.show()


def load_data(data_dir):
    """
    Load image data from directory `data_dir`.

    Assume `data_dir` has one directory named after each category, numbered
    0 through NUM_CATEGORIES - 1. Inside each category directory will be some
    number of image files.

    Return tuple `(images, labels)`. `images` should be a list of all
    of the images in the data directory, where each image is formatted as a
    numpy ndarray with dimensions IMG_WIDTH x IMG_HEIGHT x 3. `labels` should
    be a list of integer labels, representing the categories for each of the
    corresponding `images`.
    """
    #Initiate lists that will be returned as tuple
    images=[]
    labels=[]
    
    #Go in directoty and find all images within the directory
    #loop over all directories:
    for i in range(NUM_CATEGORIES-1):
        #add the directory to data-dir and get all the files:
        wildcard = os.path.join(data_dir, str(i), "*")
        files = glob.glob(wildcard)  
        #loop over each file 
        for j in files:
            #use cv2 to read image as np.ndarray with RGB colours (default)
            img = cv2.imread(j)
            
            #check size
            set_size=(IMG_WIDTH,IMG_HEIGHT)
            img2=cv2.resize(img,set_size)

            #append label (set to i), and image img2 
            labels.append(i)
            images.append(img2)
            
    #return tuple(images,labels)      
    return images,labels

def get_model():
    """
    Returns a compiled convolutional neural network model. Assume that the
    `input_shape` of the first layer is `(IMG_WIDTH, IMG_HEIGHT, 3)`.
    The output layer should have `NUM_CATEGORIES` units, one for each category.
    """
    model = tf.keras.models.Sequential([

    # Convolutional layer. Learn 32 filters using a 3x3 kernel
    tf.keras.layers.Conv2D(
        60, (5, 5), activation="relu", input_shape=(IMG_WIDTH, IMG_HEIGHT, 3)),
    
    #tf.keras.layers.Conv2D(
    #    60, (3, 3), activation="relu"),

    # Max-pooling layer, using 2x2 pool size
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),

    # Flatten units
    tf.keras.layers.Flatten(),    

    # Add a hidden layer with dropout
    tf.keras.layers.Dense(NUM_CATEGORIES*10, activation="relu"),
    tf.keras.layers.Dropout(0.25),  
    
    # Add a hidden layer with dropout
    tf.keras.layers.Dense(NUM_CATEGORIES*10, activation="relu"),
    tf.keras.layers.Dropout(0.25),    

    # Add an output layer with output units for all num categories
    tf.keras.layers.Dense(NUM_CATEGORIES-1, activation="sigmoid")
    ])
    
    # Train neural network
    opti=tf.keras.optimizers.RMSprop(learning_rate=4e-4)    
    model.compile(optimizer=opti, loss="categorical_crossentropy", metrics=["accuracy"])

    return model

if __name__ == "__main__":
    main()
