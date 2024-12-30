import os
print('This is the working directory: ', os.getcwd())

import tensorflow as tf
import datetime
import ssl
import numpy as np
from imageio import imread
from skimage.transform import resize

from tensorflow.keras import optimizers
from tensorflow.keras.models import model_from_json
from tensorflow.keras.layers import BatchNormalization
#For macOS Users:
#run on Terminal, first:    /Applications/Python\ 3.12/Install\ Certificates.command
#run on Terminal, second:    pip install --upgrade certifi
mnist = tf.keras.datasets.mnist
#For Windows computers:
#uncomment the following line:
#ssl._create_default_https_context = ssl._create_unverified_context
(x_train, y_train,), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0
#Create model:
def create_model():
    #Your code here:
    return tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape = (28,28), name = 'layers_flatten'),
        tf.keras.layers.Dense(512, activation = 'relu', name = 'layers_dense_1'),
        BatchNormalization(),
        tf.keras.layers.Dropout(0.3, name = 'layers_dropout_1'),
        tf.keras.layers.Dense(512, activation='relu', name='layers_dense_2'),
        BatchNormalization(),
        tf.keras.layers.Dropout(0.3, name='layers_dropout_2'),
        tf.keras.layers.Dense(10, activation = 'softmax', name = 'layers_output')
    ])

model = create_model()
#Compile model:
#your code here:
model.compile(optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001), loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'])
#for mac:
log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
#for Windows:
#log_dir = "logs\\fit\\" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
#tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
#Fit model:
#Your code here:
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir, histogram_freq=1)
earlystop_callback=tf.keras.callbacks.EarlyStopping(monitor='val_loss',patience=10,restore_best_weights=True)
model.fit(x = x_train, y = y_train, epochs =50, validation_data= (x_test, y_test), callbacks = [tensorboard_callback,earlystop_callback])
#Print a summary of your model:
model.summary()
#saved a trained model
model_json = model.to_json()
with open('model.json', 'w') as json_file:
    json_file.write(model_json)
model.save_weights('model.weights.h5')
#Before loading a model, comment the 'save model' code to avoid saving new models
#load a saved model:
model_architecture = 'model.json'
model_weights = 'model.weights.h5'
model = model_from_json(open(model_architecture).read())
model.load_weights(model_weights)
#load image
img_names = ['number_grayscaled.jpg','number_grayscaled.jpg']
img = [resize(imread(img_name),(28,28)).astype("float32") for img_name in img_names]
img = np.array(img) / 255
#recompile de model. The saved model needs to re-built for tensorflow
optmi = tf.keras.optimizers.SGD()
model.compile(loss = 'categorical_crossentropy', optimizer = optmi, metrics = ['accuracy'])
predict = np.argmax(model.predict(img), axis = 1)
print(predict)

'''

/Applications/Python\ 3.12/Install\ Certificates.command
 -- pip install --upgrade certifi
Requirement already satisfied: certifi in /Library/Frameworks/Python.framework/Versions/3.12/lib/python3.12/site-packages (2024.2.2)
 -- removing any existing file or link
 -- creating symlink to certifi certificate bundle
 -- setting permissions
 -- update complete.
'''

