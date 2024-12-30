# Import modules (libraries)
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras import layers, models
import datetime
import numpy as np
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator


def cnn_with_tensorboard_and_tuning(filter_size=3, epochs=10):
    # Load the MNIST dataset
    (train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()
    # Preprocess
    train_images = train_images.reshape((60000, 28, 28, 1)).astype('float32') / 255
    test_images = test_images.reshape((10000, 28, 28, 1)).astype('float32') / 255
    #  Data Augmentation
    datagen = ImageDataGenerator(
        rotation_range=20,  
        width_shift_range=0.2,
        height_shift_range=0.2,
        zoom_range=0.2,  
        shear_range=0.2,
        fill_mode='nearest'  
    )
    datagen.fit(train_images)
    # Create a Sequential model
    # First Convolutional layer
    model = tf.keras.models.Sequential()
    model.add(layers.Conv2D(32, (filter_size, filter_size), activation='relu', padding='same',kernel_regularizer=tf.keras.regularizers.l2(0.001), input_shape=(28, 28, 1),))
    model.add(layers.BatchNormalization())  
    model.add(layers.MaxPooling2D((2, 2)))
    # Second Convolutional Layer
    model.add(layers.Conv2D(64, (filter_size, filter_size), activation='relu', padding='same',kernel_regularizer=tf.keras.regularizers.l2(0.001)))
    model.add(layers.BatchNormalization())  
    model.add(layers.MaxPooling2D((2, 2)))
    # Third Convolutional Layer
    model.add(layers.Conv2D(128, (filter_size, filter_size), activation='relu', padding='same',kernel_regularizer=tf.keras.regularizers.l2(0.001)))
    model.add(layers.BatchNormalization())  
    model.add(layers.MaxPooling2D((2, 2)))
    # Flatten and Dense layer
    model.add(layers.Flatten())
    model.add(layers.Dense(128, activation='relu'))  
    model.add(layers.Dropout(0.6))  
    model.add(layers.Dense(64, activation='relu'))  
    model.add(layers.Dropout(0.6))  
    model.add(layers.Dense(10, activation='softmax'))
    # Compile the model 
    lr_schedule = tf.keras.optimizers.schedules.CosineDecay(
        initial_learning_rate=0.001, decay_steps=1000, alpha=0.1)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule), 
              loss='sparse_categorical_crossentropy', 
              metrics=['accuracy'])

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule), 
                  loss='sparse_categorical_crossentropy', 
                  metrics=['accuracy'])

    # TensorBoard log directory
    log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

    # Early stopping with reduced patience
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, min_lr=0.00001)

    # Train the model
    model.summary()
    model.fit(datagen.flow(train_images, train_labels, batch_size=32), 
              epochs=epochs, 
              validation_data=(test_images, test_labels), 
              callbacks=[early_stopping, reduce_lr, tensorboard_callback])
    test_loss, test_acc = model.evaluate(test_images, test_labels)
    print(f'Test accuracy: {test_acc}')
    
    return model
# Preprocess function to prepare an image for prediction
def preprocess_image(img_path):
    # Load the image from file
    img = image.load_img(img_path, color_mode='grayscale', target_size=(28, 28))
    # Convert the image to a numpy array
    img_array = image.img_to_array(img)
    # Reshape the array to (1, 28, 28, 1) for prediction
    img_array = np.reshape(img_array, (1, 28, 28, 1))
    # Normalize pixel values to the range [0, 1]
    img_array = img_array / 255.0
    return img_array
def make_prediction_on_image(model, img_path):
    # Preprocess the image
    preprocessed_image = preprocess_image(img_path)
    # Make the prediction
    prediction = model.predict(preprocessed_image)
    predicted_label = np.argmax(prediction, axis=1)[0]
    return predicted_label
trained_model = cnn_with_tensorboard_and_tuning(filter_size=3, epochs=10)

# Example usage:
# Assuming `trained_model` is the trained CNN model and you have an image at 'my_image.png'
img_path = 'number_grayscaled.jpg' 
predicted_label = make_prediction_on_image(trained_model, img_path)
print(f"The predicted label for the image is: {predicted_label}")
