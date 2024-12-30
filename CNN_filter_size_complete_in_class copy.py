# Import modules (libraries)



def cnn_with_tensorboard(filter_size=3, epochs=5):
    
    # Load the MNIST dataset
    (train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()
    
    # Preprocess

    

    # Create a Sequential model
    
    
    # First Convolutional Layer
    
    
    # Second Convolutional Layer
    #64 filters, filter_size x filtersize
    
    
    # Flatten the output and add Dense layers
    

    # Compile the model
    
    
    # TensorBoard log directory
    log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1, profile_batch='10,20')

    #for mac:
    #log_dir = "logs_summary/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

    
    # Train the model
    model.summary()
    model.fit(train_images, train_labels, epochs=epochs, 
              validation_data=(test_images, test_labels), 
              callbacks=[tensorboard_callback])
    
    # Evaluate the model on test data
    test_loss, test_acc = model.evaluate(test_images, test_labels)
    print(f'Test accuracy: {test_acc}')
    
    return model

# Example of how to call the function with filter size = 5 and 10 epochs
trained_model = cnn_with_tensorboard(filter_size=5, epochs=10)


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
    
    
    # Make the prediction
    
    
    # Get the predicted class label (the class with the highest probability)
   
    
    # Make return:


# Example usage:
# Assuming `trained_model` is the trained CNN model and you have an image at 'my_image.png'
img_path = 'number_grayscaled.jpg'
predicted_label = make_prediction_on_image(trained_model, img_path)
print(f"The predicted label for the image is: {predicted_label}")