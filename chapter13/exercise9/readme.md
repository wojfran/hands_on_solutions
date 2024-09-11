# Exercise 9

Practice loading and preprocessing the Fashion MNIST dataset with TensorFlow:

a. Load the Fashion MNIST dataset (introduced in Chapter 10); split it into a training set, a validation set, and a test set; shuffle the training set; and save each dataset to multiple TFRecord files. Each record should be a serialized Example protobuf with two features: the serialized image (use `tf.io.serialize_tensor()` to serialize each image), and the label. For large images, you could use `tf.io.encode_jpeg()` instead. This would save a lot of space, but it would lose a bit of image quality.

b. Then use `tf.data` to create an efficient dataset for each set. Finally, use a Keras model to train these datasets, including a preprocessing layer to standardize each input feature. Try to make the input pipeline as efficient as possible, using TensorBoard to visualize profiling data.

## Exercise Summary

A deep neural network was trained on the Fashion MNIST dataset, exploring various methods to efficiently load and preprocess image data for training.

### Library Imports

The exercise began with the importation of essential libraries, including TensorFlow, Keras, NumPy, and Matplotlib. These libraries were crucial for building, training, and evaluating the neural network.

### Data Loading and Preprocessing

-   The Fashion MNIST dataset was loaded, containing 60,000 grayscale images of clothing items.
-   The dataset was split into training, validation, and test sets. 
-   The training set was shuffled to improve model training effectiveness.
-   Each dataset was saved to multiple TFRecord files, with each record containing serialized image data and labels.

### Model Building

A Keras model was constructed to classify the Fashion MNIST images. The model architecture included:

-   A preprocessing layer to standardize the input features.
-   Convolutional layers followed by activation functions, batch normalization, and dropout layers to enhance generalization.
-   A dense output layer with 10 neurons corresponding to the 10 classes of the Fashion MNIST dataset.

### Efficient Input Pipeline

-   The TFRecord files were utilized to create efficient datasets using `tf.data`.
-   Custom functions were defined to read and preprocess the data from TFRecord format, ensuring that the images were reshaped and normalized appropriately.

### Model Training

The model was trained using the following strategies:

-   **TensorBoard Logging:** TensorBoard was utilized to visualize the training process and monitor metrics such as loss and accuracy.
-   **Early Stopping:** Training was halted if there was no improvement in validation loss for a specified number of epochs.

### Model Evaluation

-   Predictions were made on the test dataset, and the predicted classes were compared with the true classes to calculate the accuracy of the model.
-   The effect of various hyperparameters on model performance was assessed, leading to insights on the effectiveness of the input pipeline and model architecture.

### Results

The performance of the model was evaluated using the test accuracy metric. The following results were obtained:

-   The model achieved a test accuracy of **90.92%** after training and validation.

The results demonstrated the impact of an efficient input pipeline on model training speed and accuracy.

## Conclusion

The model was trained on the Fashion MNIST dataset, and the performance was evaluated on a separate test set. The input pipeline was optimized, and profiling data was visualized using TensorBoard to ensure effective monitoring of the training process.
