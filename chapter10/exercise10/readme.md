### Exercise 10

Train a deep MLP on the MNIST dataset (you can load it using `keras.datasets.mnist.load_data()`). See if you can get over 98% precision. Try searching for the optimal learning rate by using the approach presented in this chapter (i.e., by growing the learning rate exponentially, plotting the loss, and finding the point where the loss shoots up). Try adding all the bells and whistles—save checkpoints, use early stopping, and plot learning curves using TensorBoard.

## Exercise Summary:

### Library Imports:

The exercise began with the importation of essential libraries, including TensorFlow, Keras, Scikit-learn, NumPy, Matplotlib, and Scipy. These libraries were fundamental for building, training, augmenting, and visualizing the neural network.

### Logging Directory Setup:

A function was defined to create a log directory for TensorBoard, allowing effective monitoring of the training process.

### Data Loading and Preprocessing:

-   The MNIST dataset, containing 70,000 images of handwritten digits, was loaded.
-   The labels were converted into one-hot encoded vectors to prepare for multi-class classification.
-   The dataset was split into training and validation sets using `train_test_split` from Scikit-learn, with 80% for training and 20% for validation.
-   The training dataset was augmented using the `artificialAugmentation` function, significantly increasing the training data size.

### Model Building:

A function `build_model` was defined to create a Convolutional Neural Network (CNN) model. The model architecture included:

-   Convolutional layers with L2 regularization to prevent overfitting.
-   Max pooling layers to reduce the spatial dimensions.
-   Dropout layers to further prevent overfitting by randomly dropping neurons during training.
-   A dense layer with softmax activation for multi-class classification.

The model was compiled using the Adam optimizer and categorical crossentropy as the loss function.

### Learning Rate Finder:

A custom callback class `LRFinder` was implemented to search for the optimal learning rate by gradually increasing it during training and recording the corresponding loss. The learning rate that minimized loss was identified and used for further training.

### Model Training:

With the optimal learning rate determined, the model was re-initialized and trained with the following enhancements:

-   **Checkpoints:** Model checkpoints were saved to preserve the best model during training.
-   **Early Stopping:** Training was halted early if there was no improvement in validation loss for 10 consecutive epochs.
-   **TensorBoard Logging:** TensorBoard was used to visualize the training process.
-   **Learning Rate Scheduler:** A callback was used to reduce the learning rate when the validation loss plateaued.

### Model Evaluation:

-   After training, predictions were made on the test dataset.
-   The predicted classes were compared with the true classes to calculate the model's accuracy.
-   The model achieved over 99% accuracy on the test set, demonstrating strong performance.
