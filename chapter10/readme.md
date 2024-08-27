### Exercise 10

Train a deep MLP on the MNIST dataset (you can load it using keras.datasets.mnist.load_data()). See if you can get over 98% precision. Try searching for the optimal learning rate by using the approach presented in this chapter (i.e., by growing the learning rate exponentially, plotting the loss, and finding the point where the loss shoots up). Try adding all the bells and whistles—save checkpoints, use early stopping, and plot learning curves using TensorBoard.

## Exercise Summary:

### Library Imports:

The exercise began with the importation of essential libraries, including TensorFlow, Keras, Scikit-learn, NumPy, and Matplotlib. These libraries were fundamental for building, training, and visualizing the neural network.

### Logging Directory Setup:

A function was defined to create a log directory for TensorBoard. This setup allowed for effective monitoring of the training process.

### Data Loading and Preprocessing:

The MNIST dataset, containing 70,000 images of handwritten digits, was loaded.
To prepare for multi-class classification, the labels (Y) were converted into one-hot encoded vectors. This involved creating zero matrices and setting the appropriate indices based on the labels.
Train-Validation Split: The dataset was split into training and validation sets using train_test_split from Scikit-learn. Eighty percent of the data was allocated for training, while the remaining twenty percent was reserved for validation.

### Model Building:

A function was defined to build a CNN model, specifying the architecture, which included convolutional layers, max pooling, flattening, and dense layers. ReLU activation was used for the hidden layers, and softmax activation was employed for the output layer.
The model was then compiled using the Stochastic Gradient Descent (SGD) optimizer and categorical crossentropy as the loss function.
Learning Rate Finder:

To identify an optimal learning rate, a custom callback class called LRFinder was implemented. This class was designed to determine the best learning rate by gradually increasing it during training and recording the corresponding loss.
The model was trained with the LRFinder callback to identify the learning rate that minimized loss.
Training the Model:

With the learning rate determined from the previous step, the model was initialized, and callbacks for model checkpointing and TensorBoard logging were set up.
The model was trained for up to 300 epochs, utilizing early stopping based on validation loss. This meant that training would halt if there was no improvement after 10 epochs.

### Model Evaluation:

Upon completion of training, predictions were made on the test dataset, and the accuracy was calculated by comparing the predicted classes to the true classes. The final test accuracy was printed to evaluate the model's performance. The model has shown that it's able to achieve over 98% accuracy.
