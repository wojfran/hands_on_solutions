# Exercise 11

Practice training a deep neural network on the CIFAR10 image dataset:

a. Build a DNN with 20 hidden layers of 100 neurons each (that’s too many, but it’s the point of this exercise). Use He initialization and the ELU activation function.

b. Using Nadam optimization and early stopping, train the network on the CIFAR10 dataset. You can load it with `keras.datasets.cifar10.load_data()`. The dataset is composed of 60,000 32 × 32–pixel color images (50,000 for training, 10,000 for testing) with 10 classes, so you’ll need a softmax output layer with 10 neurons. Remember to search for the right learning rate each time you change the model’s architecture or hyperparameters.

c. Now try adding Batch Normalization and compare the learning curves: Is it converging faster than before? Does it produce a better model? How does it affect training speed?

d. Try replacing Batch Normalization with SELU, and make the necessary adjustments to ensure the network self-normalizes (i.e., standardize the input features, use LeCun normal initialization, make sure the DNN contains only a sequence of dense layers, etc.).

e. Try regularizing the model with alpha dropout. Then, without retraining your model, see if you can achieve better accuracy using MC Dropout.

f. Retrain your model using 1cycle scheduling and see if it improves training speed and model accuracy.

## Exercise Summary

A deep neural network was trained on the CIFAR-10 image dataset, exploring various architectures and regularization techniques. The goal was to assess the performance of a deep neural network with 20 hidden layers, each containing 100 neurons.

### Library Imports

The exercise began with the importation of essential libraries, including TensorFlow, Keras, Scikit-learn, NumPy, and Matplotlib. These libraries were crucial for building, training, and evaluating the neural network.

### Logging Directory Setup

A function was defined to create a log directory for TensorBoard, enabling effective monitoring of the training process.

### Data Loading and Preprocessing

-   The CIFAR-10 dataset, consisting of 60,000 32x32-pixel color images across 10 classes, was loaded.
-   The images were reshaped and standardized using `StandardScaler` from Scikit-learn to improve model performance.
-   Labels were converted into one-hot encoded vectors for multi-class classification.
-   The dataset was split into training and validation sets, with 80% for training and 20% for validation.

### Model Building

A function `build_model` was defined to create various configurations of the deep neural network (DNN). The models consisted of:

-   **Type 0:** 20 dense layers with 100 neurons each, using He initialization and ELU activation.
-   **Type 1:** The same architecture as Type 0, but Batch Normalization was included.
-   **Type 2:** Similar to Type 0, but LeCun initialization with SELU activation was utilized.
-   **Type 3:** The same architecture as Type 2 was augmented with a custom Alpha Dropout layer for regularization.

Each model was compiled using the Nadam optimizer and categorical crossentropy as the loss function.

### Learning Rate Finder

A custom callback class `LRFinder` was implemented to search for the optimal learning rate by gradually increasing it during training and recording the corresponding loss. The learning rate that minimized loss was identified for each model architecture.

### Model Training

Models were trained using the following strategies:

-   **Early Stopping:** Training was halted if there was no improvement in validation loss for 15 consecutive epochs.
-   **TensorBoard Logging:** TensorBoard was utilized to visualize the training process.
-   **One-Cycle Scheduling:** For specific models, a OneCycleScheduler was employed to dynamically adjust the learning rate during training.

### Model Evaluation

-   Predictions were made on the test dataset, and the predicted classes were compared with the true classes to calculate the accuracy for each model.
-   The model architectures showed varying performance metrics, with some configurations achieving high accuracy.
-   Additionally, the effect of Monte Carlo (MC) Dropout was assessed by averaging predictions over multiple passes, leading to improved accuracy for certain models.

### Results

## Results

The performance of the models was evaluated using the test accuracy metric. The following results were obtained:

-   **HE_ELU**: The model achieved a test accuracy of **41.81%** after restoring the best validation loss from epoch 8.
-   **HE_ELU_OneCycle**: This model demonstrated an improved test accuracy of **49.37%**, with the best validation loss restored from epoch 17.

-   **HE_ELU_BATCH**: With the inclusion of Batch Normalization, the test accuracy increased to **53.23%**, restoring the best validation loss from epoch 25.

-   **HE_ELU_BATCH_OneCycle**: The model achieved a marginally better test accuracy of **53.30%**, based on the best validation loss from epoch 21.

-   **LECUN_SELU**: The model using LeCun initialization and SELU activation reached a test accuracy of **46.56%**, with the best validation loss restored from epoch 10.

-   **LECUN_SELU_OneCycle**: The test accuracy improved to **51.78%**, with the best validation loss from epoch 22.

-   **LECUN_SELU_ALPHA0.05**: This model recorded a test accuracy of **40.46%**, restoring the best validation loss from epoch 17.

-   **LECUN_SELU_ALPHA0.05_OneCycle**: The model exhibited an improved test accuracy of **51.97%**, based on the best validation loss from epoch 27.

Furthermore, predictions were made with and without Monte Carlo (MC) Dropout:

-   **Prediction without MC Dropout**: The model's output probabilities were as follows:
    [[0.01 0.03 0.12 0.12 0.09 0.08 0.50 0.03 0.01 0.01]]

-   **Prediction with MC Dropout Mean**: The model’s output probabilities showed the following:
    [[0.03 0.03 0.17 0.13 0.14 0.15 0.27 0.06 0.01 0.02]]

The test accuracy for the model using MC Dropout was **41.31%**, while the accuracy with the 1Cycle model was **52.57%**. These results suggest that both Batch Normalization and 1Cycle training strategies positively influenced model performance.
