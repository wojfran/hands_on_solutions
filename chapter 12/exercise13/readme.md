# Exercise 13

In this exercise, a custom training loop was implemented to train a model on the Fashion MNIST dataset, exploring the effects of different optimizers and learning rates.

## Exercise Instructions

a. Display the epoch, iteration, mean training loss, and mean accuracy over each epoch (updated at each iteration), along with the validation loss and accuracy at the end of each epoch.

b. Experiment with different optimizers and learning rates for the upper layers and lower layers of the model.

## Exercise Summary

The exercise involved building a neural network model using Keras to classify images from the Fashion MNIST dataset. The implementation focused on training the model using a custom training loop, allowing for more granular control over the training process.

### Library Imports

Essential libraries, including TensorFlow, Keras, NumPy, Matplotlib, and Scikit-learn, were imported for data manipulation, model building, and evaluation.

### Data Loading and Preprocessing

- The Fashion MNIST dataset was loaded, consisting of 60,000 training images and 10,000 testing images of size 28x28 pixels.
- The images were reshaped and standardized using `StandardScaler` to enhance model performance.
- Labels were one-hot encoded for multi-class classification, transforming them into categorical format.
- The dataset was split into training and validation sets, with 80% for training and 20% for validation.

### Model Building

A sequential Keras model was constructed with the following layers:

- Input layer with shape `(784,)` (flattened images).
- Three hidden layers with 300, 100, and 50 neurons, respectively, using ELU activation and He initialization.
- Output layer with 10 neurons and softmax activation for multi-class classification.

### Custom Training Loop

The custom training loop included the following components:

- **Batch Sampling:** A function was implemented to randomly sample batches of data for training.
- **Loss and Metrics Calculation:** The categorical crossentropy loss was computed, and mean loss and accuracy were tracked throughout training.
- **Optimizers:** Two different optimizers were utilized:
  - Adam optimizer with a learning rate of 0.001 for the upper layers.
  - SGD optimizer with a learning rate of 0.01 for the lower layers.

### Training Process

The training process involved iterating through epochs and steps, performing the following actions:

- For each batch, the forward pass was computed using the model.
- The loss was calculated, and gradients were computed using `tf.GradientTape()`.
- Gradients were applied separately for the upper and lower layers using the defined optimizers.
- Mean training loss and accuracy were updated iteratively and displayed at each step.
- Validation loss and accuracy were computed and displayed at the end of each epoch.

### Results

The model's training performance was monitored, displaying training loss and accuracy metrics at each iteration, alongside validation metrics at the end of each epoch. This provided insight into the model's learning process and its generalization capabilities.

## Conclusion

The custom training loop effectively trained a model on the Fashion MNIST dataset, demonstrating the flexibility of using different optimizers and learning rates for different layers. The exercise highlighted the importance of monitoring training and validation metrics to ensure model performance.
