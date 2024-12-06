# Exercise 12

Implement a custom layer that performs Layer Normalization (to be used in Chapter 15):

## Exercise Instructions

a. The `build()` method should define two trainable weights, α and β, both of shape `input_shape[-1:]` and data type `tf.float32`. α should be initialized with 1s, and β with 0s.

b. The `call()` method should compute the mean μ and standard deviation σ of each instance’s features. For this, you can use `tf.nn.moments(inputs, axes=-1, keepdims=True)`, which returns the mean μ and the variance σ² of all instances (compute the square root of the variance to get the standard deviation). The function should compute and return \( \alpha \otimes \frac{(X - \mu)}{(\sigma + \epsilon)} + \beta \), where ⊗ represents itemwise multiplication (*) and ε is a smoothing term (a small constant to avoid division by zero, e.g., 0.001).

c. Ensure that your custom layer produces the same (or very similar) results as the built-in LayerNormalization layer in Keras.

## Exercise Summary

In this exercise, a custom layer for Layer Normalization was implemented and tested against Keras's built-in LayerNormalization layer to ensure functionality and performance consistency.

### Library Imports

The exercise began with the importation of essential libraries, including TensorFlow, Keras, NumPy, and Matplotlib. These libraries were crucial for building, training, and evaluating the custom layer.

### Custom Layer Implementation

A custom layer named `MyNormalization` was created, which includes:

- **Weights Initialization:** Two trainable weights (α and β) are defined in the `build()` method. 
  - α is initialized to 1s.
  - β is initialized to 0s.

- **Mean and Standard Deviation Calculation:** The `call()` method computes the mean and standard deviation of the input features using `tf.nn.moments()`. The normalized output is calculated as specified in the exercise instructions.

- **Activation Function:** An optional activation function can be applied to the output of the normalization.

### Model Building and Testing

Two sequential models were constructed to compare the custom layer with Keras's built-in LayerNormalization layer:

1. **Model with Custom Layer:**
   - Input layer with shape `[10]`.
   - Custom normalization layer `MyNormalization`.
   - Dense output layer with a sigmoid activation function.

2. **Model with Built-in LayerNormalization:**
   - Input layer with shape `[10]`.
   - Keras's `LayerNormalization`.
   - Dense output layer with a sigmoid activation function.

### Model Training

Both models were compiled using the Adam optimizer and binary crossentropy as the loss function. They were trained on random data for 10 epochs.

### Results Visualization

The predictions from both models were plotted to visually compare their outputs, showcasing their similarities.

## Conclusion

The custom Layer Normalization layer was successfully implemented and tested, demonstrating functionality comparable to Keras's built-in LayerNormalization. This exercise solidified the understanding of creating custom layers in TensorFlow/Keras.
