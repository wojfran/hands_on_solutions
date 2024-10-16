# Exercise 9: Build Your Own CNN from Scratch on MNIST

In this exercise, the goal was to build a Convolutional Neural Network (CNN) from scratch to achieve the highest possible accuracy on the MNIST dataset.

## Exercise Instructions
1. **Download the MNIST dataset**: The dataset consists of 60,000 training images and 10,000 test images of handwritten digits.
2. **Preprocessing**: Normalize the pixel values of images and one-hot encode the labels.
3. **Build a CNN Model**: Create a CNN using TensorFlow and Keras. The architecture should include Conv2D layers, MaxPooling, Dense layers, and Dropout for regularization. The optimizer, learning rate, and regularization techniques should be fine-tuned to maximize performance.
4. **Train the Model**: Train the model on the training set and validate it on a validation set created by splitting the original training data.
5. **Evaluate the Model**: Evaluate the model on the test dataset and save the accuracy and loss results.
6. **Save Results**: Save the model and the training logs, including the evaluation results in a text file.

## Model Summary

Several CNN architectures were experimented with, each yielding different results. The best model achieved over **99% accuracy** on the MNIST test dataset.

### Model Architectures and Results

1. **Model**: `c8_c32_pooling_drop0.15_128_drop0.3_l2_0.01`  
   - **Accuracy**: 0.9608  
   - **Loss**: 11.9786  

2. **Model**: `c32_c64_c128_pooling_drop0.15_128_drop0.3`  
   - **Accuracy**: 0.9917  
   - **Loss**: 7.3692  

3. **Model**: `c32_c64_pooling_drop0.15_128_drop0.3`  
   - **Accuracy**: 0.9919  
   - **Loss**: 6.2599  

4. **Model**: `pooling_c8_c32_drop0.3_l2_0.01`  
   - **Accuracy**: 0.9605  
   - **Loss**: 15.9462  

5. **Model**: `pooling_c16_c32_c64_d128_drop0.3_l20.02`  
   - **Accuracy**: 0.9014  
   - **Loss**: 26.4359  

### Training Details

- **Epochs**: 100
- **Batch Size**: 64
- **Learning Rate**: 1e-3
- **L2 Regularization**: 0.01 to 0.02 (depending on the model)
- **Dropout Rate**: 0.15 to 0.3
- **Callbacks**: 
  - EarlyStopping (patience=5, restores the best weights)
  - Learning Rate Scheduler (factor=0.5, patience=3)
  - TensorBoard for logging

### Code Summary

The CNN models were built using `keras.Sequential()` with layers including:
- Convolutional layers (`Conv2D` with varying filters and kernel sizes)
- MaxPooling layers (`MaxPooling2D`)
- Dropout layers to prevent overfitting
- Dense layers with ReLU activation, followed by a softmax output layer

All models were trained using the Adam optimizer with categorical cross-entropy loss.

### Results Storage

All models were saved in separate directories, and the results (accuracy and loss) were stored in a `results.txt` file within the log directories.

### Best Model
- **Model**: `c32_c64_pooling_drop0.15_128_drop0.3`
- **Test Accuracy**: 0.9919
- **Test Loss**: 6.2599

The best-performing model achieved nearly perfect accuracy, demonstrating the effectiveness of the chosen architecture and training strategy.

## Conclusion

This exercise showcased the importance of architectural experimentation when building CNNs for image classification tasks. By tuning hyperparameters like learning rate, dropout, and regularization, we were able to achieve excellent accuracy on the MNIST dataset.
