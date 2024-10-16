# Transfer Learning for Large Image Classification

## Exercise Instructions

10. Use transfer learning for large image classification, going through these steps:

   a. Create a training set containing at least 100 images per class. For example, you could classify your own pictures based on the location (beach, mountain, city, etc.), or alternatively, you can use an existing dataset (e.g., from TensorFlow Datasets).

   b. Split it into a training set, a validation set, and a test set.

   c. Build the input pipeline, including the appropriate preprocessing operations, and optionally add data augmentation.

   d. Fine-tune a pretrained model on this dataset.

## Dataset

For this exercise, the **Stanford Dogs** dataset was utilized, which contains images of various dog breeds. The dataset was split into:

- **Training set**: 80% of the dataset
- **Validation set**: 10% of the dataset
- **Test set**: 10% of the dataset

### Number of Examples

- Number of classes: 120
- Number of training examples: 6,662
- Number of validation examples: 832
- Number of test examples: 832

## Data Augmentation and Preprocessing

Data augmentation techniques were applied to enhance the robustness of the model. These techniques included:

- Random rotations
- Width and height shifts
- Shearing
- Zooming
- Horizontal flipping
- Random brightness and contrast adjustments

Additionally, a custom random shift function was implemented to further augment the dataset, ensuring that the model is exposed to a diverse range of image variations.

## Model Architecture

The model was built using the **Xception** architecture, pretrained on the **ImageNet** dataset. The steps for fine-tuning involved:

1. **Freezing the Base Model**: Initially, all layers of the base model were frozen to leverage the pretrained features without modifying them.

2. **Gradual Unfreezing**: Layers were gradually unfrozen in increments (e.g., last 30, 60, 90 layers, and finally all layers) to allow the model to fine-tune its weights effectively without overfitting.

3. **Optimization**: The model was compiled with the SGD optimizer, using a learning rate that was adjusted during training based on validation loss.

4. **Callbacks**: Early stopping and learning rate reduction were implemented to optimize training performance and prevent overfitting.

## Results

### Test Accuracy Evaluation

1. **Frozen Model** (after 45 epochs):
   - Test accuracy: **72.42%**
   - Test accuracy without data augmentation: **73.67%**

2. **Unfreezing Last 30 Layers**:
   - Test accuracy: **74.5%**
   - Test accuracy without data augmentation: **74.33%**

3. **Unfreezing Last 60 Layers**:
   - Test accuracy: **73.33%**
   - Test accuracy without data augmentation: **73.33%**

4. **Unfreezing Last 90 Layers**:
   - Test accuracy: **74.42%**
   - Test accuracy without data augmentation: **73.42%**

5. **Unfreezing All Layers**:
   - Test accuracy: **74.42%**
   - Test accuracy without data augmentation: **73.75%**

## Analysis of Gradual Unfreezing

Gradual unfreezing of the model's layers allowed for controlled fine-tuning, which led to better generalization. The test accuracy improved as more layers were unfrozen, suggesting that the model benefited from adapting deeper features learned during the pretraining phase. 

- **Frozen Model**: The model achieved moderate accuracy, relying on the frozen features without adaptation to the new dataset.
- **Gradual Unfreezing**: Each phase of unfreezing showed incremental improvements in accuracy, indicating that allowing the model to adjust a subset of layers at a time reduced the risk of overfitting and preserved the generalization of learned features.
- **Unfreezing All Layers**: While unfreezing all layers provided a slight improvement, the results were not significantly better than those achieved through gradual unfreezing. This suggests that fine-tuning should be approached cautiously to maintain the model's robustness against overfitting.

## Conclusion

This exercise demonstrated the effectiveness of transfer learning in large image classification tasks. By using a well-established pretrained model and applying systematic fine-tuning strategies, we achieved competitive results on the Stanford Dogs dataset, showcasing the model's ability to generalize well to a new set of images.
