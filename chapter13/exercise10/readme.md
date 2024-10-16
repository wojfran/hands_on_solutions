# Exercise: IMDB Sentiment Analysis with Pooling and Dataset Loading Methods

In this exercise you will download a dataset, split it, create a `tf.data.Dataset` to load it and preprocess it efficiently, then build and train a binary classification model containing an Embedding layer:

a. Download the Large Movie Review Dataset, which contains 50,000 movie reviews from the Internet Movie Database. The data is organized in two directories, `train` and `test`, each containing a `pos` subdirectory with 12,500 positive reviews and a `neg` subdirectory with 12,500 negative reviews. Each review is stored in a separate text file. There are other files and folders (including preprocessed bag-of-words), but we will ignore them in this exercise.

b. Split the test set into a validation set (15,000) and a test set (10,000).

c. Use `tf.data` to create an efficient dataset for each set.

d. Create a binary classification model, using a `TextVectorization` layer to preprocess each review. If the `TextVectorization` layer is not yet available (or if you like a challenge), try to create your own custom preprocessing layer: you can use the functions in the `tf.strings` package, for example `lower()` to make everything lowercase, `regex_replace()` to replace punctuation with spaces, and `split()` to split words on spaces. You should use a lookup table to output word indices, which must be prepared in the `adapt()` method.

e. Add an `Embedding` layer and compute the mean embedding for each review, multiplied by the square root of the number of words (see Chapter 16). This rescaled mean embedding can then be passed to the rest of your model.

f. Train the model and see what accuracy you get. Try to optimize your pipelines to make training as fast as possible.

g. Use TFDS to load the same dataset more easily: `tfds.load("imdb_reviews")`.

---

## Exercise Summary

In this exercise, we trained deep neural networks on the IMDB sentiment analysis dataset, exploring different pooling strategies and dataset loading methods. The results of these comparisons were stored in a log file.

### Library Imports

The exercise began by importing essential libraries such as TensorFlow, Keras, and TensorFlow Datasets (TFDS), which are necessary for building, training, and evaluating models efficiently.

### Data Loading and Preprocessing

-   The IMDB dataset was loaded using two methods: TensorFlow Datasets (TFDS) and manual loading.
-   Reviews were tokenized and vectorized into sequences of word indices.
-   The dataset was split into training, validation, and test sets.
-   Preprocessing functions were defined for both loading methods, ensuring consistent data input.

### Model Building

Two Keras models were built to classify IMDB reviews into positive and negative sentiment:

-   **Pooling Methods**:
    -   **GlobalAveragePooling1D**: Aggregates features across the sequence dimension.
    -   **Custom Mean Embedding**: A function to compute the mean embedding for each sequence.
-   **Model Architecture**:
    -   An `Embedding` layer to convert words into dense vector representations.
    -   Multiple `Conv1D` layers followed by batch normalization and activation functions.
    -   Dropout layers to prevent overfitting.
    -   A dense output layer with sigmoid activation for binary classification.

### Efficient Input Pipeline

-   `tf.data` API was used to create efficient datasets for training and evaluation.
-   Datasets were loaded either via TFDS or preprocessed manually. Both methods used similar batching strategies and augmentations.

### Model Training

The models were trained with the following strategies:

-   **TensorBoard Logging**: Visualized training metrics, such as loss and accuracy.
-   **Early Stopping**: Training was stopped if no improvement in validation loss was observed for 10 epochs.
-   **Learning Rate Scheduling**: The learning rate was reduced by 50% when validation loss plateaued for 5 epochs.

### Model Evaluation

After training, predictions were made on the test set, and the accuracy and loss were computed. Comparisons were made between the pooling techniques and dataset loading methods.

### Results

The test results were saved in a text file in the `logs` directory. Below are the contents of the file:

imdb_global_average_pooling_tfds.keras Accuracy: 0.8055999875068665 Loss: 0.9872898459434509  
imdb_global_average_pooling_manual.keras Accuracy: 0.8232799768447876 Loss: 0.9350327253341675  
imdb_early_mean_embedding_tfds.keras Accuracy: 0.8371000289916992 Loss: 0.9382879137992859  
imdb_early_mean_embedding_manual.keras Accuracy: 0.8202400207519531 Loss: 0.9754735827445984

---

NETWORK PARAMETERS:

---

No. of epochs: 100  
Embedding dim: 64  
Learning rate: 0.0001  
L2 reg: 0.02  
Network width factor: 1  
Batch size: 16  
Early stopping patience: 10  
LR scheduler factor: 0.5  
LR scheduler patience: 5

---

imdb_global_average_pooling_tfds.keras Accuracy: 0.829200029373169 Loss: 0.9291868209838867  
imdb_global_average_pooling_manual.keras Accuracy: 0.8223999738693237 Loss: 0.937877357006073  
imdb_early_mean_embedding_tfds.keras Accuracy: 0.8282999992370605 Loss: 0.9244011640548706  
imdb_early_mean_embedding_manual.keras Accuracy: 0.8276000022888184 Loss: 0.8928740620613098

---

NETWORK PARAMETERS:

---

No. of epochs: 100  
Embedding dim: 64  
Learning rate: 0.0001  
L2 reg: 0.02  
Network width factor: 2  
Batch size: 16  
Early stopping patience: 10  
LR scheduler factor: 0.5  
LR scheduler patience: 5

---

imdb_global_average_pooling_tfds.keras Accuracy: 0.8198999762535095 Loss: 0.9075161218643188  
imdb_global_average_pooling_manual.keras Accuracy: 0.826960027217865 Loss: 0.9428495168685913  
imdb_early_mean_embedding_tfds.keras Accuracy: 0.826200008392334 Loss: 0.9521739482879639  
imdb_early_mean_embedding_manual.keras Accuracy: 0.8248800039291382 Loss: 0.9450625777244568

---

NETWORK PARAMETERS:

---

No. of epochs: 100  
Embedding dim: 64  
Learning rate: 0.0001  
L2 reg: 0.02  
Network width factor: 3  
Batch size: 16  
Early stopping patience: 10  
LR scheduler factor: 0.5  
LR scheduler patience: 5

## Conclusion

This exercise demonstrated how different pooling methods and dataset loading techniques impact model performance on the IMDB sentiment analysis task. TensorBoard helped visualize training, and early stopping and learning rate scheduling improved model optimization. The results show minor differences in accuracy and loss, suggesting room for further optimization.
