# Bach Chorales Music Generation

## Project Overview

This project involves training a deep learning model to generate Bach-like chorales based on the **JSB Chorales dataset**, which consists of 382 chorales composed by Johann Sebastian Bach. The dataset contains sequences of time steps, where each time step includes four integers representing the notes being played. The goal of the model is to predict the next time step (i.e., four notes) given a sequence of previous time steps from a chorale.

### Dataset Information

-   The dataset consists of **382 chorales**, each of which contains **100 to 640 time steps**.
-   Each time step contains **four integers** representing note indices on a piano, except for the value `0`, which signifies that no note is played.

You can download the dataset from [this link](https://github.com/ageron/handson-ml2/raw/master/datasets/jsb_chorales/jsb_chorales.tgz). Unzip the dataset after downloading.

### Project Tasks

1. **Data Preparation**:

    - The dataset is processed by loading CSV files, each containing chorale sequences.
    - The chorales are divided into **windows of time steps** (where each window contains several chords). Each chord represents four notes.
    - The chords (time steps) are **flattened into a single vector** before being fed into the model using the `preprocess` function:
        ```python
        def preprocess(window):
            window = tf.where(window == 0, window, window - min_note + 1)
            return tf.reshape(window, [-1])
        ```
        - In the preprocessing step, any note with a value of `0` (indicating silence) is left unchanged, while the other notes are shifted by subtracting the `min_note` value. Then, the 4-note chord is reshaped into a single 1D array.
    - **Target Creation**: The model is trained to predict the next time step (next chord). This is achieved by shifting the window of time steps by one:
        ```python
        def create_target(batch):
            X = batch[:, :-1]
            Y = batch[:, 1:]
            return X, Y
        ```
        - Here, the input (`X`) consists of all but the last time step, while the target (`Y`) is shifted by one time step, making the model learn to predict the next chord.
    - **Data Loading**: The dataset is loaded into TensorFlow using the `load_chorales_dataset` function, which:
        - Reads the chorales from CSV files and converts them into windows of time steps.
        - Preprocesses each window and maps it to the target time step (for training).
        - Batches the windows and creates TensorFlow datasets for training, validation, and testing.
        - Example usage of the dataset loader:
        ```python
        train = load_chorales_dataset('jsb_chorales/train')
        valid = load_chorales_dataset('jsb_chorales/valid')
        test = load_chorales_dataset('jsb_chorales/test')
        ```
        This results in three datasets: `train`, `valid`, and `test`, which are preprocessed and ready for training the model.

2. **Model Design**:

    - A deep learning model is built using a combination of layers, including:
        - An **embedding layer** to convert note indices into dense vectors.
        - Multiple **1D convolutional layers** with batch normalization and dilation to capture patterns in time sequences.
        - A **GRU (Gated Recurrent Unit)** layer to model the temporal relationships in the chorale sequences.
        - A **dense output layer** with softmax activation to predict the next time step.

3. **Training**:

    - The model is trained using **sparse categorical cross-entropy** as the loss function and **Nadam** as the optimizer.
    - **Early stopping** and **learning rate scheduling** are implemented to avoid overfitting and optimize training.
    - TensorBoard is used for monitoring training logs.

4. **Chorale Generation**:

    - Once the model is trained, it can generate new chorales by predicting the next time step given a sequence of previous time steps.
    - The model starts with a sequence of notes (seed chords) and generates additional notes by predicting one time step at a time. The newly generated notes are appended to the input, and the process is repeated to create a full-length chorale.
    - The generated chorales are saved as **MIDI files** and can be played back using a MIDI player.

5. **MIDI Playback**:
    - A function is implemented to convert the generated chorale into a playable MIDI file.
    - The playback of the generated chorale is handled using the **pygame.midi** library, allowing for real-time audio output.

### Exercise Instructions

> "Download the Bach chorales dataset and unzip it. It is composed of 382 chorales composed by Johann Sebastian Bach. Each chorale is 100 to 640 time steps long, and each time step contains 4 integers, where each integer corresponds to a note’s index on a piano (except for the value 0, which means that no note is played).  
> Train a model—recurrent, convolutional, or both—that can predict the next time step (four notes), given a sequence of time steps from a chorale. Then use this model to generate Bach-like music, one note at a time: you can do this by giving the model the start of a chorale and asking it to predict the next time step, then appending these time steps to the input sequence and asking the model for the next note, and so on.  
> Also make sure to check out Google’s Coconet model, which was used for a nice Google doodle about Bach."

## Conclusion

This project demonstrates how deep learning models can be applied to generate classical music in the style of Johann Sebastian Bach. By training a model on the Bach chorales dataset and using it to predict and generate new chorales, we can explore the creative potential of neural networks in the field of music composition.
