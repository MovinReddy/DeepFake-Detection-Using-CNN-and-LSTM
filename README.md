Here’s a `README.md` section for your GitHub repository that explains the project you’ve shared. It covers data loading, model building, training, and inference:

---

# DeepFake Video Detection using CNN-LSTM

This project implements a DeepFake detection model using a combination of **Convolutional Neural Networks (CNN)** for spatial feature extraction and **Long Short-Term Memory (LSTM)** networks for capturing temporal patterns across video frames.

## 📁 Dataset

The model is trained on the **DFD (DeepFake Detection) dataset** consisting of two folders:

* `DFD_original_sequences`: Contains real videos.
* `DFD_manipulated_sequences`: Contains fake/deepfake videos.

To avoid class imbalance, a maximum of 100 fake videos are used, and real videos are duplicated to balance the dataset.

## 📦 Data Preparation

* Each video is processed by extracting **10 frames**.
* Each frame is resized to **128x128 pixels**.
* Labels: `0` for real, `1` for fake.
* Data is normalized to \[0, 1].
* Labels are one-hot encoded.
* Data is split into **80% training** and **20% testing**.

```python
X, y = load_data(video_dir, labels, num_frames=10, img_size=(128, 128))
X = X / 255.0
y = tf.keras.utils.to_categorical(y, num_classes=2)
```

## 🧠 Model Architecture

The model uses:

* **TimeDistributed Conv2D** layers to extract frame-wise spatial features.
* **LSTM** layer to learn temporal relationships across frames.
* **Fully connected layers** for classification.

```python
model.add(TimeDistributed(Conv2D(32, (3, 3), activation='relu')))
model.add(LSTM(128, return_sequences=False))
model.add(Dense(2, activation='softmax'))
```

### Loss & Optimizer

* Loss: `categorical_crossentropy`
* Optimizer: `Adam` with a learning rate of `0.0001`

## 🏋️ Training

* **Epochs**: 20
* **Batch size**: 16
* **Validation**: 20% test split from original data

```python
history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=20, batch_size=16)
```

## 🧪 Inference

The `preprocess_video()` function extracts and normalizes frames from a given video. The model can then predict whether the input is real or fake.

```python
video = preprocess_video(video_path)
video = np.expand_dims(video, axis=0)  # Add batch dimension
prediction = model.predict(video)
```

## 📝 Output

The model outputs class probabilities for two classes: **Real (0)** and **Fake (1)**.
