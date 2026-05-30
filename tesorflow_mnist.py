import tensorflow as tf 
import keras
import numpy as np
import matplotlib.pyplot as plt

(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
print(f'train: {x_train.shape}, test: {x_test.shape}')

x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

# Split validation data
from sklearn.model_selection import train_test_split
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.1, random_state=42)
print(f'train: {x_train.shape}, validation: {x_val.shape}, test: {x_test.shape}')

model = keras.Sequential([
    keras.layers.Flatten(input_shape= (28, 28)),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dropout(0.2),
    keras.layers.Dense(10, activation='softmax')


])

model.compile(
    optimizer='adam', 
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

history = model.fit(
    x_train, y_train, 
    epochs=5,
    batch_size=32,
    validation_data=(x_val, y_val),
    verbose=1
)
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
print(f'Test accuracy: {test_acc:.4f}')

# Validation testing
val_loss, val_acc = model.evaluate(x_val, y_val, verbose=0)
print(f'Validation accuracy: {val_acc:.4f}')

prediction = model.predict(x_test[:5])
print('Predicted:', np.argmax(prediction, axis=1))
print('Actual:    ', y_test[:5])

plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label= 'Train')
plt.plot(history.history['val_accuracy'], label= 'Val')
plt.title('Accuracy'); plt.legend()
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train')
plt.plot(history.history['val_loss'], label='Val')
plt.title('Loss'); plt.legend()
plt.tight_layout
plt.show()

# Save both model and weights
print("\n--- Saving Model and Weights ---")
model.save('mnist_model.h5')  # Save entire model
print("✓ Model saved as: mnist_model.h5")

model.save_weights('mnist.weights.h5')  # Save only weights
print("✓ Weights saved as: mnist.weights.h5")


