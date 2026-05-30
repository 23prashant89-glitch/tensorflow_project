import tensorflow as tf
import keras
import numpy as np
import matplotlib.pyplot as plt

# Load MNIST data
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
x_test = x_test.astype('float32') / 255.0

print("=" * 60)
print("LOADING SAVED MODELS AND WEIGHTS")
print("=" * 60)

# METHOD 1: Load entire model
print("\n1️⃣  Loading Complete Model (mnist_model.h5)...")
loaded_model = keras.models.load_model('mnist_model.h5')
print("✓ Model loaded successfully!")
print(f"   Model summary:")
loaded_model.summary()

# Evaluate loaded model
print("\n   Testing loaded model on test data:")
test_loss, test_acc = loaded_model.evaluate(x_test, y_test, verbose=0)
print(f"   ✓ Test Accuracy: {test_acc:.4f}")

# Make predictions with loaded model
predictions_model = loaded_model.predict(x_test[:10], verbose=0)
print(f"   ✓ Predictions on first 10 samples:")
print(f"     Predicted: {np.argmax(predictions_model, axis=1)}")
print(f"     Actual:    {y_test[:10]}")

# METHOD 2: Load only weights into a new model
print("\n" + "=" * 60)
print("2️⃣  Loading Only Weights (mnist.weights.h5)...")
print("   Creating new model architecture...")

new_model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dropout(0.2),
    keras.layers.Dense(10, activation='softmax')
])

new_model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

print("   Loading weights from file...")
new_model.load_weights('mnist.weights.h5')
print("✓ Weights loaded successfully!")

# Evaluate weights-loaded model
print("\n   Testing weight-loaded model on test data:")
test_loss2, test_acc2 = new_model.evaluate(x_test, y_test, verbose=0)
print(f"   ✓ Test Accuracy: {test_acc2:.4f}")

# Make predictions with weights-loaded model
predictions_weights = new_model.predict(x_test[:10], verbose=0)
print(f"   ✓ Predictions on first 10 samples:")
print(f"     Predicted: {np.argmax(predictions_weights, axis=1)}")
print(f"     Actual:    {y_test[:10]}")

# Verify both models are identical
print("\n" + "=" * 60)
print("3️⃣  Verification: Are both models identical?")
print("=" * 60)

all_same = True
for i, (w1, w2) in enumerate(zip(loaded_model.get_weights(), new_model.get_weights())):
    if np.allclose(w1, w2):
        print(f"   ✓ Layer {i}: Weights match!")
    else:
        print(f"   ✗ Layer {i}: Weights DON'T match!")
        all_same = False

if all_same:
    print("\n✅ Both models are IDENTICAL! Weights loaded correctly.")
else:
    print("\n⚠️  Models differ - check file integrity")

# Visualize predictions
print("\n" + "=" * 60)
print("4️⃣  Visualizing predictions...")
print("=" * 60)

fig, axes = plt.subplots(2, 5, figsize=(12, 5))
fig.suptitle('Model Predictions on Test Samples', fontsize=14, fontweight='bold')

for i in range(10):
    ax = axes[i // 5, i % 5]
    ax.imshow(x_test[i], cmap='gray')
    pred = np.argmax(predictions_model[i])
    actual = y_test[i]
    color = 'green' if pred == actual else 'red'
    ax.set_title(f'Pred: {pred} | Actual: {actual}', color=color, fontweight='bold')
    ax.axis('off')

plt.tight_layout()
plt.savefig('model_predictions.png', dpi=100, bbox_inches='tight')
print("✓ Visualization saved as: model_predictions.png")
plt.show()

print("\n" + "=" * 60)
print("✅ ALL MODELS LOADED AND TESTED SUCCESSFULLY!")
print("=" * 60)
